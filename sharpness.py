"""Curvature diagnostics: sharpness and batch sharpness.

All functions take a loss `loss_fn(params, X, Y, mask)` returning the average loss on (X, Y), where `mask` is a
per-neuron dropout mask (all ones means no dropout).

- Sharpness: largest eigenvalue of the Hessian of the full-batch loss without dropout, computed by Lanczos (scipy's
  eigsh) on Hessian-vector products.
- Batch sharpness (Andreyev and Beneventano, 2024, Definition 3):
      E_B [ g_B^T H_B g_B / ||g_B||^2 ],   g_B = grad L_B(params),  H_B = Hessian of L_B at params,
  where B is a minibatch drawn as in training. We also compute the variant where the expectation is over both the
  minibatch B and a dropout mask M, with L_{B,M} the minibatch loss of the network whose neurons are masked by M
  (that is, the loss whose gradient is the standard dropout update).
"""

from functools import partial

import numpy as np
from jax import numpy as jnp
from jax import grad, jit, jvp, lax, random
from jax.flatten_util import ravel_pytree
from scipy.sparse.linalg import ArpackError, LinearOperator, eigsh


def _tree_vdot(u, v):
    return sum(jnp.vdot(x, y) for x, y in zip(u, v))


def _hvp(f, params, v):
    """Hessian-vector product of the scalar function f at params, in direction v."""
    return jvp(grad(f), (params,), (v,))[1]


@partial(jit, static_argnames=["loss_fn"])
def _sum_loss_hvp(loss_fn, params, v, X, Y):
    """HVP of the summed (not averaged) loss on (X, Y), without dropout."""
    mask = jnp.ones(params[0].shape[0])
    return _hvp(lambda prm: X.shape[0] * loss_fn(prm, X, Y, mask), params, v)


def sharpness(loss_fn, params, X, Y, chunk_size=4096, v0=None, tol=1e-3):
    """Largest eigenvalue of the Hessian of the average loss on (X, Y), without dropout.

    The Hessian-vector product is accumulated over chunks of `chunk_size` samples to bound memory.
    `v0` (flat vector) warm-starts Lanczos, typically with the eigenvector returned at the previous measurement.
    Returns (eigenvalue, eigenvector as a flat numpy array).
    """
    flat_params, unravel = ravel_pytree(params)
    n = X.shape[0]

    def matvec(v):
        v = unravel(jnp.asarray(v.ravel(), dtype=flat_params.dtype))
        out = jnp.zeros_like(flat_params)
        for start in range(0, n, chunk_size):
            hv = _sum_loss_hvp(
                loss_fn, params, v, X[start : start + chunk_size], Y[start : start + chunk_size]
            )
            out = out + ravel_pytree(hv)[0]
        return np.asarray(out / n)

    op = LinearOperator((flat_params.size, flat_params.size), matvec=matvec, dtype=np.float32)
    try:
        eigvals, eigvecs = eigsh(op, k=1, which="LA", v0=v0, tol=tol)
    except ArpackError as error:  # e.g. identically zero Hessian when all ReLU neurons are inactive
        print(f"Warning: Lanczos failed ({error}), sharpness set to NaN")
        return float("nan"), None
    return float(eigvals[0]), eigvecs[:, 0]


@partial(jit, static_argnames=["loss_fn", "batch_size", "n_batches", "d", "m"])
def batch_sharpness(loss_fn, params, full_dataset, batch_size, p, key, n_batches, d, m):
    """Monte Carlo estimate of the batch sharpness, with dropout rate p (p=0 gives the minibatch-only version).

    Minibatches and masks are drawn as in the training steps: `batch_size` samples without replacement from
    `full_dataset` (features in the first d columns, label in the last), and a mask of m i.i.d. Bernoulli(1-p)
    entries rescaled by 1/(1-p). Returns (mean over the n_batches draws, standard error of the mean).
    """

    def directional_curvature(key):
        subkeys = random.split(key, 2)
        batch = random.choice(subkeys[0], full_dataset, shape=(batch_size,), replace=False)
        batch_X = batch[:, :d]
        batch_y = batch[:, -1]
        mask = random.bernoulli(subkeys[1], 1 - p, shape=(m,)) / (1 - p)
        f = lambda prm: loss_fn(prm, batch_X, batch_y, mask)
        g = grad(f)(params)
        return _tree_vdot(g, _hvp(f, params, g)) / _tree_vdot(g, g)

    values = lax.map(directional_curvature, random.split(key, n_batches))
    return jnp.mean(values), jnp.std(values) / jnp.sqrt(n_batches)
