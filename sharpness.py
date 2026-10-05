"""Curvature diagnostics: sharpness and batch sharpness.

All functions take a loss `loss_fn(params, X, Y, mask)` returning the average loss on (X, Y), where `mask` is a
per-neuron dropout mask (all ones means no dropout).

- Sharpness: largest eigenvalue of the Hessian of the full-batch loss without dropout, computed by a fixed number of
  Lanczos iterations on Hessian-vector products, on the device.
- Batch sharpness (Andreyev and Beneventano, 2024, Definition 3):
      E_B [ g_B^T H_B g_B / ||g_B||^2 ],   g_B = grad L_B(params),  H_B = Hessian of L_B at params,
  where B is a minibatch drawn as in training. We also compute the variant where the expectation is over both the
  minibatch B and a dropout mask M, with L_{B,M} the minibatch loss of the network whose neurons are masked by M
  (that is, the loss whose gradient is the standard dropout update).
"""

from functools import partial

from jax import numpy as jnp
from jax import grad, jit, jvp, lax, random, vmap
from jax.flatten_util import ravel_pytree


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


def lanczos_top_eigenpair(matvec, v0, n_iter):
    """Largest eigenvalue and eigenvector of a symmetric operator, by n_iter Lanczos iterations from v0.

    The Krylov basis is fully reorthogonalized (twice, for float32). Stops early if the Krylov space is invariant,
    e.g. for a zero operator, in which case the eigenvalue is exactly 0.
    """
    basis = [v0 / jnp.linalg.norm(v0)]
    alphas, betas = [], []
    for j in range(n_iter):
        w = matvec(basis[j])
        alphas.append(jnp.vdot(w, basis[j]))
        B = jnp.stack(basis)
        for _ in range(2):
            w = w - B.T @ (B @ w)
        beta = jnp.linalg.norm(w)
        if j == n_iter - 1 or beta <= 1e-6 * jnp.max(jnp.abs(jnp.array(alphas))):
            break
        betas.append(beta)
        basis.append(w / beta)
    tridiagonal = (
        jnp.diag(jnp.array(alphas))
        + jnp.diag(jnp.array(betas), 1)
        + jnp.diag(jnp.array(betas), -1)
    )
    eigvals, eigvecs = jnp.linalg.eigh(tridiagonal)
    return float(eigvals[-1]), eigvecs[:, -1] @ jnp.stack(basis)


def sharpness(loss_fn, params, X, Y, n_iter=15, chunk_size=4096, v0=None):
    """Largest eigenvalue of the Hessian of the average loss on (X, Y), without dropout.

    The Hessian-vector product is accumulated over chunks of `chunk_size` samples to bound memory.
    `v0` (flat vector) warm-starts Lanczos, typically with the eigenvector returned at the previous measurement;
    by default the start is a fixed random vector. Returns (eigenvalue, eigenvector as a flat array).
    """
    flat_params, unravel = ravel_pytree(params)
    n = X.shape[0]

    def matvec(v):
        v = unravel(v)
        out = jnp.zeros_like(flat_params)
        for start in range(0, n, chunk_size):
            hv = _sum_loss_hvp(
                loss_fn, params, v, X[start : start + chunk_size], Y[start : start + chunk_size]
            )
            out = out + ravel_pytree(hv)[0]
        return out / n

    if v0 is None:
        v0 = random.normal(random.PRNGKey(0), flat_params.shape, flat_params.dtype)
    return lanczos_top_eigenpair(matvec, v0, n_iter)


@partial(jit, static_argnames=["loss_fn", "batch_size", "n_batches", "d", "m", "vmap_size"])
def batch_sharpness(loss_fn, params, full_dataset, batch_size, p, key, n_batches, d, m, vmap_size=16):
    """Monte Carlo estimate of the batch sharpness, with dropout rate p (p=0 gives the minibatch-only version).

    Minibatches and masks are drawn as in the training steps: `batch_size` samples without replacement from
    `full_dataset` (features in the first d columns, label in the last), and a mask of m i.i.d. Bernoulli(1-p)
    entries rescaled by 1/(1-p). The draws are processed in vectorized groups of `vmap_size`.
    Returns (mean over the n_batches draws, standard error of the mean).
    """

    def draw(key):
        subkeys = random.split(key, 2)
        indices = random.choice(subkeys[0], full_dataset.shape[0], shape=(batch_size,), replace=False)
        mask = random.bernoulli(subkeys[1], 1 - p, shape=(m,)) / (1 - p)
        return indices, mask

    def directional_curvature(indices_and_mask):
        indices, mask = indices_and_mask
        batch = full_dataset[indices]
        batch_X = batch[:, :d]
        batch_y = batch[:, -1]
        f = lambda prm: loss_fn(prm, batch_X, batch_y, mask)
        g = grad(f)(params)
        return _tree_vdot(g, _hvp(f, params, g)) / _tree_vdot(g, g)

    # All draws at once (cheap when vectorized), then the curvatures in groups of vmap_size.
    draws = vmap(draw)(random.split(key, n_batches))
    values = lax.map(directional_curvature, draws, batch_size=vmap_size)
    return jnp.mean(values), jnp.std(values) / jnp.sqrt(n_batches)
