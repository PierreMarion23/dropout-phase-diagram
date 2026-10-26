import argparse
from functools import partial
import os

import jax
from jax import numpy as jnp
from jax import jit, vmap, value_and_grad
from jax import random
import numpy as np
import pickle
import tensorflow as tf
import time

from sharpness import batch_sharpness, sharpness

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    choices=["binary47", "full"],
    default="binary47",
    help="binary47: digits 4 vs 7 with logistic loss (paper setting); full: 10 classes with cross-entropy loss",
)
parser.add_argument(
    "--sharpness-every",
    type=int,
    default=200,
    help="compute sharpness and batch sharpness every this many steps (multiple of 100; 0 disables)",
)
parser.add_argument(
    "--sharpness-repeats",
    type=int,
    default=None,
    help="compute the curvature diagnostics only for the first this many repeats; default: all",
)
parser.add_argument(
    "--sharpness-n-samples",
    type=int,
    default=None,
    help="number of training samples (fixed subset) for the full-batch sharpness; default: whole training set",
)
parser.add_argument(
    "--lanczos-iters",
    type=int,
    default=15,
    help="number of Lanczos iterations (Hessian-vector products) for the sharpness, warm-started from the previous one",
)
parser.add_argument(
    "--batch-sharpness-n-batches",
    type=int,
    default=256,
    help="number of minibatches (and masks) in the Monte Carlo estimate of the batch sharpness",
)
parser.add_argument(
    "--variants",
    nargs="+",
    choices=[
        "zero",
        "dropout",
        "row",
        "forward",
        "for-back",
        "penalty",
        "row-penalty",
        "forward-penalty",
        "for-back-penalty",
    ],
    default=["zero", "dropout", "row", "forward", "for-back", "penalty", "for-back-penalty"],
    help="training variants to run (see the step functions); for-back-penalty is the right-hand side of the "
    "decomposition of the dropout update in Section 4.1 of the paper",
)
parser.add_argument(
    "--penalty-coefficient",
    choices=["exact", "paper"],
    default="exact",
    help="coefficient of the penalty term (see grad_penalty): exact, (1 + eta_i)(eta_i - eta~_i), for which each "
    "neuron moves as under dropout; paper, eta_i (eta_i - eta~_i), as written in Section 4.1 of the submitted paper",
)
parser.add_argument(
    "--num-steps",
    type=int,
    default=40000,
    help="number of training steps (Figure 2 of the paper: 40000, Figure 7: 20000)",
)
parser.add_argument("--output", default=None, help="path of the pickled logs")
args = parser.parse_args()
TASK = args.task
OUT_SHAPE = () if TASK == "binary47" else (10,)  # shape of the network output

# dataset = tf.keras.datasets.mnist.load_data()
# train = dataset[0]
# test = dataset[1]
print("before loading")
with np.load("mnist.npz") as f:
    train = (f["x_train"], f["y_train"])
    test = (f["x_test"], f["y_test"])


def preprocess(x, y):
    """Flatten images to [0, 1]-valued vectors; for binary47, keep digits 4 and 7 with labels +1 and -1."""
    x = x.reshape((x.shape[0], -1)) / 255
    y = y.astype(float)
    if TASK == "binary47":
        keep = (y == 4) | (y == 7)
        x, y = x[keep], np.where(y[keep] == 4, 1.0, -1.0)
    return x, y


train_x, train_y = preprocess(*train)
test_x, test_y = preprocess(*test)

val_size = 2000
train_x = random.permutation(random.PRNGKey(0), train_x, axis=0)
train_y = random.permutation(random.PRNGKey(0), train_y, axis=0)
val_x, val_y = train_x[:val_size], train_y[:val_size]
train_x_new, train_y_new = train_x[val_size:], train_y[val_size:]

full_dataset = jnp.concatenate([train_x_new, train_y_new.reshape(-1, 1)], axis=1)


def neuron_mask(mask, arr):
    """Reshape a per-neuron vector of shape (m,) to broadcast against arr of shape (m, ...)."""
    return mask.reshape(mask.shape + (1,) * (arr.ndim - 1))


def network(x, params, mask):
    a, b = params
    m = a.shape[0]
    a = a * neuron_mask(mask, a)
    return 1 / m * jnp.maximum(b @ x, 0) @ a


def loss_from_output(out, y):
    """Logistic loss (binary47, y in {-1, 1}) or cross-entropy loss (full, y in {0, ..., 9})."""
    if TASK == "binary47":
        return jnp.log(1 + jnp.exp(-y * out))
    return jax.nn.logsumexp(out) - out[y.astype(int)]


def logistic_loss(params, x, y, mask):
    return loss_from_output(network(x, params, mask), y)


def accuracy(params, x, y):
    out = network(x, params, jnp.ones(params[0].shape[0]))
    if TASK == "binary47":
        return y * out > 0.0
    return jnp.argmax(out) == y


batched_logistic_loss = vmap(logistic_loss, in_axes=[None, 0, 0, None])
batched_accuracy = vmap(accuracy, in_axes=[None, 0, 0])


@jit
def avg_logistic_loss(params, X, Y, mask):
    n = X.shape[0]
    return 1 / n * jnp.sum(batched_logistic_loss(params, X, Y, mask))


@jit
def avg_accuracy(params, X, Y):
    n = X.shape[0]
    return 1 / n * jnp.sum(batched_accuracy(params, X, Y))


def init_params(d, m, key):
    subkeys = random.split(key, 2)
    a = random.normal(subkeys[0], (m,) + OUT_SHAPE)
    b = random.normal(subkeys[1], (m, d)) / jnp.sqrt(d)
    return (a, b)


batched_network = vmap(network, in_axes=[0, None, None])


@jit
def grad_fb(params, X, Y, for_mask, back_mask):
    """Average gradient where dl/df is evaluated at the output of the network masked by for_mask,
    and df/dparams is the derivative of the network masked by back_mask."""
    n = X.shape[0]
    dl_df = vmap(jax.grad(loss_from_output))(batched_network(X, params, for_mask), Y)
    _, df_dparams_vjp = jax.vjp(lambda prm: batched_network(X, prm, back_mask), params)
    return df_dparams_vjp(dl_df / n)[0]


@partial(jit, static_argnames=["batch_size", "p", "d", "m"])
def sgd_step(params, full_dataset, batch_size, p, key, step_size, m, d):
    subkeys = random.split(key, 2)
    batch = random.choice(subkeys[0], full_dataset, shape=(batch_size,), replace=False)
    batch_X = batch[:, :d]
    batch_y = batch[:, -1]
    mask = random.bernoulli(subkeys[1], 1 - p, shape=(m,)) / (1 - p)
    loss_value, grads = value_and_grad(avg_logistic_loss)(
        params, batch_X, batch_y, mask
    )
    a, b = params
    da, db = grads
    return loss_value, (
        a - step_size * da,
        b - step_size * db,
    )


@partial(jit, static_argnames=["batch_size", "p", "d", "m"])
def masked_sgd_step(params, full_dataset, batch_size, p, key, step_size, m, d):
    subkeys = random.split(key, 2)
    batch = random.choice(subkeys[0], full_dataset, shape=(batch_size,), replace=False)
    batch_X = batch[:, :d]
    batch_y = batch[:, -1]
    mask = random.bernoulli(subkeys[1], 1 - p, shape=(m,)) / (1 - p)
    loss_value, grads = value_and_grad(avg_logistic_loss)(
        params, batch_X, batch_y, jnp.ones(m)
    )
    a, b = params
    da, db = grads
    return loss_value, (
        a - step_size * neuron_mask(mask, da) * da,
        b - step_size * neuron_mask(mask, db) * db,
    )


@partial(jit, static_argnames=["batch_size", "p", "d", "m"])
def fb_sgd_step(params, full_dataset, batch_size, p, key, step_size, m, d):
    subkeys = random.split(key, 3)
    batch = random.choice(subkeys[0], full_dataset, shape=(batch_size,), replace=False)
    batch_X = batch[:, :d]
    batch_y = batch[:, -1]
    for_mask = random.bernoulli(subkeys[1], 1 - p, shape=(m,)) / (1 - p)
    back_mask = random.bernoulli(subkeys[2], 1 - p, shape=(m,)) / (1 - p)
    a, b = params
    da, db = grad_fb(params, batch_X, batch_y, for_mask, back_mask)
    loss_value = avg_logistic_loss(params, batch_X, batch_y, back_mask)
    return loss_value, (a - step_size * da, b - step_size * db)


@partial(jit, static_argnames=["batch_size", "p", "d", "m"])
def forward_sgd_step(params, full_dataset, batch_size, p, key, step_size, m, d):
    subkeys = random.split(key, 2)
    batch = random.choice(subkeys[0], full_dataset, shape=(batch_size,), replace=False)
    batch_X = batch[:, :d]
    batch_y = batch[:, -1]
    for_mask = random.bernoulli(subkeys[1], 1 - p, shape=(m,)) / (1 - p)
    back_mask = jnp.ones(m)
    a, b = params
    da, db = grad_fb(params, batch_X, batch_y, for_mask, back_mask)
    loss_value = avg_logistic_loss(params, batch_X, batch_y, back_mask)
    return loss_value, (a - step_size * da, b - step_size * db)


def neuron_outputs(x, params):
    """Outputs phi_i(x) = a_i relu(b_i . x) of the individual neurons, of shape (m,) + OUT_SHAPE."""
    a, b = params
    return neuron_mask(jnp.maximum(b @ x, 0), a) * a


@jit
def grad_penalty(params, X, Y, for_mask, back_mask):
    """Penalty term of the decomposition of the dropout update (Section 4.1 of the paper), as a gradient.

    With eta = back_mask - 1, eta~ = for_mask - 1 and f~ the output of the network masked by for_mask, the step
    moves neuron i by
        - step_size / m * w_i * 1/n sum_s Dphi_i(x_s)^T [l'(f~_s + (eta_i - eta~_i) phi_i(x_s) / m) - l'(f~_s)],
    where l' is the derivative of the loss with respect to the output. For the squared loss, this is
        - step_size / m^2 * w_i (eta_i - eta~_i) * 1/n sum_s Dphi_i(x_s)^T phi_i(x_s).
    - exact (default), w_i = 1 + eta_i: added to the update of PN + RaM with the same masks, the term replaces the
      forward mask of neuron i by its backward mask in the update of neuron i. Each neuron then moves exactly as under
      dropout, and only the dependence between neurons differs: neuron i sees the forward mask eta~_j of neuron j,
      while neuron j moves with its backward mask eta_j.
    - paper, w_i = eta_i, as in Section 4.1 of the submitted paper: w_i (eta_i - eta~_i) has the same mean p / (1 - p)
      but a different law, and a dropped neuron (eta_i = -1) still moves.
    """
    n, m = X.shape[0], params[0].shape[0]
    phi, phi_vjp = jax.vjp(lambda prm: vmap(neuron_outputs, in_axes=[0, None])(X, prm), params)
    out = batched_network(X, params, for_mask)
    dloss = jax.grad(loss_from_output)
    shifted_out = out[:, None] + neuron_mask((back_mask - for_mask) / m, phi[0]) * phi
    dloss_diff = vmap(vmap(dloss, in_axes=[0, None]))(shifted_out, Y) - vmap(dloss)(out, Y)[:, None]
    weight = back_mask - 1 if args.penalty_coefficient == "paper" else back_mask
    return phi_vjp(neuron_mask(weight, phi[0]) * dloss_diff / (n * m))[0]


@partial(jit, static_argnames=["batch_size", "p", "d", "m", "forward", "backward"])
def penalty_sgd_step(params, full_dataset, batch_size, p, key, step_size, m, d, forward, backward):
    """Step of fb_sgd_step, with the masks applied only in the forward and/or backward pass as requested, plus the
    penalty term of grad_penalty, built from the same two masks.

    With forward and backward, the update is PN + RaM + penalty, that is the right-hand side of the decomposition
    of Section 4.1 with the joint law of its terms. Dropout (sgd_step) is the left-hand side.
    The backward mask is drawn from the same key as the mask of sgd_step (random.split(key, 3)[i] equals
    random.split(key, 2)[i] in jax 0.7), so that dropout and this step share the minibatch and the mask with which each
    neuron moves; with forward and backward, they differ only by the forward masks of the other neurons."""
    subkeys = random.split(key, 3)
    batch = random.choice(subkeys[0], full_dataset, shape=(batch_size,), replace=False)
    batch_X = batch[:, :d]
    batch_y = batch[:, -1]
    back_mask = random.bernoulli(subkeys[1], 1 - p, shape=(m,)) / (1 - p)
    for_mask = random.bernoulli(subkeys[2], 1 - p, shape=(m,)) / (1 - p)
    ones_mask = jnp.ones(m)
    for_mask_used = for_mask if forward else ones_mask
    back_mask_used = back_mask if backward else ones_mask
    a, b = params
    da, db = grad_fb(params, batch_X, batch_y, for_mask_used, back_mask_used)
    pa, pb = grad_penalty(params, batch_X, batch_y, for_mask, back_mask)
    loss_value = avg_logistic_loss(params, batch_X, batch_y, back_mask_used)
    return loss_value, (a - step_size * (da + pa), b - step_size * (db + pb))


# Step function of each variant, called as step(params, full_dataset, batch_size, p, key, step_size, m, d).
# The variants share the initialization and the key of each step; "zero" is run with p = 0.
STEP_FUNCTIONS = {
    "zero": sgd_step,  # SGD without dropout
    "dropout": sgd_step,  # standard dropout
    "row": masked_sgd_step,  # RaM: mask in the backward pass only
    "forward": forward_sgd_step,  # PN: mask in the forward pass only
    "for-back": fb_sgd_step,  # PN + RaM: independent masks in the forward and backward passes
    "penalty": partial(penalty_sgd_step, forward=False, backward=False),
    "row-penalty": partial(penalty_sgd_step, forward=False, backward=True),
    "forward-penalty": partial(penalty_sgd_step, forward=True, backward=False),
    "for-back-penalty": partial(penalty_sgd_step, forward=True, backward=True),
}


d = 784
batch_size = 64

params_4000_large_tau_new = {  #  WE USE THE LARGE TAU SETTING
    "m": 4000,
    "num_steps": args.num_steps,
    "tau": 0.25,  # master learning rate. True LR = tau * m
    "p_values": [0.4, 0.3, 0.2, 0.1],
    "n_repeats": [10],
}

superkey = random.PRNGKey(0)  # was 0
superkeys = random.split(superkey, 1)

logs = {
    "Step": [],
    "m": [],
    "step_size": [],
    "batch_size": [],
    "p": [],
    "q": [],
    "tau": [],
    "fraction_steps": [],
    "num_steps": [],
    "repeat": [],  # added repeat key
}

# Curvature diagnostics (see sharpness.py), logged every args.sharpness_every steps and NaN at the other logged steps.
# "Batch sharpness": expectation over minibatches only, loss without dropout.
# "Dropout batch sharpness": expectation over minibatches and dropout masks (rate p of the run, also for "zero").
log_every = 100
assert args.sharpness_every % log_every == 0
VARIANTS = args.variants
METRIC_KEYS = [
    "Train risk",  # minibatch loss returned by the step function
    "Full train risk",
    "Test risk",
    "Train accuracy",
    "Test accuracy",
    "Val risk",
]
CURVATURE_KEYS = [
    "Sharpness",
    "Batch sharpness",
    "Batch sharpness stderr",
    "Dropout batch sharpness",
    "Dropout batch sharpness stderr",
]
for key in METRIC_KEYS + CURVATURE_KEYS:
    for variant in VARIANTS:
        logs[key + " " + variant] = []

# Fixed subset of the training set on which the full-batch sharpness is computed (train_x_new is already shuffled).
sharpness_x = train_x_new[: args.sharpness_n_samples]
sharpness_y = train_y_new[: args.sharpness_n_samples]

output_path = args.output or (
    "logs/mnist_exp.pkl" if TASK == "binary47" else "logs/mnist_exp_full.pkl"
)
os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

for k, param_dict in enumerate([params_4000_large_tau_new]):
    m = param_dict["m"]
    num_steps = param_dict["num_steps"]
    tau = param_dict["tau"]
    print(f"Starting new set of parameters, tau = {tau}")
    p_values = param_dict["p_values"]
    n_repeats = param_dict["n_repeats"]
    keys_pvalues = random.split(
        superkeys[0], len(p_values)
    )  # modified superkeys[k] -> superkeys[0] to have the same set of keys for all LRs

    for i, p in enumerate(p_values):
        print("Start p={}".format(p))
        keys = random.split(
            keys_pvalues[0], n_repeats[0]
        )  # modified keys_pvalues[i] -> keys_pvalues[0] to have the same set of init keys for different p values
        for repeat in range(n_repeats[0]):
            print("Repeat {}".format(repeat))
            start = time.time()

            step_size = tau * m
            subkeys = random.split(keys[repeat], 2)
            subkeys_train = random.split(subkeys[1], num_steps)
            eigvecs = {variant: None for variant in VARIANTS}  # warm starts for Lanczos

            params = {variant: init_params(d, m, subkeys[0]) for variant in VARIANTS}
            loss_values = {}

            for step in range(num_steps):
                for variant in VARIANTS:
                    loss_values[variant], params[variant] = STEP_FUNCTIONS[variant](
                        params[variant],
                        full_dataset,
                        batch_size,
                        0.0 if variant == "zero" else p,
                        subkeys_train[step],
                        step_size,
                        m,
                        d,
                    )
                if (step + 1) % log_every == 0:
                    logs["Step"].append(step + 1)
                    logs["m"].append(m)
                    logs["step_size"].append(step_size)
                    logs["batch_size"].append(batch_size)
                    logs["p"].append(p)
                    logs["q"].append(1 - p)
                    logs["tau"].append(tau)
                    logs["fraction_steps"].append(float((step + 1) / num_steps))
                    logs["num_steps"].append(num_steps)
                    logs["repeat"].append(repeat)

                    for variant in VARIANTS:
                        params_variant, loss_value = params[variant], loss_values[variant]
                        ones_mask = jnp.ones(m)
                        full_train_risk = avg_logistic_loss(
                            params_variant, train_x_new, train_y_new, ones_mask
                        )
                        test_risk = avg_logistic_loss(params_variant, test_x, test_y, ones_mask)
                        full_train_acc = avg_accuracy(params_variant, train_x_new, train_y_new)
                        test_acc = avg_accuracy(params_variant, test_x, test_y)
                        val_risk = avg_logistic_loss(params_variant, val_x, val_y, ones_mask)
                        logs["Train risk " + variant].append(float(loss_value))
                        logs["Full train risk " + variant].append(
                            float(full_train_risk)
                        )
                        logs["Test risk " + variant].append(float(test_risk))
                        logs["Train accuracy " + variant].append(float(full_train_acc))
                        logs["Test accuracy " + variant].append(float(test_acc))
                        logs["Val risk " + variant].append(float(val_risk))

                        if (
                            args.sharpness_every
                            and (step + 1) % args.sharpness_every == 0
                            and (args.sharpness_repeats is None or repeat < args.sharpness_repeats)
                        ):
                            # Same minibatches and masks for all variants at a given step.
                            key_curvature = random.fold_in(subkeys[1], step)
                            sharp, eigvecs[variant] = sharpness(
                                avg_logistic_loss,
                                params_variant,
                                sharpness_x,
                                sharpness_y,
                                n_iter=args.lanczos_iters,
                                v0=eigvecs[variant],
                            )
                            curvatures = [sharp]
                            for p_curvature in [0.0, p]:
                                curvatures += batch_sharpness(
                                    avg_logistic_loss,
                                    params_variant,
                                    full_dataset,
                                    batch_size,
                                    p_curvature,
                                    key_curvature,
                                    args.batch_sharpness_n_batches,
                                    d,
                                    m,
                                )
                        else:
                            curvatures = [float("nan")] * len(CURVATURE_KEYS)
                        for curvature_key, value in zip(CURVATURE_KEYS, curvatures):
                            logs[curvature_key + " " + variant].append(float(value))

                if (step + 1) % (100 * log_every) == 0:
                    print(
                        "Step {}/{} for {} variants in {} seconds".format(
                            step + 1, num_steps, len(VARIANTS), time.time() - start
                        )
                    )
                    start = time.time()

            print("Dumping logs")
            with open(output_path, "wb") as handle:
                pickle.dump(logs, handle, protocol=pickle.HIGHEST_PROTOCOL)
