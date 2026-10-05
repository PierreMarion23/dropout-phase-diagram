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
    default=100,
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
parser.add_argument("--output", default=None, help="path of the pickled logs")
args = parser.parse_args()
TASK = args.task
OUT_SHAPE = () if TASK == "binary47" else (10,)  # shape of the network output

dataset = tf.keras.datasets.mnist.load_data()
train = dataset[0]
test = dataset[1]


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


d = 784
batch_size = 64

params_4000_large_tau_new = {  #  WE USE THE LARGE TAU SETTING
    "m": 4000,
    "num_steps": int(4 * 10**4),  # can be reduced to 20 000 steps
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
    "Train risk zero": [],  ############
    "Full train risk zero": [],
    "Test risk zero": [],
    "Train accuracy zero": [],
    "Test accuracy zero": [],
    "Val risk zero": [],
    "Train risk dropout": [],  ############
    "Full train risk dropout": [],
    "Test risk dropout": [],
    "Train accuracy dropout": [],
    "Test accuracy dropout": [],
    "Val risk dropout": [],
    "Train risk row": [],  ##############
    "Full train risk row": [],
    "Test risk row": [],
    "Train accuracy row": [],
    "Test accuracy row": [],
    "Val risk row": [],
    "Train risk forward": [],  ##############
    "Full train risk forward": [],
    "Test risk forward": [],
    "Train accuracy forward": [],
    "Test accuracy forward": [],
    "Val risk forward": [],
    "Train risk for-back": [],  ##############
    "Full train risk for-back": [],
    "Test risk for-back": [],
    "Train accuracy for-back": [],
    "Test accuracy for-back": [],
    "Val risk for-back": [],
}

# Curvature diagnostics (see sharpness.py), logged every args.sharpness_every steps and NaN at the other logged steps.
# "Batch sharpness": expectation over minibatches only, loss without dropout.
# "Dropout batch sharpness": expectation over minibatches and dropout masks (rate p of the run, also for "zero").
log_every = 100
assert args.sharpness_every % log_every == 0
VARIANTS = ["zero", "dropout", "row", "forward", "for-back"]
CURVATURE_KEYS = [
    "Sharpness",
    "Batch sharpness",
    "Batch sharpness stderr",
    "Dropout batch sharpness",
    "Dropout batch sharpness stderr",
]
for curvature_key in CURVATURE_KEYS:
    for variant in VARIANTS:
        logs[curvature_key + " " + variant] = []

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

            params_zero = init_params(d, m, subkeys[0])
            params_dropout = init_params(d, m, subkeys[0])
            params_row = init_params(d, m, subkeys[0])
            params_forward = init_params(d, m, subkeys[0])
            params_fb = init_params(d, m, subkeys[0])

            for step in range(num_steps):
                loss_value_zero, params_zero = sgd_step(
                    params_zero,
                    full_dataset,
                    batch_size,
                    0.0,
                    subkeys_train[step],
                    step_size,
                    m,
                    d,
                )

                loss_value_dropout, params_dropout = sgd_step(
                    params_dropout,
                    full_dataset,
                    batch_size,
                    p,
                    subkeys_train[step],
                    step_size,
                    m,
                    d,
                )
                loss_value_row, params_row = masked_sgd_step(
                    params_row,
                    full_dataset,
                    batch_size,
                    p,
                    subkeys_train[step],
                    step_size,
                    m,
                    d,
                )
                loss_value_forward, params_forward = forward_sgd_step(
                    params_forward,
                    full_dataset,
                    batch_size,
                    p,
                    subkeys_train[step],
                    step_size,
                    m,
                    d,
                )
                loss_value_fb, params_fb = fb_sgd_step(
                    params_fb,
                    full_dataset,
                    batch_size,
                    p,
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

                    for params, loss_value, variant in zip(
                        [
                            params_zero,
                            params_dropout,
                            params_row,
                            params_forward,
                            params_fb,
                        ],
                        [
                            loss_value_zero,
                            loss_value_dropout,
                            loss_value_row,
                            loss_value_forward,
                            loss_value_fb,
                        ],
                        VARIANTS,
                    ):
                        ones_mask = jnp.ones(m)
                        full_train_risk = avg_logistic_loss(
                            params, train_x_new, train_y_new, ones_mask
                        )
                        test_risk = avg_logistic_loss(params, test_x, test_y, ones_mask)
                        full_train_acc = avg_accuracy(params, train_x_new, train_y_new)
                        test_acc = avg_accuracy(params, test_x, test_y)
                        val_risk = avg_logistic_loss(params, val_x, val_y, ones_mask)
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
                                params,
                                sharpness_x,
                                sharpness_y,
                                n_iter=args.lanczos_iters,
                                v0=eigvecs[variant],
                            )
                            curvatures = [sharp]
                            for p_curvature in [0.0, p]:
                                curvatures += batch_sharpness(
                                    avg_logistic_loss,
                                    params,
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
                        "Step {}/{} for 4 variants in {} seconds".format(
                            step + 1, num_steps, time.time() - start
                        )
                    )
                    start = time.time()

            print("Dumping logs")
            with open(output_path, "wb") as handle:
                pickle.dump(logs, handle, protocol=pickle.HIGHEST_PROTOCOL)
