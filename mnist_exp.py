from functools import partial

from jax import numpy as jnp
from jax import jit, vmap, value_and_grad
from jax import random
import pickle
import tensorflow as tf
import time

dataset = tf.keras.datasets.mnist.load_data()
train = dataset[0]
test = dataset[1]

train_x_seq = train[0].shape[0]
train_x_len = int(jnp.prod(jnp.array(train[0].shape[1:])))
test_x_seq = test[0].shape[0]
test_x_len = int(jnp.prod(jnp.array(test[0].shape[1:])))

train_x = train[0].reshape((train_x_seq, train_x_len)) / 255
train_y = train[1].reshape(train_x_seq)
train_mask_47 = jnp.any(jnp.array([train_y == 4, train_y == 7]), axis=0)
train_x = train_x[train_mask_47]
train_y = train_y[train_mask_47].astype(float)
train_y[train_y == 4] = 1.0
train_y[train_y == 7] = -1.0

test_x = test[0].reshape((test_x_seq, test_x_len)) / 255
test_y = test[1].reshape(test_x_seq)
test_mask_47 = jnp.any(jnp.array([test_y == 4, test_y == 7]), axis=0)
test_x = test_x[test_mask_47]
test_y = test_y[test_mask_47].astype(float)
test_y[test_y == 4] = 1.0
test_y[test_y == 7] = -1.0

val_size = 2000
train_x = random.permutation(random.PRNGKey(0), train_x, axis=0)
train_y = random.permutation(random.PRNGKey(0), train_y, axis=0)
val_x, val_y = train_x[:val_size], train_y[:val_size]
train_x_new, train_y_new = train_x[val_size:], train_y[val_size:]

full_dataset = jnp.concatenate([train_x_new, train_y_new.reshape(-1, 1)], axis=1)


def network(x, params, mask):
    a, b = params
    m = a.shape[0]
    a = a * mask
    return 1 / m * jnp.maximum(b @ x, 0) @ a


def logistic_loss(params, x, y, mask):
    return jnp.log(1 + jnp.exp(-y * network(x, params, mask)))


def accuracy(params, x, y):
    return y * network(x, params, jnp.ones_like(params[0])) > 0.0


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
    a = random.normal(subkeys[0], (m,))
    b = random.normal(subkeys[1], (m, d)) / jnp.sqrt(d)
    return (a, b)


@jit
def grad_bceloss(y_hat, y):  # implemetns dl/dy_hat
    return -y * jnp.exp(-y * y_hat) / (1 + jnp.exp(-y * y_hat))


@jit
def grad_forward(x, params, mask):  # implements df/da, df/db
    a, b = params
    m = a.shape[0]
    da = jnp.maximum(b @ x, 0) * mask / m
    db = jnp.outer(jnp.maximum(jnp.sign(b @ x), 0) * a * mask, x) / m
    return da, db


def grad_forward_backward(params, x, y, for_mask, back_mask):
    dl_df = grad_bceloss(network(x, params, for_mask), y)
    df_da, df_db = grad_forward(x, params, back_mask)
    da = dl_df * df_da
    db = dl_df * df_db
    return da, db


batched_grad_fb = vmap(
    grad_forward_backward, in_axes=[None, 0, 0, None, None], out_axes=(0, 0)
)


@jit
def grad_fb(params, X, Y, for_mask, back_mask):
    dA, dB = batched_grad_fb(params, X, Y, for_mask, back_mask)
    return jnp.mean(dA, axis=0), jnp.mean(dB, axis=0)


batched_network = vmap(network, in_axes=[0, None, None])


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
    mask = random.bernoulli(subkeys[1], 1 - p, shape=(m, 1)) / (1 - p)
    loss_value, grads = value_and_grad(avg_logistic_loss)(
        params, batch_X, batch_y, jnp.ones(m)
    )
    a, b = params
    da, db = grads
    return loss_value, (
        a - step_size * mask.reshape(da.shape) * da,
        b - step_size * mask * db,
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
            log_every = 100
            subkeys = random.split(keys[repeat], 2)
            subkeys_train = random.split(subkeys[1], num_steps)

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
                        ["zero", "dropout", "row", "forward", "for-back"],
                    ):
                        ones_mask = jnp.ones_like(params[0])
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

                if (step + 1) % (100 * log_every) == 0:
                    print(
                        "Step {}/{} for 4 variants in {} seconds".format(
                            step + 1, num_steps, time.time() - start
                        )
                    )
                    start = time.time()

            print("Dumping logs")
            with open("logs/mnist_exp.pkl", "wb") as handle:
                pickle.dump(logs, handle, protocol=pickle.HIGHEST_PROTOCOL)
