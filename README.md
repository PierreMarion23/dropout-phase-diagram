# Phase Diagram of Dropout for Two-Layer Neural Networks in the Mean-Field Regime

This is the code for reproducing the experiments from the paper Phase Diagram of Dropout for Two-Layer Neural Networks in the Mean-Field Regime by Lénaïc Chizat, Pierre Marion, Yerkin Yesbay.

## Teacher-student experiment

Install the conda environment by running

```
conda env create -f environment-teacher-student.yml 
```

The figures from the paper can be reproduced by running the Jupyter notebook ```teacher_student_dropout.ipynb```.

The experiment takes about 30 minutes to run on a consumer laptop.

## MNIST experiment

Install the conda environment by running

```
conda env create -f environment-mnist.yml 
```

The figures from the paper can be reproduced by running the experiment in the Python file ```mnist_exp.py```, then the plotting in the Jupyter notebook ```mnist_exp_plots.ipynb```.

The experiment takes about 3 hours to run on a Nvidia GTX 1080Ti GPU.

Options of `mnist_exp.py`:

- `--task binary47` (default, setting of the paper: digits 4 and 7, logistic loss) or `--task full` (10 classes, cross-entropy loss, the second-layer weights become 10-dimensional). The logs go to `logs/mnist_exp.pkl` and `logs/mnist_exp_full.pkl` respectively, unless `--output` is given.
- `--num-steps` (default 40000): number of training steps. Figure 2 of the paper uses 40000 steps (`logs/mnist_exp_40k.pkl`), Figure 7 uses 20000 (`logs/mnist_exp_full.pkl`).
- `--variants` (default: `zero dropout row forward for-back penalty for-back-penalty`): the training variants, all started from the same initialization and run on the same minibatches. `zero` is SGD without dropout, `dropout` is standard dropout, `row` is RaM (mask in the backward pass only), `forward` is PN (mask in the forward pass only) and `for-back` is PN + RaM (independent masks in the two passes). `penalty`, `row-penalty`, `forward-penalty` and `for-back-penalty` add the penalty term of the decomposition of the dropout update (Section 4.1 of the paper, see `grad_penalty`) to `zero`, `row`, `forward` and `for-back` respectively.
  - `for-back-penalty` is the right-hand side of the decomposition, with the joint law of its terms (forward mask `eta~` independent of the backward mask `eta`).
  - `for-back-penalty` shares the minibatch and the backward mask with `dropout`, so its paired difference with `dropout` measures the effect of this dependence.
  - A penalty variant costs about one dropout step (on CPU, width 4000: +0% to +30% over `for-back`).
  - Penalty term: for the squared loss, it moves neuron i by `-tau/m * eta_i (eta_i - eta~_i) * Dphi_i^T phi_i`, with `eta` the backward mask minus one and `eta~` the forward mask minus one, as in Section 4.1 of the paper. For the logistic and cross-entropy losses, `(eta_i - eta~_i) phi_i / m`, the change of the residual when the forward mask of neuron i is switched from `eta~_i` to `eta_i`, is replaced by the corresponding change of the derivative of the loss, which gives back the formula above for the squared loss.
- Every `--sharpness-every` steps (default 100, 0 disables), the script logs for each variant (see `sharpness.py`):
  - `Sharpness <variant>`: largest eigenvalue of the Hessian of the training loss without dropout, by Lanczos, on the whole training set or on a fixed subset of `--sharpness-n-samples` samples;
  - `Batch sharpness <variant>`: batch sharpness of Andreyev and Beneventano (2024), E_B[g_B^T H_B g_B / |g_B|^2], with the expectation over minibatches only;
  - `Dropout batch sharpness <variant>`: the same with the expectation over minibatches and dropout masks, g and H being the gradient and Hessian of the dropout minibatch loss;
  - the standard errors of the two Monte Carlo estimates (over `--batch-sharpness-n-batches` draws, default 256).

  These columns are NaN at the other logged steps, and for the repeats beyond `--sharpness-repeats` (default: all).

  The sharpness costs `--lanczos-iters` (default 15) Hessian-vector products on the training set per variant and measurement, warm-started from the previous eigenvector. On a width-4000 network trained for 3000 steps on real MNIST, 15 iterations matched a tight-tolerance reference to float32 precision on both tasks, with or without warm start; 10 warm-started iterations were within 3e-6 (relative). The main levers on the total cost are `--sharpness-every` and `--sharpness-repeats`. Computing the sharpness on a subset (`--sharpness-n-samples`) is cheaper but biased: on the 10-class task, subsets of 4096, 8192 and 16384 samples changed it by -11%, -0.3% and -2.4%.