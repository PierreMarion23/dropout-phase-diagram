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
- Every `--sharpness-every` steps (default 1000, 0 disables), the script logs for each variant (see `sharpness.py`):
  - `Sharpness <variant>`: largest eigenvalue of the Hessian of the training loss without dropout, by Lanczos, on the whole training set or on a fixed subset of `--sharpness-n-samples` samples;
  - `Batch sharpness <variant>`: batch sharpness of Andreyev and Beneventano (2024), E_B[g_B^T H_B g_B / |g_B|^2], with the expectation over minibatches only;
  - `Dropout batch sharpness <variant>`: the same with the expectation over minibatches and dropout masks, g and H being the gradient and Hessian of the dropout minibatch loss;
  - the standard errors of the two Monte Carlo estimates (over `--batch-sharpness-n-batches` draws, default 256).

  These columns are NaN at the other logged steps.