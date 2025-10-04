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