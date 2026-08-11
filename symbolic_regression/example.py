"""Worked example: recover the Prime Number Theorem pi(x) ~ x / log x with
symbolic regression.

Dependencies: numpy, sympy, pysr (PySR needs a working Julia install, which it
provisions automatically on first use).

Run:  python -m symbolic_regression.example
"""

import numpy as np
from sympy import primepi
from pysr import PySRRegressor

# Dataset: pi(n) at logarithmically-spaced n in [100, 100000].
n_values = np.unique(np.logspace(2, 5, 200).astype(int))
pi_values = np.array([int(primepi(int(n))) for n in n_values], dtype=float)

X = n_values.reshape(-1, 1)
y = pi_values
print(f"Number of data points: {len(n_values)}")

model = PySRRegressor(
    niterations=3000,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["log", "exp"],
    maxsize=20,
    maxdepth=6,
    population_size=100,
    model_selection="best",
    elementwise_loss="loss(x, y) = (x - y)^2",
)
model.fit(X, y, variable_names=["n"])

print(model)
