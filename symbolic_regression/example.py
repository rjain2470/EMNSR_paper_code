''' 
Description: Example use of symbolic regression in empirically recovering the prime number theorem.

Dependencies: numpy, sympy, pysr
Note: PySR requires a working Julia installation (>=1.6) and will automatically install required Julia packages on first use.
'''

import numpy as np
from sympy import primepi
from pysr import PySRRegressor

# Construct dataset of input values from 100 to 100000
n_values = np.unique(
    np.logspace(2, 5, 200).astype(int) 
)

# Compute \pi(n) for each input value n
pi_values = np.array([int(primepi(int(n))) for n in n_values], dtype=float)

X = n_values.reshape(-1, 1)
y = pi_values

print(f"Number of data points: {len(n_values)}")

# Run the symbolic regression
model = PySRRegressor(
    niterations=3000,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["log", "exp"],
    maxsize=20,
    maxdepth=6,
    population_size=100,
    model_selection="best",
    loss="loss(x, y) = (x - y)^2",
    variable_names=["n"],
)
model.fit(X, y)

print(model)
