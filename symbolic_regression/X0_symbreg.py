"""
Description: Symbolic Regression for eigenvalues of the Laplacian associated to Maass newforms on X_0(N).

This script:
1. Downloads the dataset of eigenvalues from the LMFDB
2. Constructs spectral parameters r_k(N) for squarefree N ≤ 105
3. Builds a dataset:
       λ_k(N) = r_k(N)^2 + 1/4
   with features:
       (k, N, Vol(X_0(N)), φ(N), μ(N), σ(N))
4. Runs symbolic regression using PySR

Dependencies: numpy, sympy, pysr, requests
"""
import os
import requests
import numpy as np
from math import pi
from sympy import factorint, totient, mobius, divisor_sigma
from pysr import PySRRegressor

# =============================================================
# Arithmetic functions
# =============================================================
def phi(n: int) -> int:
    return int(totient(n))

def mu(n: int) -> int:
    return int(mobius(n))

def sigma(n: int) -> int:
    return int(divisor_sigma(n, 1))


# =============================================================
# Volumes of X_0(N)
# =============================================================
def get_X0_volumes(Nmax: int = 105):
    Ns = [n for n in range(1, Nmax + 1)
          if all(n % (p * p) for p in range(2, int(n**0.5) + 1))]

    vols = np.array([
        (pi / 3) * n * np.prod([1 + 1/p for p in factorint(n)])
        for n in Ns
    ], dtype=float)

    return Ns, vols


# =============================================================
# Spectral Parameters for X_0(N)
# =============================================================
_ZENODO_URL = "https://zenodo.org/records/15490636/files/MaassForms.txt?download=1"
_LOCAL_PATH = "MaassForms.txt"

squarefree_Ns = [n for n in range(1, 106)
                 if all(n % (p*p) for p in range(2, int(n**0.5) + 1))]


def _ensure_dataset(local_path=_LOCAL_PATH, url=_ZENODO_URL):
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    print("Downloading Maass dataset...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for blk in r.iter_content(chunk_size=1 << 20):
                if blk:
                    f.write(blk)
    return local_path


def first_k_spectral_params(N: int, k: int):
    _ensure_dataset()

    Rs = []
    with open(_LOCAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue

            parts = line.strip().split(":", 5)
            if len(parts) < 3:
                continue

            try:
                level = int(parts[1])
                R = float(parts[2])
            except:
                continue

            if level == N:
                Rs.append(R)

    Rs.sort()
    return Rs[:k]


# =============================================================
# Build array of spectral parameters
# =============================================================
K = 100
spectral_params_list = []
max_k = 0

for N in squarefree_Ns:
    params = first_k_spectral_params(N, K)
    spectral_params_list.append(params)
    max_k = max(max_k, len(params))

padded = []
for params in spectral_params_list:
    arr = np.array(params, dtype=float)
    if len(arr) < max_k:
        arr = np.pad(arr, (0, max_k - len(arr)))
    padded.append(arr)

r_vals = np.vstack(padded)

# remove trailing zero columns
if np.any(r_vals):
    last = np.argwhere(r_vals != 0)[:, 1].max()
    r_vals = r_vals[:, :last + 1]


# =============================================================
# Build regression dataset
# =============================================================
Ns, vols_X0 = get_X0_volumes()

rows = []
targets = []

for i, N in enumerate(Ns):
    r_row = r_vals[i]
    r_row = r_row[r_row != 0]

    for k, r in enumerate(r_row, start=1):
        lam = r**2 + 0.25

        rows.append([
            float(k),
            float(N),
            float(vols_X0[i]),
            float(phi(N)),
            float(mu(N)),
            float(sigma(N)),
        ])
        targets.append(lam)

X = np.array(rows)
y = np.array(targets)

print("Dataset:", X.shape)


# =============================================================
# Symbolic Regression
# =============================================================
model = PySRRegressor(
    niterations=3000,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["log", "sqrt"],
    maxsize=24,
    maxdepth=6,
    population_size=100,
    loss="loss(x, y) = (x - y)^2",
    variable_names=["k", "N", "vol", "phi", "mu", "sigma"],
)

model.fit(X, y)

print("\nDiscovered expressions:")
print(model)
