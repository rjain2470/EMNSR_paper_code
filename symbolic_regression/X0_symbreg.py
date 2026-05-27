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
    """Euler totient function."""
    return int(totient(n))


def mu(n: int) -> int:
    """Mobius function."""
    return int(mobius(n))


def sigma(n: int) -> int:
    """Sum-of-divisors function sigma_1(n)."""
    return int(divisor_sigma(n, 1))


def is_squarefree(n: int) -> bool:
    """Return True if n is squarefree."""
    return all(n % (p * p) != 0 for p in range(2, int(n**0.5) + 1))


# =============================================================
# Volumes of X_0(N)
# =============================================================

def get_X0_volumes(Nmax: int = 105):
    """
    Return squarefree levels N <= Nmax and volumes of X_0(N).

    Formula:
        Vol(X_0(N)) = (pi/3) * N * prod_{p | N} (1 + 1/p).
    """
    Ns = [n for n in range(1, Nmax + 1) if is_squarefree(n)]

    vols = np.array([
        (pi / 3) * n * np.prod([1 + 1 / p for p in factorint(n)])
        for n in Ns
    ], dtype=float)

    return Ns, vols


# =============================================================
# Maass newform spectral-parameter data
# =============================================================

_ZENODO_URL = "https://zenodo.org/records/15490636/files/MaassForms.txt?download=1"
_LOCAL_PATH = "MaassForms.txt"


def ensure_dataset(local_path: str = _LOCAL_PATH, url: str = _ZENODO_URL) -> str:
    """
    Ensure the Maass newform dataset exists locally; download it if missing.
    """
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    print("Downloading Maass newform dataset...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for block in r.iter_content(chunk_size=1 << 20):
                if block:
                    f.write(block)

    return local_path


def first_k_spectral_parameters(N: int, K: int, local_path: str = _LOCAL_PATH):
    """
    Return the first K Maass newform spectral parameters r_k(N) at level N.

    The file stores spectral parameters r, not Laplace eigenvalues lambda.
    We later convert by lambda = r^2 + 1/4.
    """
    if K <= 0:
        return []

    ensure_dataset(local_path)

    r_values = []
    with open(local_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue

            parts = line.strip().split(":", 5)
            if len(parts) < 3:
                continue

            try:
                level = int(parts[1])
                r = float(parts[2])
            except ValueError:
                continue

            if level == N:
                r_values.append(r)

    r_values.sort()
    return r_values[:K]


def first_k_eigenvalues(N: int, K: int, local_path: str = _LOCAL_PATH):
    """
    Return the first K Laplace eigenvalues lambda_k(N) at level N.

    These are computed from spectral parameters r_k(N) by

        lambda_k(N) = r_k(N)^2 + 1/4.
    """
    r_values = first_k_spectral_parameters(N, K, local_path)
    return [r**2 + 0.25 for r in r_values]


# =============================================================
# Build eigenvalue regression dataset
# =============================================================

NMAX = 105
K = 100

Ns, vols_X0 = get_X0_volumes(NMAX)

rows = []
targets = []

for i, N in enumerate(Ns):
    eigenvalues = first_k_eigenvalues(N, K)

    for k, lam in enumerate(eigenvalues, start=1):
        rows.append([
            float(k),
            float(N),
            float(vols_X0[i]),
            float(phi(N)),
            float(mu(N)),
            float(sigma(N)),
        ])
        targets.append(float(lam))

X = np.array(rows, dtype=float)
y = np.array(targets, dtype=float)

feature_names = ["k", "N", "vol", "phi", "mu", "sigma"]

print("Eigenvalue dataset constructed.")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Features:", feature_names)


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
    variable_names=feature_names,
)

model.fit(X, y)

print("\nDiscovered expressions for lambda_k^{new}(N):")
print(model)

# =============================================================
# Empirical slope plot (Figure 1)
# =============================================================
 
K_slope = 1000
 
# Build padded r-value matrix; zeros mark missing entries
X0N_r_vals = np.zeros((len(Ns), K_slope))
for i, N in enumerate(Ns):
    r = first_k_spectral_parameters(N, K_slope)
    X0N_r_vals[i, :len(r)] = r
 
# Least-squares slope of lambda vs k (no intercept) per level
c_emp = []
for i in range(len(Ns)):
    r = X0N_r_vals[i].copy()
    r = r[(r != 0.0) & np.isfinite(r)]
    lam    = r**2 + 0.25
    k_vals = np.arange(1, lam.size + 1, dtype=float)
    c_emp.append(np.dot(k_vals, lam) / np.dot(k_vals, k_vals))
 
c_emp   = np.array(c_emp,  dtype=float)
c_model = 12.0 / np.array([phi(int(N)) for N in Ns], dtype=float)
c_weyl  = (4.0 * pi) / vols_X0
 
plt.figure(figsize=(9, 6))
plt.scatter(Ns, c_emp,    label=r"Empirical slope $\hat{c}(N)$", alpha=0.8)
plt.plot(   Ns, c_model,  color="green", label=r"$12/\varphi(N)$")
plt.plot(   Ns, c_weyl,   color="red",   label=r"Weyl $4\pi/V_0(N)$")
plt.yscale("log")
plt.xlabel(r"$N$")
plt.ylabel(r"$\hat{c}(N)$")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("figure1_slopes.pdf")
plt.show()
 
 
# =============================================================
# Evaluation metrics for 12k / phi(N)
# =============================================================
 
k_col   = X[:, 0]
phi_col = X[:, 3]
y_hat   = 12.0 * k_col / phi_col
 
residuals  = y - y_hat
rel_errors = np.abs(residuals) / y
 
# Spearman rank correlation
spearman_r, _ = spearmanr(y, y_hat)
 
# Median per-level R²
per_level_r2 = []
for N in Ns:
    idx = X[:, 1] == N
    if idx.sum() < 2:
        continue
    ss_r = np.sum((y[idx] - y_hat[idx]) ** 2)
    ss_t = np.sum((y[idx] - y[idx].mean()) ** 2)
    per_level_r2.append(1 - ss_r / ss_t)
median_r2 = np.median(per_level_r2)
 
# MRE: prime vs composite levels
for label, mask_fn in [("prime",     isprime),
                        ("composite", lambda n: not isprime(n))]:
    idx   = np.array([i for i, row in enumerate(X) if mask_fn(int(row[1]))])
    mre_l = rel_errors[idx].mean() * 100
    print(f"MRE ({label:>9s}) = {mre_l:.2f}%")
 
print(f"Spearman r         = {spearman_r:.4f}")
print(f"Median per-level R²= {median_r2:.4f}")
