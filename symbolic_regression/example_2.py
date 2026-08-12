"""Arithmetic symbolic-regression experiment.

PySR searches over (k, N, Vol(X_0(N)), phi(N), mu(N), sigma(N)) -> lambda_k(N),
recovering the newform counting law lambda_k^new(N) ~ 12 k / phi(N). Weyl's law
is plotted for comparison. Every metric and figure is computed from the
discovered equation.

Run:  python -m symbolic_regression.example_2
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from emnsr import phi, mu, sigma, volume_X0, first_k_spectral_parameters
from emnsr.spectral_features import MAASS_CACHE, squarefree_Ns
from emnsr.config import K
from symbolic_regression.features import build_arithmetic_dataset
from symbolic_regression.utils import (evaluate, mre_prime_composite,
                                       plot_per_level_r2_hist,
                                       run_pysr, top_equations, SR_TIME_LIMIT)


def _slope_vs_k(lam):
    """Least-squares slope of lambda vs k with no intercept."""
    if lam.size == 0:
        return np.nan
    kk = np.arange(1, lam.size + 1, dtype=float)
    return np.dot(kk, lam) / np.dot(kk, kk)


def main(cache=MAASS_CACHE, Ns=squarefree_Ns):
    X32, y32, names32 = build_arithmetic_dataset(Ns, K, cache)
    levels32 = X32[:, 1].astype(int)
    # 'N' clashes with sympy.N in PySR's exporter; alias to 'Nlev' (column order
    # is unchanged, so the positional predictor is unaffected).
    safe_names32 = ["k", "Nlev", "vol", "phi", "mu", "sigma"]
    print("Dataset:", X32.shape, "features:", names32)

    print("\nRunning PySR...", flush=True)
    _t = time.time()
    model32, expr32, predict32 = run_pysr(X32, y32, safe_names32,
                                          niterations=120, timeout_s=SR_TIME_LIMIT, seed=0)
    print(f"PySR finished in {time.time() - _t:.0f}s")
    print("\nDiscovered law  lambda_k(N) =", expr32, "   [Nlev == N]")
    print("\nPareto tail (complexity / loss / equation):")
    print(top_equations(model32))

    m32 = evaluate(X32, y32, predict32, levels32)
    mre32 = mre_prime_composite(y32, m32["y_hat"], levels32)
    print("\n-- metrics (discovered formula) --")
    print(f"  eigenvalue R^2       = {m32['r2']:.4f}")
    print(f"  overall MRE          = {m32['mre']:.2f}%")
    print(f"  Spearman rho         = {m32['spearman']:.4f}")
    print(f"  median per-level R^2 = {m32['median_per_level_r2']:.4f}")
    print(f"  MRE prime levels     = {mre32['prime']:.2f}%")
    print(f"  MRE composite levels = {mre32['composite']:.2f}%")

    # empirical vs discovered per-level slope c(N) = lambda / k. Restrict to the
    # levels that actually have spectral data (a level with no entries has no
    # empirical slope to compare against).
    K_slope = 1000
    Ns_slope = [N for N in Ns if len(first_k_spectral_parameters(N, 1, cache)) > 0]
    c_emp, c_disc = [], []
    for N in Ns_slope:
        r = np.asarray(first_k_spectral_parameters(N, K_slope, cache), float)
        lam = r ** 2 + 0.25
        c_emp.append(_slope_vs_k(lam))
        kk = np.arange(1, lam.size + 1, dtype=float)
        Xn = np.column_stack([kk, np.full_like(kk, N), np.full_like(kk, volume_X0(N)),
                              np.full_like(kk, phi(N)), np.full_like(kk, mu(N)),
                              np.full_like(kk, sigma(N))])
        c_disc.append(_slope_vs_k(predict32(Xn)))
    c_emp, c_disc = np.array(c_emp), np.array(c_disc)
    c_weyl = 4.0 * np.pi / np.array([volume_X0(N) for N in Ns_slope], float)

    r2_slope = 1.0 - np.sum((c_emp - c_disc) ** 2) / np.sum((c_emp - c_emp.mean()) ** 2)
    print(f"\n  slope-level R^2 (discovered vs empirical c(N)) = {r2_slope:.4f}")

    plt.figure(figsize=(9, 6))
    plt.scatter(Ns_slope, c_emp, label=r"Empirical slope $c(N)$", alpha=0.8, zorder=3)
    plt.plot(Ns_slope, c_disc, color="C1", lw=2, label="Discovered (PySR)")
    plt.plot(Ns_slope, c_weyl, color="red", ls=":", lw=1,
             label=r"Weyl $4\pi/\mathrm{Vol}(X_0)$")
    plt.yscale("log"); plt.xlabel(r"$N$"); plt.ylabel(r"$c(N)$")
    plt.grid(True, which="both", alpha=0.3); plt.legend(); plt.tight_layout()
    plt.show()

    plot_per_level_r2_hist(m32["per_level_r2"])
    plt.show()


if __name__ == "__main__":
    main()
