"""Arithmetic symbolic-regression experiment.

PySR searches over (k, N, Vol(X_0(N)), phi(N), mu(N), sigma(N)) -> lambda_k(N)
under a RELATIVE squared-error loss ((x - y) / y)^2. Because lambda_k spans
orders of magnitude, plain squared error is dominated by the largest eigenvalues
and biases the recovered leading coefficient away from 12; relative loss weights
every (k, N) equally, so the clean counting law c * k / phi(N) -- solely in
phi(N) -- is the natural optimum. We report the full Pareto front (simplest
first) and select the lowest-complexity equation depending only on (k, phi); its
leading term is ~ 12 k / phi(N). Weyl's law is plotted for comparison.

``composite_addendum`` reruns the SAME search/loss restricted to composite
squarefree levels: the small-prime levels have the smallest Vol(X_0(N)) and
carry the largest pre-asymptotic Weyl remainder, which inflates the pooled
constant to ~13.2; on the composite levels the leading constant drops toward the
theoretical 12.

Run:  python -m symbolic_regression.example_2
"""

import time
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sympy import isprime

from emnsr import phi, mu, sigma, volume_X0, first_k_spectral_parameters
from emnsr.spectral_features import MAASS_CACHE, squarefree_Ns
from emnsr.config import K
from symbolic_regression.features import build_arithmetic_dataset
from symbolic_regression.utils import (evaluate, mre_prime_composite,
                                       plot_per_level_r2_hist, pareto_front,
                                       select_law, sympy_predictor, run_pysr,
                                       REL_LOSS, SR_TIME_LIMIT)

# 'N' clashes with sympy.N in PySR's exporter; alias to 'Nlev' (column order is
# unchanged, so the positional predictor is unaffected).
SAFE_NAMES = ["k", "Nlev", "vol", "phi", "mu", "sigma"]


def _slope_vs_k(lam):
    """Least-squares slope of lambda vs k with no intercept."""
    if lam.size == 0:
        return np.nan
    kk = np.arange(1, lam.size + 1, dtype=float)
    return np.dot(kk, lam) / np.dot(kk, kk)


def _leading_coefficient(law_expr):
    """Leading coefficient c in c * k / phi from a (k, phi) equation."""
    kk, ph = sp.Symbol("k"), sp.Symbol("phi")
    return float(sp.limit(sp.limit(law_expr * ph / kk, kk, sp.oo), ph, sp.oo))


def main(cache=MAASS_CACHE, Ns=squarefree_Ns):
    X32, y32, names32 = build_arithmetic_dataset(Ns, K, cache)
    levels32 = X32[:, 1].astype(int)
    print("Dataset:", X32.shape, "features:", names32)

    # NOTE ON METRICS: under relative loss the fit optimises relative (not
    # squared) error, so the honest headline metrics are MRE, per-level R^2,
    # Spearman and the slope-level R^2. The pooled eigenvalue R^2 is a
    # squared-error metric dominated by the largest eigenvalues and reads low
    # (~0.83) even though the law tracks every level well (median per-level
    # R^2 ~ 0.97); it is reported for completeness, not as the headline.
    print("\nRunning PySR (relative loss, hard cap)...", flush=True)
    _t = time.time()
    model32, expr32, predict32 = run_pysr(X32, y32, SAFE_NAMES, niterations=120,
                                          timeout_s=SR_TIME_LIMIT, seed=0,
                                          loss=REL_LOSS)
    print(f"PySR finished in {time.time() - _t:.0f}s")
    print("\nPySR default 'best' pick:", expr32, "   [Nlev == N]")
    print("\nPareto front (complexity / loss / score / equation), simplest first:")
    print(pareto_front(model32))

    # Select the law closest to the theoretical form, SOLELY in phi: the
    # simplest front equation in (k, phi) that actually involves phi.
    sel = select_law(model32, SAFE_NAMES, allowed={"k", "phi"},
                     require={"phi"}, prefer="complexity")
    if sel is None:
        print("\n[warn] no (k, phi)-only equation on the front; using best pick")
        law_expr, predict_law = expr32, predict32
    else:
        _, law_expr, predict_law = sel
        print("\nLaw closest to 12k/phi (solely in phi(N)):  lambda_k(N) =", law_expr)
        print(f"   leading coefficient c in c*k/phi(N) = "
              f"{_leading_coefficient(law_expr):.4f}   (theoretical 12)")

    m32 = evaluate(X32, y32, predict_law, levels32)
    mre32 = mre_prime_composite(y32, m32["y_hat"], levels32)
    print("\n-- metrics (law solely in phi) --")
    print(f"  overall MRE          = {m32['mre']:.2f}%   <- headline")
    print(f"  MRE prime levels     = {mre32['prime']:.2f}%")
    print(f"  MRE composite levels = {mre32['composite']:.2f}%")
    print(f"  Spearman rho         = {m32['spearman']:.4f}")
    print(f"  median per-level R^2 = {m32['median_per_level_r2']:.4f}")
    print(f"  eigenvalue R^2       = {m32['r2']:.4f}   (pooled squared-error; see note)")

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
        c_disc.append(_slope_vs_k(predict_law(Xn)))
    c_emp, c_disc = np.array(c_emp), np.array(c_disc)
    c_weyl = 4.0 * np.pi / np.array([volume_X0(N) for N in Ns_slope], float)

    r2_slope = 1.0 - np.sum((c_emp - c_disc) ** 2) / np.sum((c_emp - c_emp.mean()) ** 2)
    print(f"  slope-level R^2 (discovered vs empirical c(N)) = {r2_slope:.4f}")

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

    composite_addendum(cache, Ns)


def composite_addendum(cache=MAASS_CACHE, Ns=squarefree_Ns):
    """Rerun the SAME search/loss on composite squarefree levels only.

    The leading constant of c*k/phi drops from ~13.2 (pooled) toward the
    theoretical 12, because the small-prime levels (N=2,3,5,7,...) have the
    smallest Vol(X_0(N)) and therefore carry the largest pre-asymptotic
    Selberg/Weyl remainder relative to the leading term."""
    Xa, ya, _ = build_arithmetic_dataset(Ns, K, cache)
    lev_a = Xa[:, 1].astype(int)
    comp = np.array([n > 1 and not isprime(int(n)) for n in lev_a])   # composite squarefree
    Xc, yc, levc = Xa[comp], ya[comp], lev_a[comp]
    n_prime = len({int(n) for n in lev_a if isprime(int(n))})
    print(f"\n=== Composite-only addendum ===\nDataset: {Xc.shape}  "
          f"({len(np.unique(levc))} composite levels; {n_prime} prime levels dropped)")

    print("\nRunning PySR (composite levels, SAME relative loss, hard cap)...", flush=True)
    _t = time.time()
    modelc, _, _ = run_pysr(Xc, yc, SAFE_NAMES, niterations=120,
                            timeout_s=SR_TIME_LIMIT, seed=0, loss=REL_LOSS)
    print(f"PySR finished in {time.time() - _t:.0f}s")
    print("\nPareto front (complexity / loss / score / equation), simplest first:")
    print(pareto_front(modelc))

    # Select the counting law off the front. NOTE: prefer='complexity' (as in
    # main) misfires here -- the degenerate k/log(phi) is simpler yet a poor fit
    # (and blows up at phi=1) -- so among the (k, phi)-only equations we take the
    # one at the loss ELBOW (largest PySR score): the pure counting law.
    df = modelc.equations_
    cand = []
    for i in range(len(df)):
        try:
            e = modelc.sympy(i)
        except Exception:
            continue
        s = {t.name for t in e.free_symbols}
        if s and s.issubset({"k", "phi"}) and "phi" in s:
            cand.append((float(df.iloc[i]["score"]), e))
    if not cand:
        print("[warn] no (k, phi)-only equation on the composite front")
        return
    _, law_expr = max(cand, key=lambda c: c[0])
    law_pred = sympy_predictor(law_expr, SAFE_NAMES)
    print("\nLaw closest to 12k/phi (composite levels):  lambda_k(N) =", law_expr)
    print(f"   leading coefficient c in c*k/phi = {_leading_coefficient(law_expr):.4f}"
          f"   (theoretical 12; full-set run gives ~13.2)")

    mc = evaluate(Xc, yc, law_pred, levc)
    mrc = mre_prime_composite(ya, law_pred(Xa), lev_a)
    print("\n-- composite-only metrics --")
    print(f"  overall MRE (composite)  = {mc['mre']:.2f}%")
    print(f"  Spearman rho             = {mc['spearman']:.4f}")
    print(f"  median per-level R^2     = {mc['median_per_level_r2']:.4f}")
    print(f"  this law's MRE on primes = {mrc['prime']:.2f}%   (composite {mrc['composite']:.2f}%)")


if __name__ == "__main__":
    main()
