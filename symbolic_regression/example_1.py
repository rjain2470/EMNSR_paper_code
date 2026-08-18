"""Geometric symbolic-regression experiment.

TARGET: the eigenvalue lambda_k(N) = r_k(N)^2 + 1/4 (eigenvalue level, so the
metrics are directly comparable to Example 3.2).

Stage A: for each k, PySR searches the geometric features
    (Vol(X_1(N)), genus, l_1, l_2) -> lambda_k(N)
with no assumed form. We keep (a_k, b_k) for the k whose best equation is
(algebraically) the volume-only law a / (sqrt(Vol(X_1)) - b) -- the lambda-native
form of r ~ a/(Vol^{1/4}-b), since lambda = r^2 + 1/4 and leading order
lambda ~ a/sqrt(Vol).

Stage B: second-level symbolic regression discovers a(k) and b(k), which resolve
into a single closed law
    lambda_k(N) = a(k) / (sqrt(Vol(X_1(N))) - b(k)).

The per-k searches use multithreading (faster, non-deterministic), so the exact
set of matched k varies between runs; the discovered laws are stable in form.

Run:  python -m symbolic_regression.example_1
"""

import time
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from emnsr import K, volume_X1, first_k_spectral_parameters
from emnsr.spectral_features import MAASS_CACHE, squarefree_Ns
from emnsr.feature_vectors import GEO_POOL
from symbolic_regression.features import build_geometric_dataset
from symbolic_regression.utils import (evaluate, mre_prime_composite, per_k_errors,
                                       plot_pred_vs_actual, plot_mse_vs_k,
                                       pareto_front, run_pysr, per_k_volume_search,
                                       SR_TIME_LIMIT)


def stage_a(cache=MAASS_CACHE, Ns=squarefree_Ns):
    """Per-k search on the lambda target; return
    (Xg, yg, namesg, levelsg, ks_matched, a_k, b_k)."""
    Xg, yg_r, namesg, levelsg = build_geometric_dataset(Ns, K, cache, GEO_POOL)
    yg = yg_r ** 2 + 0.25                     # lambda target (eigenvalue level)
    print("Dataset:", Xg.shape, "features:", namesg, "| target = lambda_k")

    kcounts = {int(k): int((Xg[:, 0] == k).sum()) for k in np.unique(Xg[:, 0])}
    k_values = [k for k in sorted(kcounts) if kcounts[k] >= 10]
    print(f"Per-k searches queued: {len(k_values)} (k with >=10 levels)\n", flush=True)

    _t = time.time()
    res = per_k_volume_search(Xg, yg, namesg, k_values, k_col=0,
                              deadline_s=SR_TIME_LIMIT, per_k_iters=30, seed=0,
                              square=True)
    print(f"\nper-k search wall-clock: {time.time() - _t:.0f}s")

    ks_matched = np.array([k for k in sorted(res)
                           if res[k]["matched"]
                           and np.isfinite(res[k]["a"]) and np.isfinite(res[k]["b"])
                           and res[k]["a"] > 0])
    a_k = np.array([res[k]["a"] for k in ks_matched])
    b_k = np.array([res[k]["b"] for k in ks_matched])
    print(f"searches done: {len(res)};  matched a/(sqrt(Vol) - b): {len(ks_matched)}")
    print("matched k:", list(map(int, ks_matched)))
    for k, a, b in zip(ks_matched, a_k, b_k):
        print(f"  k={int(k):>3}  a={a:9.4f}  b={b:8.4f}  a/k={a / max(int(k), 1):7.4f}"
              f"   {res[int(k)]['expr']}")
    return Xg, yg, namesg, levelsg, ks_matched, a_k, b_k


def stage_b(Xg, yg, levelsg, ks_matched, a_k, b_k, cache=MAASS_CACHE):
    """Regress a(k), b(k); resolve the single law and report metrics + figures.

    maxsize is kept small (9) so the 1-D fits stay pole-free and extrapolate
    cleanly beyond the matched k (a larger maxsize overfits the ~20 noisy points
    with rational poles -> catastrophic extrapolation)."""
    assert len(ks_matched) >= 5, "too few matched k for a stable second-level fit"
    Xk = ks_matched.reshape(-1, 1).astype(float)

    print("Regressing a(k) ...", flush=True)
    mA, a_expr, a_pred = run_pysr(Xk, a_k, ["k"], niterations=80, maxsize=9,
                                  timeout_s=SR_TIME_LIMIT, seed=0)
    print("Regressing b(k) ...", flush=True)
    mB, b_expr, b_pred = run_pysr(Xk, b_k, ["k"], niterations=80, maxsize=9,
                                  timeout_s=SR_TIME_LIMIT, seed=0)
    print("\nDiscovered  a(k) =", a_expr)
    print("Discovered  b(k) =", b_expr)

    kS, vS = sp.Symbol("k"), sp.Symbol("Vol_X1")
    print("\nResolved law  lambda_k(N) =", a_expr / (vS ** sp.Rational(1, 2) - b_expr))
    print("\na(k) Pareto front (complexity / loss / equation):\n", pareto_front(mA))
    print("\nb(k) Pareto front (complexity / loss / equation):\n", pareto_front(mB))

    # discovered a_k, b_k sequences and their fitted trends
    kk = np.linspace(ks_matched.min(), ks_matched.max(), 200)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].scatter(ks_matched, a_k, s=35, zorder=3, label=r"$a_k$ (per-$k$ SR)")
    ax[0].plot(kk, a_pred(kk.reshape(-1, 1)), "C1", lw=2, label="discovered $a(k)$")
    ax[0].set_xlabel("$k$"); ax[0].set_ylabel("$a_k$"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].scatter(ks_matched, b_k, s=35, zorder=3, label=r"$b_k$ (per-$k$ SR)")
    ax[1].plot(kk, b_pred(kk.reshape(-1, 1)), "C1", lw=2, label="discovered $b(k)$")
    ax[1].set_xlabel("$k$"); ax[1].set_ylabel("$b_k$"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.show()

    def predict(X):
        k = X[:, 0].reshape(-1, 1)
        vol1 = X[:, 1]
        return a_pred(k) / (vol1 ** 0.5 - b_pred(k))

    mg = evaluate(Xg, yg, predict, levelsg)
    mre = mre_prime_composite(yg, mg["y_hat"], levelsg)
    per_k = per_k_errors(Xg, yg, predict)
    print("\n-- metrics (resolved lambda formula) --")
    print(f"  eigenvalue R^2       = {mg['r2']:.4f}")
    print(f"  Spearman rho         = {mg['spearman']:.4f}")
    print(f"  median per-level R^2 = {mg['median_per_level_r2']:.4f}")
    print(f"  overall MRE          = {mg['mre']:.2f}%")
    print(f"  MRE prime levels     = {mre['prime']:.2f}%")
    print(f"  MRE composite levels = {mre['composite']:.2f}%")
    print("  per-k MSE:")
    for k in [1, 5, 10, 20, 40]:
        if k in per_k:
            print(f"    k={k:>2}: MSE={per_k[k]['mse']:.4f}, MAE={per_k[k]['mae']:.4f}")

    # predicted vs actual at N = 3 (lambda level)
    N_show = 3
    r_actual = np.asarray(first_k_spectral_parameters(N_show, 20, cache), float)
    lam_actual = r_actual ** 2 + 0.25
    Xshow = np.column_stack([
        np.arange(1, lam_actual.size + 1, dtype=float),
        np.full(lam_actual.size, volume_X1(N_show)),
        np.zeros(lam_actual.size), np.zeros(lam_actual.size), np.zeros(lam_actual.size),
    ])
    plot_pred_vs_actual(lam_actual, predict(Xshow), N_show,
                        ylabel=r"Eigenvalue $\lambda_k$", kmax=20)
    plt.show()
    plot_mse_vs_k(per_k); plt.show()


if __name__ == "__main__":
    Xg, yg, namesg, levelsg, ks_matched, a_k, b_k = stage_a()
    stage_b(Xg, yg, levelsg, ks_matched, a_k, b_k)
