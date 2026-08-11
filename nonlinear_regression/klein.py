"""Klein quartic: recover the first K spectral parameters by STF-constrained
regression, reusing the shared trace-formula machinery.

Run:  python -m nonlinear_regression.klein
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from emnsr import K, SEED
from emnsr.spectral import klein_r_vals
from emnsr.features import klein_lengths, klein_mults
from nonlinear_regression.stf import (
    gl_quadrature, h_bump, h_trivial, identity_term, geometric_term,
    compress_multiplicities,
)

A_KQ, VOL_KQ, Q_KQ = 8.0, 8.0 * np.pi, 512
L_CUTOFF_KQ = 33.0        # covers all klein_lengths (max ~ 32.58)
u_kq, wq_kq, phi_kq = gl_quadrature(Q_KQ, A_KQ)

r_klein = np.asarray(klein_r_vals, float)


def regress(n_starts=300):
    true_r = r_klein[:K]
    true_rho, spec_mults = compress_multiplicities(true_r)
    M = len(true_rho)
    print(f"Klein quartic: K={K}, M={M} distinct locations")

    T_lo = max(2.5, true_rho[-1] + 0.05)
    T_hi = 20.0
    T_vals = np.unique(np.concatenate([
        np.linspace(T_lo, min(T_lo + 1.0, T_hi), 100),
        np.linspace(min(T_lo + 1.0, T_hi), T_hi, 150),
    ]))
    J = len(T_vals)
    print(f"Probe functions: J={J},  T in [{T_vals[0]:.3f}, {T_vals[-1]:.3f}]")

    c_h = identity_term(T_vals, u_kq, wq_kq, phi_kq, VOL_KQ)
    geo = geometric_term(T_vals, klein_lengths, klein_mults, L_CUTOFF_KQ, u_kq, wq_kq, phi_kq)
    trivial = h_trivial(T_vals, A_KQ)
    tail = np.array([h_bump(r_klein[K:], T, A_KQ).sum() for T in T_vals])
    modified_target = c_h + geo - trivial - tail

    direct_K = np.array([(h_bump(true_rho, T, A_KQ) * spec_mults).sum() for T in T_vals])
    err = np.linalg.norm(modified_target - direct_K) / np.linalg.norm(direct_K)
    print(f"Target consistency (STF check): {err:.3e}")

    def _h_and_dh(rho, T_arr):
        T = np.asarray(T_arr, float)[:, None]
        y = rho[None, :] / T
        mask = np.abs(y) < 1.0
        safe = np.where(mask, 1.0 - y ** 2, 1.0)
        H = np.where(mask, np.exp(-A_KQ / safe), 0.0)
        dH = np.where(mask, H * (-2.0 * A_KQ * rho[None, :] / T ** 2) / safe ** 2, 0.0)
        return H, dH

    def residual_fn(rho):
        H, _ = _h_and_dh(rho, T_vals)
        return (H * spec_mults[None, :]).sum(1) - modified_target

    def jacobian_fn(rho):
        _, dH = _h_and_dh(rho, T_vals)
        return dH * spec_mults[None, :]

    rng = np.random.default_rng(SEED)
    best_sol, best_cost = None, np.inf
    print(f"Running {n_starts} restarts (M={M}, J={J})...")
    for s in range(n_starts):
        rho0 = np.sort(rng.uniform(1.0, T_vals[-1] - 1.0, size=M))
        try:
            sol = least_squares(residual_fn, rho0, jac=jacobian_fn,
                                bounds=(0.01, T_vals[-1] - 0.01), method='trf',
                                xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=10_000)
            if np.isfinite(sol.cost) and sol.cost < best_cost:
                best_cost, best_sol = sol.cost, sol
        except Exception:
            pass
        if (s + 1) % 100 == 0:
            print(f"  restart {s+1}/{n_starts},  best cost = {best_cost:.4e}")

    rho_rec = np.sort(best_sol.x)
    r_rec = np.repeat(rho_rec, spec_mults.astype(int))
    r2 = 1 - np.sum((r_rec - true_r) ** 2) / np.sum((true_r - true_r.mean()) ** 2)
    rmse = np.sqrt(np.mean((r_rec - true_r) ** 2))
    print(f"\nR^2 = {r2:.6f},  RMSE = {rmse:.4e},  Cost = {best_cost:.4e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    lo = min(true_r.min(), r_rec.min()) * 0.92
    hi = max(true_r.max(), r_rec.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='y = x')
    ax.scatter(true_r, r_rec, s=40, c='steelblue', zorder=3)
    ax.set_xlabel('True spectral parameters $r_k$'); ax.set_ylabel('Recovered $r_k$')
    ax.set_title('Klein quartic: recovered vs true'); ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.bar(range(1, M + 1), np.abs(rho_rec - true_rho), color='steelblue', alpha=0.8)
    ax.set_xlabel('Distinct location index $a$'); ax.set_ylabel(r'$|\hat{\rho}_a - \rho_a|$')
    ax.set_title('Recovery error per distinct location'); ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    regress()
