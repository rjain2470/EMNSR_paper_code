"""Bolza surface: verify the Selberg trace formula against the known spectrum,
then recover the first K spectral parameters by STF-constrained regression.

Spectrum: Stohmaier & Uski (2011, arXiv:1110.2150).
Geodesic lengths and multiplicities: Aurich & Steiner (1988, Physica D 32:451).

Run:  python -m nonlinear_regression.bolza
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from emnsr import K, SEED
from emnsr.spectral_features import bolza_r_vals
from emnsr.feature_vectors import bolza_lengths, bolza_mults
from nonlinear_regression.stf import (
    gl_quadrature, h_bump, h_trivial, identity_term, geometric_term,
    compress_multiplicities,
)

A_BOLZA, VOL_BOLZA, Q_BOLZA, L_CUTOFF = 8.0, 4.0 * np.pi, 512, 12.5
_u, _wq, _phi = gl_quadrature(Q_BOLZA, A_BOLZA)
r_bolza = np.asarray(bolza_r_vals, float)


def verify():
    """Check that the spectral side matches the identity + geometric side."""
    T_vals = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

    trivial = h_trivial(T_vals, A_BOLZA)
    spec_sum = np.array([h_bump(r_bolza, T, A_BOLZA).sum() for T in T_vals])
    lhs = trivial + spec_sum

    c_h = identity_term(T_vals, _u, _wq, _phi, VOL_BOLZA)
    geo = geometric_term(T_vals, bolza_lengths, bolza_mults, L_CUTOFF, _u, _wq, _phi)
    rhs = c_h + geo
    err = np.abs(lhs - rhs) / np.abs(lhs)

    sep = "-" * 78
    print(f"{'BOLZA STF VERIFICATION  (L_cutoff = ' + str(L_CUTOFF) + ')':^78}")
    print(sep)
    print(f"{'T':>5}  {'trivial':>9}  {'spec_sum':>10}  {'LHS':>10}  "
          f"{'c_h':>10}  {'geo':>10}  {'rel err':>9}")
    print(sep)
    for i, T in enumerate(T_vals):
        print(f"{T:5.1f}  {trivial[i]:9.5f}  {spec_sum[i]:10.5f}  {lhs[i]:10.5f}  "
              f"{c_h[i]:10.5f}  {geo[i]:10.5f}  {err[i]:9.2e}")
    print(sep)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(T_vals, lhs, 'ko-', label='LHS (spectral)', ms=7, lw=1.5)
    ax.plot(T_vals, rhs, 'b^--', label='RHS (identity + geometric)', ms=7, lw=1.5)
    ax.set_xlabel('T'); ax.set_ylabel('Value'); ax.set_title('Bolza STF: LHS vs RHS')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.semilogy(T_vals, err, 'b^-', ms=7, lw=1.5)
    ax.axvline(2.5, color='gray', ls=':', lw=1.5, label='T = 2.5')
    ax.set_xlabel('T'); ax.set_ylabel('Relative error |LHS - RHS| / |LHS|')
    ax.set_title('Bolza STF verification error'); ax.legend(); ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout(); plt.show()


def regress(n_starts=100):
    """Recover the first K spectral parameters by multi-start TRF least squares."""
    true_r = r_bolza[:K]
    true_rho, spec_mults = compress_multiplicities(true_r)
    M = len(true_rho)
    print(f"Bolza: K={K} eigenvalues -> M={M} distinct locations to recover")

    # Probe range: T >= 2.5 (STF converged, per verify) and just above the
    # largest rho, so every unknown sits inside the bump support.
    T_lo = max(2.5, true_rho[-1] + 0.05)
    T_hi = 20.0
    T_vals = np.unique(np.concatenate([
        np.linspace(T_lo, min(T_lo + 1.0, T_hi), 100),
        np.linspace(min(T_lo + 1.0, T_hi), T_hi, 150),
    ]))
    J = len(T_vals)
    print(f"Probe functions: J={J},  T in [{T_vals[0]:.3f}, {T_vals[-1]:.3f}]")

    # Move all known terms to the RHS so the target equals sum_a m_a h_T(rho_a).
    c_h = identity_term(T_vals, _u, _wq, _phi, VOL_BOLZA)
    geo = geometric_term(T_vals, bolza_lengths, bolza_mults, L_CUTOFF, _u, _wq, _phi)
    trivial = h_trivial(T_vals, A_BOLZA)
    tail = np.array([h_bump(r_bolza[K:], T, A_BOLZA).sum() for T in T_vals])
    modified_target = c_h + geo - trivial - tail

    direct_K = np.array([(h_bump(true_rho, T, A_BOLZA) * spec_mults).sum() for T in T_vals])
    rel = np.linalg.norm(modified_target - direct_K) / np.linalg.norm(direct_K)
    print(f"Target consistency (STF check): rel error = {rel:.3e}")

    def _h_and_dh(rho, T_arr):
        T = np.asarray(T_arr, float)[:, None]
        y = rho[None, :] / T
        mask = np.abs(y) < 1.0
        safe = np.where(mask, 1.0 - y ** 2, 1.0)
        H = np.where(mask, np.exp(-A_BOLZA / safe), 0.0)
        dH = np.where(mask, H * (-2.0 * A_BOLZA * rho[None, :] / T ** 2) / safe ** 2, 0.0)
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
        rho0 = np.sort(rng.uniform(0.5, T_vals[-1] - 0.2, size=M))
        try:
            sol = least_squares(residual_fn, rho0, jac=jacobian_fn,
                                bounds=(0.01, T_vals[-1] - 0.01), method='trf',
                                xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=10_000)
            if np.isfinite(sol.cost) and sol.cost < best_cost:
                best_cost, best_sol = sol.cost, sol
        except Exception:
            pass
        if (s + 1) % 50 == 0:
            print(f"  restart {s+1:3d}/{n_starts},  best cost = {best_cost:.4e}")

    rho_rec = np.sort(best_sol.x)
    r_rec = np.repeat(rho_rec, spec_mults.astype(int))
    r2 = 1.0 - np.sum((r_rec - true_r) ** 2) / np.sum((true_r - true_r.mean()) ** 2)
    rmse = np.sqrt(np.mean((r_rec - true_r) ** 2))
    print(f"\nR^2 = {r2:.6f},  RMSE = {rmse:.4e},  Cost = {best_cost:.4e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    lo = min(true_r.min(), r_rec.min()) * 0.92
    hi = max(true_r.max(), r_rec.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='y = x')
    ax.scatter(true_r, r_rec, s=70, c='steelblue', zorder=3)
    ax.set_xlabel('True spectral parameters $r_k$'); ax.set_ylabel('Recovered $r_k$')
    ax.set_title(f'Bolza: recovered vs true  (K={K})'); ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.bar(range(1, M + 1), np.abs(rho_rec - true_rho), color='steelblue', alpha=0.8)
    ax.set_xlabel('Distinct location index $a$'); ax.set_ylabel(r'$|\hat{\rho}_a - \rho_a|$')
    ax.set_title('Recovery error per distinct location'); ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y'); ax.set_xticks(range(1, M + 1))
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    verify()
    regress()
