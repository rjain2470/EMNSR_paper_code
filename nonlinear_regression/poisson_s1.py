"""Poisson summation on S^1: recover the integer spectral parameters of the flat
circle by nonlinear regression against Gaussian test functions.

On S^1 the Selberg trace formula reduces to the Poisson summation formula, whose
spectrum is r_k = k (k = 1..K), each with multiplicity 2 (the +-n eigenvalues of
the flat Laplacian). Recovered by multi-start Trust-Region Reflective least
squares.

Run:  python -m nonlinear_regression.poisson_s1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from emnsr import SEED

K_S1 = 40
N_STARTS = 100
J_S1 = 200
sig_lo, sig_hi = 0.25, 5.00

rng_S1 = np.random.default_rng(SEED)
sigmas = rng_S1.uniform(sig_lo, sig_hi, size=J_S1)

true_rho_S1 = np.arange(1, K_S1 + 1, dtype=float)
spec_mults_S1 = 2.0 * np.ones(K_S1)
M_S1 = K_S1

# Poisson: sum_{n in Z} h(n) = sum_{n in Z} Hhat(n). For h_j(r) = exp(-r^2/(2 sigma^2)),
# Hhat_j(n) = sigma_j sqrt(2 pi) exp(-2 pi^2 sigma_j^2 n^2).
# Target = sum_n Hhat_j(n) - h_j(0) = 2 sum_{n>=1} h_j(n); the n-sum truncates at |n| <= 50.
ns = np.arange(-50, 51, dtype=float)
Hhat_j = (sigmas[:, None] * np.sqrt(2 * np.pi)
          * np.exp(-2.0 * np.pi ** 2 * sigmas[:, None] ** 2 * ns[None, :] ** 2))
target_S1 = Hhat_j.sum(axis=1) - 1.0


def residual_S1(rho):
    H = np.exp(-rho[None, :] ** 2 / (2.0 * sigmas[:, None] ** 2))
    return (H * spec_mults_S1[None, :]).sum(axis=1) - target_S1


def jacobian_S1(rho):
    H = np.exp(-rho[None, :] ** 2 / (2.0 * sigmas[:, None] ** 2))
    dH = H * (-rho[None, :] / sigmas[:, None] ** 2)
    return dH * spec_mults_S1[None, :]


def main():
    best_sol, best_cost = None, np.inf
    print(f"S1  K={K_S1}, J={J_S1}, M={M_S1}, N_STARTS={N_STARTS}")
    for s in range(N_STARTS):
        rho0 = np.sort(rng_S1.uniform(0.5, K_S1 + 0.5, size=M_S1))
        try:
            sol = least_squares(residual_S1, rho0, jac=jacobian_S1,
                                bounds=(0.01, K_S1 + 1.0), method='trf',
                                xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=300)
            if np.isfinite(sol.cost) and sol.cost < best_cost:
                best_cost, best_sol = sol.cost, sol
        except Exception:
            pass
        if (s + 1) % 100 == 0:
            print(f"  restart {s+1}/{N_STARTS},  best cost = {best_cost:.4e}")

    rho_rec = np.sort(best_sol.x)
    true_r = np.repeat(true_rho_S1, 2)
    r_rec = np.repeat(rho_rec, 2)
    r2 = 1 - np.sum((r_rec - true_r) ** 2) / np.sum((true_r - true_r.mean()) ** 2)
    rmse = np.sqrt(np.mean((r_rec - true_r) ** 2))
    print(f"\nR^2 = {r2:.6f},  RMSE = {rmse:.4e},  Cost = {best_cost:.4e}")

    sym_true = np.concatenate([-true_rho_S1[::-1], true_rho_S1])
    sym_rec = np.concatenate([-rho_rec[::-1], rho_rec])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    lo, hi = -K_S1 - 1, K_S1 + 1
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='y = x')
    ax.scatter(sym_true, sym_rec, s=40, c='steelblue', zorder=3)
    ax.set_xlabel('Real spectral parameters $r_k$')
    ax.set_ylabel('Recovered spectral parameters $r_k$')
    ax.set_title('Recovered vs real Poisson spectral parameters')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.bar(range(1, M_S1 + 1), np.abs(rho_rec - true_rho_S1), color='steelblue', alpha=0.8)
    ax.set_xlabel('Index $k$'); ax.set_ylabel(r'$|\hat{\rho}_k - \rho_k|$')
    ax.set_title('Recovery error per parameter')
    ax.set_yscale('log'); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
