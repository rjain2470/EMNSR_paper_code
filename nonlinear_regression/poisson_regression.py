# =============================================================================
# Poisson Summation Nonlinear Regression
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

K_S1      = 40
N_STARTS  = 100
SEED      = 42
J_S1      = 100
sig_lo, sig_hi = 0.25, 0.80

rng_S1 = np.random.default_rng(SEED)
sigmas = rng_S1.uniform(sig_lo, sig_hi, size=J_S1)  

# True spectral parameters: positive integers 1…K, each with multiplicity 2 (from ±n eigenvalues of the flat Laplacian on S¹)
true_rho_S1  = np.arange(1, K_S1 + 1, dtype=float)
spec_mults_S1 = 2.0 * np.ones(K_S1)
M_S1 = K_S1

# ── Poisson target ────────────────────────────────────────────────────────────
ns = np.arange(-50, 51, dtype=float)                
Hhat_j = (sigmas[:, None] * np.sqrt(2*np.pi)
          * np.exp(-2.0 * np.pi**2 * sigmas[:, None]**2 * ns[None, :]**2))
target_S1 = Hhat_j.sum(axis=1) - 1.0                        # (J,)

# ── Residual and Jacobian ─────────────────────────────────────────────────────
def residual_S1(rho):
    H = np.exp(-rho[None, :]**2 / (2.0 * sigmas[:, None]**2))  # (J, M)
    return (H * spec_mults_S1[None, :]).sum(axis=1) - target_S1

def jacobian_S1(rho):
    H  = np.exp(-rho[None, :]**2 / (2.0 * sigmas[:, None]**2))
    dH = H * (-rho[None, :] / sigmas[:, None]**2)
    return dH * spec_mults_S1[None, :]                          # (J, M)

# ── Multi-start TRF ───────────────────────────────────────────────────────────
best_sol_S1, best_cost_S1 = None, np.inf

print(f"S¹  K={K_S1}, J={J_S1}, M={M_S1}, N_STARTS={N_STARTS}")
for s in range(N_STARTS):
    rho0 = np.sort(rng_S1.uniform(0.5, K_S1 + 0.5, size=M_S1))
    try:
        sol = least_squares(
            residual_S1, rho0, jac=jacobian_S1,
            bounds=(0.01, K_S1 + 1.0),
            method='trf',
            xtol=1e-13, ftol=1e-13, gtol=1e-13,
            max_nfev=300
        )
        if np.isfinite(sol.cost) and sol.cost < best_cost_S1:
            best_cost_S1, best_sol_S1 = sol.cost, sol
    except Exception:
        pass
    if (s + 1) % 100 == 0:
        print(f"  restart {s+1}/{N_STARTS},  best cost = {best_cost_S1:.4e}")

rho_rec_S1 = np.sort(best_sol_S1.x)
true_r_S1  = np.repeat(true_rho_S1, 2)         
r_rec_S1   = np.repeat(rho_rec_S1,  2)

r2_S1   = 1 - np.sum((r_rec_S1 - true_r_S1)**2) / np.sum((true_r_S1 - true_r_S1.mean())**2)
rmse_S1 = np.sqrt(np.mean((r_rec_S1 - true_r_S1)**2))
print(f"\nR² = {r2_S1:.6f},  RMSE = {rmse_S1:.4e},  Cost = {best_cost_S1:.4e}")

# ── Plot ─────────────────────────────────
sym_true = np.concatenate([-true_rho_S1[::-1], true_rho_S1])
sym_rec  = np.concatenate([-rho_rec_S1[::-1],  rho_rec_S1])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
lo, hi = -K_S1 - 1, K_S1 + 1
ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='y = x')
ax.scatter(sym_true, sym_rec, s=40, c='steelblue', zorder=3)
ax.set_xlabel('Real spectral parameters $r_k$')
ax.set_ylabel('Recovered spectral parameters $r_k$')
ax.set_title(f'Recovered vs real Poisson spectral parameters')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.bar(range(1, M_S1 + 1), np.abs(rho_rec_S1 - true_rho_S1),
       color='steelblue', alpha=0.8)
ax.set_xlabel('Index $k$')
ax.set_ylabel(r'$|\hat{\rho}_k - \rho_k|$')
ax.set_title('Recovery error per parameter')
ax.set_yscale('log'); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout(); plt.show()

sym_true = np.concatenate([-true_rho_S1[::-1], true_rho_S1])
sym_rec  = np.concatenate([-rho_rec_S1[::-1],  rho_rec_S1])

ss_res = np.sum((sym_rec  - sym_true)**2)
ss_tot = np.sum((sym_true - sym_true.mean())**2)
r2 = 1.0 - ss_res / ss_tot
print(f"R² = {r2:.6f}")
