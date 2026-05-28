"""
bolza_regression.py  —  Recovers the first K Bolza spectral parameters via STF regression.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from bolza_stf import (
    A, VOL,
    load_eigenvalues,
    h_trivial, h_bump,
    identity_term, geometric_term,
    compress_multiplicities,
)

# =============================================================================
# Hyperparameters
# =============================================================================
K        = 100     # spectral parameters to recover, counting multiplicity
N_STARTS = 300    # random restarts for the multi-start TRF
SEED     = 42
T_HI = 20.0       # max support of our test functions

# =============================================================================
# Load data
# =============================================================================
print("Loading eigenvalues...", flush=True)
r_all = load_eigenvalues()
print(f"  {len(r_all)} positive spectral parameters loaded.")
print(f"  First {min(K+2, 12)} r_n: {np.round(r_all[:K+2], 6)}\n")

# =============================================================================
# Compress first K eigenvalues
# =============================================================================
true_r               = r_all[:K]
true_rho, spec_mults = compress_multiplicities(true_r)
M                    = len(true_rho)

print("First K spectrum (distinct locations + multiplicities):")
for a, (rho, mult) in enumerate(zip(true_rho, spec_mults), 1):
    print(f"  rho_{a} = {rho:.8f}   mult = {int(mult)}")
print(f"  -> M = {M} unknowns to recover\n")

# =============================================================================
# Probe functions T_vals
# =============================================================================
T_lo = max(2.5, true_rho[-1] + 0.05)   # just above the largest true parameter
T_vals = np.unique(np.concatenate([
    np.linspace(T_lo, min(T_lo + 1.0, T_HI), 100),
    np.linspace(min(T_lo + 1.0, T_HI), T_HI, 150),
]))
J = len(T_vals)
print(f"Probe functions: J = {J},  T in [{T_vals[0]:.3f}, {T_vals[-1]:.3f}]\n")


print("Building target...", flush=True)
c_h     = identity_term(T_vals)
geo     = geometric_term(T_vals)
trivial = h_trivial(T_vals)

y_tail  = r_all[K:][None, :] / T_vals[:, None]
tail    = np.where(
    y_tail < 1.0,
    np.exp(-A / np.where(y_tail < 1.0, 1.0 - y_tail**2, 1.0)), 0.0
).sum(axis=1)

modified_target = c_h + geo - trivial - tail
y_K      = true_rho[None, :] / T_vals[:, None]
h_K_mat  = np.where(y_K < 1.0, np.exp(-A / np.where(y_K < 1.0, 1.0 - y_K**2, 1.0)), 0.0)
direct_K = (h_K_mat * spec_mults[None, :]).sum(axis=1)

target_err = np.linalg.norm(modified_target - direct_K) / np.linalg.norm(direct_K)
print(f"  Target consistency: rel error = {target_err:.3e}  (expect < 1%)\n")

# =============================================================================
# Residual and Jacobian
# =============================================================================

def h_and_dh(rho, T_arr):
    """
    H[j, a]  = h_{T_j}(rho_a)
    dH[j, a] = (d/dr) h_{T_j}(rho_a)
    """
    rho  = np.asarray(rho, dtype=float)
    T    = T_arr[:, None]
    y    = rho[None, :] / T
    mask = np.abs(y) < 1.0
    safe = np.where(mask, 1.0 - y**2, 1.0)
    H    = np.where(mask, np.exp(-A / safe), 0.0)
    dH   = np.where(mask, H * (-2.0 * A * rho[None, :] / T**2) / safe**2, 0.0)
    return H, dH

def residual_fn(rho):
    H, _ = h_and_dh(rho, T_vals)
    return (H * spec_mults[None, :]).sum(axis=1) - modified_target

def jacobian_fn(rho):
    _, dH = h_and_dh(rho, T_vals)
    return dH * spec_mults[None, :]   # (J, M)

# =============================================================================
# Multi-start TRF
# =============================================================================
rng = np.random.default_rng(SEED)
best_sol, best_cost = None, np.inf

print(f"Running {N_STARTS} random restarts (M={M} unknowns, J={J} equations)...")
for s in range(N_STARTS):
    rho0 = np.sort(rng.uniform(0.5, T_vals[-1] - 0.2, size=M))
    try:
        sol = least_squares(
            residual_fn, rho0, jac=jacobian_fn,
            bounds=(0.01, T_vals[-1] - 0.01),
            method='trf',
            xtol=1e-13, ftol=1e-13, gtol=1e-13,
            max_nfev=10_000,
        )
        if np.isfinite(sol.cost) and sol.cost < best_cost:
            best_cost, best_sol = sol.cost, sol
    except Exception:
        pass
    if (s + 1) % 100 == 0:
        print(f"  restart {s+1:3d}/{N_STARTS},  best cost = {best_cost:.4e}")

rho_rec = np.sort(best_sol.x)
r_rec   = np.repeat(rho_rec, spec_mults.astype(int))

# =============================================================================
# Diagnostics
# =============================================================================
r2   = 1.0 - np.sum((r_rec - true_r)**2) / np.sum((true_r - true_r.mean())**2)
rmse = np.sqrt(np.mean((r_rec - true_r)**2))

print(f"\n{'─'*54}")
print(f"  R2   = {r2:.8f}")
print(f"  RMSE = {rmse:.4e}")
print(f"  Cost = {best_cost:.4e}")
print(f"{'─'*54}\n")

print(f"{'a':>3}  {'true rho_a':>14}  {'recov rho_a':>14}  {'mult':>5}  {'error':>11}")
print("-" * 54)
for a, (rt, rr, mm) in enumerate(zip(true_rho, rho_rec, spec_mults), 1):
    print(f"{a:3d}  {rt:14.8f}  {rr:14.8f}  {int(mm):5d}  {rr - rt:+.4e}")
print("-" * 54)

# =============================================================================
# Plots
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
lo = min(true_r.min(), r_rec.min()) * 0.92
hi = max(true_r.max(), r_rec.max()) * 1.05
ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='y = x')
ax.scatter(true_r, r_rec, s=70, c='steelblue', zorder=3)
ax.set_xlabel('True spectral parameters $r_k$',     fontsize=12)
ax.set_ylabel('Recovered spectral parameters $r_k$', fontsize=12)
ax.set_title(f'Bolza: recovered vs true  (K={K})', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.bar(range(1, M+1), np.abs(rho_rec - true_rho), color='steelblue', alpha=0.8)
ax.set_xlabel('Distinct location index $a$', fontsize=12)
ax.set_ylabel('$|\\hat{\\rho}_a - \\rho_a|$',     fontsize=12)
ax.set_title('Recovery error per distinct location', fontsize=12)
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(range(1, M+1))

plt.tight_layout()
plt.savefig('bolza_regression.pdf', dpi=150)
plt.show()
print("Plot saved to bolza_regression.pdf")
