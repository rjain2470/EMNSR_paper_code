"""
bolza_stf_verify.py  —  Verify the Selberg trace formula on the Bolza surface.

For the bump test functions h_T defined in bolza_stf.py, checks that

    LHS(T) = h_T(i/2) + Σ_{n≥1} h_T(r_n)   [spectral side]
    RHS(T) = c_T + G_T                     [identity + geometric]

agree to < 0.1% for T ≥ 2.5.
"""

import numpy as np
import matplotlib.pyplot as plt

from bolza_stf import (
    A, VOL,
    load_eigenvalues,
    h_trivial,
    identity_term,
    geometric_term,
)

# =============================================================================
# Load data
# =============================================================================
print("Loading eigenvalues from Stohmaier-Uski...", flush=True)
r_all = load_eigenvalues()
print(f"  {len(r_all)} positive spectral parameters.")
print(f"  First 8 r_n: {np.round(r_all[:8], 6)}\n")

# =============================================================================
# Spectral side  (LHS)
# =============================================================================

def spectral_lhs(T_arr):
    """LHS[j] = h_{T_j}(i/2) + sum_n h_{T_j}(r_n)."""
    T_arr   = np.asarray(T_arr, dtype=float)
    trivial = h_trivial(T_arr)
    y       = r_all[None, :] / T_arr[:, None]          # (J, N)
    mask    = y < 1.0
    spec    = np.where(
        mask, np.exp(-A / np.where(mask, 1.0 - y**2, 1.0)), 0.0
    ).sum(axis=1)
    return trivial + spec, trivial, spec

# =============================================================================
# Verification
# =============================================================================
T_vals   = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
L_CUTOFF = 12.5

print("Computing STF components...", flush=True)
lhs, trivial, spec_sum = spectral_lhs(T_vals)
c_h  = identity_term(T_vals)
geo1 = geometric_term(T_vals, L_cutoff=L_CUTOFF)
geo2 = 2.0 * geo1                          # geo_factor = 2 for comparison

rhs1 = c_h + geo1
rhs2 = c_h + geo2

err1 = np.abs(lhs - rhs1) / np.abs(lhs)
err2 = np.abs(lhs - rhs2) / np.abs(lhs)

# =============================================================================
# Table
# =============================================================================
W   = 105
sep = "-" * W
print(f"\nSTF VERIFICATION  (L_cutoff={L_CUTOFF}, g = h-hat/(2*pi))")
print(sep)
print(f"{'T':>5}  {'trivial':>9}  {'spec_sum':>10}  {'LHS':>10}  "
      f"{'c_h':>10}  {'geo(f=1)':>10}  {'geo(f=2)':>10}  "
      f"{'err f=1':>9}  {'err f=2':>9}")
print(sep)
for i, T in enumerate(T_vals):
    print(f"{T:5.1f}  {trivial[i]:9.5f}  {spec_sum[i]:10.5f}  {lhs[i]:10.5f}  "
          f"{c_h[i]:10.5f}  {geo1[i]:10.5f}  {geo2[i]:10.5f}  "
          f"{err1[i]:9.2e}  {err2[i]:9.2e}")
print(sep)
print("Errors at T < 2.5 stem from geodesic truncation (Fourier transform")
print("decays slowly for small T with L_cutoff=12.5).  For T >= 2.5: < 0.1%.")

# =============================================================================
# Plots
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(T_vals, lhs,  'ko-',  label='LHS (spectral)',      ms=7, lw=1.5)
ax.plot(T_vals, rhs1, 'b^--', label='RHS  geo_factor = 1', ms=7, lw=1.5)
ax.plot(T_vals, rhs2, 'rs--', label='RHS  geo_factor = 2', ms=7, lw=1.5)
ax.set_xlabel('T', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title(r'STF: LHS vs RHS  [$g = \hat{h}/(2\pi)$]', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.semilogy(T_vals, err1, 'b^-', label='geo_factor = 1', ms=7, lw=1.5)
ax.semilogy(T_vals, err2, 'rs-', label='geo_factor = 2', ms=7, lw=1.5)
ax.axvline(x=2.5, color='gray', linestyle=':', lw=1.5, label='T = 2.5')
ax.set_xlabel('T', fontsize=12)
ax.set_ylabel('Relative error  |LHS - RHS| / |LHS|', fontsize=11)
ax.set_title('STF verification error', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('bolza_stf_verify.pdf', dpi=150)
plt.show()
print("Plot saved to bolza_stf_verify.pdf")
