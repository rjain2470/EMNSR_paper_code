"""
Recovers Poisson spectral support points {s_k} via stochastic
Levenberg-Marquardt optimization with optional Adam warm-up.
 
Exports:
  - gaussian_probe: Gaussian test function and its Fourier transform
  - build_rhs: vectorized RHS of the probe equations
  - lhs_and_jac: LHS values and Jacobian for a batch of probes
  - adam_warmup: Adam-based global exploration
  - lm_step: one stochastic LM iteration
  - run_solver: full solver; returns (best_s, loss_hist)
"""
 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
 
 
# =============================================================================
#  1.  Gaussian test function
# =============================================================================
 
def gaussian_probe(c=0.0, sigma=1.0):
    def h(x):    return np.exp(-0.5*((x - c)/sigma)**2)
    def hhat(xi): return np.sqrt(2*np.pi)*sigma*np.exp(-2*(np.pi*sigma*xi)**2)*np.exp(-2j*np.pi*c*xi)
    return h, hhat
 
 
# =============================================================================
#  2.  Hyperparameters
# =============================================================================
 
K           = 250    # number of support points to learn
L_TRUNC     = 100    # Dirac comb truncation
N_PROBES    = 10000  # number of Gaussian probes
SIGMA_MIN   = 0.25
SIGMA_MAX   = 0.8
 
LM_ITERS    = 200    # LM iterations
LM_LAMBDA0  = 1e-2   # initial LM damping
 
BATCH_SIZE  = 2000   # probes per LM/Adam step
 
USE_ADAM    = True   # enable Adam warm-up
ADAM_STEPS  = 1000
ADAM_LR     = 0.015
SEED        = 0
 
# Fixed trust-region defaults
_LM_DEC, _LM_INC = 0.5, 2.0
 
 
# =============================================================================
#  3.  Core utilities
# =============================================================================
 
def build_rhs(centers, sigmas, L):
    """Vectorized RHS_j = sum_{|n|≤L} σ√(2π)e^{-2(πσn)^2}cos(2πnc)."""
    n     = np.arange(-L, L + 1)
    n_mat = n[None, :]
    c     = centers[:, None]
    s     = sigmas[:, None]
    return np.sum(
        s * np.sqrt(2*np.pi) * np.exp(-2*(np.pi*s*n_mat)**2) * np.cos(2*np.pi*n_mat*c),
        axis=1
    )
 
 
def lhs_and_jac(s, centers, sigmas, idx=None):
    """Compute LHS and Jacobian on (optionally) a mini-batch of probes."""
    if idx is None:
        c, sg = centers, sigmas
    else:
        c, sg = centers[idx], sigmas[idx]
    invsig2 = 1.0 / (sg[:, None]**2)
    x       = s[None, :] - c[:, None]
    h       = np.exp(-0.5 * (x**2) * invsig2)
    lhs     = np.sum(h, axis=1)
    J       = -x * invsig2 * h
    return lhs, J
 
 
# =============================================================================
#  4.  Optimizers
# =============================================================================
 
def adam_warmup(s, centers, sigmas, rhs, rng):
    """Global exploration via Adam on random mini-batches."""
    m = np.zeros_like(s)
    v = np.zeros_like(s)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, ADAM_STEPS + 1):
        idx      = rng.choice(len(centers), size=BATCH_SIZE, replace=False)
        lhs, J   = lhs_and_jac(s, centers, sigmas, idx)
        r        = lhs - rhs[idx]
        g        = 2.0 * (J.T @ r)
        m        = b1*m + (1-b1)*g
        v        = b2*v + (1-b2)*(g*g)
        mhat     = m / (1 - b1**t)
        vhat     = v / (1 - b2**t)
        s        = np.sort(s - ADAM_LR * mhat / (np.sqrt(vhat) + eps))
    lhs_full, _  = lhs_and_jac(s, centers, sigmas)
    loss         = float(np.dot(lhs_full - rhs, lhs_full - rhs))
    return s, loss
 
 
def lm_step(s, centers, sigmas, rhs, lam, rng):
    """One stochastic LM step (mini-batch residuals, full-loss check)."""
    idx      = rng.choice(len(centers), size=BATCH_SIZE, replace=False)
    lhs, J   = lhs_and_jac(s, centers, sigmas, idx)
    r        = lhs - rhs[idx]
    A        = J.T @ J + lam * np.eye(len(s))
    delta    = np.linalg.solve(A, -J.T @ r)
    s_cand   = np.sort(s + delta)
    lhs_full, _ = lhs_and_jac(s_cand, centers, sigmas)
    loss_full   = float(np.dot(lhs_full - rhs, lhs_full - rhs))
    return s_cand, loss_full
 
 
# =============================================================================
#  5.  Main solver
# =============================================================================
 
def run_solver():
    rng   = np.random.default_rng(SEED)
    s_min, s_max = -(K - 1)/2, (K - 1)/2
    s     = np.linspace(s_min, s_max, K)
 
    # Build probe data
    centers = rng.uniform(s_min, s_max, N_PROBES)
    sigmas  = rng.uniform(SIGMA_MIN, SIGMA_MAX, N_PROBES)
    rhs     = build_rhs(centers, sigmas, L_TRUNC)
 
    lam = LM_LAMBDA0
 
    # Optional Adam warm-up
    if USE_ADAM:
        s, loss = adam_warmup(s, centers, sigmas, rhs, rng)
    else:
        lhs0, _ = lhs_and_jac(s, centers, sigmas)
        loss    = float(np.dot(lhs0 - rhs, lhs0 - rhs))
 
    loss_hist, lam_hist = [loss], [lam]
 
    # LM iterations
    for it in range(LM_ITERS):
        s_cand, loss_cand = lm_step(s, centers, sigmas, rhs, lam, rng)
        if loss_cand < loss:
            s, loss, lam = s_cand, loss_cand, max(lam * _LM_DEC, 1e-12)
        else:
            lam *= _LM_INC
        loss_hist.append(loss)
        lam_hist.append(lam)
 
    return s, np.array(loss_hist)
 
 
# =============================================================================
#  6.  Run and evaluate
# =============================================================================
 
best_s, loss_hist = run_solver()
 
k = 20  # Range of values
n = np.arange(125 - k, 125 + k)
 
# Coordinates: (x, y) = (real r_k, recovered s_k)
x = n - 124
y = best_s[n]
 
# Plot recovered vs. real spectral parameters
plt.figure(figsize=(6, 5))
plt.scatter(x, y, color='royalblue', s=35)
plt.xlabel("Real spectral parameters r_k")
plt.ylabel("Recovered spectral parameters r_k")
plt.grid(True, linestyle='--', alpha=0.6)
plt.title("Recovered vs. real Poisson spectral parameters")
 
min_val = min(x.min(), y.min())
max_val = max(x.max(), y.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
plt.legend()
plt.show()
# R^2 score
r2 = r2_score(x, y)
print(f"R^2 value: {r2:.6f}")
