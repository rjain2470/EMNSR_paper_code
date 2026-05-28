"""
bolza_stf.py  —  Shared utilities for Bolza surface Selberg trace formula computations.

References
----------
Stohmaier & Uski (2011), arXiv:1110.2150  — eigenvalue data
Aurich & Steiner (1988), Physica D 32:451  — geodesic lengths and multiplicities
"""

import numpy as np
import urllib.request

# =============================================================================
# Constants
# =============================================================================
A   = 8.0           # bump exponent
VOL = 4.0 * np.pi  # Bolza surface volume  (genus 2, χ = −2, Vol = −2πχ)
Q   = 512           # Gauss–Legendre nodes

# =============================================================================
# Eigenvalue data
# =============================================================================
_EIGENVALUE_URL = (
    "https://arxiv.org/src/1110.2150v4/anc/eig-bolza-refined0-1000.txt"
)

def load_eigenvalues(url=_EIGENVALUE_URL):
    """
    Return positive spectral parameters r_n = sqrt(λ_n − 1/4) for the Bolza
    surface (Stohmaier & Uski 2011).  The trivial eigenvalue λ_0 = 0 is
    excluded; its contribution h_T(i/2) is handled analytically by h_trivial().
    """
    raw     = urllib.request.urlopen(url).read().decode()
    lam_all = np.array([float(x) for x in raw.strip().split()])
    lam_pos = lam_all[lam_all > 0.0]
    return np.sqrt(np.maximum(lam_pos - 0.25, 0.0))

# =============================================================================
# Primitive geodesic data  (Aurich & Steiner 1988)
#
# Lengths in strictly ascending order.
# Indices 35–36 corrected from some circulated versions:
#   (10.7270…, mult 96) precedes (10.7308…, mult 272).
# =============================================================================
BOLZA_LENGTHS = np.array([
  3.0571148390, 4.8969048954, 5.8280707754, 6.1128364779, 6.6720057699,
  7.1073578414, 7.2631634751, 7.5956918304, 7.8806928877, 8.1300753289,
  8.2249036323, 8.4368496405, 8.6284635656, 8.7027505564, 8.8714798107,
  9.0270171797, 9.1714255169, 9.2282950896, 9.3592716579, 9.4821914493,
  9.5309770571, 9.6440665486, 9.7510997583, 9.7938097907, 9.8980946367,
 10.0785887303,10.1149054144,10.1999558888,10.2815633765,10.3143770353,
 10.3915072941,10.4657729697,10.5373792543,10.5663604642,10.6344594159,
 10.7270445783,10.7308060283,10.7901078989,10.8510687003,10.8758208811,
 10.9343449467,10.9912049969,11.0464930504,11.0689538604,11.1226185852,
 11.1739905306,11.2450675676,11.2938444648,11.3416400077,11.3608557221,
 11.4060920010,11.4519475300,11.4703055581,11.5139343106,11.5566494092,
 11.5984625221,11.6159252909,11.6561415509,11.6954535600,11.7122036393,
 11.7509179318,11.7858970234,11.8044196533,11.8410454395,11.8777196407,
 11.9133862138,11.9279754971,11.9627646508,11.9969589861,12.0443403136,
 12.0771791386,12.1098474621,12.1227186314,12.1543053200,12.1854008904,
 12.1981480207,12.2285673558,12.2585379275,12.2708217905
])

BOLZA_MULTS = np.array([
 24,24,48,24,96,48,48,8,96,48,192,48,96,48,288,12,48,96,192,48,192,96,336,24,192,
 192,96,384,96,192,384,96,288,96,272,96,272,560,40,96,432,96,288,288,384,544,272,
 272,96,272,672,192,464,48,648,40,544,192,416,288,496,96,192,352,544,512,352,384,
 384,544,288,352,96,800,352,368,48,736,96
])

assert len(BOLZA_LENGTHS) == len(BOLZA_MULTS) == 79
assert np.all(np.diff(BOLZA_LENGTHS) > 0), \
    "Lengths not strictly increasing — check swap at indices 35–36."

# =============================================================================
# Gauss–Legendre quadrature
#
# Integrals on [0, T] are mapped to [0, 1] via r = T·u.
# φ(u) = h_T(T·u) = exp(−A/(1−u²)) is identical for every T.
# =============================================================================
_xgl, _wgl = np.polynomial.legendre.leggauss(Q)
_u   = 0.5 * (_xgl + 1.0)         # nodes  in [0, 1]
_w   = 0.5 * _wgl                  # weights (sum to 1)
_phi = np.exp(-A / (1.0 - _u**2)) # φ(u),  shape (Q,)

# =============================================================================
# Test function family  h_T(r) = exp(−A/(1−(r/T)²)),  |r| < T
# =============================================================================

def h_bump(r, T):
    """Evaluate h_T at an array of r values.  h_T is even and compactly supported."""
    r    = np.asarray(r, dtype=float)
    y    = r / T
    mask = np.abs(y) < 1.0
    return np.where(mask, np.exp(-A / np.where(mask, 1.0 - y**2, 1.0)), 0.0)

def h_trivial(T_arr):
    """
    Contribution of the trivial eigenvalue λ_0 = 0  (spectral parameter r_0 = i/2):
        h_T(i/2) = exp(−A / (1 + 1/(4T²)))
    This term is real-valued and O(1); omitting it causes O(1) errors.
    """
    return np.exp(-A / (1.0 + 1.0 / (4.0 * np.asarray(T_arr, dtype=float)**2)))

# =============================================================================
# STF components
# =============================================================================

def identity_term(T_arr):
    """
    c_T = (Vol/4π) ∫_{−∞}^{∞} r h_T(r) tanh(πr) dr
         = (Vol/2π) ∫_0^T  r h_T(r) tanh(πr) dr     [h_T even]
    """
    T_arr     = np.asarray(T_arr, dtype=float)         # (J,)
    r_nodes   = T_arr[:, None] * _u[None, :]           # (J, Q)
    r_weights = T_arr[:, None] * _w[None, :]
    return (VOL / (2.0 * np.pi)) * (
        r_weights * r_nodes * _phi[None, :] * np.tanh(np.pi * r_nodes)
    ).sum(axis=1)


def hhat_matrix(T_arr, ells):
    """
    Iwaniec Fourier transform evaluated at geodesic lengths:
        g_{T_j}(ℓ_e) = (T_j/π) Σ_q w_q φ(u_q) cos(T_j u_q ℓ_e)

    Returns shape (J, E).
    """
    T_arr = np.asarray(T_arr, dtype=float)   # (J,)
    ells  = np.asarray(ells,  dtype=float)   # (E,)
    arg   = T_arr[:, None, None] * _u[None, :, None] * ells[None, None, :]
    return (T_arr[:, None] / np.pi) * np.einsum(
        'q,jqe->je', _w, _phi[None, :, None] * np.cos(arg)
    )


def build_geodesic_list(L_cutoff=12.5):
    """
    Expand all primitive geodesics to iterates γ_0^m with m·ℓ_0 ≤ L_cutoff.
    Returns (ells, weights) where weight = mult · ℓ_0 / (2 sinh(ℓ/2)).
    """
    ells, weights = [], []
    for ell0, mult in zip(BOLZA_LENGTHS, BOLZA_MULTS):
        m = 1
        while m * ell0 <= L_cutoff:
            ell = m * ell0
            ells.append(ell)
            weights.append(float(mult) * ell0 / (2.0 * np.sinh(0.5 * ell)))
            m += 1
    return np.array(ells), np.array(weights)


def geometric_term(T_arr, L_cutoff=12.5):
    """
    G_T = Σ_{γ_0^m} weight(γ_0, m) · g_T(m ℓ_0)

    Uses Iwaniec's g = ĥ/(2π) and geo_factor = 1 (verified in
    bolza_stf_verify.py to give errors < 0.1% for T ≥ 2.5).
    """
    geo_ells, geo_wts = build_geodesic_list(L_cutoff)
    return hhat_matrix(T_arr, geo_ells) @ geo_wts     # (J,)

# =============================================================================
# Utility
# =============================================================================

def compress_multiplicities(x, tol=1e-6):
    """
    Compress a sorted array of spectral parameters (with repetitions) into
    (distinct_values, multiplicities).
    """
    x = np.sort(np.asarray(x, dtype=float))
    vals, mults = [], []
    for y in x:
        if not vals or abs(y - vals[-1]) > tol:
            vals.append(y); mults.append(1)
        else:
            mults[-1] += 1
    return np.array(vals), np.array(mults, dtype=float)
