"""Spectral-parameter arrays for the three surfaces.

Building this module downloads the Maass newform dataset (Zenodo 15490636) and
the Bolza eigenvalue file (arXiv:1110.2150), then assembles:

    X0N_r_vals    padded matrix of Maass newform spectral parameters, one row
                  per squarefree level in ``squarefree_Ns``
    bolza_r_vals  positive spectral parameters of the Bolza surface
    klein_r_vals  spectral parameters of the Klein quartic (with multiplicity)

``MAASS_CACHE`` is the single parse of the Maass file, reused downstream.
"""

import numpy as np
import urllib.request

from .config import NMAX, K
from .nt_features import squarefree_levels
from .data import spectral_parameters_by_level, first_k_spectral_parameters

# ---- X_0(N): padded matrix of Maass newform spectral parameters ------
squarefree_Ns = squarefree_levels(NMAX)
MAASS_CACHE = spectral_parameters_by_level()

_rows = [first_k_spectral_parameters(N, K, MAASS_CACHE) for N in squarefree_Ns]
_maxk = max(len(r) for r in _rows)
X0N_r_vals = np.zeros((len(squarefree_Ns), _maxk), dtype=np.float32)
for i, r in enumerate(_rows):
    X0N_r_vals[i, :len(r)] = r

# ---- Bolza surface: eigenvalues from Stohmaier & Uski (arXiv:1110.2150)
# r_n = sqrt(lambda_n - 1/4); only positive eigenvalues are retained.
_BOLZA_URL = "https://arxiv.org/src/1110.2150v4/anc/eig-bolza-refined0-1000.txt"
_bolza_lambda = np.array(
    [float(x) for x in urllib.request.urlopen(_BOLZA_URL).read().decode().split()]
)
_bolza_lambda = _bolza_lambda[_bolza_lambda > 0.0]
bolza_r_vals = np.sqrt(np.maximum(_bolza_lambda - 0.25, 0.0))

# ---- Klein quartic: spectral parameters (with multiplicity) ----------
klein_r_vals = np.array([
    1.555177, 1.555177, 1.555177, 1.555177, 1.555177, 1.555177, 1.555177, 1.555177,
    2.507492, 2.507492, 2.507492, 2.507492, 2.507492, 2.507492, 2.507492,
    3.252553, 3.252553, 3.252553, 3.252553, 3.252553, 3.252553,
    3.456486, 3.456486, 3.456486, 3.456486, 3.456486, 3.456486, 3.456486, 3.456486,
    4.140797, 4.140797, 4.140797, 4.140797, 4.140797, 4.140797, 4.140797,
    4.658500, 4.658500, 4.658500, 4.658500, 4.658500, 4.658500, 4.658500,
    4.889904, 4.889904, 4.889904, 4.889904, 4.889904, 4.889904, 4.889904, 4.889904,
    5.068157, 5.068157, 5.068157, 5.068157, 5.068157, 5.068157,
    5.481241, 5.481241, 5.481241, 5.481241, 5.481241, 5.481241,
    6.022913, 6.022913, 6.022913, 6.022913, 6.022913, 6.022913, 6.022913, 6.022913,
    6.106100, 6.106100, 6.106100, 6.106100, 6.106100, 6.106100, 6.106100, 6.106100,
    6.424379, 6.424379, 6.424379, 6.424379, 6.424379, 6.424379,
    6.682302, 6.682302, 6.682302, 6.682302, 6.682302, 6.682302, 6.682302, 6.682302,
    6.958684, 6.958684, 6.958684, 6.958684, 6.958684, 6.958684,
    7.062952, 7.062952, 7.062952, 7.062952, 7.062952, 7.062952,
])
