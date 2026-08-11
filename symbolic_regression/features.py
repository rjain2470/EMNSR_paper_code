"""Feature-matrix builders for the two symbolic-regression experiments. Both cap
at K spectral parameters per level and range over the squarefree levels passed
in."""

import numpy as np

from emnsr import (phi, mu, sigma, volume_X0, volume_X1, genus_X0,
                   geodesic_lengths_X0, first_k_eigenvalues,
                   first_k_spectral_parameters)


def build_arithmetic_dataset(Ns, K, cache=None):
    """Arithmetic experiment dataset.

    Features (k, N, Vol(X_0(N)), phi(N), mu(N), sigma(N)).
    Target   lambda_k(N) = r_k(N)^2 + 1/4.
    """
    rows, y = [], []
    for N in Ns:
        vol0 = volume_X0(N)
        p, m, s = phi(N), mu(N), sigma(N)
        for k, lam in enumerate(first_k_eigenvalues(N, K, cache), start=1):
            rows.append([float(k), float(N), vol0, float(p), float(m), float(s)])
            y.append(float(lam))
    names = ["k", "N", "vol", "phi", "mu", "sigma"]
    return np.array(rows, float), np.array(y, float), names


def build_geometric_dataset(Ns, K, cache=None, geo_pool=None):
    """Geometric experiment dataset.

    Features (k, Vol(X_1(N)), genus(N), l_1(N), l_2(N)).
    Target   r_k(N) = sqrt(lambda_k(N) - 1/4)  (the spectral parameter).

    Also returns ``levels``, the level N of each row (N is not a feature here
    but is needed for per-level / prime-vs-composite reporting).
    """
    rows, y, levels = [], [], []
    for N in Ns:
        vol1 = volume_X1(N)
        g = genus_X0(N)
        gl = geodesic_lengths_X0(N, 2, geo_pool)
        l1 = gl[0] if len(gl) > 0 else 0.0
        l2 = gl[1] if len(gl) > 1 else 0.0
        for k, r in enumerate(first_k_spectral_parameters(N, K, cache), start=1):
            rows.append([float(k), vol1, float(g), l1, l2])
            y.append(float(r))
            levels.append(N)
    names = ["k", "vol_X1", "genus", "l1", "l2"]
    return np.array(rows, float), np.array(y, float), names, np.array(levels, int)
