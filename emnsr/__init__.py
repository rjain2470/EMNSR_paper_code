"""Shared library for the EMNSR pipeline.

Lightweight helpers (config, number-theoretic, geometric, data access) are
re-exported here. The data-heavy arrays live in the submodules
``emnsr.spectral_features`` and ``emnsr.features`` and are imported explicitly,
since building them downloads the LMFDB/Zenodo and Bolza datasets.
"""

from .config import NMAX, K, SEED
from .nt_features import (phi, mu, sigma, is_squarefree, isprime,
                          squarefree_levels)
from .geometric_features import (index_gamma0, volume_X0, volume_X1, genus_X0,
                                 sl2z_primitive_geodesics, geodesic_lengths_X0)
from .data import (ensure_dataset, spectral_parameters_by_level,
                   first_k_spectral_parameters, first_k_eigenvalues)

__all__ = [
    "NMAX", "K", "SEED",
    "phi", "mu", "sigma", "is_squarefree", "isprime", "squarefree_levels",
    "index_gamma0", "volume_X0", "volume_X1", "genus_X0",
    "sl2z_primitive_geodesics", "geodesic_lengths_X0",
    "ensure_dataset", "spectral_parameters_by_level",
    "first_k_spectral_parameters", "first_k_eigenvalues",
]
