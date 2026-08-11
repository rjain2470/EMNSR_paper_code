"""Primitive geodesic length spectra and MLP feature vectors for the three
surfaces.

Feature vectors have the form ``[genus, volume, l_1, l_2, ...]``:

    X0N_features    one row per squarefree level, first K lengths
    bolza_features  Bolza surface (Aurich & Steiner 1988, Physica D 32:451)
    klein_features  Klein quartic
"""

import numpy as np

from .config import K
from .geometry import (genus_X0, volume_X0, geodesic_lengths_X0,
                       sl2z_primitive_geodesics)
from .spectral import squarefree_Ns

# Shared SL_2(Z) geodesic pool, reused by the geometric SR experiment.
GEO_POOL = sl2z_primitive_geodesics(50000)


# ---- X_0(N): [genus, volume, first K geodesic lengths] per level -----
def _x0n_feature_row(N):
    lens = geodesic_lengths_X0(N, K, GEO_POOL)
    return np.concatenate([[float(genus_X0(N)), volume_X0(N)], lens])


_raw = [_x0n_feature_row(N) for N in squarefree_Ns]
_maxlen = max(len(f) for f in _raw)
X0N_features = np.vstack(
    [np.pad(f, (0, _maxlen - len(f))) for f in _raw]
).astype(np.float32)


# ---- Bolza surface: primitive geodesic length spectrum + multiplicities
# Aurich & Steiner 1988. Lengths are strictly ascending.
bolza_lengths = np.array([
    3.0571148390, 4.8969048954, 5.8280707754, 6.1128364779, 6.6720057699,
    7.1073578414, 7.2631634751, 7.5956918304, 7.8806928877, 8.1300753289,
    8.2249036323, 8.4368496405, 8.6284635656, 8.7027505564, 8.8714798107,
    9.0270171797, 9.1714255169, 9.2282950896, 9.3592716579, 9.4821914493,
    9.5309770571, 9.6440665486, 9.7510997583, 9.7938097907, 9.8980946367,
    10.0785887303, 10.1149054144, 10.1999558888, 10.2815633765, 10.3143770353,
    10.3915072941, 10.4657729697, 10.5373792543, 10.5663604642, 10.6344594159,
    10.7270445783, 10.7308060283, 10.7901078989, 10.8510687003, 10.8758208811,
    10.9343449467, 10.9912049969, 11.0464930504, 11.0689538604, 11.1226185852,
    11.1739905306, 11.2450675676, 11.2938444648, 11.3416400077, 11.3608557221,
    11.4060920010, 11.4519475300, 11.4703055581, 11.5139343106, 11.5566494092,
    11.5984625221, 11.6159252909, 11.6561415509, 11.6954535600, 11.7122036393,
    11.7509179318, 11.7858970234, 11.8044196533, 11.8410454395, 11.8777196407,
    11.9133862138, 11.9279754971, 11.9627646508, 11.9969589861, 12.0443403136,
    12.0771791386, 12.1098474621, 12.1227186314, 12.1543053200, 12.1854008904,
    12.1981480207, 12.2285673558, 12.2585379275, 12.2708217905,
])
bolza_mults = np.array([
    24, 24, 48, 24, 96, 48, 48, 8, 96, 48, 192, 48, 96, 48, 288, 12, 48, 96, 192,
    48, 192, 96, 336, 24, 192, 192, 96, 384, 96, 192, 384, 96, 288, 96, 272, 96,
    272, 560, 40, 96, 432, 96, 288, 288, 384, 544, 272, 272, 96, 272, 672, 192,
    464, 48, 648, 40, 544, 192, 416, 288, 496, 96, 192, 352, 544, 512, 352, 384,
    384, 544, 288, 352, 96, 800, 352, 368, 48, 736, 96,
], dtype=float)
assert len(bolza_lengths) == len(bolza_mults) == 79
assert np.all(np.diff(bolza_lengths) > 0), "Bolza lengths not strictly increasing"

bolza_features = np.concatenate([[2.0, 4.0 * np.pi], bolza_lengths]).astype(np.float32)


# ---- Klein quartic: primitive geodesic length spectrum ---------------
klein_lengths = np.array([
    3.935962, 5.208017, 7.358318, 7.609408, 7.985792, 8.205601, 8.524421,
    8.694483, 9.040478, 9.173079, 9.450298, 9.643831, 9.866002, 9.954731,
    10.026434, 10.076370, 10.240213, 10.345908, 10.515278, 10.569689,
    10.632507, 10.873258, 10.996456, 11.047355, 11.283088, 11.327265,
    11.390759, 11.629027, 12.147440, 12.503292, 12.605421, 12.717043,
    13.467771, 13.927559, 14.170841, 14.281298, 14.509264, 14.525393,
    16.758943, 17.565838, 17.788878, 18.418937, 18.865298, 20.507876,
    20.562704, 21.079984, 21.990249, 22.083771, 22.123374, 22.720066,
    23.379660, 24.505599, 24.871774, 26.800749, 27.551624, 28.845985,
    29.063801, 30.212253, 32.204814, 32.578100,
])
klein_mults = np.ones(len(klein_lengths))
klein_features = np.concatenate([[3.0, 8.0 * np.pi], klein_lengths]).astype(np.float32)
