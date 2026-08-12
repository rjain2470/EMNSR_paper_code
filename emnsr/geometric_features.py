"""Geometry of X_0(N): the covolume index, the volumes of X_0(N) and X_1(N),
the genus of X_0(N), and primitive geodesic lengths.

    Vol(X_0(N)) = (pi/3) * [SL_2(Z):Gamma_0(N)] = (pi/3) * N * prod_{p|N}(1 + 1/p),
    Vol(X_1(N)) = (phi(N)/2) * Vol(X_0(N)).
"""

import math
from math import pi
from sympy import factorint, divisors

from .nt_features import phi


def index_gamma0(N: int) -> int:
    """mu_0(N) = [SL_2(Z):Gamma_0(N)] = N * prod_{p|N}(1 + 1/p)."""
    m = N
    for p in factorint(N):
        m = m * (p + 1) // p
    return int(m)


def volume_X0(N: int) -> float:
    """Vol(X_0(N)) = (pi/3) * [SL_2(Z):Gamma_0(N)]."""
    return (pi / 3.0) * index_gamma0(N)


def volume_X1(N: int) -> float:
    """Vol(X_1(N)) = (phi(N)/2) * Vol(X_0(N))."""
    return 0.5 * phi(N) * volume_X0(N)


# ---- genus of X_0(N) -------------------------------------------------

def _legendre_m1(p: int) -> int:
    return 1 if p % 4 == 1 else -1


def _legendre_m3(p: int) -> int:
    if p == 3:
        return 0
    return 1 if p % 3 == 1 else -1


def _elliptic_2(N: int) -> int:
    if N % 4 == 0:
        return 0
    e = 1
    for p in factorint(N):
        if p != 2:
            e *= (1 + _legendre_m1(p))
    return e


def _elliptic_3(N: int) -> int:
    if N % 2 == 0 or N % 9 == 0:
        return 0
    e = 1
    for p in factorint(N):
        if p != 2:
            e *= (1 + _legendre_m3(p))
    return e


def _cusps(N: int) -> int:
    return sum(phi(math.gcd(d, N // d)) for d in divisors(N))


def genus_X0(N: int) -> int:
    """Genus g = 1 + mu_0/12 - e_2/4 - e_3/3 - cusps/2 (rounded to the exact
    integer value)."""
    mu0 = index_gamma0(N)
    g = 1 + mu0 / 12 - _elliptic_2(N) / 4 - _elliptic_3(N) / 3 - _cusps(N) / 2
    return round(g)


# ---- primitive geodesic lengths --------------------------------------
# SL_2(Z) primitive hyperbolic classes enumerated by trace, with a single
# representative matrix per trace tested for membership in Gamma_0(N) via
# c = 0 (mod N). l_1, l_2 provide the two shortest lengths used as features
# in the geometric symbolic-regression experiment.

def sl2z_primitive_geodesics(num: int):
    """First `num` SL_2(Z) primitive geodesic lengths, each paired with the
    lower-left entry c of a representative matrix."""
    out, nonprim = [], set()
    trace, tmax = 3, 100000
    while len(out) < num and trace <= tmax:
        if trace in nonprim:
            trace += 1
            continue
        disc = trace * trace - 4
        length = 2.0 * math.log((trace + math.sqrt(disc)) / 2.0)
        if trace % 2 == 0:
            k = trace // 2
            c = k * k - 1
        else:
            k = (trace - 1) // 2
            c = k * (k + 1) - 1
        out.append((length, c))
        t0, t1 = 2, trace
        while True:
            tn = trace * t1 - t0
            if tn > tmax:
                break
            nonprim.add(tn)
            t0, t1 = t1, tn
        trace += 1
    out.sort(key=lambda x: x[0])
    return out[:num]


def geodesic_lengths_X0(N: int, K: int, pool=None):
    """First K primitive geodesic lengths on X_0(N)."""
    if pool is None:
        pool = sl2z_primitive_geodesics(50000)
    lengths = [L for (L, c) in pool if c % N == 0]
    return lengths[:K]
