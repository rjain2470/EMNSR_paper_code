"""Shared Selberg-trace-formula machinery, reused by the Bolza and Klein
experiments.

For the smooth compactly-supported bump test function
    h_T(r) = exp(-A / (1 - (r/T)^2))   for |r| < T,  else 0,
the STF on a compact hyperbolic surface X reads
    h_T(i/2) + sum_{n>=1} h_T(r_n)  =  c_T + G_T,
with identity term  c_T = (Vol/4pi) integral r h_T(r) tanh(pi r) dr  and
geometric term
    G_T = sum_{gamma_0^m} [l(gamma_0) / (2 sinh(m l(gamma_0)/2))] * g_T(m l(gamma_0)),
    g_T(l) = hhat_T(l)/(2pi) = (1/pi) integral_0^T h_T(r) cos(r l) dr.
The trivial eigenvalue lambda_0 = 0 gives r_0 = i/2, h_T(i/2) = exp(-A/(1+1/(4T^2))).

The helpers are parameterised by (A, VOL, quadrature, lengths, multiplicities).
"""

import numpy as np


def gl_quadrature(Q, A):
    """Gauss-Legendre nodes/weights on [0,1] plus phi(u) = exp(-A/(1-u^2))."""
    x, w = np.polynomial.legendre.leggauss(Q)
    u = 0.5 * (x + 1.0)
    wq = 0.5 * w
    phi = np.exp(-A / (1.0 - u ** 2))
    return u, wq, phi


def h_bump(r, T, A):
    """h_T(r) = exp(-A/(1-(r/T)^2)) for |r|<T, else 0."""
    r = np.asarray(r, float)
    y = r / T
    m = np.abs(y) < 1.0
    return np.where(m, np.exp(-A / np.where(m, 1.0 - y ** 2, 1.0)), 0.0)


def h_trivial(T, A):
    """h_T(i/2) = exp(-A/(1+1/(4T^2)))  (trivial eigenvalue lambda_0 = 0)."""
    return np.exp(-A / (1.0 + 1.0 / (4.0 * np.asarray(T, float) ** 2)))


def identity_term(T, u, wq, phi, VOL):
    """c_T = (VOL/2pi) integral_0^T r h_T(r) tanh(pi r) dr  (integrand even)."""
    T = np.asarray(T, float)
    rn = T[:, None] * u[None, :]
    rw = T[:, None] * wq[None, :]
    return (VOL / (2.0 * np.pi)) * (rw * rn * phi[None, :] * np.tanh(np.pi * rn)).sum(1)


def g_transform(T, ells, u, wq, phi):
    """g_T(l) = hhat_T(l)/(2pi) = (T/pi) sum_q w_q phi(u_q) cos(T u_q l). Shape (J, E)."""
    T = np.asarray(T, float)
    ells = np.asarray(ells, float)
    arg = T[:, None, None] * u[None, :, None] * ells[None, None, :]
    return (T[:, None] / np.pi) * np.einsum('q,jqe->je', wq, phi[None, :, None] * np.cos(arg))


def expand_geodesics(lengths, mults, L_cutoff):
    """Expand primitives gamma_0 into iterates gamma_0^m with m*l_0 <= L_cutoff;
    return (iterate lengths, STF weights mult*l_0/(2 sinh(m l_0/2)))."""
    ells, wts = [], []
    for l0, mlt in zip(lengths, mults):
        m = 1
        while m * l0 <= L_cutoff:
            l = m * l0
            ells.append(l)
            wts.append(float(mlt) * l0 / (2.0 * np.sinh(0.5 * l)))
            m += 1
    return np.array(ells), np.array(wts)


def geometric_term(T, lengths, mults, L_cutoff, u, wq, phi, geo_factor=1):
    """G_T = geo_factor * sum_{gamma_0^m} weight * g_T(m l_0)."""
    ells, wts = expand_geodesics(lengths, mults, L_cutoff)
    return geo_factor * (g_transform(T, ells, u, wq, phi) @ wts)


def compress_multiplicities(x, tol=1e-6):
    """Collapse near-equal sorted values into (distinct values, multiplicities)."""
    x = np.sort(np.asarray(x, float))
    vals, mults = [], []
    for y in x:
        if not vals or abs(y - vals[-1]) > tol:
            vals.append(y); mults.append(1)
        else:
            mults[-1] += 1
    return np.array(vals), np.array(mults, dtype=float)
