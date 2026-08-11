"""Arithmetic functions: Euler phi, Moebius mu, divisor sum sigma, and the
squarefree/prime tests used to enumerate the levels N."""

from sympy import totient, mobius, divisor_sigma, isprime  # noqa: F401 (isprime re-exported)


def phi(n: int) -> int:
    """Euler totient phi(n)."""
    return int(totient(n))


def mu(n: int) -> int:
    """Moebius mu(n)."""
    return int(mobius(n))


def sigma(n: int) -> int:
    """Sum-of-divisors sigma_1(n)."""
    return int(divisor_sigma(n, 1))


def is_squarefree(n: int) -> bool:
    """True iff n is squarefree."""
    return all(n % (p * p) != 0 for p in range(2, int(n ** 0.5) + 1))


def squarefree_levels(Nmax: int = 105):
    """Sorted squarefree levels 1 <= N <= Nmax."""
    return [n for n in range(1, Nmax + 1) if is_squarefree(n)]
