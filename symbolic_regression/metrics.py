"""Evaluation metrics. Each takes a predictor (a callable X -> y_hat), so the
same code scores any formula -- a discovered PySR model or a closed form."""

import numpy as np
from scipy.stats import spearmanr
from sympy import isprime


def evaluate(X, y, predict, levels):
    """Aggregate metrics for ``predict`` against ``y``.

    ``levels`` is the per-row level N (passed explicitly so this works whether
    or not N is one of the feature columns).
    """
    y_hat = np.asarray(predict(X), float)
    res = y - y_hat
    rel = np.abs(res) / y

    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    rho, _ = spearmanr(y, y_hat)

    per_level = {}
    for N in np.unique(levels):
        idx = levels == N
        if idx.sum() < 2:
            continue
        sr = np.sum((y[idx] - y_hat[idx]) ** 2)
        st = np.sum((y[idx] - y[idx].mean()) ** 2)
        per_level[int(N)] = 1.0 - sr / st if st > 0 else np.nan

    return {
        "r2": r2,
        "mre": rel.mean() * 100.0,
        "spearman": rho,
        "median_per_level_r2": float(np.nanmedian(list(per_level.values()))),
        "per_level_r2": per_level,
        "y_hat": y_hat,
    }


def mre_prime_composite(y, y_hat, levels):
    """Mean relative error (%) split by prime vs composite level. N = 1 is
    neither prime nor composite and is excluded from both buckets."""
    rel = np.abs(y - y_hat) / y
    levels = np.asarray(levels)
    prime_mask = np.array([bool(isprime(int(N))) for N in levels])
    comp_mask = np.array([int(N) > 1 and not isprime(int(N)) for N in levels])
    out = {}
    out["prime"] = rel[prime_mask].mean() * 100.0 if prime_mask.any() else float("nan")
    out["composite"] = (rel[comp_mask].mean() * 100.0
                        if comp_mask.any() else float("nan"))
    return out


def per_k_errors(X, y, predict, k_col=0):
    """Per-k MSE and MAE (X[:, k_col] holds k)."""
    y_hat = np.asarray(predict(X), float)
    out = {}
    for k in np.unique(X[:, k_col]):
        idx = X[:, k_col] == k
        e = y[idx] - y_hat[idx]
        out[int(k)] = {"mse": float(np.mean(e ** 2)),
                       "mae": float(np.mean(np.abs(e))),
                       "n": int(idx.sum())}
    return out
