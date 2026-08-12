"""Shared utilities for the symbolic-regression experiments: evaluation metrics,
plotting helpers, and the PySR toolkit.

Each metric takes a predictor (a callable X -> y_hat), so the same code scores
any formula -- a discovered PySR model or a closed form. Each PySR search is
bounded by a runtime cap (PySR's ``timeout_in_seconds``); the best equation and a
short Pareto tail are reported rather than the full hall of fame.
"""

import importlib.util
import subprocess
import sys
import time
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sympy import isprime

if importlib.util.find_spec("pysr") is None:
    print("Installing PySR (one-time; downloads the Julia backend on first fit)...",
          flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pysr"], check=True)
from pysr import PySRRegressor

SR_TIME_LIMIT = 900   # runtime cap per search (seconds)


# ---- metrics ---------------------------------------------------------

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


# ---- plots -----------------------------------------------------------

def plot_empirical_slope(Ns, c_emp, c_model, c_weyl):
    """Empirical slope c(N) vs 12/phi(N) vs Weyl 4pi/Vol(X_0)."""
    fig = plt.figure(figsize=(9, 6))
    plt.scatter(Ns, c_emp, label=r"Empirical slope $c(N)$", alpha=0.8)
    plt.plot(Ns, c_model, color="green", label=r"$12/\varphi(N)$")
    plt.plot(Ns, c_weyl, color="red", label=r"Weyl $4\pi/\mathrm{Vol}(X_0(N))$")
    plt.yscale("log")
    plt.xlabel(r"$N$")
    plt.ylabel(r"$c(N)$")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    return fig


def plot_pred_vs_actual(actual, predicted, N, ylabel, kmax=20):
    """Predicted vs actual, first kmax indices at level N."""
    n = min(kmax, len(actual), len(predicted))
    idx = np.arange(1, n + 1)
    w = 0.4
    fig = plt.figure(figsize=(10, 5))
    plt.bar(idx - w / 2, np.asarray(actual)[:n], w, label="Actual", alpha=0.85)
    plt.bar(idx + w / 2, np.asarray(predicted)[:n], w,
            label="Predicted (symbolic regression)", alpha=0.85)
    plt.xlabel(r"Index $k$")
    plt.ylabel(ylabel)
    plt.title(rf"Maass newforms on $X_0({N})$: predicted vs actual")
    plt.xticks(idx)
    plt.legend()
    plt.tight_layout()
    return fig


def plot_residuals_vs_level(levels, y, y_hat):
    """Residuals (y - y_hat) against level N."""
    fig = plt.figure(figsize=(9, 5))
    plt.scatter(levels, y - y_hat, s=10, alpha=0.4)
    plt.axhline(0.0, color="k", lw=0.8)
    plt.xlabel(r"$N$")
    plt.ylabel(r"residual $y - \hat{y}$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_per_level_r2_hist(per_level_r2, bins=20):
    """Distribution of per-level R^2."""
    vals = np.array([v for v in per_level_r2.values() if np.isfinite(v)])
    fig = plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=bins, alpha=0.8)
    plt.axvline(np.median(vals), color="red",
                label=f"median = {np.median(vals):.3f}")
    plt.xlabel(r"per-level $R^2$")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    return fig


def plot_mse_vs_k(per_k):
    """MSE as a function of k (per_k from per_k_errors)."""
    ks = sorted(per_k)
    mse = [per_k[k]["mse"] for k in ks]
    fig = plt.figure(figsize=(9, 5))
    plt.plot(ks, mse, marker="o", ms=3)
    plt.xlabel(r"$k$")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ---- PySR toolkit ----------------------------------------------------

def make_pysr_model(variable_names, niterations=200, timeout_s=SR_TIME_LIMIT,
                    maxsize=24, maxdepth=6, population_size=100, seed=0,
                    parallelism="serial"):
    """A PySRRegressor with the standard operator set and a runtime cap.
    ``parallelism='serial'`` gives a deterministic search; ``'multithreading'``
    is faster (used for the many per-k searches)."""
    serial = (parallelism == "serial")
    return PySRRegressor(
        niterations=niterations,
        timeout_in_seconds=timeout_s,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "log"],
        maxsize=maxsize,
        maxdepth=maxdepth,
        population_size=population_size,
        elementwise_loss="loss(x, y) = (x - y)^2",
        deterministic=serial,
        parallelism=parallelism,
        random_state=(seed if serial else None),
        verbosity=0, progress=False,
    )


def sympy_predictor(expr, variable_names):
    """Turn a discovered sympy expression into a vectorised predictor X -> y_hat."""
    syms = [sp.Symbol(n) for n in variable_names]
    f = sp.lambdify(syms, expr, modules=["numpy"])

    def predict(X):
        X = np.asarray(X, float)
        out = f(*[X[:, i] for i in range(len(variable_names))])
        return np.broadcast_to(np.asarray(out, float), (X.shape[0],)).astype(float)
    return predict


def top_equations(model, n=6):
    """Compact Pareto tail (complexity / loss / equation)."""
    df = model.equations_
    cols = [c for c in ["complexity", "loss", "score", "equation"] if c in df.columns]
    return df[cols].tail(n).to_string(index=False)


def run_pysr(X, y, variable_names, **kw):
    """Fit one PySR model over the whole matrix; return (model, expr, predictor)."""
    model = make_pysr_model(variable_names, **kw)
    model.fit(np.asarray(X, float), np.asarray(y, float),
              variable_names=list(variable_names))
    expr = model.sympy()
    return model, expr, sympy_predictor(expr, variable_names)


# ---- geometric experiment: per-k search + volume-form parsing --------

def parse_volume_form(expr, vol_samples, vol_name="vol_X1", r2_tol=0.9995):
    """If ``expr`` is (numerically) a / (Vol^(1/4) - b) in the volume alone,
    return (a, b), else None. The test: ``expr`` depends only on ``vol_name``,
    and 1/expr(vol) is affine in w = vol^(1/4) over the sampled volumes
    (R^2 >= r2_tol). The pure power law a*Vol^(-1/4) is admitted as the b=0 case."""
    if expr is None:
        return None
    syms = {s.name: s for s in expr.free_symbols}
    if vol_name not in syms or (set(syms) - {vol_name}):
        return None
    v = syms[vol_name]
    try:
        f = sp.lambdify(v, expr, modules=["numpy"])
        y = np.asarray(f(np.asarray(vol_samples, float)), float)
    except Exception:
        return None
    if y.shape != vol_samples.shape or not np.all(np.isfinite(y)) or np.any(y == 0):
        return None
    w = np.asarray(vol_samples, float) ** 0.25
    inv = 1.0 / y
    A = np.vstack([w, np.ones_like(w)]).T
    (c1, c0), *_ = np.linalg.lstsq(A, inv, rcond=None)
    pred = A @ np.array([c1, c0])
    ss_tot = np.sum((inv - inv.mean()) ** 2)
    if ss_tot <= 0:
        return None
    r2 = 1.0 - np.sum((inv - pred) ** 2) / ss_tot
    if r2 < r2_tol or abs(c1) < 1e-9:
        return None
    return 1.0 / c1, -c0 / c1             # expr == a / (w - b)


def per_k_volume_search(X, y, variable_names, k_values, k_col=0, vol_col=1,
                        deadline_s=SR_TIME_LIMIT, per_k_iters=30, seed=0,
                        maxsize=16, population_size=40, vol_name="vol_X1"):
    """Run PySR at each k (the k column is dropped) under a shared wall-clock
    deadline. Returns {k: {expr, a, b, matched}} for whatever finished in time."""
    t0 = time.time()
    cols = [j for j in range(X.shape[1]) if j != k_col]
    sub_names = [variable_names[j] for j in cols]
    vol_all = X[:, vol_col]
    results = {}
    for k in k_values:
        elapsed = time.time() - t0
        if elapsed > deadline_s:
            print(f"[deadline] hit {deadline_s}s after {len(results)} searches "
                  f"(last completed before k={int(k)})", flush=True)
            break
        idx = X[:, k_col] == k
        if idx.sum() < 5:
            continue
        model = make_pysr_model(sub_names, niterations=per_k_iters,
                                timeout_s=max(5.0, deadline_s - elapsed),
                                maxsize=maxsize, population_size=population_size,
                                seed=seed, parallelism="multithreading")
        model.fit(X[idx][:, cols], y[idx], variable_names=sub_names)
        expr = model.sympy()
        vol_samples = np.unique(vol_all[idx])
        ab = parse_volume_form(expr, vol_samples, vol_name=vol_name)
        results[int(k)] = dict(expr=expr, matched=ab is not None,
                               a=(ab[0] if ab else np.nan),
                               b=(ab[1] if ab else np.nan))
    return results
