"""PySR toolkit: build a regressor, turn a discovered sympy expression into a
predictor, and the helpers used by the geometric experiment (per-k search and
volume-form parsing).

Each search is bounded by a runtime cap (PySR's ``timeout_in_seconds``); the best
equation and a short Pareto tail are reported rather than the full hall of fame.
"""

import importlib.util
import subprocess
import sys
import time
import numpy as np
import sympy as sp

if importlib.util.find_spec("pysr") is None:
    print("Installing PySR (one-time; downloads the Julia backend on first fit)...",
          flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pysr"], check=True)
from pysr import PySRRegressor

SR_TIME_LIMIT = 900   # runtime cap per search (seconds)


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
