"""Plotting helpers for the symbolic-regression experiments."""

import numpy as np
import matplotlib.pyplot as plt


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
    """MSE as a function of k (per_k from metrics.per_k_errors)."""
    ks = sorted(per_k)
    mse = [per_k[k]["mse"] for k in ks]
    fig = plt.figure(figsize=(9, 5))
    plt.plot(ks, mse, marker="o", ms=3)
    plt.xlabel(r"$k$")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
