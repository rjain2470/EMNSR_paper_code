"""Train the MLP on the X_0(N) family, evaluate it, and transfer it to the Bolza
surface and the Klein quartic.

Feature representation: [genus, volume, first L_COMMON primitive geodesic
lengths]. Every surface supplies the same number of lengths, so the transfer
surfaces need no zero-padding. Bolza provides 79 primitive lengths and the Klein
quartic 60, so L_COMMON = 50 is used for all three (input dim 52).

Run:  python -m mlp.run
"""

import numpy as np
import matplotlib.pyplot as plt

from emnsr.spectral import X0N_r_vals, bolza_r_vals, klein_r_vals, squarefree_Ns
from emnsr.features import X0N_features, bolza_features, klein_features
from mlp.model import (
    train_model, evaluate_X0N_spectrum, evaluate_spectrum,
    analyze_saliency, analyze_single_surface_saliency, X0N_LOOCV,
    _bar_compare,
)

NUM_PARAMS = 40   # spectral parameters predicted per surface
L_COMMON = 50     # primitive geodesic lengths used per surface (<= min(79, 60))
IN_DIM = 2 + L_COMMON


def weyl_r(index, vol):
    """r for the index-th Weyl-law eigenvalue: lambda ~ 4 pi index / vol."""
    return np.sqrt(4.0 * np.pi * index / vol - 0.25)


def main(run_loocv=False):
    # Common [genus, volume, first L_COMMON lengths] slice for each surface.
    X0N_feat = X0N_features[:, :IN_DIM]
    bolza_feat = bolza_features[:IN_DIM]
    klein_feat = klein_features[:IN_DIM]
    assert (bolza_feat[2:] > 0).all() and (klein_feat[2:] > 0).all(), \
        "a transfer surface has fewer than L_COMMON geodesic lengths"

    # ---- Train on the X_0(N) family ----------------------------------
    model = train_model(features=X0N_feat, labels=X0N_r_vals[:, :NUM_PARAMS],
                         epochs=20000, lr=1e-3, width=256)

    # ---- Evaluate on a chosen level + saliency -----------------------
    evaluate_X0N_spectrum(model, X0N_feat, X0N_r_vals, squarefree_Ns, N=10, K=40)
    analyze_saliency(model, X0N_feat, all_outputs=False, target_idx=0)  # r_1
    analyze_saliency(model, X0N_feat, all_outputs=True)                 # all r_j

    if run_loocv:
        loo = X0N_LOOCV(X0N_feat, X0N_r_vals, k_spectral=40, epochs=2000)
        print(f"LOOCV mean MSE = {loo.mean():.6f}")
        plt.figure(figsize=(12, 6))
        plt.bar(squarefree_Ns, loo, edgecolor="black", alpha=0.7)
        plt.ylabel("MSE"); plt.title("LOOCV MSE per level N"); plt.show()

    # ---- Transfer to other surfaces (Weyl-normalized at index 20) ----
    print("\n=== Bolza surface (Vol = 4 pi) ===")
    bolza_pred = evaluate_spectrum(model, bolza_feat, actual_r=bolza_r_vals)
    bolza_norm = bolza_pred * (weyl_r(20, 4.0 * np.pi) / bolza_pred[19])
    mse_b = np.mean((bolza_norm[:len(bolza_r_vals)] - bolza_r_vals[:len(bolza_norm)]) ** 2)
    print(f"Bolza normalized MSE vs actual: {mse_b:.4f}")
    _bar_compare(bolza_r_vals, bolza_norm,
                 "Bolza: normalized predicted vs actual", "Normalized $r_j$")
    analyze_single_surface_saliency(model, bolza_feat, "Bolza surface")

    print("\n=== Klein quartic (Vol = 8 pi) ===")
    klein_pred = evaluate_spectrum(model, klein_feat, actual_r=klein_r_vals)
    klein_norm = klein_pred * (weyl_r(20, 8.0 * np.pi) / klein_pred[19])
    mse_k = np.mean((klein_norm[:len(klein_r_vals)] - klein_r_vals[:len(klein_norm)]) ** 2)
    print(f"Klein normalized MSE vs actual: {mse_k:.4f}")
    _bar_compare(klein_r_vals, klein_norm,
                 "Klein: normalized predicted vs actual", "Normalized $r_j$")
    analyze_single_surface_saliency(model, klein_feat, "Klein quartic")


if __name__ == "__main__":
    main()
