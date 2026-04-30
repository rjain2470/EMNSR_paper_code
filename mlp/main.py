"""
main.py
-------
Trains an MLP to predict Maass spectral parameters from geometric features,
then evaluates predictions on X_0(N), the Bolza surface, and the Klein quartic.
"""
 
import numpy as np
import matplotlib.pyplot as plt
 
from surface_features import X0N_features, bolza_features, klein_features, squarefree_Ns
from spectral_parameters import X0N_r_vals, bolza_r_vals, klein_r_vals
from utils import (train_model, evaluate_X0N_spectrum, analyze_r1_saliency, analyze_global_saliency, X0N_LOOCV, predict_spectrum, evaluate_spectrum)
         
# =============================================================================
#  1.  Train model on X_0(N) dataset
# =============================================================================
 
num_params = 40   # Number of parameters to predict
model = train_model(
    features=X0N_features,
    labels=X0N_r_vals[:, :num_params],
    epochs=20000,
    lr=1e-3,
    width=256,
)
 
 
# =============================================================================
#  2.  Evaluate X_0(N) spectrum and saliency
# =============================================================================
 
evaluate_X0N_spectrum(
    model=model,
    feature_vector=X0N_features,
    label_vector=X0N_r_vals,
    squarefree_Ns=squarefree_Ns,
    N=10,
    K=40,
)
 
analyze_r1_saliency(model, X0N_features, target_output_idx=0)
analyze_global_saliency(model, X0N_features)
 
 
# =============================================================================
#  3.  Leave-one-out cross-validation
# =============================================================================
 
loo_mses = X0N_LOOCV(X0N_features, X0N_r_vals, k_spectral=40, epochs=2000)
 
print(f"\nFinal LOOCV Results:")
print(f"Mean MSE: {np.mean(loo_mses):.6f}")
print(f"Std MSE:  {np.std(loo_mses):.6f}")
print(f"Mean MSE (excluding N=1,2,3): {np.mean(loo_mses[3:]):.6f}")
 
plt.figure(figsize=(12, 6))
plt.bar(squarefree_Ns, loo_mses, color='blue', edgecolor='black', alpha=0.7)
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Leave-One-Out Cross Validation MSE per Level N')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
 
 
# =============================================================================
#  4.  Bolza surface evaluation
# =============================================================================
 
# Perform evaluation, and capture the predicted spectrum
bolza_spectrum_pred = evaluate_spectrum(model, bolza_features, actual_r=bolza_r_vals)
 
# Normalize by the 20th entry (index 19 in a 0-indexed array)
normalized_bolza_pred   = bolza_spectrum_pred * (np.sqrt(20 - 0.25) / bolza_spectrum_pred[19])
normalized_bolza_actual = bolza_r_vals
 
# Plot the normalized arrays
k_display = min(len(normalized_bolza_pred), len(normalized_bolza_actual))
plt.figure(figsize=(12, 6))
indices   = np.arange(1, k_display + 1)
width_bar = 0.35
plt.bar(indices - width_bar/2, normalized_bolza_actual[:k_display], width_bar, label='Actual (Normalized)',    alpha=0.7)
plt.bar(indices + width_bar/2, normalized_bolza_pred[:k_display],   width_bar, label='Predicted (Normalized)', alpha=0.7)
 
plt.xlabel('Index j')
plt.ylabel('Normalized Spectral Parameter r_j')
plt.title('Normalized Spectral Parameters: Predicted vs Actual')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
 
print("\nNormalized Predicted first 20 spectral parameters (r_j):")
print(normalized_bolza_pred[:20])
 
 
# =============================================================================
#  5.  Klein quartic evaluation
# =============================================================================
 
# Perform evaluation, and capture the predicted spectrum
klein_spectrum_pred = evaluate_spectrum(model, klein_features, actual_r=klein_r_vals)
 
# Normalize by the 20th entry (index 19 in a 0-indexed array)
normalized_klein_pred   = klein_spectrum_pred * (np.sqrt(20/2 - 0.25) / klein_spectrum_pred[19])
normalized_klein_actual = klein_r_vals
 
# Plot the normalized arrays
k_display = min(len(normalized_klein_pred), len(normalized_klein_actual))
plt.figure(figsize=(12, 6))
indices   = np.arange(1, k_display + 1)
width_bar = 0.35
plt.bar(indices - width_bar/2, normalized_klein_actual[:k_display], width_bar, label='Actual (Normalized)',    alpha=0.7)
plt.bar(indices + width_bar/2, normalized_klein_pred[:k_display],   width_bar, label='Predicted (Normalized)', alpha=0.7)
 
plt.xlabel('Index j')
plt.ylabel('Normalized Klein Spectral Parameter r_j')
plt.title('Normalized Klein Spectral Parameters: Predicted vs Actual')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
 
print("\nNormalized predicted first 20 spectral parameters (r_j):")
print(normalized_klein_pred[:20])
 
print("--- Klein Quartic Saliency ---")
analyze_single_surface_saliency(model, klein_features, "Klein Quartic")
