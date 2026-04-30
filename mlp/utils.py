"""
utils.py
--------
Shared utilities for MLP training, evaluation, and saliency analysis on spectral data.

Exports:
  - MLP: two-hidden-layer ReLU network
  - build_dataset: PyTorch tensors
  - train_model: full training loop with LR scheduler
  - evaluate_X0N_spectrum: predict and plot spectral params for one X_0(N)
  - X0N_LOOCV: leave-one-out cross-validation over X_0(N) dataset
  - analyze_r1_saliency: gradient saliency for a single output index
  - analyze_global_saliency: gradient saliency averaged over all outputs
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import matplotlib.pyplot as plt


# =============================================================================
#  1.  Model definition
# =============================================================================

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
#  2.  Dataset construction and training
# =============================================================================

def build_dataset(features, labels):
    """
    Constructs a PyTorch dataset from arbitrary numpy arrays.
    """
    if isinstance(features, list):
        features = np.array(features)
    if isinstance(labels, list):
        labels = np.array(labels)

    X_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.float32)

    print(f"Dataset ready. X: {X_tensor.shape}, y: {y_tensor.shape}")
    return X_tensor, y_tensor


def train_model(features, labels, epochs=10000, lr=1e-3, width=256):
    """
    Trains the MLP on the provided features and labels.
    """
    X, y = build_dataset(features, labels)

    model = MLP(input_dim=X.shape[1], output_dim=y.shape[1], width=width)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=100)

    print("Training model...\n")
    for t in range(1, epochs + 1):
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        scheduler.step(loss.item())

        if t % (epochs // 5) == 0 or t == 1:
            current_lr = opt.param_groups[0]['lr']
            print(f"Epoch {t:4d}, Loss = {loss.item():.6f}, LR = {current_lr:.2e}")

    print("\nTraining complete.")
    return model


# =============================================================================
#  3.  Evaluation
# =============================================================================

def evaluate_X0N_spectrum(model, feature_vector, label_vector, squarefree_Ns, N=1, K=20):
    """
    Predict and compare the first K spectral parameters r_j for X_0(N)
    using a trained MLP model and the provided numpy datasets.

    Parameters
    ----------
    model : Trained MLP model.
    feature_vector : Feature matrix of shape (num_samples, num_features).
    label_vector : Label matrix of shape (num_samples, num_labels).
    squarefree_Ns : List of squarefree levels N corresponding to the rows.
    N : Target level to evaluate.
    K : Number of spectral parameters to evaluate.
    """
    print(f"\n--- Predicting Spectral Parameters for X_0({N}) ---\n")

    try:
        predicted_r = predict_X0N_spectrum(model, feature_vector, N, K, squarefree_Ns)
    except Exception as e:
        print("Prediction error:", e)
        return

    if isinstance(squarefree_Ns, np.ndarray):
        Ns_list = squarefree_Ns.tolist()
    else:
        Ns_list = squarefree_Ns

    try:
        idx = Ns_list.index(N)
        actual_r = np.asarray(label_vector[idx], dtype=float)
    except ValueError:
        print(f"Level N={N} is not in the provided squarefree list.")
        return

    actual_r = actual_r[:K]

    if len(actual_r) < K:
        print(f"Warning: requested K={K}, but only {len(actual_r)} ground-truth values are available.")
        predicted_r = predicted_r[:len(actual_r)]
        K = len(actual_r)

    mse = np.mean((predicted_r - actual_r) ** 2)
    print(f"Mean Squared Error: {mse:.6f}\n")

    plt.figure(figsize=(12, 6))
    indices = np.arange(1, K + 1)
    width = 0.35

    plt.bar(indices - width / 2, actual_r, width, label="Actual", alpha=0.7)
    plt.bar(indices + width / 2, predicted_r, width, label="Predicted (MLP)", alpha=0.7)

    plt.xlabel("Index $j$")
    plt.ylabel("Spectral Parameter $r_j$")
    plt.title(f"Spectral Parameters for $X_0({N})$: Predicted vs Actual")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(indices)
    plt.show()


# =============================================================================
#  4.  Leave-one-out cross-validation
# =============================================================================

def X0N_LOOCV(features, labels, k_spectral=40, epochs=2000, width=256):
    """
    Performs leave-one-out cross validation (LOOCV) on the X0(N) dataset.
    """
    X_all = features
    y_all = labels[:, :k_spectral]

    print(f"Starting LOOCV on {len(X_all)} samples with {epochs} epochs per fold...")
    loo_mses = []

    for i in range(len(X_all)):
        # 1. Split data
        X_train = np.delete(X_all, i, axis=0)
        y_train = np.delete(y_all, i, axis=0)
        X_test = X_all[i:i+1]
        y_test = y_all[i:i+1]

        # 2. Prepare tensors
        t_X_train = torch.tensor(X_train, dtype=torch.float32)
        t_y_train = torch.tensor(y_train, dtype=torch.float32)
        t_X_test  = torch.tensor(X_test,  dtype=torch.float32)

        # 3. Train model 
        model = MLP(input_dim=X_train.shape[1], output_dim=k_spectral, width=width)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        # Training loop without scheduler for speed inside the LOOCV loop
        for _ in range(epochs):
            opt.zero_grad()
            pred = model(t_X_train)
            loss = loss_fn(pred, t_y_train)
            loss.backward()
            opt.step()

        # 4. Evaluate
        model.eval()
        with torch.no_grad():
            pred_test = model(t_X_test).cpu().numpy()
            mse = np.mean((pred_test - y_test) ** 2)
            loo_mses.append(mse)

        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/{len(X_all)} samples...")

    return np.array(loo_mses)


# =============================================================================
#  5.  Saliency analysis
# =============================================================================

def analyze_r1_saliency(model, feature_vector, target_output_idx=0):
    """
    Computes the average saliency map across the dataset of X0(N) surfaces.
    """
    print(f"Computing saliency for all {len(feature_vector)} samples...")

    input_dim = feature_vector.shape[1]
    total_saliency = np.zeros(input_dim)

    model.eval()

    for i in range(len(feature_vector)):
        input_np = feature_vector[i]
        input_tensor = torch.tensor(input_np, dtype=torch.float32).unsqueeze(0)
        input_tensor.requires_grad = True

        model.zero_grad()

        output = model(input_tensor)

        prediction_scalar = output[0, target_output_idx]
        prediction_scalar.backward()

        gradients = input_tensor.grad.data.numpy().flatten()
        total_saliency += np.abs(gradients)

    avg_saliency = total_saliency / len(feature_vector)

    # Visualization
    num_features = len(avg_saliency)
    feature_names = ['Genus', 'Volume'] + [f'L_{i+1}' for i in range(num_features - 2)]

    plt.figure(figsize=(14, 6))
    plt.bar(range(num_features), avg_saliency, color='teal', alpha=0.7)

    plt.xticks(range(num_features), feature_names, rotation=90)
    plt.ylabel('Average Gradient Magnitude')
    plt.title(
        f'Global Average Saliency Map for the X0(N)\n'
        f'(Averaged over {len(feature_vector)} samples, target: $r_{{{target_output_idx+1}}}$)'
    )
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print top 3 features
    top_indices = avg_saliency.argsort()[-3:][::-1]
    print(f"Top three most influential features on average for r_{target_output_idx+1}:")
    for idx in top_indices:
        print(f"  - {feature_names[idx]}: {avg_saliency[idx]:.4f}")


def analyze_global_saliency(model, feature_vector):
    """
    Computes the average saliency map across the dataset of X0(N) surfaces
    and across all output spectral parameters.
    """
    print(f"Computing saliency for all {len(feature_vector)} samples across all spectral parameters...")

    input_dim = feature_vector.shape[1]
    total_saliency = np.zeros(input_dim)

    model.eval()

    for i in range(len(feature_vector)):
        input_np = feature_vector[i]
        input_tensor = torch.tensor(input_np, dtype=torch.float32).unsqueeze(0)
        input_tensor.requires_grad = True

        model.zero_grad()

        output = model(input_tensor)
        num_outputs = output.shape[1]

        sample_saliency_sum = np.zeros(input_dim)

        for j in range(num_outputs):
            if input_tensor.grad is not None:
                input_tensor.grad.zero_()

            output[0, j].backward(retain_graph=True)
            gradients = input_tensor.grad.data.numpy().flatten()
            sample_saliency_sum += np.abs(gradients)

        sample_avg_saliency = sample_saliency_sum / num_outputs
        total_saliency += sample_avg_saliency

    avg_saliency = total_saliency / len(feature_vector)

    # Visualization
    num_features = len(avg_saliency)
    feature_names = ['Genus', 'Volume'] + [f'L_{i+1}' for i in range(num_features - 2)]

    plt.figure(figsize=(14, 6))
    plt.bar(range(num_features), avg_saliency, color='teal', alpha=0.7)

    plt.xticks(range(num_features), feature_names, rotation=90)
    plt.ylabel('Average Gradient Magnitude')
    plt.title(
        f'Global Average Saliency Map for the X0(N)\n'
        f'(Averaged over {len(feature_vector)} samples and all {num_outputs} spectral parameters)'
    )
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    top_indices = avg_saliency.argsort()[-3:][::-1]
    print("Top three most influential features on average (across all spectral parameters):")
    for idx in top_indices:
        print(f"  - {feature_names[idx]}: {avg_saliency[idx]:.4f}")
