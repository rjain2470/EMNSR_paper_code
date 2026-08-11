"""MLP model, training, prediction, saliency, and leave-one-out cross-validation.

Inputs are standardized (z-scored by the training mean/std) so that features on
very different raw scales -- genus O(1), volume up to ~226, geodesic lengths
~2-18 -- are treated on an equal footing, which also stabilises the transfer
from X_0(N) to the out-of-sample surfaces. The standardizer (``feat_mu``,
``feat_sd``) is fit on the training data and stored on the model as buffers, so
prediction and saliency apply the identical transform.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

_FEATURE_NAMES = ["Genus", "Volume"]  # + L_1, L_2, ... appended dynamically


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, output_dim),
        )
        # Standardization buffers (identity until fit); kept as buffers so they
        # move with .to()/.eval() and are saved with the model.
        self.register_buffer("feat_mu", torch.zeros(input_dim))
        self.register_buffer("feat_sd", torch.ones(input_dim))

    def standardize(self, x):
        return (x - self.feat_mu) / self.feat_sd

    def forward(self, x):
        return self.net(self.standardize(x))


def build_dataset(features, labels):
    """numpy arrays -> float32 tensors."""
    X = torch.tensor(np.asarray(features), dtype=torch.float32)
    y = torch.tensor(np.asarray(labels), dtype=torch.float32)
    print(f"Dataset ready. X: {tuple(X.shape)}, y: {tuple(y.shape)}")
    return X, y


def train_model(features, labels, epochs=20000, lr=1e-3, width=256):
    """Train the MLP with Adam + ReduceLROnPlateau on standardized inputs."""
    X, y = build_dataset(features, labels)
    model = MLP(input_dim=X.shape[1], output_dim=y.shape[1], width=width)
    with torch.no_grad():
        model.feat_mu.copy_(X.mean(0))
        model.feat_sd.copy_(X.std(0) + 1e-8)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                       factor=0.5, patience=100)
    print("Training model...\n")
    for t in range(1, epochs + 1):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
        sched.step(loss.item())
        if t % (epochs // 5) == 0 or t == 1:
            print(f"Epoch {t:5d}  loss={loss.item():.6f}  lr={opt.param_groups[0]['lr']:.2e}")
    print("\nTraining complete.")
    return model


def _fit_feature(feature_vector, dim):
    """Slice/pad one raw feature vector to the model's input dimension."""
    feat = np.asarray(feature_vector, dtype=np.float32)
    return feat[:dim] if len(feat) > dim else np.pad(feat, (0, dim - len(feat)))


def predict_spectrum(model, feature_vector):
    """Predict the spectrum for one raw feature vector (the model standardizes
    internally)."""
    dim = model.net[0].in_features
    feat = _fit_feature(feature_vector, dim)
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(feat).unsqueeze(0)).cpu().numpy().flatten()


def _bar_compare(actual, predicted, title, ylabel="Spectral parameter $r_j$"):
    n = min(len(actual), len(predicted))
    idx = np.arange(1, n + 1)
    plt.figure(figsize=(12, 6))
    plt.bar(idx - 0.175, np.asarray(actual)[:n], 0.35, label="Actual", alpha=0.7)
    plt.bar(idx + 0.175, np.asarray(predicted)[:n], 0.35, label="Predicted", alpha=0.7)
    plt.xlabel("Index $j$"); plt.ylabel(ylabel); plt.title(title)
    plt.legend(); plt.grid(True, alpha=0.3); plt.show()


def evaluate_spectrum(model, feature_vector, actual_r=None):
    """Predict (and optionally compare) the spectrum for one surface."""
    pred = predict_spectrum(model, feature_vector)
    print(f"\n--- Predicted first {len(pred)} spectral parameters ---")
    print(np.round(pred, 4))
    if actual_r is not None:
        limit = min(len(pred), len(actual_r))
        mse = np.mean((pred[:limit] - np.asarray(actual_r, float)[:limit]) ** 2)
        print(f"MSE vs actual: {mse:.6f}")
        _bar_compare(actual_r, pred, "Predicted vs actual spectrum")
    return pred


def evaluate_X0N_spectrum(model, feature_vector, label_vector, squarefree_Ns, N=1, K=20):
    """Compare predicted vs actual spectral parameters at level N."""
    Ns_list = list(np.asarray(squarefree_Ns).tolist())
    if N not in Ns_list:
        print(f"Level N={N} is not in the squarefree list."); return
    idx = Ns_list.index(N)
    pred = predict_spectrum(model, feature_vector[idx])
    actual = np.asarray(label_vector[idx], dtype=float)
    K = min(K, len(actual), len(pred))
    pred, actual = pred[:K], actual[:K]
    print(f"\n--- X_0({N}) ---  MSE: {np.mean((pred - actual) ** 2):.6f}")
    _bar_compare(actual, pred, f"Spectral parameters for $X_0({N})$: predicted vs actual")


def _saliency_of(model, feat_row):
    """Gradient of the output w.r.t. the standardized input for one surface."""
    dim = model.net[0].in_features
    x = model.standardize(torch.tensor(_fit_feature(feat_row, dim))).detach().clone()
    x.requires_grad_(True)
    model.eval()
    out = model.net(x.unsqueeze(0))   # bypass forward's own standardize
    return out, x


def _saliency_plot(saliency, title, color="teal"):
    names = _FEATURE_NAMES + [f"L_{i+1}" for i in range(len(saliency) - 2)]
    plt.figure(figsize=(14, 6))
    plt.bar(range(len(saliency)), saliency, color=color, alpha=0.7)
    plt.xticks(range(len(saliency)), names, rotation=90)
    plt.ylabel("Average gradient magnitude"); plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.3); plt.tight_layout(); plt.show()
    for i in saliency.argsort()[-3:][::-1]:
        print(f"  - {names[i]}: {saliency[i]:.4f}")


def analyze_saliency(model, feature_vector, all_outputs=True, target_idx=0):
    """Average |d r_j / d (standardized feature)| over the dataset."""
    dim = model.net[0].in_features
    total = np.zeros(dim)
    for row in feature_vector:
        out, x = _saliency_of(model, row)
        js = range(out.shape[1]) if all_outputs else [target_idx]
        acc = np.zeros(dim)
        for j in js:
            if x.grad is not None:
                x.grad.zero_()
            out[0, j].backward(retain_graph=True)
            acc += np.abs(x.grad.data.numpy().flatten())
        total += acc / len(js)
    avg = total / len(feature_vector)
    scope = "all spectral parameters" if all_outputs else f"$r_{{{target_idx+1}}}$"
    _saliency_plot(avg, f"Global average saliency for $X_0(N)$ (target: {scope})")
    return avg


def analyze_single_surface_saliency(model, feature_vector, surface_name, target_idx=0):
    """Saliency map for a single surface (w.r.t. standardized inputs)."""
    out, x = _saliency_of(model, feature_vector)
    out[0, target_idx].backward()
    _saliency_plot(np.abs(x.grad.data.numpy().flatten()),
                   f"Saliency map for {surface_name} (target $r_{{{target_idx+1}}}$)",
                   color="darkorange")


def X0N_LOOCV(features, labels, k_spectral=40, epochs=2000, width=256):
    """Leave-one-out CV over the X_0(N) levels; returns per-fold MSE. Each fold
    re-fits the input standardizer on its own training split."""
    X_all, y_all = features, labels[:, :k_spectral]
    mses = []
    for i in range(len(X_all)):
        Xtr = torch.tensor(np.delete(X_all, i, 0), dtype=torch.float32)
        ytr = torch.tensor(np.delete(y_all, i, 0), dtype=torch.float32)
        Xte = torch.tensor(X_all[i:i+1], dtype=torch.float32)
        model = MLP(X_all.shape[1], k_spectral, width=width)
        with torch.no_grad():
            model.feat_mu.copy_(Xtr.mean(0)); model.feat_sd.copy_(Xtr.std(0) + 1e-8)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        for _ in range(epochs):
            opt.zero_grad(); loss_fn(model(Xtr), ytr).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            mses.append(float(np.mean((model(Xte).numpy() - y_all[i:i+1]) ** 2)))
        if (i + 1) % 5 == 0:
            print(f"  LOOCV {i+1}/{len(X_all)}")
    return np.array(mses)
