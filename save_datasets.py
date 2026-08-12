"""Build the feature vectors and spectral arrays once and cache them to
``datasets.npz`` so later runs can skip regeneration.

Run:  python save_datasets.py

Reload elsewhere with:
    import numpy as np
    data = np.load("datasets.npz")
    X0N_features = data["X0N_features"]   # etc.
"""

import numpy as np

from emnsr.features import X0N_features, bolza_features, klein_features
from emnsr.spectral_features import X0N_r_vals, bolza_r_vals, klein_r_vals

if __name__ == "__main__":
    np.savez(
        "datasets.npz",
        X0N_features=X0N_features,
        bolza_features=bolza_features,
        klein_features=klein_features,
        X0N_r_vals=X0N_r_vals,
        bolza_r_vals=bolza_r_vals,
        klein_r_vals=klein_r_vals,
    )
    print("Saved datasets.npz")
