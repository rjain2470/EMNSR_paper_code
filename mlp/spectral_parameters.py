"""
Computes and exports spectral parameter arrays for three hyperbolic surfaces:
  - X0N_r_vals   : Maass spectral parameters for X_0(N) over squarefree N
  - bolza_r_vals : spectral parameters for the Bolza surface
  - klein_r_vals : spectral parameters for the Klein quartic
"""
 
import os
import numpy as np
import requests
import urllib.request
 
 
# =============================================================================
#  1.  Squarefree N in [1, 105]
# =============================================================================
 
squarefree_Ns = [n for n in range(1, 106) if all(n % (p*p) for p in range(2, int(n**0.5) + 1))]
 
 
# =============================================================================
#  2.  X_0(N) – Maass spectral parameters
# =============================================================================
 
_ZENODO_URL = "https://zenodo.org/records/15490636/files/MaassForms.txt?download=1"
_LOCAL_PATH  = "MaassForms.txt"
 
 
def _ensure_dataset(local_path: str = _LOCAL_PATH,
                    url: str = _ZENODO_URL,
                    chunk: int = 1 << 20,
                    timeout: int = 60):
    """
    Ensure the Maass form dataset exists locally; download if missing.
    """
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
 
    print("⏬ Downloading Maass form dataset...")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for blk in r.iter_content(chunk_size=chunk):
                if blk:
                    f.write(blk)
 
    print(f"Downloaded to {local_path}")
    return local_path
 
 
def first_k_spectral_params(N: int, k: int, local_path: str = _LOCAL_PATH):
    """
    Return the first k Maass spectral parameters R for Γ_0(N).
 
    Note:
    The dataset consists of Maass newforms (LMFDB), so this returns
    newform spectral parameters rather than the full spectrum.
    """
    if k <= 0:
        return []
 
    _ensure_dataset(local_path)
 
    Rs = []
    with open(local_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
 
            parts = line.strip().split(":", 5)
            if len(parts) < 3:
                continue
 
            try:
                level = int(parts[1])
            except ValueError:
                continue
 
            if level != N:
                continue
 
            try:
                R = float(parts[2])
            except ValueError:
                continue
 
            Rs.append(R)
 
    Rs.sort()
    return Rs[:k]
 
 
# =============================================================================
#  3.  X_0(N) spectral parameter matrix
# =============================================================================
 
K = 100
 
spectral_params_list = []
max_current_k = 0
 
for N_val in squarefree_Ns:
    params = first_k_spectral_params(N_val, K)
    spectral_params_list.append(params)
    max_current_k = max(max_current_k, len(params))
 
# Pad to uniform length
padded_spectral_params = []
for params in spectral_params_list:
    pad_len = max_current_k - len(params)
 
    if pad_len > 0:
        padded = np.pad(params, (0, pad_len),
                        mode="constant", constant_values=0.0)
    else:
        padded = np.array(params, dtype=np.float32)
 
    padded_spectral_params.append(padded)
 
X0N_r_vals = np.vstack(padded_spectral_params).astype(np.float32)
 
# Remove trailing zero columns
if X0N_r_vals.size > 0 and np.any(X0N_r_vals):
    nonzero_indices = np.argwhere(X0N_r_vals != 0)
    last_nonzero_col = nonzero_indices[:, 1].max()
    X0N_r_vals = X0N_r_vals[:, :last_nonzero_col + 1]
else:
    X0N_r_vals = np.array([], dtype=np.float32)
 
 
# =============================================================================
#  4.  Bolza surface spectral parameters
# =============================================================================
 
# Download eigenvalues directly from Stohmaier & Uski
url  = "https://arxiv.org/src/1110.2150v4/anc/eig-bolza-refined0-1000.txt"
data = urllib.request.urlopen(url).read().decode()
 
# Parse plain-text floats (one per line)
lambda_vals = np.array([float(x) for x in data.strip().split()])
 
# Compute spectral parameters r_n
bolza_r_vals = np.sqrt(lambda_vals - 0.25)
 
print(f"Loaded {len(bolza_r_vals)} eigenvalues.")
 
 
# =============================================================================
#  V.  Klein quartic spectral parameters
# =============================================================================
 
klein_r_vals = np.array([1.555177,1.555177,1.555177,1.555177,1.555177,1.555177,1.555177,1.555177,2.507492,2.507492,2.507492,2.507492,2.507492,2.507492,2.507492,3.252553,3.252553,3.252553,3.252553,3.252553,3.252553,3.456486,3.456486,3.456486,3.456486,3.456486,3.456486,3.456486,3.456486,4.140797,4.140797,4.140797,4.140797,4.140797,4.140797,4.140797,4.658500,4.658500,4.658500,4.658500,4.658500,4.658500,4.658500,4.889904,4.889904,4.889904,4.889904,4.889904,4.889904,4.889904,4.889904,5.068157,5.068157,5.068157,5.068157,5.068157,5.068157,5.481241,5.481241,5.481241,5.481241,5.481241,5.481241,6.022913,6.022913,6.022913,6.022913,6.022913,6.022913,6.022913,6.022913,6.106100,6.106100,6.106100,6.106100,6.106100,6.106100,6.106100,6.106100,6.424379,6.424379,6.424379,6.424379,6.424379,6.424379,6.682302,6.682302,6.682302,6.682302,6.682302,6.682302,6.682302,6.682302,6.958684,6.958684,6.958684,6.958684,6.958684,6.958684,7.062952,7.062952,7.062952,7.062952,7.062952,7.062952])
