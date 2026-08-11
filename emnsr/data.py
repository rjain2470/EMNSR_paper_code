"""Maass newform spectral parameters from the LMFDB, mirrored on Zenodo record
15490636.

File format: colon-separated, level = field[1], r = field[2]. The stored values
are spectral parameters r; Laplace eigenvalues are lambda = r^2 + 1/4. The file
is parsed once into ``{level: sorted r-values}`` and reused everywhere.
"""

import os
import requests

ZENODO_URL = "https://zenodo.org/records/15490636/files/MaassForms.txt?download=1"
LOCAL_PATH = "MaassForms.txt"


def _download(url: str, dest: str) -> None:
    """Stream `url` to `dest`, verifying the whole file arrived.

    The data is written to a temporary ``.part`` file and only renamed into
    place once the full ``Content-Length`` has been received, so a truncated
    download (network drop mid-stream) never leaves a partial file that later
    calls would mistake for a complete one.
    """
    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        expected = r.headers.get("Content-Length")
        expected = int(expected) if expected is not None else None
        written = 0
        with open(tmp, "wb") as f:
            for blk in r.iter_content(chunk_size=1 << 20):
                if blk:
                    f.write(blk)
                    written += len(blk)
    if expected is not None and written != expected:
        os.remove(tmp)
        raise IOError(f"Incomplete download from {url}: "
                      f"got {written} of {expected} bytes")
    os.replace(tmp, dest)   # atomic; a partial .part never becomes `dest`


def ensure_dataset(local_path: str = LOCAL_PATH, url: str = ZENODO_URL) -> str:
    """Download the Maass newform dataset if not already present."""
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    print("Downloading Maass newform dataset...")
    _download(url, local_path)
    return local_path


def spectral_parameters_by_level(local_path: str = LOCAL_PATH):
    """Parse the whole dataset once into ``{level: sorted list of r-values}``."""
    ensure_dataset(local_path)
    by_level = {}
    with open(local_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            parts = line.strip().split(":", 5)
            if len(parts) < 3:
                continue
            try:
                level = int(parts[1])
                r = float(parts[2])
            except ValueError:
                continue
            by_level.setdefault(level, []).append(r)
    for lv in by_level:
        by_level[lv].sort()
    return by_level


def first_k_spectral_parameters(N: int, K: int, cache=None,
                                local_path: str = LOCAL_PATH):
    """First K spectral parameters r_k(N) at level N (ascending)."""
    if K <= 0:
        return []
    if cache is None:
        cache = spectral_parameters_by_level(local_path)
    return cache.get(N, [])[:K]


def first_k_eigenvalues(N: int, K: int, cache=None,
                        local_path: str = LOCAL_PATH):
    """First K Laplace eigenvalues lambda_k(N) = r_k(N)^2 + 1/4."""
    return [r * r + 0.25
            for r in first_k_spectral_parameters(N, K, cache, local_path)]
