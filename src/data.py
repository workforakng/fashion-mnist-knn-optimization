"""Fashion-MNIST loading, normalization and splits.

Data files are the official Zalando IDX archives (also mirrored on Kaggle:
zalando-dataset/fashionmnist). Loader expects them under <repo>/data/.
"""
from __future__ import annotations

import gzip
import os
import struct
import urllib.request

import numpy as np
from sklearn.model_selection import train_test_split

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = {
    "train-images-idx3-ubyte.gz": (60000, 28, 28),
    "train-labels-idx1-ubyte.gz": (60000,),
    "t10k-images-idx3-ubyte.gz": (10000, 28, 28),
    "t10k-labels-idx1-ubyte.gz": (10000,),
}


def _read_idx(path: str) -> np.ndarray:
    """Parse a gzipped IDX file into a numpy array."""
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    dims = struct.unpack(">" + "I" * ndim, data[4:4 + 4 * ndim])
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def ensure_downloaded() -> None:
    """Download any missing IDX file and verify its header shape."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for fname, expected in FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"[data] downloading {fname} ...")
            urllib.request.urlretrieve(BASE_URL + fname, path)
        got = _read_idx(path).shape
        if got != expected:
            raise RuntimeError(f"{fname}: corrupt header {got}, expected {expected}")


def load_raw() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train uint8 [60000,784], y_train, X_test uint8 [10000,784], y_test)."""
    ensure_downloaded()
    x_train = _read_idx(os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz")).reshape(60000, -1)
    y_train = _read_idx(os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz"))
    x_test = _read_idx(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz")).reshape(10000, -1)
    y_test = _read_idx(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz"))
    return x_train, y_train, x_test, y_test


def load_splits(val_size: int = 6000, seed: int = 42):
    """Normalized float32 splits: (X_fit, y_fit, X_val, y_val, X_test, y_test).

    Pixels scaled to [0, 1]. Validation is stratified out of the 60k train pool;
    the 10k test set stays untouched.
    """
    x_train, y_train, x_test, y_test = load_raw()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train, y_train, test_size=val_size, random_state=seed, stratify=y_train
    )
    return x_fit, y_fit, x_val, y_val, x_test, y_test
