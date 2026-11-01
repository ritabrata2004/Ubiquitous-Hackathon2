"""Model-side normalization of (150, 256) amplitude windows.

1. Per-window gain removal: divide by the window's median amplitude. The two rooms
   report amplitude on unrelated scales (Room A: decoded block-float, Room B: integer
   export) and per-session gain differs, so absolute level is not activity information.
2. log(x + eps): compresses the multipath dynamic range (deep fades vs strong taps).
3. Per-subcarrier standardization with mean/std fitted on DEVELOPMENT windows only.
   Steps 1-2 have no fitted parameters; step 3 is the only learned statistic.
"""
import numpy as np

EPS = 1e-3


def log_gain_normalized(windows):
    w = np.asarray(windows, dtype=np.float32)
    med = np.median(w, axis=(-2, -1), keepdims=True)
    return np.log(w / np.maximum(med, 1e-12) + EPS)


def fit(dev_windows):
    x = log_gain_normalized(dev_windows)
    mean = x.mean(axis=(0, 1))
    std = x.std(axis=(0, 1)) + 1e-6
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def apply(windows, stats):
    return ((log_gain_normalized(windows) - stats["mean"]) / stats["std"]).astype(np.float32)
