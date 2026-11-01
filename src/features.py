"""Handcrafted baseline features from a normalized (T, 256) window.

Per subcarrier: temporal mean (static multipath profile) and temporal std (motion-induced
fluctuation), plus the mean absolute first difference averaged over subcarriers
(overall motion energy). 513 features.
"""
import numpy as np


def handcrafted(x):
    """x: (n, T, 256) -> (n, 513)."""
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    motion = np.abs(np.diff(x, axis=1)).mean(axis=(1, 2))[:, None]
    return np.concatenate([mean, std, motion], axis=1)
