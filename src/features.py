"""Handcrafted baseline features from a normalized (T, 256) window.

Per subcarrier: temporal mean (static multipath profile) and temporal std (motion-induced
fluctuation), plus the mean absolute first difference averaged over subcarriers
(overall motion energy). 513 features.
"""
import numpy as np
from scipy.stats import kurtosis


def event_features(x):
    """Transition-vs-steady-state descriptors of a normalized (n, T, 256) window, 11 features.

    Motion energy e(t) = mean |x[t+1] - x[t]| over subcarriers, divided by its per-capture
    median so absolute (room/session) motion level cancels. Features: log peak ratio, peak
    position, std and kurtosis of e, and the log of e's 7-segment time profile relative to its
    mean. A transition (fall, sit-down, stand-up) is a burst then stillness; SIT/STAND/WALK
    are steady states.
    """
    e = np.abs(np.diff(x, axis=1)).mean(axis=2)
    e = e / (np.median(e, axis=1, keepdims=True) + 1e-6)
    prof = np.stack([seg.mean(axis=1) for seg in np.array_split(e, 7, axis=1)], axis=1)
    return np.column_stack([np.log(e.max(axis=1)), e.argmax(axis=1) / e.shape[1], e.std(axis=1),
                            kurtosis(e, axis=1), np.log(prof / prof.mean(axis=1, keepdims=True) + 1e-6)])


def handcrafted(x):
    """x: (n, T, 256) -> (n, 513)."""
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    motion = np.abs(np.diff(x, axis=1)).mean(axis=(1, 2))[:, None]
    return np.concatenate([mean, std, motion], axis=1)
