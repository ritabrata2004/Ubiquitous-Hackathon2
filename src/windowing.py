"""Turn a capture's valid observation sequence into one fixed (150, 256) window.

The whole valid sequence of a capture is resampled to `length` steps with a polyphase
FIR filter (scipy.signal.resample_poly), which low-passes before decimation. Room A
captures (~60 full-band observations) are upsampled and Room B captures (~480) are
downsampled by the same rule, so every window spans its full capture. The procedure is
deterministic and never mixes captures.
"""
import numpy as np
from scipy.signal import resample_poly


def capture_window(amp, length=150, padtype="line"):
    n = len(amp)
    if n == length:
        out = amp
    else:
        out = resample_poly(amp, up=length, down=n, axis=0, padtype=padtype)
    # FIR ringing can push amplitudes slightly below zero; amplitudes are non-negative.
    return np.clip(out[:length], 0.0, None).astype(np.float32)


def center_crop(window, length):
    """Central `length` rows, used for the window-length study."""
    start = (len(window) - length) // 2
    return window[start:start + length]
