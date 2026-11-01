"""Turn a capture's valid observation sequence into one fixed (150, 256) window.

The whole valid sequence of a capture is resampled to `length` steps with a polyphase
FIR filter (scipy.signal.resample_poly), which low-passes before decimation. Room A
captures (~60 full-band observations) are upsampled and Room B captures (~480) are
downsampled by the same rule, so every window spans its full capture. The procedure is
deterministic and never mixes captures.

Resampling leaves each room with a different temporal bandwidth (Room A content stops at
~n_valid/2 cycles per window), which identifies the room and tracks packet count. A common
zero-phase low-pass at `lowpass_cycles` cycles per window (~5 Hz for a 4 s capture at 20)
gives both rooms the same bandwidth.
"""
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


def capture_window(amp, length=150, padtype="line", lowpass_cycles=None):
    n = len(amp)
    if n == length:
        out = amp
    else:
        out = resample_poly(amp, up=length, down=n, axis=0, padtype=padtype)[:length]
    if lowpass_cycles:
        sos = butter(4, lowpass_cycles / (length / 2), output="sos")
        out = sosfiltfilt(sos, out, axis=0)
    # Filter ringing can push amplitudes slightly below zero; amplitudes are non-negative.
    return np.clip(out, 0.0, None).astype(np.float32)


def center_crop(window, length):
    """Central `length` rows, used for the window-length study."""
    start = (len(window) - length) // 2
    return window[start:start + length]
