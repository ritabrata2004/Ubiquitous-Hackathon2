"""Observation-level quality control on the 256-column amplitude form."""
import numpy as np


def qc_mask(amp, min_subband_energy_share):
    """Return (keep_mask, reject_counts). Each rejected row is counted under the first
    rule it fails, in this order:

    nonfinite    any NaN/inf amplitude
    all_zero     no CSI information
    duplicate    exact repeat of an earlier row in the same capture
    partial_band some 20 MHz sub-band carries < min share of energy (e.g. a 20 MHz frame
                 reported in the 80 MHz layout: 192 subcarriers of noise floor)
    """
    n = len(amp)
    keep = np.ones(n, dtype=bool)
    counts = {}

    def reject(name, bad):
        bad = bad & keep
        counts[name] = int(bad.sum())
        keep[bad] = False

    reject("nonfinite", ~np.isfinite(amp).all(axis=1))
    safe = np.where(np.isfinite(amp), amp, 0.0)
    reject("all_zero", ~(safe > 0).any(axis=1))

    seen, dup = set(), np.zeros(n, dtype=bool)
    for i, row in enumerate(safe):
        key = row.tobytes()
        dup[i] = key in seen
        seen.add(key)
    reject("duplicate", dup)

    energy = (safe ** 2).reshape(n, 4, 64).sum(axis=2)
    share = energy / np.maximum(energy.sum(axis=1, keepdims=True), 1e-30)
    reject("partial_band", share.min(axis=1) < min_subband_energy_share)
    return keep, counts
