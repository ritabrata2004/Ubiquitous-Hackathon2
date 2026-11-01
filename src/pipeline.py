"""Per-capture preprocessing shared by the dataset build and episode inference:
raw capture -> 256-column amplitude -> QC -> (150, 256) window."""
from pathlib import Path

from . import cleaning, intermediate, windowing


def load_amplitude(path, cfg):
    """Dispatch on file type: Nexmon .pcap (Room-A format) or amplitude .csv (Room-B format)."""
    if Path(path).suffix.lower() == ".pcap":
        return intermediate.room_A_to_amplitude(path, cfg["preprocessing"]["nexmon"])
    return intermediate.room_B_to_amplitude(path)


def amplitude_to_window(amp, cfg):
    """Return (window or None if the capture is not eligible, n_valid, reject counts)."""
    qc, win = cfg["preprocessing"]["qc"], cfg["preprocessing"]["window"]
    keep, rejects = cleaning.qc_mask(amp, qc["min_subband_energy_share"])
    n_valid = int(keep.sum())
    if n_valid < qc["min_valid_obs"]:
        return None, n_valid, rejects
    return windowing.capture_window(amp[keep], win["length"], win["resample_padtype"],
                                    win["lowpass_cycles"]), n_valid, rejects
