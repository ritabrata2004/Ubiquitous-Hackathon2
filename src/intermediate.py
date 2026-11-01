"""Common intermediate form: one CSV per capture, 256 amplitude columns, one row per
observation in temporal (record) order. Downstream stages read only these files."""
import numpy as np

from .csi_decode import amplitude, decode_complex
from .pcap_reader import read_csi_packets

COLUMNS = [f"subcarrier_{i:03d}" for i in range(1, 257)]


def room_A_to_amplitude(pcap_path, nexmon_cfg):
    words, stats = read_csi_packets(pcap_path, nexmon_cfg["udp_port"], nexmon_cfg["magic"],
                                    nexmon_cfg["n_subcarriers"])
    return amplitude(decode_complex(words)), stats


def room_B_to_amplitude(csv_path):
    # Header naming varies (Subcarrier_1.. or 0..255); only the 256 numeric columns matter.
    a = np.genfromtxt(csv_path, delimiter=",", skip_header=1, dtype=np.float64)
    a = np.atleast_2d(a)
    if a.shape[1] != 256:
        raise ValueError(f"{csv_path}: expected 256 columns, got {a.shape[1]}")
    return a, {"packets": len(a)}


def write_csv(path, amp):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, amp, delimiter=",", fmt="%.7g", header=",".join(COLUMNS), comments="")


def read_csv(path):
    return np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64))
