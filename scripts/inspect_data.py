"""Hint 1: inspect the raw source files and record the facts the pipeline design relies on.

Usage: python scripts/inspect_data.py
Writes results/inspection.json and results/figures/{subband_energy_share,valid_obs_per_capture}.png
"""
import collections
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import dpkt
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import csi_decode, intermediate, splits  # noqa: E402
from src.pcap_reader import HEADER  # noqa: E402


def min_subband_share(amp):
    e = (amp ** 2).reshape(len(amp), 4, 64).sum(axis=2)
    return (e / np.maximum(e.sum(axis=1, keepdims=True), 1e-30)).min(axis=1)


def inspect_pcap(path):
    ts, lens, seq, chanspec, chip, words = [], [], [], set(), set(), []
    with open(path, "rb") as fh:
        for t, buf in dpkt.pcap.Reader(fh):
            p = bytes(dpkt.ethernet.Ethernet(buf).data.data.data)
            ts.append(t)
            lens.append(len(p))
            if len(p) == HEADER.size + 1024:
                h = HEADER.unpack_from(p)
                seq.append(h[4])
                chanspec.add(h[6])
                chip.add(h[7])
                words.append(np.frombuffer(p, "<u4", offset=HEADER.size))
    amp = csi_decode.amplitude(csi_decode.decode_complex(np.array(words)))
    share = min_subband_share(amp)
    seq = np.array(seq)
    return {"packets": len(ts), "payload_lengths": dict(collections.Counter(lens)),
            "duration_s": float(ts[-1] - ts[0]), "monotonic": bool(np.all(np.diff(ts) >= 0)),
            "seq_none_frac": float(np.mean(seq == 65535)), "chanspec": sorted(chanspec),
            "chip": sorted(chip), "share": share}


def inspect_csv(path):
    amp, _ = intermediate.room_B_to_amplitude(path)
    with open(path) as fh:
        header = fh.readline().strip().split(",")
    return {"rows": len(amp), "header_first": header[0], "integer_valued": bool(np.all(amp == np.round(amp))),
            "share": min_subband_share(amp)}


def main():
    cfg = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
    thr = cfg["preprocessing"]["qc"]["min_subband_energy_share"]
    a_files = [c["source"] for c in splits.room_A_captures(ROOT / cfg["paths"]["room_A_raw"])]
    b_files = sorted(str(p) for p in (ROOT / cfg["paths"]["room_B_raw"]).glob("*.csv"))
    with ProcessPoolExecutor() as ex:
        ra = list(ex.map(inspect_pcap, a_files, chunksize=16))
        rb = list(ex.map(inspect_csv, b_files, chunksize=16))

    lens = collections.Counter()
    for r in ra:
        lens.update(r["payload_lengths"])
    va = np.array([(r["share"] >= thr).sum() for r in ra])
    vb = np.array([(r["share"] >= thr).sum() for r in rb])
    report = {
        "room_A": {
            "captures": len(ra),
            "packets_per_capture": np.percentile([r["packets"] for r in ra], [0, 50, 100]).tolist(),
            "duration_s_median": float(np.median([r["duration_s"] for r in ra])),
            "udp_payload_lengths": {str(k): v for k, v in lens.items()},
            "chanspec": sorted({c for r in ra for c in r["chanspec"]}),
            "chip_version": sorted({c for r in ra for c in r["chip"]}),
            "non_monotonic_timestamp_captures": int(sum(not r["monotonic"] for r in ra)),
            "seq_65535_fraction": float(np.mean([r["seq_none_frac"] for r in ra])),
            "full_band_fraction": float(np.mean(np.concatenate([r["share"] for r in ra]) >= thr)),
            "full_band_obs_per_capture_pct_0_5_50_95_100": np.percentile(va, [0, 5, 50, 95, 100]).tolist(),
        },
        "room_B": {
            "captures": len(rb),
            "rows_per_capture": sorted({r["rows"] for r in rb}),
            "header_variants": dict(collections.Counter(r["header_first"] for r in rb)),
            "integer_valued": all(r["integer_valued"] for r in rb),
            "full_band_fraction": float(np.mean(np.concatenate([r["share"] for r in rb]) >= thr)),
            "full_band_obs_per_capture_pct_0_5_50_95_100": np.percentile(vb, [0, 5, 50, 95, 100]).tolist(),
        },
    }
    out = ROOT / cfg["paths"]["results"]
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "inspection.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    bins = np.logspace(-8, np.log10(0.25), 60)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name, rs in [("Room A", ra), ("Room B", rb)]:
        s = np.concatenate([r["share"] for r in rs])
        ax.hist(np.clip(s, bins[0], None), bins=bins, histtype="step", weights=np.full(len(s), 1 / len(s)),
                label=name)
    ax.axvline(thr, color="k", ls="--", lw=1, label=f"QC threshold {thr}")
    ax.set(xscale="log", xlabel="smallest 20 MHz sub-band share of CSI energy", ylabel="fraction of observations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures/subband_energy_share.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist([va, vb], bins=np.arange(0, 510, 10), label=["Room A", "Room B"])
    ax.axvline(cfg["preprocessing"]["window"]["length"], color="k", ls="--", lw=1, label="window length")
    ax.set(xlabel="valid (full-band) observations per capture", ylabel="captures")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures/valid_obs_per_capture.png", dpi=150)


if __name__ == "__main__":
    main()
