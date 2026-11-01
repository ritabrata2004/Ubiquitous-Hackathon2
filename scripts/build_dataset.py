"""Raw captures -> intermediate CSV -> QC -> (150, 256) windows, with provenance.

Usage: python scripts/build_dataset.py [--config configs/experiment.yaml] [--force-csv]

Outputs
  intermediate_csv/room_{A,B}/<capture_id>.csv   common 256-column amplitude form
  splits/room_B_validation.txt                   frozen Room-B validation captures
  splits/captures.csv                            every capture: split, label, QC counts
  windows/{development,local_validation}/window_XXXX.npy
  windows/index.csv                              window -> capture provenance
  results/qc_report.json                         QC summary per room/split
"""
import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import cleaning, intermediate, splits, windowing  # noqa: E402

SPLIT_DIR = {"development": "development", "validation": "local_validation"}


def sha1(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def process(cap, cfg, force_csv):
    pre = cfg["preprocessing"]
    out_csv = ROOT / cfg["paths"]["intermediate_csv"] / f"room_{cap['room']}" / f"{cap['capture_id']}.csv"
    stats_file = out_csv.with_suffix(".ingest.json")
    if force_csv or not out_csv.exists():
        if cap["room"] == "A":
            amp, ingest = intermediate.room_A_to_amplitude(ROOT / cap["source"], pre["nexmon"])
        else:
            amp, ingest = intermediate.room_B_to_amplitude(ROOT / cap["source"])
        intermediate.write_csv(out_csv, amp)
        stats_file.write_text(json.dumps(ingest))
    ingest = json.loads(stats_file.read_text())

    amp = intermediate.read_csv(out_csv)
    keep, rejects = cleaning.qc_mask(amp, pre["qc"]["min_subband_energy_share"])
    rec = dict(cap, source_sha1=sha1(ROOT / cap["source"]), n_obs=len(amp),
               n_valid=int(keep.sum()), **{f"rej_{k}": v for k, v in rejects.items()},
               **{f"ingest_{k}": v for k, v in ingest.items()})
    rec["eligible"] = rec["n_valid"] >= pre["qc"]["min_valid_obs"]
    win = windowing.capture_window(amp[keep], pre["window"]["length"],
                                   pre["window"]["resample_padtype"]) if rec["eligible"] else None
    return rec, win


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--force-csv", action="store_true", help="re-ingest even if intermediate CSV exists")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    paths = cfg["paths"]
    config_hash = hashlib.sha1(json.dumps(cfg["preprocessing"], sort_keys=True).encode()).hexdigest()[:12]

    caps = splits.room_A_captures(paths["room_A_raw"]) + splits.room_B_captures(
        paths["room_B_raw"], ROOT / paths["splits"] / "room_B_validation.txt",
        cfg["splits"]["room_B_validation_per_class"], cfg["splits"]["room_B_seed"])
    for c in caps:  # store sources relative to the repo for portability
        c["source"] = str(Path(c["source"]).resolve().relative_to(ROOT))

    with ProcessPoolExecutor() as ex:
        results = list(ex.map(process, caps, [cfg] * len(caps), [args.force_csv] * len(caps), chunksize=8))

    # Window ids are a seeded permutation so that id order carries no room/label/split signal.
    eligible = [i for i, (r, _) in enumerate(results) if r["eligible"]]
    ids = np.random.default_rng(cfg["seed"]).permutation(len(eligible)) + 1
    win_root = ROOT / paths["windows"]
    for d in SPLIT_DIR.values():
        (win_root / d).mkdir(parents=True, exist_ok=True)
        for old in (win_root / d).glob("window_*.npy"):
            old.unlink()
    index = []
    for wid, i in zip(ids, eligible):
        rec, win = results[i]
        name = f"window_{wid:04d}"
        rel = Path(paths["windows"]) / SPLIT_DIR[rec["split"]] / f"{name}.npy"
        np.save(ROOT / rel, win)
        rec["window_id"] = name
        index.append({"window_id": name, "path": str(rel), "split": rec["split"], "room": rec["room"],
                      "label": rec["label"], "capture_id": rec["capture_id"], "source": rec["source"],
                      "source_sha1": rec["source_sha1"], "n_valid_obs": rec["n_valid"],
                      "config_hash": config_hash})
    index.sort(key=lambda r: r["window_id"])
    write_rows(win_root / "index.csv", index)
    records = [r for r, _ in results]
    write_rows(ROOT / paths["splits"] / "captures.csv", records)

    summary = {"config_hash": config_hash}
    for room in "AB":
        for split in SPLIT_DIR:
            rs = [r for r in records if r["room"] == room and r["split"] == split]
            nv = np.array([r["n_valid"] for r in rs])
            summary[f"{room}/{split}"] = {
                "captures": len(rs), "eligible": sum(r["eligible"] for r in rs),
                "excluded": sorted(r["capture_id"] for r in rs if not r["eligible"]),
                "observations": int(sum(r["n_obs"] for r in rs)), "valid_observations": int(nv.sum()),
                "valid_per_capture_median": float(np.median(nv)),
                "rejected": {k: int(sum(r.get(k, 0) or 0 for r in rs)) for k in records[0] if k.startswith("rej_")},
                "per_label": {lab: sum(1 for r in rs if r["label"] == lab and r["eligible"])
                              for lab in sorted({r["label"] for r in rs})}}
    (ROOT / paths["results"]).mkdir(exist_ok=True)
    (ROOT / paths["results"] / "qc_report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
