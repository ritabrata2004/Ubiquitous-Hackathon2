"""End-to-end inference on a challenge episode: raw captures -> predictions.

Usage:
  python run_episode.py EPISODE_DIR [--out predictions.csv] [--k 5] [--method proto]

EPISODE_DIR layout (captures may be Nexmon .pcap or 256-column amplitude .csv):
  EPISODE_DIR/query/**/*.{pcap,csv}          captures to predict (or directly in EPISODE_DIR)
  EPISODE_DIR/support/<label>/*.{pcap,csv}   optional labeled target-room support captures

Steps per capture: ingest -> QC -> (150, 256) window -> normalization (development stats)
-> room CNN (Task 1) routes the window -> SSL encoder (Task 2) -> the routed room's
standardization -> that room's few-shot head (Task 3).
Each room's head is trained on the other room's development SIT/STAND/WALK windows plus
target support: the episode's support windows routed to that room, or, if the episode has
none, K development captures per target-exclusive class (seed 0).

Requires scripts/build_dataset.py, scripts/task1_room.py and scripts/task2_ssl.py outputs.
Writes CSV columns: window_id, room, label, confidence.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import data, fewshot, models, normalization, pipeline, ssl, train, windowing  # noqa: E402

OTHER = {"A": "B", "B": "A"}


def captures(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in (".pcap", ".csv"))


def to_window(path, cfg):
    amp, _ = pipeline.load_amplitude(path, cfg)
    win, n_valid, _ = pipeline.amplitude_to_window(amp, cfg)
    if win is None:  # too few full-band observations: fall back to every observation
        print(f"warning: {path} has {n_valid} valid observations; using all {len(amp)}", file=sys.stderr)
        w = cfg["preprocessing"]["window"]
        win = windowing.capture_window(np.nan_to_num(amp), w["length"], w["resample_padtype"], w["lowpass_cycles"])
    return win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--k", type=int, default=5, help="fallback support captures per exclusive class")
    ap.add_argument("--method", default="logreg", choices=["proto", "proto_tkm", "logreg"])
    ap.add_argument("--config", default="configs/experiment.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    episode = Path(args.episode)

    # Development reference data (normalization and per-room statistics; never validation).
    index = data.load_index()
    dev_rows, dev_x = data.load_split(index, "development")
    stats = normalization.fit(dev_x)
    dev_x = normalization.apply(dev_x, stats)
    dev_labels, dev_rooms = data.column(dev_rows, "label"), data.column(dev_rows, "room")

    room_clf = models.Classifier(2)
    room_clf.load_state_dict(torch.load(ROOT / "results/models/room_cnn.pt"))
    encoder = models.Encoder()
    encoder.load_state_dict(torch.load(ROOT / "results/models/ssl_encoder.pt"))

    def prepare(paths):
        x = normalization.apply(np.stack([to_window(p, cfg) for p in paths]), stats)
        proba = train.predict_proba(room_clf, x)
        return x, np.array(["A", "B"])[proba.argmax(1)], ssl.embed(encoder, x)

    dev_f = ssl.embed(encoder, dev_x)
    dom = fewshot.domain_stats(dev_f, dev_rooms, np.ones(len(dev_f), bool))
    dev_z = fewshot.standardize(dev_f, dev_rooms, dom)

    support_dir = episode / "support"
    sup_paths = captures(support_dir) if support_dir.exists() else []
    if sup_paths:
        _, sup_rooms, sup_f = prepare(sup_paths)
        sup_z = fewshot.standardize(sup_f, sup_rooms, dom)
        sup_labels = np.array([p.relative_to(support_dir).parts[0].lower() for p in sup_paths])
        print(f"episode support: {len(sup_paths)} captures, routed rooms {dict(zip(*np.unique(sup_rooms, return_counts=True)))}")

    rng = np.random.default_rng(0)
    all_dev = np.ones(len(dev_rows), bool)
    heads = {}
    for t in "AB":
        src = np.flatnonzero((dev_rooms == OTHER[t]) & np.isin(dev_labels, data.SHARED))
        if sup_paths and (sup_rooms == t).any():
            f_sup, y_sup = sup_z[sup_rooms == t], sup_labels[sup_rooms == t]
        else:
            idx = fewshot.support_indices(dev_labels, dev_rooms, all_dev, t, args.k, rng)
            f_sup, y_sup = dev_z[idx], dev_labels[idx]
        heads[t] = {"f_train": np.concatenate([dev_z[src], f_sup]),
                    "y_train": np.concatenate([dev_labels[src], y_sup]),
                    "f_unlab": dev_z[dev_rooms == t], "support": (f_sup, y_sup)}

    query_dir = episode / "query" if (episode / "query").exists() else episode
    q_paths = [p for p in captures(query_dir) if support_dir not in p.parents]
    _, q_rooms, q_f = prepare(q_paths)
    q_z = fewshot.standardize(q_f, q_rooms, dom)
    labels, conf = np.empty(len(q_paths), dtype=object), np.zeros(len(q_paths))
    for t in "AB":
        m = q_rooms == t
        if m.any():
            labels[m], conf[m] = fewshot.fit_predict(args.method, f_test=q_z[m], **heads[t])

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["window_id", "room", "label", "confidence"])
        for p, r, lab, c in zip(q_paths, q_rooms, labels, conf):
            w.writerow([str(p.relative_to(query_dir).with_suffix("")), r, lab, round(float(c), 4)])
    print(f"wrote {len(q_paths)} predictions to {args.out}")


if __name__ == "__main__":
    main()
