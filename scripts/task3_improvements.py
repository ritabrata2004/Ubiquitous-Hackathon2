"""Task 3 follow-up: targeted fixes for the diagnosed failure modes, on the primary
configuration (SSL embeddings, logreg head, per-room standardization, oracle routing, which
equals predicted-room routing here). Same 10 support episodes per K as task3_fewshot.py.

Variants (all decided before running, none tuned on validation):
  base            primary configuration
  crops           + 4 extra crops (120 steps at offsets 0/10/20/30) per support capture;
                  still one capture = one shot
  inlp_room       + project out room-predictive directions (INLP) fitted on ALL development
                  windows with room as the only label, until 5-fold room accuracy <= 0.6
  inlp_head       + project out directions separating the head's source-room training windows
                  from the target support windows (10 directions, per episode)
  inlp_room_crops inlp_room + crops
  selftrain       base + self-training on target unlabeled dev windows (p >= 0.9, 3 rounds)
  inlp_selftrain  inlp_room + self-training

Extra diagnostics: shortcut_rate = share of true shared-class windows predicted as a target-
exclusive class; sitstand_rate = share of true SIT/STAND windows predicted as the other one.

Usage: python scripts/task3_improvements.py [--episodes 10]
Writes results/task3_improvements.csv and results/figures/task3_improvements.png.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.task3_fewshot import KS, OTHER, metrics, write_csv  # noqa: E402
from src import data, fewshot, models, ssl  # noqa: E402

VARIANTS = ["base", "crops", "inlp_room", "inlp_head", "inlp_room_crops", "selftrain", "inlp_selftrain"]
CROP_LEN, CROP_OFFSETS = 120, (0, 10, 20, 30)


def diagnostics(y, p, t):
    shared = np.isin(y, data.SHARED)
    ss = np.isin(y, ["sit", "stand"])
    swap = ((y == "sit") & (p == "stand")) | ((y == "stand") & (p == "sit"))
    return {"shortcut_rate": float(np.isin(p[shared], data.EXCLUSIVE[t]).mean()),
            "sitstand_rate": float(swap[ss].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()
    torch.set_num_threads(12)
    out = ROOT / "results"

    dev_rows, dev_x, val_rows, val_x, _ = data.load_normalized()
    rows = dev_rows + val_rows
    x = np.concatenate([dev_x, val_x])
    labels, rooms = data.column(rows, "label"), data.column(rows, "room")
    dev = data.column(rows, "split") == "development"
    val = ~dev

    enc = models.Encoder()
    enc.load_state_dict(torch.load(ROOT / "results/models/ssl_encoder.pt"))
    f = ssl.embed(enc, x)
    crops = np.stack([ssl.embed(enc, np.ascontiguousarray(x[:, o:o + CROP_LEN])) for o in CROP_OFFSETS], 1)

    stats = fewshot.domain_stats(f, rooms, dev)
    z = fewshot.standardize(f, rooms, stats)
    zc = np.stack([fewshot.standardize(crops[:, i], rooms, stats) for i in range(len(CROP_OFFSETS))], 1)

    p_room = fewshot.nullspace_projection(z[dev], rooms[dev])
    n_removed = int(round(z.shape[1] - np.trace(p_room)))
    print(f"INLP room: removed {n_removed} directions")

    def head(feats, crop_feats, t, sup, use_crops, inlp_head, selftrain):
        src = np.flatnonzero(dev & (rooms == OTHER[t]) & np.isin(labels, data.SHARED))
        f_tr = np.concatenate([feats[src], feats[sup]])
        y_tr = np.concatenate([labels[src], labels[sup]])
        if use_crops:
            f_tr = np.concatenate([f_tr, crop_feats[sup].reshape(-1, feats.shape[1])])
            y_tr = np.concatenate([y_tr, np.repeat(labels[sup], len(CROP_OFFSETS))])
        test = np.flatnonzero(val & (rooms == t))
        f_te, f_un = feats[test], feats[dev & (rooms == t)]
        if inlp_head:
            g = np.r_[np.zeros(len(src)), np.ones(len(f_tr) - len(src))]
            P = fewshot.nullspace_projection(f_tr, g, max_iter=10, stop_acc=0.0, cv=False)
            f_tr, f_te, f_un = f_tr @ P, f_te @ P, f_un @ P
        if selftrain:
            return test, fewshot.self_train_logreg(f_tr, y_tr, f_te, f_un)[0]
        return test, fewshot.fit_predict("logreg", f_tr, y_tr, f_te)[0]

    zp, zcp = z @ p_room, zc @ p_room
    config = {"base": (z, zc, False, False, False), "crops": (z, zc, True, False, False),
              "inlp_room": (zp, zcp, False, False, False), "inlp_head": (z, zc, False, True, False),
              "inlp_room_crops": (zp, zcp, True, False, False), "selftrain": (z, zc, False, False, True),
              "inlp_selftrain": (zp, zcp, False, False, True)}

    groups = defaultdict(list)
    for k in KS:
        for seed in range(args.episodes):
            rng = np.random.default_rng(seed)  # identical support draws to task3_fewshot.py
            sup = {t: fewshot.support_indices(labels, rooms, dev, t, k, rng) for t in "AB"}
            for v in VARIANTS:
                feats, cfeats, use_crops, inlp_head, st = config[v]
                for t in "AB":
                    test, p = head(feats, cfeats, t, sup[t], use_crops, inlp_head, st)
                    for name, val_ in {**metrics(labels[test], p, t), **diagnostics(labels[test], p, t)}.items():
                        groups[(v, t, k, name)].append(val_)
        print("K", k, "done")

    summary = [{"experiment": f"task3_improve_{v}", "target_room": t, "K": k, "metric": m,
                "value": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4),
                "n_episodes": len(vals)} for (v, t, k, m), vals in groups.items()]
    write_csv(out / "task3_improvements.csv", summary)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for col, t in enumerate("AB"):
        for row, metric in enumerate(["macro_f1", "shortcut_rate"]):
            ax = axes[row, col]
            for v in VARIANTS:
                ax.plot(KS, [np.mean(groups[(v, t, k, metric)]) for k in KS], marker="o", label=v)
            ax.set(title=f"{OTHER[t]} -> {t}: {metric}", xscale="log", xticks=KS, xticklabels=KS,
                   ylim=(0, 1))
            ax.grid(alpha=0.3)
    axes[1, 0].set_xlabel("K")
    axes[1, 1].set_xlabel("K")
    axes[0, 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figures/task3_improvements.png", dpi=150)

    for t in "AB":
        print(f"\n== target {t}: macro / exclusive / shared F1, shortcut, sit<->stand  (K=5 | K=20)")
        for v in VARIANTS:
            cells = []
            for k in (5, 20):
                cells.append(" ".join(f"{np.mean(groups[(v, t, k, m)]):.3f}" for m in
                                      ["macro_f1", "exclusive_macro_f1", "shared_macro_f1",
                                       "shortcut_rate", "sitstand_rate"]))
            print(f"{v:16s} " + " | ".join(cells))


if __name__ == "__main__":
    main()
