"""Task 3: cross-room few-label adaptation (A->B and B->A), K in {1, 5, 10, 20}.

Usage: python scripts/task3_fewshot.py [--episodes 10]

For target room T with source room S, the head for T is trained on
  - S development windows of SIT/STAND/WALK (all S labels allowed), and
  - K distinct T development captures per T-exclusive class (support),
and scored on ALL T validation windows over T's class space. T's shared-class labels are
never used, validation is never used for support or fitting, and support is redrawn for
each of `episodes` seeds.

Conditions
  oracle       window routed by its true room (domain standardization + T's head)
  routed       window routed by the Task-1 room CNN's prediction (deployment setting)
  no_routing   one global standardization and one head over S's classes + T's exclusive classes
  no_align     oracle class space, but global instead of per-room standardization
Representations: ssl (Task-2 encoder), random (same architecture), handcrafted.
Methods: proto, proto_tkm (transductive prototypes), logreg.

Writes results/task3_fewshot_episodes.csv (per episode), results/task3_fewshot.csv
(mean/std; experiment, target_room, K, metric, value), per-class tables and confusion
matrices at K=5, predictions, and figures.
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
from sklearn.metrics import (ConfusionMatrixDisplay, balanced_accuracy_score, confusion_matrix,  # noqa: E402
                             f1_score, precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data, features, fewshot, models, ssl, train  # noqa: E402

KS = [1, 5, 10, 20]
# Primary configuration, chosen on validation (best mean macro-F1 over both directions and all K).
PRIMARY = ("ssl", "logreg")
OTHER = {"A": "B", "B": "A"}


def representations(x):
    enc = models.Encoder()
    enc.load_state_dict(torch.load(ROOT / "results/models/ssl_encoder.pt"))
    train.set_seed(0)
    rand = models.Encoder()
    return {"ssl": ssl.embed(enc, x), "random": ssl.embed(rand, x), "handcrafted": features.handcrafted(x)}


def predicted_rooms(x):
    clf = models.Classifier(2)
    clf.load_state_dict(torch.load(ROOT / "results/models/room_cnn.pt"))
    return np.array(["A", "B"])[train.predict_proba(clf, x).argmax(1)]


def metrics(y_true, y_pred, target):
    ex = list(data.EXCLUSIVE[target])
    return {"macro_f1": f1_score(y_true, y_pred, labels=fewshot.target_classes(target), average="macro",
                                 zero_division=0),
            "exclusive_macro_f1": f1_score(y_true, y_pred, labels=ex, average="macro", zero_division=0),
            "shared_macro_f1": f1_score(y_true, y_pred, labels=list(data.SHARED), average="macro",
                                        zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)}


def run_episode(f, labels, rooms, dev, val, pred_room, k, seed, method):
    """Return {(condition, target): (val indices, predictions, confidences)}."""
    rng = np.random.default_rng(seed)
    support = {t: fewshot.support_indices(labels, rooms, dev, t, k, rng) for t in "AB"}
    per_room = fewshot.standardize(f, rooms, fewshot.domain_stats(f, rooms, dev))
    glob_stats = {r: (f[dev].mean(0), f[dev].std(0) + 1e-6) for r in "AB"}
    glob = fewshot.standardize(f, rooms, glob_stats)
    # Routed windows are standardized with the stats of the room they are routed to.
    routed = fewshot.standardize(f, pred_room, fewshot.domain_stats(f, rooms, dev))

    def fit(train_feats, t, source_classes, test_feats):
        """Head for target t: source-room windows of `source_classes` + t's support."""
        src = np.flatnonzero(dev & (rooms == OTHER[t]) & np.isin(labels, source_classes))
        tr = np.concatenate([src, support[t]])
        return fewshot.fit_predict(method, train_feats[tr], labels[tr], test_feats,
                                   f_unlab=train_feats[dev & (rooms == t)],
                                   support=(train_feats[support[t]], labels[support[t]]))

    out = {}
    for t in "AB":
        test = np.flatnonzero(val & (rooms == t))
        out[("oracle", t)] = (test, *fit(per_room, t, data.SHARED, per_room[test]))
        p, c = np.empty(len(test), dtype=object), np.empty(len(test))
        for r in "AB":  # each window goes to the head of its predicted room
            m = pred_room[test] == r
            if m.any():
                p[m], c[m] = fit(per_room, r, data.SHARED, routed[test[m]])
        out[("routed", t)] = (test, p.astype(str), c)
        out[("no_routing", t)] = (test, *fit(glob, t, np.unique(labels[rooms == OTHER[t]]), glob[test]))
        out[("no_align", t)] = (test, *fit(glob, t, data.SHARED, glob[test]))
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--plot-only", action="store_true",
                    help="reuse results/task3_fewshot_episodes.csv; rerun only the K=5 primary episode")
    args = ap.parse_args()
    torch.set_num_threads(12)
    out = ROOT / "results"

    dev_rows, dev_x, val_rows, val_x, _ = data.load_normalized()
    rows = dev_rows + val_rows
    x = np.concatenate([dev_x, val_x])
    labels, rooms = data.column(rows, "label"), data.column(rows, "room")
    dev = data.column(rows, "split") == "development"
    val = ~dev
    pred_room = predicted_rooms(x)
    print("routing accuracy on validation:", (pred_room[val] == rooms[val]).mean())
    reps = representations(x)

    if args.plot_only:
        with open(out / "task3_fewshot_episodes.csv") as fh:
            episodes = [dict(e, K=int(e["K"]), value=float(e["value"])) for e in csv.DictReader(fh)]
        res = run_episode(reps[PRIMARY[0]], labels, rooms, dev, val, pred_room, 5, 0, PRIMARY[1])
        keep = {key: v for key, v in res.items()}
    else:
        episodes, keep = [], {}
        for rep, f in reps.items():
            for method in ("proto", "proto_tkm", "logreg"):
                for k in KS:
                    for seed in range(args.episodes):
                        res = run_episode(f, labels, rooms, dev, val, pred_room, k, seed, method)
                        for (cond, t), (idx, p, c) in res.items():
                            if rep != PRIMARY[0] and cond != "oracle":
                                continue
                            for metric, v in metrics(labels[idx], p, t).items():
                                episodes.append({"representation": rep, "method": method, "condition": cond,
                                                 "target_room": t, "K": k, "seed": seed, "metric": metric,
                                                 "value": round(float(v), 4)})
                            if (rep, method) == PRIMARY and k == 5 and seed == 0:
                                keep[(cond, t)] = (idx, p, c)
                    print(rep, method, "K", k, "done")
        write_csv(out / "task3_fewshot_episodes.csv", episodes)

    groups = defaultdict(list)
    for e in episodes:
        groups[(e["representation"], e["method"], e["condition"], e["target_room"], e["K"], e["metric"])].append(e["value"])
    summary = [{"experiment": f"task3_{rep}_{method}_{cond}", "target_room": t, "K": k, "metric": metric,
                "value": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4), "n_episodes": len(v)}
               for (rep, method, cond, t, k, metric), v in groups.items()]
    write_csv(out / "task3_fewshot.csv", summary)

    def curve(rep, method, cond, t, metric):
        g = [groups[(rep, method, cond, t, k, metric)] for k in KS]
        return np.array([np.mean(v) for v in g]), np.array([np.std(v) for v in g])

    rep, method = PRIMARY
    # Label-efficiency curves.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, t in zip(axes, "AB"):
        for metric, name in [("macro_f1", "all target classes"), ("exclusive_macro_f1", "target-labeled (exclusive)"),
                             ("shared_macro_f1", "transferred shared")]:
            m, s = curve(rep, method, "oracle", t, metric)
            ax.errorbar(KS, m, yerr=s, marker="o", capsize=3, label=name)
        for r in ("random", "handcrafted"):
            m, _ = curve(r, method, "oracle", t, "macro_f1")
            ax.plot(KS, m, ls="--", marker=".", label=f"{r} repr., all classes")
        ax.set(title=f"{OTHER[t]} -> {t} ({rep}, {method})", xlabel="K (support captures per exclusive class)",
               xscale="log", xticks=KS, xticklabels=KS, ylim=(0, 1))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("validation macro-F1")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "figures/task3_label_efficiency.png", dpi=150)

    # Routing contribution.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, t in zip(axes, "AB"):
        for cond in ("oracle", "routed", "no_routing", "no_align"):
            m, s = curve(rep, method, cond, t, "macro_f1")
            ax.errorbar(KS, m, yerr=s, marker="o", capsize=3, label=cond)
        ax.set(title=f"{OTHER[t]} -> {t}: routing contribution", xlabel="K", xscale="log", xticks=KS,
               xticklabels=KS, ylim=(0, 1))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("validation macro-F1")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures/task3_routing.png", dpi=150)

    # Confusion matrices, per-class P/R/F1 and predictions at K=5 (episode 0, routed = deployment).
    (out / "predictions").mkdir(exist_ok=True)
    per_class = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, t in zip(axes, "AB"):
        idx, p, c = keep[("routed", t)]
        cls = fewshot.target_classes(t)
        ConfusionMatrixDisplay(confusion_matrix(labels[idx], p, labels=cls), display_labels=cls).plot(
            ax=ax, colorbar=False)
        ax.set_title(f"{OTHER[t]} -> {t}, K=5 ({rep}, {method}, routed)")
        pr, rc, f1, n = precision_recall_fscore_support(labels[idx], p, labels=cls, zero_division=0)
        per_class += [{"target_room": t, "K": 5, "activity": a, "precision": round(float(a1), 4),
                       "recall": round(float(a2), 4), "f1": round(float(a3), 4), "support": int(a4)}
                      for a, a1, a2, a3, a4 in zip(cls, pr, rc, f1, n)]
        with open(out / f"predictions/task3_target{t}_K5.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["window_id", "room", "label", "confidence"])
            for i, lab, conf in zip(idx, p, c):
                w.writerow([rows[i]["window_id"], pred_room[i], lab, round(float(conf), 4)])
    fig.tight_layout()
    fig.savefig(out / "figures/task3_confusion_K5.png", dpi=150)
    write_csv(out / "task3_per_class_K5.csv", per_class)

    for s in summary:
        if s["metric"] in ("macro_f1", "exclusive_macro_f1") and s["experiment"].startswith("task3_ssl"):
            print(f"{s['experiment']:30s} {s['target_room']} K={s['K']:2d} {s['metric']:20s} "
                  f"{s['value']:.3f} ± {s['std']:.3f}")


if __name__ == "__main__":
    main()
