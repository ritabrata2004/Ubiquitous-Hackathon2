"""Task 1: recorded-environment identification (Room A vs Room B) from CSI windows.

Usage: python scripts/task1_room.py [--seeds 3] [--epochs 30]

Models (all trained on development windows of both rooms, scored on validation windows):
  static_profile  logistic regression on the per-subcarrier temporal mean (multipath profile)
  dynamics_only   logistic regression on the subcarrier-averaged temporal power spectrum
                  (shortcut control: no spectral/multipath information at all)
  handcrafted     logistic regression on src/features.handcrafted
  cnn             src/models.Classifier trained with random-length crops, saved to
                  results/models/room_cnn.pt for routing in Task 3
Headline numbers use the official 150-observation windows. The window-length study
center-crops validation windows; the logistic models are refit on equally cropped
development windows, the CNN is trained once on random-length crops.

Writes results/task1_room.csv (experiment, target_room, K, metric, value),
results/predictions/task1_room_cnn.csv, results/figures/room_acc_vs_window_length.png.
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data, features, models, train, windowing  # noqa: E402

LENGTHS = [10, 25, 50, 75, 100, 150]


def temporal_psd(x):
    z = x - x.mean(axis=1, keepdims=True)
    p = (np.abs(np.fft.rfft(z, axis=1)) ** 2).mean(axis=2)[:, 1:]
    return np.log(p / p.sum(axis=1, keepdims=True) + 1e-12)


FEATURES = {"static_profile": lambda x: x.mean(axis=1), "dynamics_only": temporal_psd,
            "handcrafted": features.handcrafted}


def crop(x, length):
    return np.stack([windowing.center_crop(w, length) for w in x])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    torch.set_num_threads(12)

    dev_rows, dev_x, val_rows, val_x, _ = data.load_normalized()
    y_dev = (data.column(dev_rows, "room") == "B").astype(np.int64)
    y_val = (data.column(val_rows, "room") == "B").astype(np.int64)
    results = []

    def record(model, length, y_pred, seed=""):
        for metric, fn in [("accuracy", accuracy_score), ("balanced_accuracy", balanced_accuracy_score)]:
            results.append({"experiment": f"task1_room_{model}", "target_room": "A+B", "K": "",
                            "window_length": length, "seed": seed, "metric": metric,
                            "value": round(float(fn(y_val, y_pred)), 4)})

    for name, feat in FEATURES.items():
        for length in LENGTHS:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
            clf.fit(feat(crop(dev_x, length)), y_dev)
            record(name, length, clf.predict(feat(crop(val_x, length))))
            print(name, length, results[-2]["value"])

    out = ROOT / "results"
    (out / "models").mkdir(parents=True, exist_ok=True)
    for seed in range(args.seeds):
        model = train.fit_classifier(models.Classifier(2), dev_x, y_dev, epochs=args.epochs,
                                     min_crop=LENGTHS[0], seed=seed)
        for length in LENGTHS:
            proba = train.predict_proba(model, crop(val_x, length))
            record("cnn", length, proba.argmax(1), seed)
            print("cnn seed", seed, length, results[-2]["value"])
        if seed == 0:
            torch.save(model.state_dict(), out / "models/room_cnn.pt")
            proba = train.predict_proba(model, val_x)
            (out / "predictions").mkdir(exist_ok=True)
            with open(out / "predictions/task1_room_cnn.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["window_id", "room", "label", "confidence"])
                for r, p in zip(val_rows, proba):
                    w.writerow([r["window_id"], "AB"[p.argmax()], "", round(float(p.max()), 4)])

    with open(out / "task1_room.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for name in [*FEATURES, "cnn"]:
        acc = np.array([[r["value"] for r in results if r["experiment"] == f"task1_room_{name}"
                         and r["window_length"] == length and r["metric"] == "accuracy"] for length in LENGTHS])
        ax.errorbar(LENGTHS, acc.mean(1), yerr=acc.std(1) if acc.shape[1] > 1 else None,
                    marker="o", capsize=3, label=name)
    ax.set(xlabel="window length (observations, center crop)", ylabel="validation room accuracy",
           ylim=(0.4, 1.02), xticks=LENGTHS)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures/room_acc_vs_window_length.png", dpi=150)

    for name in [*FEATURES, "cnn"]:
        acc = [r["value"] for r in results if r["experiment"] == f"task1_room_{name}"
               and r["window_length"] == 150 and r["metric"] == "accuracy"]
        print(f"{name:15s} L=150 accuracy {np.mean(acc):.4f}")


if __name__ == "__main__":
    main()
