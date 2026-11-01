"""Task 2: learn without labels, then linear-probe the frozen representation.

Usage: python scripts/task2_ssl.py [--epochs 60] [--seed 0]

1. SimCLR pretraining of src/models.Encoder on development windows of BOTH rooms with no
   activity or room labels -> results/models/ssl_encoder.pt
2. Frozen linear probes (standardize + logistic regression) fitted on labeled development
   windows, scored on validation windows, for three representations:
     ssl          pretrained encoder
     random       same architecture, random weights
     handcrafted  src/features.handcrafted
   Probe settings: joint (8 activity classes over both rooms), room_A (6), room_B (5).
3. UMAP of validation embeddings colored by room and by activity.

Writes results/task2_ssl.csv and results/figures/task2_embeddings_{ssl,random}.png.
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
import umap  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data, features, models, ssl, train  # noqa: E402

PROBES = {"joint": ("A", "B"), "room_A": ("A",), "room_B": ("B",)}


def probe(f_dev, y_dev, f_val, y_val):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    clf.fit(f_dev, y_dev)
    p = clf.predict(f_val)
    return {"macro_f1": f1_score(y_val, p, average="macro"), "balanced_accuracy": balanced_accuracy_score(y_val, p)}


def plot_embedding(emb, rooms, labels, title, path):
    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(emb)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, values, name in [(axes[0], rooms, "room"), (axes[1], labels, "activity")]:
        for v in sorted(set(values)):
            m = values == v
            ax.scatter(xy[m, 0], xy[m, 1], s=8, label=v, alpha=0.8)
        ax.set(title=f"{title}: colored by {name}", xticks=[], yticks=[])
        ax.legend(fontsize=7, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(12)
    out = ROOT / "results"
    (out / "models").mkdir(parents=True, exist_ok=True)

    dev_rows, dev_x, val_rows, val_x, _ = data.load_normalized()
    room_dev, room_val = data.column(dev_rows, "room"), data.column(val_rows, "room")
    lab_dev, lab_val = data.column(dev_rows, "label"), data.column(val_rows, "label")

    encoder = ssl.pretrain(models.Encoder(), dev_x, epochs=args.epochs, seed=args.seed)
    torch.save(encoder.state_dict(), out / "models/ssl_encoder.pt")
    train.set_seed(args.seed)
    random_encoder = models.Encoder()

    reps = {"ssl": (ssl.embed(encoder, dev_x), ssl.embed(encoder, val_x)),
            "random": (ssl.embed(random_encoder, dev_x), ssl.embed(random_encoder, val_x)),
            "handcrafted": (features.handcrafted(dev_x), features.handcrafted(val_x))}

    results = []
    for rep, (f_dev, f_val) in reps.items():
        for setting, rooms in PROBES.items():
            md, mv = np.isin(room_dev, rooms), np.isin(room_val, rooms)
            for metric, value in probe(f_dev[md], lab_dev[md], f_val[mv], lab_val[mv]).items():
                results.append({"experiment": f"task2_probe_{rep}", "target_room": setting, "K": "",
                                "metric": metric, "value": round(float(value), 4)})
                print(f"{rep:12s} {setting:7s} {metric:18s} {value:.4f}")
    with open(out / "task2_ssl.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    for rep in ("ssl", "random"):
        plot_embedding(reps[rep][1], room_val, lab_val, f"{rep} encoder (validation)",
                       out / f"figures/task2_embeddings_{rep}.png")


if __name__ == "__main__":
    main()
