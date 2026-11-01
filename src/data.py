"""Load windows and their index for modeling. Labels and rooms come from the index
(which is built from folder/filename metadata) and are used only as targets, never as
features."""
import csv
from pathlib import Path

import numpy as np

from . import normalization

ROOT = Path(__file__).resolve().parents[1]
SHARED = ("sit", "stand", "walk")
EXCLUSIVE = {"A": ("fall", "sitdown", "standup"), "B": ("empty", "lying")}


def load_index(path=ROOT / "windows/index.csv"):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def load_split(index, split):
    rows = [r for r in index if r["split"] == split]
    x = np.stack([np.load(ROOT / r["path"]) for r in rows])
    return rows, x


def load_normalized():
    """Return (dev_rows, dev_x, val_rows, val_x, stats) with normalization fitted on development."""
    index = load_index()
    dev_rows, dev_x = load_split(index, "development")
    val_rows, val_x = load_split(index, "validation")
    stats = normalization.fit(dev_x)
    return dev_rows, normalization.apply(dev_x, stats), val_rows, normalization.apply(val_x, stats), stats


def column(rows, key):
    return np.array([r[key] for r in rows])
