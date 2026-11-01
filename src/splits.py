"""Capture enumeration, activity labels, and the fixed development/validation split.

Room A: the supplied validation/ folder is validation; everything else is development.
Room B: per class, sort filenames lexicographically and draw 25 with
numpy.random.default_rng(42) (one generator, classes in sorted order). The result is
frozen to splits/room_B_validation.txt on first creation and only read afterwards.
"""
import re
from pathlib import Path

import numpy as np

ROOM_B_LABELS = {"Empty": "empty", "Lying": "lying", "Sitting": "sit",
                 "Standing": "stand", "Walking": "walk"}


def room_A_captures(root):
    root = Path(root)
    caps = []
    for p in sorted(root.rglob("*.pcap")):
        rel = p.relative_to(root)
        split = "validation" if rel.parts[0] == "validation" else "development"
        label = rel.parts[-2]
        caps.append({"capture_id": "A_" + "_".join(rel.with_suffix("").parts),
                     "room": "A", "label": label, "split": split, "source": str(p)})
    return caps


def room_B_label(name):
    return ROOM_B_LABELS[re.match(r"[A-Za-z]+", name).group()]


def room_B_validation(root, per_class, seed):
    names = sorted(p.name for p in Path(root).glob("*.csv"))
    rng = np.random.default_rng(seed)
    chosen = []
    for label in sorted(set(map(room_B_label, names))):
        cands = sorted(n for n in names if room_B_label(n) == label)
        chosen += [cands[i] for i in sorted(rng.choice(len(cands), per_class, replace=False))]
    return chosen


def room_B_captures(root, frozen_file, per_class, seed):
    frozen_file = Path(frozen_file)
    if frozen_file.exists():
        val = set(frozen_file.read_text().split())
    else:
        val = room_B_validation(root, per_class, seed)
        frozen_file.parent.mkdir(parents=True, exist_ok=True)
        frozen_file.write_text("\n".join(val) + "\n")
        val = set(val)
    caps = []
    for p in sorted(Path(root).glob("*.csv")):
        caps.append({"capture_id": "B_" + p.stem, "room": "B", "label": room_B_label(p.name),
                     "split": "validation" if p.name in val else "development", "source": str(p)})
    return caps
