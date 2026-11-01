"""Build a demo episode from VALIDATION captures of both rooms, for rehearsing run_episode.py.

Usage: python scripts/make_demo_episode.py [--per-class 3] [--out episodes/demo]
       python run_episode.py episodes/demo --out episodes/demo/predictions.csv
       python scripts/make_demo_episode.py --score episodes/demo

Query files get neutral names (q001.pcap, ...) so nothing about room or label is visible;
the answer key is written to answers.csv next to (not inside) query/.
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data  # noqa: E402


def build(per_class, out):
    rows = [r for r in data.load_index() if r["split"] == "validation"]
    rng = np.random.default_rng(0)
    picked = []
    for key in sorted({(r["room"], r["label"]) for r in rows}):
        cands = [r for r in rows if (r["room"], r["label"]) == key]
        picked += [cands[i] for i in rng.choice(len(cands), per_class, replace=False)]
    picked = [picked[i] for i in rng.permutation(len(picked))]
    if out.exists():
        shutil.rmtree(out)
    (out / "query").mkdir(parents=True)
    with open(out / "answers.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["window_id", "room", "label"])
        for i, r in enumerate(picked, 1):
            src = ROOT / r["source"]
            shutil.copy(src, out / "query" / f"q{i:03d}{src.suffix}")
            w.writerow([f"q{i:03d}", r["room"], r["label"]])
    print(f"{len(picked)} query captures in {out / 'query'}")


def score(out):
    ans = {r["window_id"]: r for r in csv.DictReader(open(out / "answers.csv"))}
    pred = {r["window_id"]: r for r in csv.DictReader(open(out / "predictions.csv"))}
    ids = sorted(ans)
    print("room accuracy:", np.mean([pred[i]["room"] == ans[i]["room"] for i in ids]).round(3))
    for room in "AB":
        sel = [i for i in ids if ans[i]["room"] == room]
        y, p = [ans[i]["label"] for i in sel], [pred[i]["label"] for i in sel]
        print(f"room {room}: activity accuracy {np.mean(np.array(y) == np.array(p)):.3f}, "
              f"macro-F1 {f1_score(y, p, labels=sorted(set(y)), average='macro', zero_division=0):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--out", default="episodes/demo")
    ap.add_argument("--score", metavar="EPISODE_DIR")
    a = ap.parse_args()
    score(Path(a.score)) if a.score else build(a.per_class, ROOT / a.out)
