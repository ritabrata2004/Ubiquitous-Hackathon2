"""Task 3 follow-up: targeted fixes for the diagnosed failure modes, on the primary
configuration (SSL embeddings, logreg head, per-room standardization, oracle routing, which
equals predicted-room routing here). Same 10 support episodes per K as task3_fewshot.py.

Round 1 (decided before running, none tuned on validation):
  base            previous primary configuration (ssl / logreg)
  crops           + 4 extra crops (120 steps at offsets 0/10/20/30) per support capture;
                  still one capture = one shot
  inlp_room       + project out room-predictive directions (INLP) fitted on ALL development
                  windows with room as the only label, until 5-fold room accuracy <= 0.6
  inlp_head       + project out directions separating the head's source-room training windows
                  from the target support windows (10 directions, per episode)
  inlp_room_crops inlp_room + crops
  selftrain       base + self-training on target unlabeled dev windows (p >= 0.9, 3 rounds)
  inlp_selftrain  inlp_room + self-training
Round 2 (prior correction and event features; group_prior was designed after seeing the
em_prior / balanced_prior results, so its selection is validation-based):
  em_prior        base + EM re-estimate of target class priors (Saerens et al. 2002)
  balanced_prior  base + per-class weights making predicted target-dev proportions uniform
  group_prior     base + shared-vs-exclusive group weights only (fewshot.prior_weights "group")
  event           base + 11 event features (src/features.event_features, pooled standardization)
  event_balanced  event + balanced_prior
  event_group     event + group_prior = the current primary (ssl_event / logreg_gp)

Extra diagnostics: shortcut_rate = share of true shared-class windows predicted as a target-
exclusive class; sitstand_rate = share of true SIT/STAND windows predicted as the other one.

Usage: python scripts/task3_improvements.py [--episodes 10] [--bootstrap]
Writes results/task3_improvements.csv and results/figures/task3_improvements.png; with
--bootstrap also results/task3_bootstrap.csv (paired bootstrap over validation windows of
group_prior and event_group vs base, 500 resamples, episodes averaged).
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
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from src import data, features, fewshot, models, ssl  # noqa: E402

VARIANTS = ["base", "crops", "inlp_room", "inlp_head", "inlp_room_crops", "selftrain", "inlp_selftrain",
            "em_prior", "balanced_prior", "group_prior", "event", "event_balanced", "event_group"]
FIG_VARIANTS = ["base", "inlp_head", "selftrain", "em_prior", "balanced_prior", "group_prior", "event_group"]
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
    ap.add_argument("--bootstrap", action="store_true")
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

    fe = np.hstack([f, features.event_features(x)])
    ze = fewshot.standardize(fe, rooms, fewshot.domain_stats(fe, rooms, dev, np.arange(f.shape[1], fe.shape[1])))

    def head(t, sup, feats=z, crop_feats=zc, crops=False, inlp_head=False, selftrain=False, prior=None):
        src = np.flatnonzero(dev & (rooms == OTHER[t]) & np.isin(labels, data.SHARED))
        f_tr = np.concatenate([feats[src], feats[sup]])
        y_tr = np.concatenate([labels[src], labels[sup]])
        if crops:
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
        if prior is None:
            return test, fewshot.fit_predict("logreg", f_tr, y_tr, f_te)[0]
        clf = LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced").fit(f_tr, y_tr)
        p = clf.predict_proba(f_te) * fewshot.prior_weights(clf.predict_proba(f_un), clf.classes_, prior)
        return test, clf.classes_[p.argmax(1)]

    zp, zcp = z @ p_room, zc @ p_room
    config = {"base": {}, "crops": {"crops": True},
              "inlp_room": {"feats": zp, "crop_feats": zcp}, "inlp_head": {"inlp_head": True},
              "inlp_room_crops": {"feats": zp, "crop_feats": zcp, "crops": True},
              "selftrain": {"selftrain": True}, "inlp_selftrain": {"feats": zp, "selftrain": True},
              "em_prior": {"prior": "em"}, "balanced_prior": {"prior": "balanced"},
              "group_prior": {"prior": "group"}, "event": {"feats": ze},
              "event_balanced": {"feats": ze, "prior": "balanced"}, "event_group": {"feats": ze, "prior": "group"}}

    groups, preds = defaultdict(list), defaultdict(list)
    for k in KS:
        for seed in range(args.episodes):
            rng = np.random.default_rng(seed)  # identical support draws to task3_fewshot.py
            sup = {t: fewshot.support_indices(labels, rooms, dev, t, k, rng) for t in "AB"}
            for v in VARIANTS:
                for t in "AB":
                    test, p = head(t, sup[t], **config[v])
                    preds[(v, t, k)].append(p)
                    for name, val_ in {**metrics(labels[test], p, t), **diagnostics(labels[test], p, t)}.items():
                        groups[(v, t, k, name)].append(val_)
        print("K", k, "done")

    if args.bootstrap:
        boot_rng, rows_out = np.random.default_rng(123), []
        for t in "AB":
            y = labels[val & (rooms == t)]
            cls = fewshot.target_classes(t)
            idx_all = boot_rng.integers(0, len(y), (500, len(y)))
            for k in KS:
                def mf(v, idx):
                    return np.mean([f1_score(y[idx], p[idx], labels=cls, average="macro", zero_division=0)
                                    for p in preds[(v, t, k)]])
                for v in ("group_prior", "event_group"):
                    d = np.array([mf(v, b) - mf("base", b) for b in idx_all])
                    full = np.arange(len(y))
                    rows_out.append({"variant": v, "target_room": t, "K": k,
                                     "diff_vs_base": round(mf(v, full) - mf("base", full), 4),
                                     "ci_low": round(float(np.percentile(d, 2.5)), 4),
                                     "ci_high": round(float(np.percentile(d, 97.5)), 4),
                                     "p_diff_le_0": round(float(np.mean(d <= 0)), 4)})
                    print(rows_out[-1])
        write_csv(out / "task3_bootstrap.csv", rows_out)

    summary = [{"experiment": f"task3_improve_{v}", "target_room": t, "K": k, "metric": m,
                "value": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4),
                "n_episodes": len(vals)} for (v, t, k, m), vals in groups.items()]
    write_csv(out / "task3_improvements.csv", summary)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for col, t in enumerate("AB"):
        for row, metric in enumerate(["macro_f1", "shortcut_rate"]):
            ax = axes[row, col]
            for v in FIG_VARIANTS:
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
