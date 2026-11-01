"""Cross-room few-label adaptation on frozen window representations.

Domain alignment: each room's features are standardized with the mean/std of that room's
UNLABELED development windows (no labels, no validation data). This removes the room-
specific offset of the representation so SIT/STAND/WALK learned in the source room land
in the same place in the target room.

Classifiers, trained on: source-room labeled windows of the shared classes + the K target
support captures of each target-exclusive class (one window per capture).
  proto   nearest class prototype (cosine), Snell et al. 2017
  logreg  class-balanced logistic regression
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from .data import EXCLUSIVE, SHARED


def domain_stats(feats, rooms, dev_mask):
    return {r: (feats[dev_mask & (rooms == r)].mean(0), feats[dev_mask & (rooms == r)].std(0) + 1e-6)
            for r in np.unique(rooms)}


def standardize(feats, rooms, stats):
    out = np.empty_like(feats)
    for r, (mu, sd) in stats.items():
        m = rooms == r
        out[m] = (feats[m] - mu) / sd
    return out


def support_indices(labels, rooms, dev_mask, target, k, rng):
    """K distinct target development captures per target-exclusive class (1 window = 1 capture)."""
    idx = []
    for c in EXCLUSIVE[target]:
        cands = np.flatnonzero(dev_mask & (rooms == target) & (labels == c))
        idx += list(rng.choice(cands, k, replace=False))
    return np.array(idx)


def fit_predict(method, f_train, y_train, f_test):
    """Return (predicted labels, confidence)."""
    classes = np.array(sorted(set(y_train)))
    if method == "proto":
        def unit(v):
            return v / np.linalg.norm(v, axis=-1, keepdims=True)
        protos = unit(np.stack([f_train[y_train == c].mean(0) for c in classes]))
        sim = unit(f_test) @ protos.T
        p = np.exp(sim / 0.1)
        p /= p.sum(1, keepdims=True)
    else:
        clf = LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced").fit(f_train, y_train)
        p = clf.predict_proba(f_test)
        classes = clf.classes_
    return classes[p.argmax(1)], p.max(1)


def target_classes(target):
    return list(SHARED) + list(EXCLUSIVE[target])
