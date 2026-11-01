"""Cross-room few-label adaptation on frozen window representations.

Domain alignment: each room's features are standardized with the mean/std of that room's
UNLABELED development windows (no labels, no validation data). This removes the room-
specific offset of the representation so SIT/STAND/WALK learned in the source room land
in the same place in the target room.

Classifiers, trained on: source-room labeled windows of the shared classes + the K target
support captures of each target-exclusive class (one window per capture).
  proto      nearest class prototype (cosine), Snell et al. 2017
  proto_tkm  proto refined by spherical k-means on the target room's UNLABELED development
             windows (transductive adaptation; no target labels beyond the support)
  logreg     class-balanced logistic regression

Extensions evaluated in scripts/task3_improvements.py: removal of room-predictive directions
(nullspace_projection) and self-training on unlabeled target windows (self_train_logreg).
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


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def refine_prototypes(protos, classes, f_unlab, f_anchor, y_anchor, iters=10):
    """Transductive refinement: spherical k-means over the target room's UNLABELED development
    windows, initialized at the prototypes. Labeled target support windows stay assigned to
    their own class, so exclusive-class prototypes remain anchored."""
    u = unit(f_unlab)
    a = unit(f_anchor)
    for _ in range(iters):
        assign = (u @ protos.T).argmax(1)
        new = []
        for i, c in enumerate(classes):
            s = u[assign == i].sum(0) + a[y_anchor == c].sum(0)
            new.append(s if np.any(s) else protos[i])
        protos = unit(np.stack(new))
    return protos


def nullspace_projection(feats, groups, max_iter=30, stop_acc=0.6, cv=True):
    """Iterative nullspace projection (INLP, Ravfogel et al. 2020).

    Repeatedly fit a linear classifier for `groups` (e.g. room) and project its direction out
    of the feature space, until the group is no longer linearly predictable (5-fold accuracy
    <= stop_acc) or max_iter directions are removed. Returns the (d, d) projection matrix P;
    apply as feats @ P. With cv=False the stop criterion is training accuracy (for tiny groups).
    """
    from sklearn.model_selection import cross_val_score
    P = np.eye(feats.shape[1])
    for _ in range(max_iter):
        x = feats @ P
        clf = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced")
        acc = cross_val_score(clf, x, groups, cv=5).mean() if cv else clf.fit(x, groups).score(x, groups)
        if acc <= stop_acc:
            break
        w = P @ clf.fit(x, groups).coef_[0]
        u = w / np.linalg.norm(w)
        P = P - np.outer(u, u)
    return P


def self_train_logreg(f_train, y_train, f_test, f_unlab, threshold=0.9, rounds=3):
    """Self-training: label the target room's UNLABELED development windows with the current
    head, add those predicted with probability >= threshold, refit; `rounds` times."""
    x, y = f_train, y_train
    for _ in range(rounds):
        clf = LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced").fit(x, y)
        p = clf.predict_proba(f_unlab)
        keep = p.max(1) >= threshold
        x = np.concatenate([f_train, f_unlab[keep]])
        y = np.concatenate([y_train, clf.classes_[p[keep].argmax(1)]])
    clf = LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced").fit(x, y)
    p = clf.predict_proba(f_test)
    return clf.classes_[p.argmax(1)], p.max(1)


def fit_predict(method, f_train, y_train, f_test, f_unlab=None, support=None):
    """Return (predicted labels, confidence).
    proto_tkm additionally needs f_unlab (target unlabeled dev features) and
    support = (features, labels) of the target support windows."""
    classes = np.array(sorted(set(y_train)))
    if method in ("proto", "proto_tkm"):
        protos = unit(np.stack([f_train[y_train == c].mean(0) for c in classes]))
        if method == "proto_tkm":
            protos = refine_prototypes(protos, classes, f_unlab, *support)
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
