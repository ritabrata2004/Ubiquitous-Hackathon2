"""Supervised training loop for window classifiers (CPU)."""
import numpy as np
import torch
from torch import nn


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def random_crop(xb, min_len, rng):
    """Same random length and per-sample random offset for one batch."""
    t = xb.shape[1]
    length = int(rng.integers(min_len, t + 1))
    starts = rng.integers(0, t - length + 1, len(xb))
    return np.stack([x[s:s + length] for x, s in zip(xb, starts)])


def fit_classifier(model, x, y, epochs=30, batch=64, lr=2e-3, min_crop=None, seed=0):
    """x: (n, T, 256) float32, y: int labels. min_crop enables random-length crop augmentation."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * ((len(x) + batch - 1) // batch))
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(len(x)), (len(x) + batch - 1) // batch):
            xb = x[idx] if min_crop is None else random_crop(x[idx], min_crop, rng)
            loss = loss_fn(model(torch.from_numpy(xb)), torch.from_numpy(y[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model


@torch.no_grad()
def predict_proba(model, x, batch=256):
    model.eval()
    return torch.cat([model(torch.from_numpy(x[i:i + batch])).softmax(1)
                      for i in range(0, len(x), batch)]).numpy()
