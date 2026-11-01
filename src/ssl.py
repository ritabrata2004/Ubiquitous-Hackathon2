"""Self-supervised contrastive pretraining (SimCLR / NT-Xent, Chen et al. 2020) for CSI windows.

Two augmented views of the same window are pulled together and pushed away from the other
windows in the batch. No activity or room labels are used. Augmentations target nuisances
that should not change the activity:
  random temporal crop        position/duration of the activity within the capture
  per-subcarrier gain jitter  frequency-selective gain / AGC differences
  subcarrier block masking    missing or faded sub-bands
  time masking                dropped observations
  additive Gaussian noise     measurement noise
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .train import set_seed


def augment(xb, rng, min_len=75):
    n, t, c = xb.shape
    length = int(rng.integers(min_len, t + 1))
    starts = rng.integers(0, t - length + 1, n)
    x = np.stack([w[s:s + length] for w, s in zip(xb, starts)]).copy()
    x *= np.exp(rng.normal(0, 0.2, (n, 1, c))).astype(np.float32)
    for i in range(n):
        w = int(rng.integers(8, 48))
        s = int(rng.integers(0, c - w))
        x[i, :, s:s + w] = 0.0
        w = int(rng.integers(0, length // 5 + 1))
        s = int(rng.integers(0, length - w + 1))
        x[i, s:s + w] = 0.0
    x += rng.normal(0, 0.1, x.shape).astype(np.float32)
    return torch.from_numpy(x)


def nt_xent(z1, z2, tau=0.2):
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.T / tau
    sim.fill_diagonal_(float("-inf"))
    n = len(z1)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    return F.cross_entropy(sim, target)


def pretrain(encoder, x, epochs=60, batch=128, lr=1e-3, seed=0, log=print):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    proj = nn.Sequential(nn.Linear(encoder.emb, encoder.emb), nn.ReLU(), nn.Linear(encoder.emb, 64))
    params = list(encoder.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    steps = len(x) // batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * steps)
    encoder.train()
    for epoch in range(epochs):
        perm, total = rng.permutation(len(x)), 0.0
        for k in range(steps):
            xb = x[perm[k * batch:(k + 1) * batch]]
            loss = nt_xent(proj(encoder(augment(xb, rng))), proj(encoder(augment(xb, rng))))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        if epoch % 5 == 0 or epoch == epochs - 1:
            log(f"ssl epoch {epoch:3d} loss {total / steps:.4f}")
    return encoder


@torch.no_grad()
def embed(encoder, x, batch=256):
    encoder.eval()
    return torch.cat([encoder(torch.from_numpy(x[i:i + batch])) for i in range(0, len(x), batch)]).numpy()
