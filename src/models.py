"""CSI window encoder shared by the room classifier, SSL pretraining and few-shot heads.

Input (batch, T, 256): subcarriers are channels, convolution runs over time, and global
average pooling makes the embedding independent of T (needed for the window-length study).
"""
import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, in_ch=256, width=64, emb=128):
        super().__init__()

        def block(i, o, stride):
            return nn.Sequential(nn.Conv1d(i, o, 5, stride, padding=2), nn.BatchNorm1d(o), nn.GELU())

        self.net = nn.Sequential(block(in_ch, width, 1), block(width, width, 2),
                                 block(width, 2 * width, 2), block(2 * width, emb, 1))
        self.emb = emb

    def forward(self, x):
        return self.net(x.transpose(1, 2)).mean(dim=2)


class Classifier(nn.Module):
    def __init__(self, n_classes, **enc_kw):
        super().__init__()
        self.encoder = Encoder(**enc_kw)
        self.head = nn.Linear(self.encoder.emb, n_classes)

    def forward(self, x):
        return self.head(self.encoder(x))
