"""DS-CNN multi-tugas untuk ESP32: kepala KATA (6 kelas) + kepala PENUTUR (cosine, untuk enrol few-shot).

Input (B, 1, 98, 40) log-Mel CMN.
  conv1 10x4 stride 2 -> BN -> ReLU                       (49 x 20 x C)
  4 blok depthwise-separable (dw3x3 -> BN -> ReLU -> pw1x1 -> BN -> ReLU), blok ke-2 stride 2 (25 x 10 x C)
  Global Average Pool -> fitur f (C)            <- embedding KATA (prototipe enrol) + dasar CAM
  kepala kata   : Linear(C, 6)
  kepala penutur: Linear(C, E) -> e (embedding PENUTUR) -> cosine ke bobot kelas x skala s
Semua lapisan punya padanan INT8 sederhana (conv, depthwise, pointwise, GAP, linear) -> diport ke C.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSBlock(nn.Module):
    def __init__(self, c, stride):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, stride, 1, groups=c, bias=False); self.bn1 = nn.BatchNorm2d(c)
        self.pw = nn.Conv2d(c, c, 1, bias=False); self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        return F.relu(self.bn2(self.pw(F.relu(self.bn1(self.dw(x))))))


class KWSNet(nn.Module):
    def __init__(self, width=48, n_kw=6, n_spk=4, emb=32, strides=(1, 2, 1, 1), spk_scale=16.0):
        super().__init__()
        self.width, self.emb_dim, self.strides = width, emb, tuple(strides)
        self.conv1 = nn.Conv2d(1, width, (10, 4), (2, 2), (4, 1), bias=False); self.bn1 = nn.BatchNorm2d(width)
        self.blocks = nn.ModuleList([DSBlock(width, s) for s in strides])
        self.drop = nn.Dropout(0.1)
        self.kw = nn.Linear(width, n_kw)
        self.spk_emb = nn.Linear(width, emb)
        self.spk_w = nn.Parameter(torch.randn(n_spk, emb)*0.1)
        self.spk_scale = spk_scale

    def features(self, x):
        """Return (peta fitur terakhir A [B,C,25,10], fitur GAP f [B,C])."""
        x = F.relu(self.bn1(self.conv1(x)))
        for b in self.blocks:
            x = b(x)
        return x, x.mean(dim=(2, 3))

    def forward(self, x):
        A, f = self.features(x)
        fd = self.drop(f)
        kw = self.kw(fd)
        e = self.spk_emb(fd)
        spk = self.spk_scale*F.normalize(e, dim=1)@F.normalize(self.spk_w, dim=1).T
        return kw, spk, f, e, A

    def macs(self, h=98, w=40):
        c = self.width; h1, w1 = (h+8-10)//2+1, (w+2-4)//2+1
        total = 40*c*h1*w1; hh, ww = h1, w1
        for s in self.strides:
            hh, ww = (hh-1)//s+1, (ww-1)//s+1
            total += 9*c*hh*ww+c*c*hh*ww
        return int(total+c*6+c*self.emb_dim)


def param_count(m):
    return sum(p.numel() for p in m.parameters())
