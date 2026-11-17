"""Kuantisasi INT8 DS-CNN + referensi numpy yang meniru firmware C bit-per-bit.

Skema (sama gaya TFLite Micro, tanpa zero-point):
  * input log-Mel: q = clip(round(x / s_in), -127, 127)
  * conv/dw/pw (BN dilipat): bobot int8 per-kanal (s_w[o] = max|W[o]|/127), bias int32 = round(b / (s_x*s_w[o]))
  * acc int32 -> requant fixed-point: q = (acc*M0 + 2^(rs-1)) >> rs, M0*2^-rs = s_x*s_w[o]/s_y ; ReLU -> clamp [0,127]
  * GAP: jumlah int32 -> float f = sum*s_A/(H*W); kepala kata & penutur float32 (kecil)
Tata letak tensor HWC (baris = waktu, kolom = mel, kanal terakhir) seperti di C.
"""
from __future__ import annotations

import numpy as np
import torch

import common as C
from model import KWSNet


def fold_bn(conv_w, bn):
    g = (bn.weight/torch.sqrt(bn.running_var+bn.eps)).detach()
    w = conv_w.detach()*g.reshape(-1, 1, 1, 1)
    b = (bn.bias-bn.running_mean*g).detach()
    return w.numpy().astype(np.float64), b.numpy().astype(np.float64)


def quant_mult(m):
    """M (float, >0) -> (M0 int32, rs) dengan M ~= M0 * 2^-rs, M0 di [2^30, 2^31)."""
    mant, ex = np.frexp(m)
    m0 = np.round(mant*(1 << 31)).astype(np.int64)
    over = m0 == (1 << 31); m0[over] //= 2; ex = ex+over
    return m0.astype(np.int64), (31-ex).astype(np.int64)


def requant(acc, m0, rs):
    acc = acc.astype(np.int64)
    return np.clip((acc*m0+(np.int64(1) << (rs-1))) >> rs, 0, 127).astype(np.int8)


def load_float(path=C.OUT/'kwsnet.pt'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = KWSNet(width=ck['width'], emb=ck['emb'], strides=ck['strides'])
    m.load_state_dict(ck['state_dict']); m.eval()
    return m


@torch.no_grad()
def calibrate(model, X, pct=99.99):
    """Skala aktivasi per lapisan (persentil |nilai|) dari data TRAIN saja."""
    acts = {}
    x = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    import torch.nn.functional as F
    h = F.relu(model.bn1(model.conv1(x))); acts['conv1'] = h
    for i, b in enumerate(model.blocks):
        d = F.relu(b.bn1(b.dw(h))); acts[f'dw{i}'] = d
        h = F.relu(b.bn2(b.pw(d))); acts[f'pw{i}'] = h
    s = {k: float(np.percentile(v.numpy().ravel()[::7], pct))/127.0 for k, v in acts.items()}
    s['input'] = float(np.percentile(np.abs(X).ravel()[::7], pct))/127.0
    return s


class Int8KWS:
    def __init__(self, model: KWSNet, act_scale: dict):
        self.width, self.strides = model.width, model.strides
        self.s_in = act_scale['input']
        self.layers = []
        s_x = self.s_in
        specs = [('conv1', model.conv1, model.bn1, 'conv')]
        for i, b in enumerate(model.blocks):
            specs += [(f'dw{i}', b.dw, b.bn1, 'dw'), (f'pw{i}', b.pw, b.bn2, 'pw')]
        for name, conv, bn, kind in specs:
            w, bias = fold_bn(conv.weight, bn)
            s_w = np.maximum(np.abs(w).reshape(w.shape[0], -1).max(1), 1e-12)/127.0
            wq = np.clip(np.round(w/s_w.reshape(-1, 1, 1, 1)), -127, 127).astype(np.int8)
            bq = np.round(bias/(s_x*s_w)).astype(np.int32)
            s_y = act_scale[name]
            m0, rs = quant_mult(s_x*s_w/s_y)
            self.layers.append({'name': name, 'kind': kind, 'w': wq, 'b': bq, 'm0': m0, 'rs': rs, 's_x': s_x, 's_y': s_y,
                                'stride': conv.stride, 'pad': conv.padding, 'k': conv.kernel_size})
            s_x = s_y
        self.s_last = s_x
        self.kw_w = model.kw.weight.detach().numpy().astype(np.float32); self.kw_b = model.kw.bias.detach().numpy().astype(np.float32)
        self.e_w = model.spk_emb.weight.detach().numpy().astype(np.float32); self.e_b = model.spk_emb.bias.detach().numpy().astype(np.float32)

    # ---------------- lapisan (HWC) ----------------
    @staticmethod
    def _pad(x, ph, pw):
        return np.pad(x, ((ph, ph), (pw, pw), (0, 0)))

    def _conv1(self, x, L):                          # x: (98, 40, 1) int8
        kh, kw = L['k']; sh, sw = L['stride']; ph, pw = L['pad']
        xp = self._pad(x, ph, pw).astype(np.int32)
        Ho, Wo = (xp.shape[0]-kh)//sh+1, (xp.shape[1]-kw)//sw+1
        cols = np.stack([xp[i:i+sh*Ho:sh, j:j+sw*Wo:sw, 0] for i in range(kh) for j in range(kw)], axis=-1)  # Ho,Wo,kh*kw
        wt = L['w'][:, 0].reshape(L['w'].shape[0], -1).astype(np.int32)                                  # Cout, kh*kw
        acc = cols.reshape(-1, kh*kw)@wt.T+L['b']
        return requant(acc, L['m0'], L['rs']).reshape(Ho, Wo, -1)

    def _dw(self, x, L):                             # depthwise 3x3
        s = L['stride'][0]
        xp = self._pad(x, 1, 1).astype(np.int32)
        Ho, Wo = (xp.shape[0]-3)//s+1, (xp.shape[1]-3)//s+1
        acc = np.zeros((Ho, Wo, x.shape[2]), np.int64)
        w = L['w'][:, 0].astype(np.int64)                                                                # C,3,3
        for i in range(3):
            for j in range(3):
                acc += xp[i:i+s*Ho:s, j:j+s*Wo:s, :]*w[:, i, j]
        acc += L['b']
        return requant(acc.reshape(-1, x.shape[2]), L['m0'], L['rs']).reshape(Ho, Wo, -1)

    def _pw(self, x, L):
        H, W, Cc = x.shape
        acc = x.reshape(-1, Cc).astype(np.int32)@L['w'][:, :, 0, 0].astype(np.int32).T+L['b']
        return requant(acc, L['m0'], L['rs']).reshape(H, W, -1)

    def quantize_input(self, lm):
        return np.clip(np.round(lm/self.s_in), -127, 127).astype(np.int8)

    def forward_q(self, xq):
        """xq: (98, 40) int8 -> dict(kw_logits, f, e, A_q). Integer-exact dengan C."""
        h = xq.reshape(xq.shape[0], xq.shape[1], 1)
        for L in self.layers:
            h = {'conv': self._conv1, 'dw': self._dw, 'pw': self._pw}[L['kind']](h, L)
        H, W, Cc = h.shape
        ssum = h.reshape(-1, Cc).astype(np.int32).sum(0)
        f = (ssum.astype(np.float32)*np.float32(self.s_last/(H*W))).astype(np.float32)
        kw = (self.kw_w@f+self.kw_b).astype(np.float32)
        e = (self.e_w@f+self.e_b).astype(np.float32)
        return {'kw': kw, 'f': f, 'e': e, 'A': h, 'gap_sum': ssum}

    def forward(self, lm):
        return self.forward_q(self.quantize_input(lm))

    def cam(self, A, k):
        """Class Activation Map kelas k: (25, 10) float >= 0 (dinormalisasi 0..1)."""
        c = (A.astype(np.float32)*np.float32(self.s_last))@self.kw_w[k]
        c = np.maximum(c, 0); return c/(c.max()+1e-9)

    def sizes(self):
        wb = sum(L['w'].size for L in self.layers); bb = sum(4*L['b'].size+8*L['m0'].size for L in self.layers)
        hb = 4*(self.kw_w.size+self.kw_b.size+self.e_w.size+self.e_b.size)
        return {'int8_weight_bytes': int(wb), 'bias_requant_bytes': int(bb), 'head_float_bytes': int(hb), 'total_bytes': int(wb+bb+hb)}


def softmax(z):
    z = np.asarray(z, np.float64); e = np.exp(z-z.max()); return e/e.sum()
