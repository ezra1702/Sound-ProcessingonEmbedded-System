"""Training DS-CNN multi-tugas (Siklus 2). Seleksi lebar model HANYA dengan validation (block 5); test tidak disentuh.

Policy (sama dengan MP10/MP11/MP15): ambil lebar terkecil yang macro-F1 validation-nya dalam toleransi 0,01 dari terbaik.
Output: artifacts/kwsnet.pt, artifacts/train_report.json, artifacts/speaker_prototypes.npz
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

import common as C
from model import KWSNet, param_count

def _pick_device():
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device='cuda').add_(1); torch.cuda.synchronize(); return 'cuda'
        except Exception:
            pass                                      # GPU terlalu baru untuk build PyTorch ini -> CPU
    return 'cpu'


DEV = _pick_device()
torch.set_num_threads(max(1, torch.get_num_threads()))
KEYWORD_IDX = list(range(4))
TOL = 0.01


def load(split):
    d = np.load(C.OUT/f'dataset_{split}.npz', allow_pickle=True)
    return {k: d[k] for k in d.files}


def to_dev(d, mask=None):
    m = np.ones(len(d['kw']), bool) if mask is None else mask
    return (torch.tensor(d['X'][m], dtype=torch.float32, device=DEV).unsqueeze(1),
            torch.tensor(d['kw'][m], dtype=torch.long, device=DEV), torch.tensor(d['spk'][m], dtype=torch.long, device=DEV))


def spec_augment(x, g):
    B, _, T, Fq = x.shape
    x = x.clone()
    for _ in range(2):
        t = torch.randint(0, 16, (B,), generator=g, device='cpu'); t0 = (torch.rand(B, generator=g)*(T-t)).long()
        f = torch.randint(0, 7, (B,), generator=g, device='cpu'); f0 = (torch.rand(B, generator=g)*(Fq-f)).long()
        ar_t = torch.arange(T)[None]; ar_f = torch.arange(Fq)[None]
        mt = ((ar_t >= t0[:, None]) & (ar_t < (t0+t)[:, None])).to(x.device)
        mf = ((ar_f >= f0[:, None]) & (ar_f < (f0+f)[:, None])).to(x.device)
        keep = torch.rand(B, generator=g).to(x.device) < 0.8
        x = x.masked_fill((mt[:, None, :, None] & keep[:, None, None, None]), 0.0)
        x = x.masked_fill((mf[:, None, None, :] & keep[:, None, None, None]), 0.0)
    return x


@torch.no_grad()
def predict(model, X, bs=1024):
    model.eval(); outs = []
    for i in range(0, len(X), bs):
        kw, spk, f, e, _ = model(X[i:i+bs]); outs.append((kw.softmax(1), spk.softmax(1), f, F.normalize(e, dim=1)))
    return [torch.cat([o[j] for o in outs]).cpu().numpy() for j in range(4)]


def kw_metrics(y, p, clean=None, spk_true=None, spk_prob=None):
    pred = p.argmax(1)
    is_kw_t, is_kw_p = y < 4, pred < 4
    m = {'macro_f1': float(f1_score(y, pred, average='macro', labels=list(range(6)))),
         'accuracy': float(np.mean(pred == y)),
         'keyword_recall': float(np.mean(pred[is_kw_t] == y[is_kw_t])),
         'false_accept_rate': float(np.mean(is_kw_p[~is_kw_t])),            # diam/lain dianggap perintah
         'unknown_reject_rate': float(np.mean(~is_kw_p[(y == 5)])),
         'silence_reject_rate': float(np.mean(~is_kw_p[(y == 4)]))}
    if clean is not None and clean.any():
        m['clean_keyword_accuracy'] = float(np.mean(pred[clean] == y[clean])); m['n_clean'] = int(clean.sum())
    if spk_true is not None:
        s = spk_true >= 0
        m['speaker_accuracy_all'] = float(np.mean(spk_prob[s].argmax(1) == spk_true[s]))
        if clean is not None:
            cs = clean & s
            m['speaker_accuracy_clean'] = float(np.mean(spk_prob[cs].argmax(1) == spk_true[cs]))
    return m


def train_model(tr, va, width=48, epochs=40, seed=C.SEED, lam_spk=0.5, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed); g = torch.Generator().manual_seed(seed)
    Xtr, ytr, str_ = tr; Xva, yva, sva = va
    model = KWSNet(width=width).to(DEV)
    cnt = torch.bincount(ytr, minlength=6).float(); w = (cnt.sum()/(6*cnt.clamp(min=1)))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    steps = epochs*int(np.ceil(len(ytr)/256))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.15)
    best, best_state, hist = -1, None, []
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(ytr), generator=g).to(DEV); tl = 0.0
        for i in range(0, len(ytr), 256):
            idx = perm[i:i+256]; xb = spec_augment(Xtr[idx], g)
            kw, spk, *_ = model(xb)
            loss = F.cross_entropy(kw, ytr[idx], weight=w, label_smoothing=0.05)
            sm = str_[idx] >= 0
            if sm.any():
                loss = loss+lam_spk*F.cross_entropy(spk[sm], str_[idx][sm], label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tl += float(loss)*len(idx)
        pk, ps, *_ = predict(model, Xva)
        f1 = float(f1_score(yva.cpu().numpy(), pk.argmax(1), average='macro', labels=list(range(6))))
        sm = sva.cpu().numpy() >= 0
        sacc = float(np.mean(ps[sm].argmax(1) == sva.cpu().numpy()[sm])) if sm.any() else float('nan')
        hist.append({'epoch': ep+1, 'train_loss': tl/len(ytr), 'val_macro_f1': f1, 'val_speaker_acc': sacc})
        if f1 > best:
            best, best_state = f1, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 5 == 4 or ep == epochs-1):
            print(f'  w{width} ep{ep+1:3d} loss {tl/len(ytr):.3f} val F1 {f1:.3f} spk {sacc:.3f}', flush=True)
    model.load_state_dict(best_state)
    return model, hist, best


def main():
    t0 = time.time()
    tr_d, va_d = load('train'), load('val')
    tr, va = to_dev(tr_d), to_dev(va_d)
    print(f'device {DEV} | train {len(tr[1])} | val {len(va[1])}', flush=True)
    results, models = {}, {}
    for width in (32, 48, 64):
        m, hist, best = train_model(tr, va, width=width)
        pk, ps, *_ = predict(m, va[0])
        met = kw_metrics(va_d['kw'], pk, va_d['clean'], va_d['spk'], ps)
        results[width] = {'params': param_count(m), 'macs': m.macs(), 'best_val_macro_f1': best, 'val': met,
                          'best_epoch': int(np.argmax([h['val_macro_f1'] for h in hist])+1), 'history': hist}
        models[width] = m
        print(f'width {width}: params {param_count(m)}, MAC {m.macs():,}, val {met}', flush=True)
    best_f1 = max(r['best_val_macro_f1'] for r in results.values())
    chosen = min(w for w, r in results.items() if r['best_val_macro_f1'] >= best_f1-TOL)
    model = models[chosen]
    torch.save({'state_dict': model.state_dict(), 'width': chosen, 'emb': model.emb_dim, 'strides': model.strides,
                'kw_classes': C.KW_CLASSES, 'speakers': C.SPEAKERS}, C.OUT/'kwsnet.pt')
    # prototipe penutur yang dikenal: rata-rata embedding ternormalisasi dari data TRAIN (contoh keyword)
    m_kw = (tr_d['spk'] >= 0) & (tr_d['kw'] < 4)
    _, _, _, e = predict(model, tr[0][torch.tensor(m_kw, device=DEV)])
    protos = np.stack([e[tr_d['spk'][m_kw] == i].mean(0) for i in range(4)])
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    # ambang "penutur dikenal" sementara dari val: kemiripan cosine ke prototipe sendiri (kuantil 5%)
    _, _, _, ev = predict(model, va[0]); sv = va_d['spk']; mv = (sv >= 0) & (va_d['kw'] < 4)
    own = np.sum(ev[mv]*protos[sv[mv]], axis=1)
    np.savez(C.OUT/'speaker_prototypes.npz', protos=protos.astype(np.float32), names=np.array(C.SPEAKERS),
             genuine_cos_val_q05=np.float32(np.quantile(own, 0.05)))
    C.save_json(C.OUT/'train_report.json', {'device': DEV, 'selection_policy': f'lebar terkecil dalam {TOL} dari macro-F1 val terbaik',
                                             'chosen_width': chosen, 'results': results, 'test_used': False,
                                             'speaker_genuine_cos_val': {'min': float(own.min()), 'q05': float(np.quantile(own, .05)),
                                                                         'median': float(np.median(own))},
                                             'seconds': time.time()-t0})
    print(f'DIPILIH width {chosen} | waktu {time.time()-t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
