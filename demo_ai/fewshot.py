"""Eksperimen pengguna baru (few-shot enrol) dengan leave-one-speaker-out (LOSO).

Untuk tiap anggota s:
  * latih model TANPA data s (train block 1-4 & val block 5 milik 3 anggota lain + data 'lain'/'diam')
  * zero-shot  : akurasi kata s pada block 4-5 sebelum didaftarkan (= "orang baru yang belum dikenal")
  * enrol      : prototipe dari 3 rekaman per kata milik s (block 1-3, total 12) -> fusi seperti firmware
  * penutur    : kemiripan cosine embedding s ke 3 prototipe anggota lain (harus RENDAH = "tidak dikenal"),
                 dan ke prototipe enrol s sendiri (harus TINGGI)
Block 6 TIDAK dipakai sama sekali di sini (disimpan untuk evaluasi akhir).
Ambang penutur dipilih dari gabungan semua fold (titik EER) -> dipakai firmware.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

import build_dataset as BD
import common as C
from train import DEV, load, predict, to_dev, train_model

TAU = 10.0          # suhu softmax prototipe kata (tetap, tidak dituning)
EPOCHS = 30


def fuse(p_model, f, kw_protos):
    """Sama persis dengan kws_decide() di firmware."""
    out = p_model.copy()
    if p_model.argmax() >= 4:
        return out
    fn = f/(np.linalg.norm(f)+1e-9)
    z = TAU*(kw_protos@fn); pp = np.exp(z-z.max()); pp /= pp.sum()
    pk = p_model[:4].sum()
    out[:4] = 0.5*p_model[:4]+0.5*pk*pp
    return out


def feats(Y):
    return torch.tensor(np.stack([C.logmel(C.center_window(y)[0]) for y in Y]), dtype=torch.float32).unsqueeze(1).to(DEV)


def main():
    t0 = time.time()
    tr_d, va_d = load('train'), load('val')
    g = np.load(C.OUT/'group_clips.npz', allow_pickle=True)
    width = int(torch.load(C.OUT/'kwsnet.pt', weights_only=False)['width'])
    folds, genuine, impostor = {}, [], []
    for si, s in enumerate(C.SPEAKERS):
        tr = to_dev(tr_d, tr_d['spk'] != si); va = to_dev(va_d, va_d['spk'] != si)
        # label penutur harus 0..2 untuk 3 penutur tersisa -> petakan ulang
        remap = {old: new for new, old in enumerate([i for i in range(4) if i != si])}
        for d in (tr, va):
            sp = d[2].cpu().numpy(); d[2].copy_(torch.tensor([remap.get(int(v), -1) for v in sp], device=DEV))
        print(f'[LOSO {s}] train {len(tr[1])} val {len(va[1])}', flush=True)
        model, hist, best = train_model(tr, va, width=width, epochs=EPOCHS, verbose=False)
        others = [i for i in range(4) if i != si]
        # prototipe penutur 3 anggota lain dari data train mereka
        protos = []
        for o in others:
            m = (tr_d['spk'] == o) & (tr_d['kw'] < 4)
            _, _, _, e = predict(model, torch.tensor(tr_d['X'][m][:600], dtype=torch.float32).unsqueeze(1).to(DEV))
            p = e.mean(0); protos.append(p/np.linalg.norm(p))
        protos = np.stack(protos)
        sel = g['spk'] == s
        enr = sel & np.isin(g['block'], [1, 2, 3]); tst = sel & np.isin(g['block'], [4, 5])
        Xe, Xt = feats(g['Y'][enr]), feats(g['Y'][tst])
        ye = np.array([C.KW_CLASSES.index(k) for k in g['kw'][enr]]); yt = np.array([C.KW_CLASSES.index(k) for k in g['kw'][tst]])
        # versi berderau dari klip uji (5 variasi tiap klip, derau dari rekaman anggota lain block 5)
        nb = BD.NoiseBank([{'y': y} for y in g['Y'][(g['spk'] != s) & (g['block'] == 5)]])
        r = np.random.default_rng(C.SEED+si)
        Wn = np.stack([BD.aug_speech(y, r, nb)[0] for y in g['Y'][tst] for _ in range(5)])
        Xn = torch.tensor(C.logmel_batch(Wn), dtype=torch.float32).unsqueeze(1).to(DEV); yn = np.repeat(yt, 5)
        pe, _, fe, ee = predict(model, Xe)
        res = {}
        for tag, X, y in (('clean', Xt, yt), ('noisy', Xn, yn)):
            p, _, f, e = predict(model, X)
            kw_protos = np.stack([(lambda v: v/np.linalg.norm(v))(np.mean(fe[ye == k]/np.linalg.norm(fe[ye == k], axis=1, keepdims=True), 0))
                                  for k in range(4)])
            pf = np.stack([fuse(p[i], f[i], kw_protos) for i in range(len(p))])
            res[f'{tag}_zero_shot_acc'] = float(np.mean(p.argmax(1) == y))
            res[f'{tag}_enrolled_acc'] = float(np.mean(pf.argmax(1) == y))
            res[f'{tag}_n'] = int(len(y))
            sp_own = ee/np.linalg.norm(ee, axis=1, keepdims=True); own = sp_own.mean(0); own /= np.linalg.norm(own)
            imp = (e@protos.T).max(1)                 # sebelum enrol: mirip anggota lain?
            gen = e@own                               # sesudah enrol: mirip dirinya sendiri?
            res[f'{tag}_impostor_cos_mean'] = float(imp.mean()); res[f'{tag}_genuine_cos_mean'] = float(gen.mean())
            if tag == 'clean':
                genuine += gen.tolist(); impostor += imp.tolist()
        # penutur dikenal (3 anggota lain) pada val block 5 terhadap prototipe mereka sendiri
        for j, o in enumerate(others):
            m = (va_d['spk'] == o) & va_d['clean']
            _, _, _, e = predict(model, torch.tensor(va_d['X'][m], dtype=torch.float32).unsqueeze(1).to(DEV))
            genuine += (e@protos[j]).tolist()
        res['best_val_macro_f1'] = best
        folds[s] = res
        print(f'  {s}: {res}', flush=True)
    genuine, impostor = np.array(genuine), np.array(impostor)
    ths = np.linspace(-0.2, 1.0, 241)
    far = np.array([np.mean(impostor >= t) for t in ths]); frr = np.array([np.mean(genuine < t) for t in ths])
    k = int(np.argmin(np.abs(far-frr))); thr = float(ths[k])
    summ = {k2: float(np.mean([f[k2] for f in folds.values()])) for k2 in next(iter(folds.values())) if k2.endswith('_acc')}
    C.save_json(C.OUT/'fewshot_report.json', {
        'protocol': 'LOSO: latih tanpa 1 anggota; enrol 3 rekaman/kata (block 1-3); uji block 4-5; block 6 tidak dipakai',
        'epochs': EPOCHS, 'width': width, 'proto_tau': TAU, 'folds': folds, 'mean': summ,
        'speaker_threshold': thr, 'speaker_eer': float((far[k]+frr[k])/2), 'n_genuine': int(len(genuine)), 'n_impostor': int(len(impostor)),
        'note': 'Ambang dari titik EER gabungan 4 fold. Hanya 4 orang -> estimasi kasar.', 'seconds': time.time()-t0})
    print('RATA-RATA', summ, '| ambang penutur', round(thr, 3), '| EER', round(float((far[k]+frr[k])/2), 3), flush=True)


if __name__ == '__main__':
    main()
