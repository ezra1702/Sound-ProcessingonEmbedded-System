"""Evaluasi AKHIR siklus 2 pada split test (block 6 + kata 'lain' TTS yang tidak pernah dilatih). Jalankan SEKALI.

Membandingkan pada jendela uji yang SAMA:
  * DS-CNN float32 (PyTorch)
  * DS-CNN INT8 (referensi numpy = bit-exact dengan firmware C)
  * pipeline kelas (MP16): front-end MP9 + logreg INT8 MP15 + gerbang OOD Mahalanobis
Keluaran: artifacts/eval_report.json + artifacts/figures/*.png
"""
from __future__ import annotations

import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

import common as C
from int8_ref import Int8KWS, load_float, softmax
from train import kw_metrics, load, predict

FIG = C.OUT/'figures'
SNR_BINS = [(-1, 5), (5, 10), (10, 20), (20, 31)]


def classical_pipeline():
    sys.path.insert(0, str(C.ROOT.parent/'demo_web'))
    from mock_board import Pipeline
    return Pipeline(C.ROOT.parent/'Mini Project 16')


def by_snr(y, pred, snr):
    out = {}
    kw = y < 4
    for lo, hi in SNR_BINS:
        m = kw & ~np.isnan(snr) & (snr > lo) & (snr <= hi)
        if m.any():
            out[f'snr_{max(lo, 0)}_{hi-1}dB'] = {'keyword_accuracy': float(np.mean(pred[m] == y[m])), 'n': int(m.sum())}
    m = kw & np.isnan(snr)
    if m.any():
        out['no_added_noise'] = {'keyword_accuracy': float(np.mean(pred[m] == y[m])), 'n': int(m.sum())}
    return out


def compare_with_classical(q, n_var=5, snrs=(20, 10, 5, 0)):
    """Sinyal yang SAMA untuk kedua sistem: klip mentah block 6 (+derau terkontrol), kata 'lain' TTS test, suara non-ucapan."""
    import build_dataset as BD
    pipe = classical_pipeline(); name2idx = {n: C.KW_CLASSES.index(n) for n in ('nyala', 'mati', 'naik', 'turun')}
    g = np.load(C.OUT/'group_clips.npz', allow_pickle=True); m6 = g['block'] == 6
    Y6, y6 = g['Y'][m6], np.array([C.KW_CLASSES.index(k) for k in g['kw'][m6]])
    nb = BD.NoiseBank([{'y': y} for y in Y6]); r = np.random.default_rng(C.SEED+66)

    def both(y):
        lab, _, dist, _ = pipe.run(y)
        c = 5 if dist > pipe.ood_thr else name2idx[lab]
        d = int(q.forward(C.logmel(C.center_window(y)[0]))['kw'].argmax())
        return c, d

    res = {}
    for cond in ['bersih']+[f'{s} dB' for s in snrs]:
        P = []
        for y, lab in zip(Y6, y6):
            for v in range(1 if cond == 'bersih' else n_var):
                x = y.copy()
                if cond != 'bersih':
                    snr = float(cond.split()[0]); s0, s1 = BD.speech_bounds(y)
                    nz = nb.sample(len(y), r, ['babble', 'room', 'pink', 'hum'])
                    x = np.clip(y+nz*BD.rms(y[s0:s1])/10**(snr/20), -1, 1).astype(np.float32)
                P.append((lab,)+both(x))
        P = np.array(P)
        res[cond] = {'n': int(len(P)), 'klasik_mp16_akurasi': float(np.mean(P[:, 1] == P[:, 0])),
                     'dscnn_int8_akurasi': float(np.mean(P[:, 2] == P[:, 0]))}
    tts = BD.get_tts('test'); ns = [BD.nonspeech_event(r) for _ in range(60)]
    for tag, sigs in (('kata_lain_tts', tts), ('suara_non_ucapan', ns)):
        P = []
        for x in sigs:
            x = np.concatenate([np.zeros(1600), np.asarray(x, float), np.zeros(1600)])
            P.append(both((x+1e-4*r.normal(size=len(x))).astype(np.float32)))
        P = np.array(P)
        res[tag] = {'n': int(len(P)), 'klasik_mp16_salah_terima': float(np.mean(P[:, 0] < 4)),
                    'dscnn_int8_salah_terima': float(np.mean(P[:, 1] < 4))}
    res['catatan'] = ('klasik = front-end MP9 + logreg INT8 MP15 + gerbang OOD MP16 (ditolak = bukan perintah). '
                      'Derau babble/room diambil dari rekaman block 6 sendiri; kata lain = suara TTS en-US (bukan manusia).')
    ks = [k for k in res if k.endswith('dB') or k == 'bersih']
    fig, ax = plt.subplots(figsize=(6.4, 3.3)); xs = np.arange(len(ks))
    ax.bar(xs-0.18, [res[k]['klasik_mp16_akurasi'] for k in ks], 0.36, label='klasik (MP16)', color='#a0a7b4')
    ax.bar(xs+0.18, [res[k]['dscnn_int8_akurasi'] for k in ks], 0.36, label='DS-CNN INT8', color='#3b6fd4')
    ax.set_xticks(xs, ks); ax.set_ylim(0, 1); ax.set_ylabel('akurasi kata (block 6)'); ax.legend(fontsize=8)
    ax.set_title('Ketahanan terhadap derau: sinyal yang sama untuk kedua sistem', fontsize=10)
    fig.tight_layout(); fig.savefig(FIG/'classical_vs_dscnn.png', dpi=130); plt.close(fig)
    return res


def plot_cm(y, pred, title, path):
    cm = confusion_matrix(y, pred, labels=range(6))
    fig, ax = plt.subplots(figsize=(5, 4.3))
    ax.imshow(cm/np.maximum(cm.sum(1, keepdims=True), 1), cmap='Blues', vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=8, color='white' if cm[i, j] > cm[i].sum()*0.6 else 'black')
    ax.set_xticks(range(6), C.KW_CLASSES, rotation=30); ax.set_yticks(range(6), C.KW_CLASSES)
    ax.set_xlabel('prediksi'); ax.set_ylabel('label'); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_cam_examples(q, X, y, path):
    fig, axs = plt.subplots(2, 4, figsize=(12, 5))
    for k in range(4):
        i = int(np.where(y == k)[0][0])
        o = q.forward(X[i]); cam = q.cam(o['A'], k)
        axs[0, k].imshow(X[i].T, origin='lower', aspect='auto', cmap='magma'); axs[0, k].set_title(f'log-Mel "{C.KW_CLASSES[k]}"')
        big = np.kron(cam, np.ones((4, 4)))[:C.N_FRAMES, :C.N_MELS]
        axs[1, k].imshow(X[i].T, origin='lower', aspect='auto', cmap='gray')
        axs[1, k].imshow(big.T, origin='lower', aspect='auto', cmap='jet', alpha=0.45, extent=(0, C.N_FRAMES, 0, C.N_MELS))
        axs[1, k].set_title('CAM (bagian yang diperhatikan)')
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    te = load('test'); y = te['kw'].astype(int); snr = te['snr'].astype(float); clean = te['clean'].astype(bool)
    X = te['X'].astype(np.float32)
    model = load_float()
    pf, ps, *_ = predict(model, torch.tensor(X).unsqueeze(1))
    sc = dict(np.load(C.OUT/'int8_params.npz'))
    q = Int8KWS(model, {k[6:]: float(v) for k, v in sc.items() if k.startswith('scale_')})
    protos = sc['protos']
    outs = [q.forward(x) for x in X]
    pq = np.stack([softmax(o['kw']) for o in outs])
    E = np.stack([o['e']/np.linalg.norm(o['e']) for o in outs])

    rep = {'test_split': 'block 6 (4 anggota) + kata lain TTS split test + derau/diam test', 'n': int(len(y)),
           'dscnn_float': kw_metrics(y, pf, clean, te['spk'], ps), 'dscnn_int8': kw_metrics(y, pq, clean)}
    rep['dscnn_float']['by_snr'] = by_snr(y, pf.argmax(1), snr); rep['dscnn_int8']['by_snr'] = by_snr(y, pq.argmax(1), snr)
    rep['int8_vs_float_agreement'] = float(np.mean(pq.argmax(1) == pf.argmax(1)))

    # identifikasi penutur INT8 (prototipe dari header firmware, ambang dari LOSO)
    import json
    thr = json.loads((C.OUT/'export_report.json').read_text(encoding='utf-8'))['speaker_threshold']
    sp = te['spk'].astype(int); cos = E@protos.T; best = cos.argmax(1)
    m = (sp >= 0) & clean
    rep['speaker_int8'] = {'closed_set_accuracy_clean': float(np.mean(best[m] == sp[m])),
                           'closed_set_accuracy_all_keyword_windows': float(np.mean(best[sp >= 0] == sp[sp >= 0])),
                           'accepted_as_known_clean': float(np.mean(cos[m].max(1) >= thr)), 'threshold': thr,
                           'n_clean': int(m.sum()), 'catatan': 'hanya 4 penutur yang juga ada di training; uji orang asing ada di fewshot_report.json'}

    rep['robustness_comparison'] = compare_with_classical(q)
    rep['mp13_reference'] = 'MP13 (16 klip block 6): logreg F1 0,937; tiny-CNN 0,560'

    plot_cm(y, pq.argmax(1), 'DS-CNN INT8 — test (semua kondisi)', FIG/'cm_test_int8.png')
    plot_cam_examples(q, X[clean], y[clean], FIG/'cam_examples.png')
    accs = [[rep['dscnn_int8']['by_snr'].get(k, {}).get('keyword_accuracy', np.nan) for k in rep['dscnn_int8']['by_snr']]]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ks = list(rep['dscnn_int8']['by_snr']); ax.bar(range(len(ks)), accs[0], color='#3b6fd4')
    ax.set_xticks(range(len(ks)), [k.replace('snr_', '').replace('dB', ' dB') for k in ks], fontsize=8); ax.set_ylim(0, 1)
    ax.set_ylabel('akurasi kata'); ax.set_title('DS-CNN INT8 vs tingkat derau (test)', fontsize=10)
    fig.tight_layout(); fig.savefig(FIG/'acc_vs_snr.png', dpi=130); plt.close(fig)
    rep['seconds'] = time.time()-t0
    C.save_json(C.OUT/'eval_report.json', rep)
    print(json.dumps({k: v for k, v in rep.items() if k not in ('dscnn_float',)}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
