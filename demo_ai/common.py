"""Konstanta & front-end bersama untuk Siklus Pengembangan 2 (deep learning, target ESP32).

Front-end sengaja sederhana agar identik di Python dan C (firmware ESP32):
  jendela 1,0 s @16 kHz -> DC removal -> pre-emphasis 0,97 -> frame 400 (25 ms) hop 160 (10 ms), Hamming
  -> FFT 512 -> power -> 40 mel (20..7600 Hz, HTK) -> log(mel + 1e-6) -> CMN per band (kurangi rata-rata waktu)
Hasil: matriks 98 x 40 (frame x mel). CMN membuang perbedaan respons frekuensi mikrofon (kanal konstan) dan gain.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent            # demo_ai/
PROJECT = ROOT.parent
DATASET_DIR = PROJECT/'dataset_rekaman'
OUT = ROOT/'artifacts'

FS = 16000
WIN = 16000                    # 1,0 s jendela model
FRAME, HOP, NFFT = 400, 160, 512
N_FRAMES = 1 + (WIN-FRAME)//HOP  # 98
N_MELS, FMIN, FMAX = 40, 20.0, 7600.0
PRE_EMPH = 0.97
LOG_EPS = 1e-6

KEYWORDS = ['nyala', 'mati', 'naik', 'turun']
KW_CLASSES = KEYWORDS+['diam', 'lain']           # 6 kelas output kata
SPEAKERS = ['ezra', 'abdillah', 'ivang', 'jahzeel']
SPEAKER_NAMES = {'ezra': 'Christama Ezra Yudianto', 'abdillah': 'Abdillah Fatah Suryanto',
                 'ivang': 'Ivang Arfandia', 'jahzeel': 'Jahzeel Erason Nicodemus'}
SPLIT_OF_BLOCK = {1: 'train', 2: 'train', 3: 'train', 4: 'train', 5: 'val', 6: 'test'}
SEED = 2026


def hz2mel(f):
    return 2595.0*np.log10(1.0+np.asarray(f, float)/700.0)


def mel2hz(m):
    return 700.0*(10.0**(np.asarray(m, float)/2595.0)-1.0)


def mel_filterbank():
    """Matriks (N_MELS, NFFT/2+1) segitiga HTK, float32."""
    nb = NFFT//2+1
    freqs = np.arange(nb)*FS/NFFT
    pts = mel2hz(np.linspace(hz2mel(FMIN), hz2mel(FMAX), N_MELS+2))
    fb = np.zeros((N_MELS, nb), np.float32)
    for m in range(N_MELS):
        lo, ce, hi = pts[m], pts[m+1], pts[m+2]
        up = (freqs-lo)/(ce-lo)
        dn = (hi-freqs)/(hi-ce)
        fb[m] = np.maximum(0.0, np.minimum(up, dn))
    return fb


MEL_FB = mel_filterbank()
WINDOW = np.hamming(FRAME).astype(np.float32)


def logmel(y: np.ndarray) -> np.ndarray:
    """y: float sinyal panjang WIN (dipotong/dipad). Return float32 (98, 40) setelah CMN."""
    y = np.asarray(y, np.float32)
    if len(y) < WIN:
        y = np.pad(y, (0, WIN-len(y)))
    y = y[:WIN]
    y = y-np.mean(y, dtype=np.float32)
    y = np.concatenate([y[:1], y[1:]-np.float32(PRE_EMPH)*y[:-1]]).astype(np.float32)
    idx = np.arange(FRAME)[None, :]+HOP*np.arange(N_FRAMES)[:, None]
    fr = y[idx]*WINDOW[None, :]
    spec = np.fft.rfft(fr, n=NFFT, axis=1)
    pw = (spec.real**2+spec.imag**2).astype(np.float32)
    lm = np.log(pw@MEL_FB.T+np.float32(LOG_EPS)).astype(np.float32)
    return (lm-lm.mean(axis=0, keepdims=True)).astype(np.float32)


def logmel_batch(Y: np.ndarray) -> np.ndarray:
    """Versi batch (N, WIN) -> (N, 98, 40) untuk membangun dataset besar dengan cepat."""
    Y = np.asarray(Y, np.float32)
    Y = Y-Y.mean(axis=1, keepdims=True)
    Y = np.concatenate([Y[:, :1], Y[:, 1:]-np.float32(PRE_EMPH)*Y[:, :-1]], axis=1)
    idx = np.arange(FRAME)[None, :]+HOP*np.arange(N_FRAMES)[:, None]
    fr = Y[:, idx]*WINDOW[None, None, :]
    spec = np.fft.rfft(fr, n=NFFT, axis=2)
    pw = (spec.real**2+spec.imag**2).astype(np.float32)
    lm = np.log(pw@MEL_FB.T+np.float32(LOG_EPS)).astype(np.float32)
    return (lm-lm.mean(axis=1, keepdims=True)).astype(np.float32)


def center_window(y: np.ndarray, n: int = WIN, hop: int = 160) -> tuple[np.ndarray, int]:
    """Ambil jendela n sampel yang memuat energi terbesar (dipakai saat inferensi & di firmware)."""
    y = np.asarray(y, np.float32)
    if len(y) <= n:
        return np.pad(y, (0, n-len(y))), 0
    e = np.convolve(y.astype(np.float64)**2, np.ones(n), 'valid')[::hop]
    s0 = int(np.argmax(e))*hop
    return y[s0:s0+n], s0


def read_wav(path) -> np.ndarray:
    from scipy.io import wavfile
    fs, y = wavfile.read(path)
    assert fs == FS, f'{path}: fs {fs}'
    if y.ndim > 1:
        y = y.mean(axis=1)
    return (y.astype(np.float32)/32768.0) if np.issubdtype(y.dtype, np.integer) else y.astype(np.float32)


def load_group_clips():
    """96 klip kelompok: list dict(id, label, speaker, block, split, y)."""
    folder = {v: k for k, v in SPEAKER_NAMES.items()}
    out = []
    for d in sorted(DATASET_DIR.iterdir()):
        spk = folder[d.name]
        for b in range(1, 7):
            for kw in KEYWORDS:
                p = d/f'block_0{b}'/f'{kw}_b0{b}.wav'
                out.append({'id': f'{spk}_{kw}_b0{b}', 'label': kw, 'speaker': spk, 'block': b,
                            'split': SPLIT_OF_BLOCK[b], 'y': read_wav(p)})
    assert len(out) == 96, len(out)
    return out


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, 'item') else str(o)),
                          encoding='utf-8')
