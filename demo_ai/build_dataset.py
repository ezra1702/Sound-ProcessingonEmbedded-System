"""Bangun dataset augmentasi untuk model deep learning (Siklus 2).

Sumber data (semua dipisah per acquisition block seperti MP09: block 1-4 train, 5 val, 6 test):
  * keyword  : 96 rekaman asli kelompok (nyala/mati/naik/turun, 4 penutur)
  * 'lain'   : (a) kata non-keyword TTS Windows SAPI - DAFTAR KATA berbeda untuk train/val/test,
               (b) turunan rekaman kelompok dari block yang sama: dibalik, disambung (separuh kata A + separuh kata B),
                   dan kata terpotong; (c) suara non-ucapan (tepuk, ketukan, nada, klik)
  * 'diam'   : derau latar (putih/pink/coklat/hum/babble/derau ruang dari rekaman) tanpa ucapan
Augmentasi kanal: speed perturbation, reverb (RIR sintetis), respons mikrofon acak (HPF/LPF/tilt),
gain, derau pada SNR 0..30 dB, clipping ringan, self-noise mikrofon.

Output: artifacts/dataset_{train,val,test}.npz (fitur log-Mel 98x40 float16 + label kata + label penutur + meta).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

import common as C

rng_master = np.random.default_rng(C.SEED)
TTS_DIR = C.OUT/'tts'

# Kata non-keyword. Dipisah PER KATA agar val/test berisi kata yang tidak pernah dilihat saat training.
# Banyak yang mirip bunyi keyword (nyanyi, mata, naga, turis, ...) supaya model belajar membedakan.
WORDS = {
    'train': ['halo', 'lampu', 'satu', 'dua', 'tolong', 'buka', 'tutup', 'kiri', 'kanan', 'maju', 'rumah', 'makan',
              'tidur', 'pagi', 'nyanyi', 'nanti', 'nasi', 'mata', 'mana', 'turis', 'taman', 'tahun', 'bisa', 'kamu',
              'saya', 'apa', 'kenapa', 'selamat', 'oke', 'tidak', 'jangan', 'sudah', 'kursi', 'meja', 'pintu', 'kopi',
              'motor', 'nyamuk', 'malam', 'naga', 'tukar', 'tiga'],
    'val': ['minum', 'mundur', 'belum', 'jendela', 'mobil', 'nyata', 'masuk', 'tunggu', 'siapa', 'empat'],
    'test': ['lima', 'keluar', 'nanas', 'matahari', 'tulis', 'sekarang', 'nyonya', 'naik', 'baik', 'terima kasih'],
}
WORDS['test'].remove('naik')          # jaga-jaga: keyword tidak boleh masuk daftar 'lain'
VOICES = ['Microsoft Zira Desktop', 'Microsoft David Desktop']
RATES = [-3, 0, 3]
N_AUG = {'train': {'kw': 160, 'tts': 10, 'derived': 6, 'event': 500, 'silence': 2400},
         'val': {'kw': 20, 'tts': 6, 'derived': 3, 'event': 60, 'silence': 300},
         'test': {'kw': 20, 'tts': 6, 'derived': 3, 'event': 60, 'silence': 300}}


# ----------------------------------------------------------------------------- TTS
def tts_render(jobs):
    lines = ["Add-Type -AssemblyName System.Speech",
             "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)"]
    for text, voice, rate, out in jobs:
        lines += ["$s = New-Object System.Speech.Synthesis.SpeechSynthesizer", f"$s.SelectVoice('{voice}')",
                  f"$s.Rate = {int(rate)}", f"$s.SetOutputToWaveFile('{out}', $fmt)", f"$s.Speak('{text}')", "$s.Dispose()"]
    with tempfile.NamedTemporaryFile('w', suffix='.ps1', delete=False, encoding='utf-8') as f:
        f.write("\n".join(lines)); ps1 = f.name
    try:
        subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ps1], check=True, capture_output=True, timeout=900)
    finally:
        os.unlink(ps1)


def get_tts(split):
    TTS_DIR.mkdir(parents=True, exist_ok=True)
    items, jobs = [], []
    for w in WORDS[split]:
        for v in VOICES:
            for r in RATES:
                p = TTS_DIR/f"{w.replace(' ', '_')}_{v.split()[1].lower()}_{r:+d}.wav"
                items.append(p)
                if not p.exists():
                    jobs.append((w, v, r, str(p)))
    if jobs:
        tts_render(jobs)
    return [C.read_wav(p) for p in items]


# ----------------------------------------------------------------------------- utilitas sinyal
def speech_bounds(y, rel=0.1):
    fr = y[:len(y)//160*160].reshape(-1, 160)
    e = np.sqrt(np.mean(fr**2, axis=1)+1e-12)
    a = np.where(e >= rel*e.max())[0]
    return (int(a[0])*160, int(a[-1]+1)*160) if len(a) else (0, len(y))


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, np.float64)**2)+1e-12))


def colored_noise(n, kind, r):
    if kind == 'white':
        return r.normal(size=n)
    X = np.fft.rfft(r.normal(size=n))
    f = np.maximum(np.fft.rfftfreq(n, 1/C.FS), 10.0)
    X = X/np.sqrt(f) if kind == 'pink' else X/f
    return np.fft.irfft(X, n)


def hum(n, r):
    t = np.arange(n)/C.FS; f0 = r.choice([50.0, 60.0])
    return sum(r.uniform(.2, 1)/(k)*np.sin(2*np.pi*f0*k*t+r.uniform(0, 6.3)) for k in range(1, 6))


def synthetic_rir(r):
    rt60 = r.uniform(0.12, 0.7); n = int(min(0.6, rt60)*C.FS); t = np.arange(n)/C.FS
    h = r.normal(size=n)*np.exp(-6.9*t/rt60)*r.uniform(0.1, 0.5)
    h[:int(r.uniform(0.001, 0.004)*C.FS)] = 0; h[0] = 1.0
    return h/np.sqrt(np.sum(h**2))


def mic_eq(y, r):
    """Respons frekuensi mikrofon acak (mic MEMS ESP32, laptop, HP berbeda-beda)."""
    if r.random() < 0.8:
        y = sosfilt(butter(2, r.uniform(60, 400), 'highpass', fs=C.FS, output='sos'), y)
    if r.random() < 0.7:
        y = sosfilt(butter(int(r.choice([2, 4])), r.uniform(3000, 7600), 'lowpass', fs=C.FS, output='sos'), y)
    if r.random() < 0.5:                                    # tilt: tambah/kurangi komponen tinggi
        hp = sosfilt(butter(1, 1500, 'highpass', fs=C.FS, output='sos'), y)
        y = y+r.uniform(-0.6, 0.8)*hp
    return y


def speed(y, f):
    n = int(len(y)/f)
    return np.interp(np.arange(n)*f, np.arange(len(y)), y)


class NoiseBank:
    """Sumber derau per split (babble & derau ruang hanya dari rekaman split yang sama)."""

    def __init__(self, clips):
        self.babble_src = [c['y'] for c in clips]
        room = []
        for c in clips:                       # bagian non-ucapan rekaman = derau ruang asli mic penutur
            s0, s1 = speech_bounds(c['y'])
            for seg in (c['y'][:max(0, s0-800)], c['y'][s1+800:]):
                if len(seg) > 800:
                    room.append(seg-np.mean(seg))
        self.room = room

    def sample(self, n, r, kinds=None):
        kind = r.choice(kinds or ['white', 'pink', 'brown', 'hum', 'babble', 'room', 'room', 'mix'])
        if kind in ('white', 'pink', 'brown'):
            x = colored_noise(n, kind, r)
        elif kind == 'hum':
            x = hum(n, r)+0.05*r.normal(size=n)
        elif kind == 'babble':
            x = np.zeros(n)
            for _ in range(r.integers(3, 7)):
                s = self.babble_src[r.integers(len(self.babble_src))]
                s = np.roll(np.resize(s, n), r.integers(n)); x += s/(rms(s)+1e-9)
        elif kind == 'room':
            segs = []
            while sum(len(s) for s in segs) < n:
                segs.append(self.room[r.integers(len(self.room))])
            x = np.concatenate(segs)[:n]
        else:
            x = self.sample(n, r, ['white', 'pink', 'hum', 'babble'])+0.5*self.sample(n, r, ['room', 'brown'])
        return x/(rms(x)+1e-12)


def place_in_window(y, r, bounds=None):
    """Potong/tempatkan sinyal ke jendela 1,0 s dengan ucapan utuh di dalamnya (geser acak)."""
    s0, s1 = bounds if bounds else speech_bounds(y)
    dur = s1-s0
    if len(y) < C.WIN:
        pad = C.WIN-len(y); a = r.integers(0, pad+1)
        return np.pad(y, (a, pad-a)).astype(np.float64)
    lo = max(0, s1-C.WIN+int(0.03*C.FS)) if dur < C.WIN else max(0, s0-(C.WIN-dur)//2)
    hi = min(len(y)-C.WIN, s0-int(0.03*C.FS)) if dur < C.WIN else lo
    st = int(r.integers(min(lo, hi), max(lo, hi)+1)) if hi >= 0 else 0
    st = int(np.clip(st, 0, len(y)-C.WIN))
    out = y[st:st+C.WIN].astype(np.float64)
    return np.pad(out, (0, C.WIN-len(out)))


def channel(x, r, nb: NoiseBank, speech_ref=None, snr_range=(0, 30), p_noise=0.85):
    """Augmentasi kanal pada jendela 1,0 s. speech_ref = rms ucapan untuk menghitung SNR."""
    if r.random() < 0.4:
        x = fftconvolve(x, synthetic_rir(r))[:C.WIN]
    x = mic_eq(x, r)
    x = x/(np.max(np.abs(x))+1e-9)*r.uniform(0.05, 0.9)
    ref = rms(x) if speech_ref is None else speech_ref*np.max(np.abs(x))
    snr = None
    if r.random() < p_noise:
        snr = float(r.uniform(*snr_range))
        x = x+nb.sample(C.WIN, r)*ref/10**(snr/20)
    x = x+r.normal(size=C.WIN)*10**(r.uniform(-75, -55)/20)
    if r.random() < 0.05:
        lim = r.uniform(0.4, 0.8)*np.max(np.abs(x)); x = np.clip(x, -lim, lim)
    return np.clip(x, -1, 1).astype(np.float32), snr


def aug_speech(y, r, nb, sp_range=(0.88, 1.12)):
    y = speed(y, r.uniform(*sp_range)) if r.random() < 0.8 else y
    w = place_in_window(y, r)
    s0, s1 = speech_bounds(w)
    pk = np.max(np.abs(w))+1e-9
    return channel(w, r, nb, speech_ref=rms(w[s0:s1])/pk)


# ----------------------------------------------------------------------------- 'lain' turunan & non-ucapan
def derived_unknowns(clips, r):
    """Dari rekaman split yang sama: dibalik, disambung (separuh A + separuh B, kata beda), kata terpotong."""
    out = []
    by_spk = {}
    for c in clips:
        by_spk.setdefault(c['speaker'], []).append(c)
    for c in clips:
        y = c['y']; s0, s1 = speech_bounds(y)
        out.append(('reversed', c['speaker'], y[::-1].copy()))
        cut = s0+int(r.uniform(0.3, 0.5)*(s1-s0))
        tr = y.copy(); tr[cut:] = tr[cut:]*0.02           # kata terpotong (hanya awal kata)
        out.append(('truncated', c['speaker'], tr))
        for _ in range(2):
            o = [d for d in by_spk[c['speaker']] if d['label'] != c['label']]
            d = o[r.integers(len(o))]; t0, t1 = speech_bounds(d['y'])
            a = y[s0:s0+(s1-s0)//2]; b = d['y'][(t0+t1)//2:t1]
            sp = np.concatenate([np.zeros(2400), a, b, np.zeros(2400)])
            out.append(('spliced', c['speaker'], sp))
    return out


def nonspeech_event(r):
    n = C.WIN; x = np.zeros(n); kind = r.choice(['clap', 'knock', 'tone', 'click', 'whistle', 'thud'])
    t0 = int(r.uniform(0.05, 0.6)*n)
    if kind == 'clap':
        for _ in range(r.integers(1, 4)):
            L = int(0.03*C.FS); seg = r.normal(size=L)*np.exp(-np.arange(L)/(0.005*C.FS))
            x[t0:t0+L] += seg; t0 = min(n-L-1, t0+int(r.uniform(0.15, 0.3)*C.FS))
    elif kind in ('knock', 'thud'):
        for _ in range(r.integers(1, 4)):
            L = int(0.06*C.FS); tt = np.arange(L)/C.FS
            seg = np.sin(2*np.pi*r.uniform(60, 250)*tt)*np.exp(-tt/0.015)+0.3*r.normal(size=L)*np.exp(-tt/0.004)
            x[t0:t0+L] += seg; t0 = min(n-L-1, t0+int(r.uniform(0.12, 0.25)*C.FS))
    elif kind in ('tone', 'whistle'):
        L = int(r.uniform(0.2, 0.7)*C.FS); tt = np.arange(L)/C.FS
        f0, f1 = (r.uniform(300, 3000), r.uniform(300, 3000)) if kind == 'tone' else (r.uniform(1000, 2500),)*2
        ph = 2*np.pi*(f0*tt+(f1-f0)*tt**2/(2*tt[-1]))
        seg = np.sin(ph)*np.minimum(1, np.minimum(tt, tt[-1]-tt)/0.02); L = min(L, n-t0); x[t0:t0+L] += seg[:L]
    else:
        for _ in range(r.integers(3, 12)):
            p = r.integers(0, n-50); x[p:p+30] += r.normal(size=30)*np.exp(-np.arange(30)/5)
    return x


# ----------------------------------------------------------------------------- bangun split
def build_split(split, clips_all, r):
    t = time.time()
    clips = [c for c in clips_all if c['split'] == split]
    nb = NoiseBank(clips)
    cfg = N_AUG[split]
    W, KW, SPK, KIND, SRC, SNR, CLEAN = [], [], [], [], [], [], []

    def add(w, kw, spk, kind, src, snr, clean=False):
        W.append(w); KW.append(C.KW_CLASSES.index(kw)); SPK.append(C.SPEAKERS.index(spk) if spk else -1)
        KIND.append(kind); SRC.append(src); SNR.append(np.nan if snr is None else snr); CLEAN.append(clean)

    for c in clips:                                   # keyword: versi bersih + augmentasi
        if split != 'train':
            w, _ = C.center_window(c['y']); add(w, c['label'], c['speaker'], 'kw', c['id'], None, True)
        for _ in range(cfg['kw']):
            w, snr = aug_speech(c['y'], r, nb); add(w, c['label'], c['speaker'], 'kw', c['id'], snr)
    tts = get_tts(split)
    for i, y in enumerate(tts):                       # 'lain' TTS (kata berbeda per split)
        s0, s1 = speech_bounds(y); y = y[max(0, s0-1600):s1+1600]
        for _ in range(cfg['tts']):
            w, snr = aug_speech(y, r, nb); add(w, 'lain', None, 'tts', f'tts{i}', snr)
    for kind, spk, y in derived_unknowns(clips, r):   # 'lain' turunan rekaman kelompok
        for _ in range(cfg['derived']):
            w, snr = aug_speech(y, r, nb, sp_range=(0.9, 1.1)); add(w, 'lain', spk, kind, kind, snr)
    for _ in range(cfg['event']):                     # 'lain' non-ucapan (tepuk, ketuk, nada)
        ev = nonspeech_event(r); w, snr = channel(ev, r, nb, snr_range=(5, 30)); add(w, 'lain', None, 'event', 'event', snr)
    for _ in range(cfg['silence']):                   # 'diam': derau latar saja
        x = mic_eq(nb.sample(C.WIN, r), r); x = x/(rms(x)+1e-12)*10**(r.uniform(-70, -25)/20)
        add(np.clip(x, -1, 1).astype(np.float32), 'diam', None, 'silence', 'silence', None)

    Wn = np.stack(W).astype(np.float32)
    X = np.concatenate([C.logmel_batch(Wn[i:i+2048]) for i in range(0, len(Wn), 2048)]).astype(np.float16)
    np.savez(C.OUT/f'dataset_{split}.npz', X=X, kw=np.array(KW, np.int8), spk=np.array(SPK, np.int8),
             kind=np.array(KIND), src=np.array(SRC), snr=np.array(SNR, np.float32), clean=np.array(CLEAN))
    if split != 'train':                              # simpan waveform bersih keyword untuk verifikasi C & simulasi enrol
        np.savez(C.OUT/f'waves_{split}.npz', W=Wn[np.array(CLEAN)], kw=np.array(KW)[np.array(CLEAN)],
                 spk=np.array(SPK)[np.array(CLEAN)], src=np.array(SRC)[np.array(CLEAN)])
    counts = {k: int(np.sum(np.array(KW) == i)) for i, k in enumerate(C.KW_CLASSES)}
    print(f'[{split}] {len(W)} contoh {counts} ({time.time()-t:.0f} s)', flush=True)
    return counts


def main():
    C.OUT.mkdir(parents=True, exist_ok=True)
    clips = C.load_group_clips()
    np.savez(C.OUT/'group_clips.npz', Y=np.stack([c['y'] for c in clips]), ids=np.array([c['id'] for c in clips]),
             kw=np.array([c['label'] for c in clips]), spk=np.array([c['speaker'] for c in clips]),
             block=np.array([c['block'] for c in clips]), split=np.array([c['split'] for c in clips]))
    summary = {}
    for i, split in enumerate(['train', 'val', 'test']):
        summary[split] = build_split(split, clips, np.random.default_rng(C.SEED+17*i))
    C.save_json(C.OUT/'dataset_summary.json', {'counts': summary, 'words_lain': WORDS, 'voices': VOICES, 'rates': RATES,
                                                'n_aug': N_AUG, 'frontend': {'win': C.WIN, 'frame': C.FRAME, 'hop': C.HOP,
                                                                            'nfft': C.NFFT, 'n_mels': C.N_MELS}})


if __name__ == '__main__':
    main()
