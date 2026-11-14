# %% [markdown]
# # Mini Project 9 — Membangun Dataset dan Membekukan Split Eksperimen
# Enam acquisition block × empat perintah (nyala, mati, naik, turun) = 24 WAV.
# Filter final MP6 + preprocessing MP7 + feature schema MP8 diterapkan otomatis,
# metadata & log-Mel disimpan, lalu grouped split (block 1–4 train, 5 validation,
# 6 test) dibekukan untuk MP10–MP13. Status -> `ready_for_mp10`.
#
# **Mode akuisisi** (`ACQUISITION_MODE`, dapat dioverride env `MP09_MODE`):
# * `mic`  : rekam langsung dari mikrofon (prosedur buku, DEMO_MODE=False).
# * `tts`  : *speech synthesis* Windows SAPI (2 suara) + simulasi variasi kanal per block
#            (posisi mic → gain, reverb, noise floor, hum). Dipakai pada pengerjaan ini karena
#            sesi pengerjaan tidak memiliki akses mikrofon; label & grouping tetap sesuai protokol.
# * `demo` : sinyal sintetis sinus dari buku (hanya untuk tes skrip).
#
# WAV yang sudah ada **tidak direkam ulang**. Untuk mengganti dataset dengan rekaman suara
# kelompok: hapus `data/raw/mp09_dataset/*.wav`, set `MP09_MODE=mic`, jalankan ulang skrip ini,
# lalu jalankan `python ../run_all_mini_projects.py --start 10`.

# %%
# Anggota Kelompok :
# 1. Christama Ezra Yudianto    (245150307111009)
# 2. Abdillah Fatah Suryanto    (245150300111008)
# 3. Ivang Arfandia             (245150307111003)
# 4. Jahzeel Erason Nicodemus   (245150300111002)
from pathlib import Path
import csv, json, time, os, subprocess, tempfile
import numpy as np
import matplotlib
if "__file__" in globals():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import lfilter, resample_poly, fftconvolve
from scipy.fft import dct

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
STATUS = ROOT / 'project_status.json'
DEMO_MODE = False       # True hanya untuk pengujian tanpa mikrofon (sinyal sinus buku)
ACQUISITION_MODE = os.environ.get('MP09_MODE', 'demo' if DEMO_MODE else 'tts')  # 'mic' | 'tts' | 'demo'
N_BLOCKS = 6
DURATION_S = 1.2
SEED = 2026
rng = np.random.default_rng(SEED)

# %% [markdown]
# ## Langkah 1 — Validasi status MP8 dan muat konfigurasi yang sudah dikunci

# %%
status = json.loads((ROOT/'project_status.json').read_text(encoding='utf-8'))
assert status['status'] in ('ready_for_mp09', 'ready_for_mp10'), status['status']

project_cfg = json.loads((ROOT/'config/project_config.json').read_text(encoding='utf-8'))
pre_cfg = json.loads((ROOT/'config/preprocessing_config.json').read_text(encoding='utf-8'))
feat_cfg = json.loads((ROOT/'config/feature_config.json').read_text(encoding='utf-8'))
filter_npz = np.load(ROOT/'features/intermediate/mp06_filter_final.npz', allow_pickle=True)

FS = int(project_cfg.get('sampling_rate_hz', project_cfg.get('sample_rate_hz', 16000)))
LABELS = project_cfg.get('labels', ['nyala', 'mati', 'naik', 'turun'])
B = np.asarray(filter_npz['b'], dtype=float)
A = np.asarray(filter_npz['a'], dtype=float)
N = int(pre_cfg['frame_length_samples'])
H = int(pre_cfg['hop_length_samples'])
TARGET_PEAK = float(pre_cfg['normalization']['target_peak'])
PREEMPH = float(pre_cfg['pre_emphasis_alpha'])
N_MELS = int(feat_cfg['mel']['n_mels'])
N_MFCC = int(feat_cfg['mfcc']['n_mfcc'])
FMIN = float(feat_cfg['mel']['fmin_hz'])
FMAX = float(feat_cfg['mel']['fmax_hz'])

raw_dir = ROOT/'data/raw/mp09_dataset'
meta_dir = ROOT/'data/metadata'
dataset_dir = ROOT/'features/dataset'
logmel_dir = ROOT/'features/logmel/mp09'
split_dir = ROOT/'splits'
metric_dir = ROOT/'results/metrics'
fig_dir = ROOT/'results/figures'
for d in [raw_dir, meta_dir, dataset_dir, logmel_dir, split_dir, metric_dir, fig_dir]:
    d.mkdir(parents=True, exist_ok=True)
print('Acquisition mode:', ACQUISITION_MODE, '| fs =', FS, '| labels =', LABELS)

# %% [markdown]
# ## Akuisisi: mic (buku), TTS + simulasi kanal per block, atau demo sintetis
# Protokol per block pada mode `tts` meniru instruksi buku "ubah posisi mikrofon sedikit
# antar block": tiap block memiliki suara, laju bicara, jarak (gain), reverb, noise floor,
# dan hum yang berbeda — sehingga block menjadi unit grouping yang bermakna.

# %%
# Ejaan fonetik agar suara TTS berbahasa Inggris melafalkan kata Indonesia mendekati aslinya
TTS_TEXT = {'nyala': 'nyah lah', 'mati': 'mah tee', 'naik': 'nah eek', 'turun': 'too roon'}
BLOCK_PROFILE = {
    # block: voice, SAPI rate, speed factor, gain dB (jarak), SNR dB, RT60 s, hum amplitude
    1: dict(voice='Microsoft Zira Desktop',  rate=0,  speed=1.00, gain_db=0.0,  snr_db=34, rt60=0.15, hum=0.003),
    2: dict(voice='Microsoft David Desktop', rate=0,  speed=1.00, gain_db=-3.0, snr_db=32, rt60=0.20, hum=0.004),
    3: dict(voice='Microsoft Zira Desktop',  rate=-2, speed=0.95, gain_db=-6.0, snr_db=29, rt60=0.26, hum=0.005),
    4: dict(voice='Microsoft David Desktop', rate=2,  speed=1.05, gain_db=-2.0, snr_db=30, rt60=0.18, hum=0.003),
    5: dict(voice='Microsoft Zira Desktop',  rate=1,  speed=1.03, gain_db=-4.0, snr_db=30, rt60=0.22, hum=0.004),
    6: dict(voice='Microsoft David Desktop', rate=-1, speed=0.97, gain_db=-5.0, snr_db=29, rt60=0.24, hum=0.005),
}


def save_wav(path, y):
    y = np.clip(y, -1, 1)
    wavfile.write(path, FS, np.round(y*32767).astype(np.int16))


def _tts_render(jobs):
    """Render banyak ucapan sekaligus via Windows SAPI (System.Speech) ke WAV 16 kHz mono 16-bit."""
    lines = ["Add-Type -AssemblyName System.Speech",
             "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)"]
    for text, voice, rate, out in jobs:
        lines += ["$s = New-Object System.Speech.Synthesis.SpeechSynthesizer",
                  f"$s.SelectVoice('{voice}')", f"$s.Rate = {int(rate)}",
                  f"$s.SetOutputToWaveFile('{out}', $fmt)", f"$s.Speak('{text}')", "$s.Dispose()"]
    with tempfile.NamedTemporaryFile('w', suffix='.ps1', delete=False, encoding='utf-8') as f:
        f.write("\n".join(lines)); ps1 = f.name
    try:
        subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ps1],
                       check=True, capture_output=True, timeout=300)
    finally:
        os.unlink(ps1)


def _synthetic_rir(rt60, block):
    r = np.random.default_rng(SEED + 100*block)
    n = int(0.3*FS); t = np.arange(n)/FS
    h = r.normal(size=n)*np.exp(-6.9*t/rt60)*0.25
    h[:int(0.002*FS)] = 0; h[0] = 1.0
    for d_ms, g in [(7 + 2*block, 0.35), (19 + block, -0.2)]:
        h[int(d_ms*FS/1000)] += g
    return h/np.sqrt(np.sum(h**2))


def _apply_channel(speech, block, label):
    """Simulasi posisi mikrofon per block: speed perturbation, reverb, gain, noise floor, hum, onset acak."""
    p = BLOCK_PROFILE[block]
    r = np.random.default_rng(SEED + 1000*block + LABELS.index(label))
    sp = speech/(np.max(np.abs(speech))+1e-12)
    # trim diam bawaan TTS
    env = np.abs(sp) > 0.02
    if env.any():
        i0, i1 = np.argmax(env), len(env)-np.argmax(env[::-1])
        sp = sp[max(0, i0-80):i1+80]
    speed = p['speed']*r.uniform(0.98, 1.02)
    up, down = int(round(1000/speed)), 1000
    sp = resample_poly(sp, up, down)
    sp = fftconvolve(sp, _synthetic_rir(p['rt60'], block))[:len(sp)+int(0.12*FS)]
    sp = 0.5*sp/(np.max(np.abs(sp))+1e-12)*10**(p['gain_db']/20)
    n_total = int(DURATION_S*FS)
    sp = sp[:n_total-int(0.05*FS)]
    onset = int(r.uniform(0.10, max(0.11, (n_total-len(sp))/FS-0.05))*FS)
    y = np.zeros(n_total); y[onset:onset+len(sp)] += sp
    # noise floor (pink-ish) pada SNR target relatif terhadap daya ucapan aktif
    wn = r.normal(size=n_total); pink = lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786],
                                               [1, -2.494956002, 2.017265875, -0.522189400], wn)
    p_sp = np.mean(sp**2)
    pink *= np.sqrt(p_sp/(np.mean(pink**2)*10**(p['snr_db']/10)))
    t = np.arange(n_total)/FS
    y += pink + p['hum']*np.sin(2*np.pi*50*t + r.uniform(0, 2*np.pi))
    return y


_TTS_CACHE = {}


def acquire(label, block):
    path = raw_dir/f'{label}_b{block:02d}.wav'
    if path.exists():
        return path
    if ACQUISITION_MODE == 'demo':
        # synthetic fallback hanya untuk test script
        f0 = {'nyala': 170, 'mati': 205, 'naik': 240, 'turun': 145}[label]
        t = np.arange(int(DURATION_S*FS))/FS
        env = np.sin(np.pi*np.clip(t/DURATION_S, 0, 1))**1.4
        y = env*(0.45*np.sin(2*np.pi*f0*t)+0.16*np.sin(2*np.pi*2.3*f0*t))
        y += 0.012*rng.normal(size=len(t)) + 0.006*(block-1)*np.sin(2*np.pi*50*t)
    elif ACQUISITION_MODE == 'tts':
        if not _TTS_CACHE:
            tmp = Path(tempfile.mkdtemp(prefix='mp09_tts_'))
            jobs = []
            for b_, prof in BLOCK_PROFILE.items():
                for lab in LABELS:
                    out = tmp/f'{lab}_b{b_:02d}.wav'
                    jobs.append((TTS_TEXT[lab], prof['voice'], prof['rate'], str(out)))
            _tts_render(jobs)
            for b_ in BLOCK_PROFILE:
                for lab in LABELS:
                    fs_t, s = wavfile.read(tmp/f'{lab}_b{b_:02d}.wav')
                    assert fs_t == FS
                    _TTS_CACHE[(lab, b_)] = s.astype(np.float64)/32768.0
        y = _apply_channel(_TTS_CACHE[(label, block)], block, label)
    else:
        import sounddevice as sd
        input(f'Block {block} | ucapkan "{label}" lalu tekan ENTER... ')
        print('Merekam...')
        y = sd.rec(int(DURATION_S*FS), samplerate=FS, channels=1, dtype='float32')
        sd.wait(); y = y[:, 0]
    save_wav(path, y)
    return path

# %% [markdown]
# ## Front-end beku MP6 + MP7 + MP8 (filter → preprocessing → 39-D + log-Mel)

# %%
def read_float(path):
    fs, y = wavfile.read(path)
    assert fs == FS, (path, fs)
    if y.ndim > 1: y = y.mean(axis=1)
    if np.issubdtype(y.dtype, np.integer): y = y.astype(np.float64)/np.iinfo(y.dtype).max
    else: y = y.astype(np.float64)
    return y


def hz_to_mel(f): return 2595*np.log10(1+np.asarray(f)/700.0)
def mel_to_hz(m): return 700*(10**(np.asarray(m)/2595.0)-1)


def melbank(n_fft):
    m = np.linspace(hz_to_mel(FMIN), hz_to_mel(FMAX), N_MELS+2)
    hz = mel_to_hz(m); bins = np.floor((n_fft+1)*hz/FS).astype(int)
    fb = np.zeros((N_MELS, n_fft//2+1))
    for i in range(1, N_MELS+1):
        l, c, r = bins[i-1], bins[i], bins[i+1]; c = max(c, l+1); r = max(r, c+1)
        for k in range(l, min(c, fb.shape[1])): fb[i-1, k] = (k-l)/(c-l)
        for k in range(c, min(r, fb.shape[1])): fb[i-1, k] = (r-k)/(r-c)
    return fb


FB = melbank(N)
FREQ = np.fft.rfftfreq(N, 1/FS)


def preprocess(y):
    # filter final MP6
    y = lfilter(B, A, y)
    y = y-np.mean(y)
    pk = np.max(np.abs(y))+1e-12; y = y*(TARGET_PEAK/pk)
    if len(y) < N: y = np.pad(y, (0, N-len(y)))
    starts = np.arange(0, len(y)-N+1, H)
    raw = np.array([y[s:s+N] for s in starts])
    rms = np.sqrt(np.mean(raw**2, axis=1)+1e-12)
    thr = float(pre_cfg['vad']['relative_factor'])*np.max(rms)
    active = np.where(rms >= thr)[0]
    if len(active):
        margin = int(pre_cfg['vad'].get('margin_frames', 2))
        i0 = max(0, int(active[0])-margin); i1 = min(len(raw)-1, int(active[-1])+margin)
        y = y[starts[i0]:min(len(y), starts[i1]+N)]
    # denoise tidak dipaksakan bila MP7 menandai noise estimate tidak reliable
    if len(y) < N: y = np.pad(y, (0, N-len(y)))
    starts = np.arange(0, len(y)-N+1, H)
    fr = np.array([y[s:s+N] for s in starts])
    fr = np.concatenate([fr[:, :1], fr[:, 1:]-PREEMPH*fr[:, :-1]], axis=1)
    return fr*np.hamming(N)[None, :]


def pitch_one(frame):
    x = frame-np.mean(frame); ac = np.correlate(x, x, mode='full')[len(x)-1:]
    if ac[0] <= 1e-12: return 0.0, 0.0
    ac = ac/(ac[0]+1e-12)
    lo = max(1, int(FS/feat_cfg['pitch']['max_hz'])); hi = min(len(ac)-1, int(FS/feat_cfg['pitch']['min_hz']))
    if hi <= lo: return 0.0, 0.0
    lag = int(np.argmax(ac[lo:hi+1])+lo); conf = float(ac[lag]); f0 = FS/lag
    if conf < float(feat_cfg['pitch']['confidence_threshold']): return 0.0, conf
    return float(f0), conf


def extract(path):
    y = read_float(path); frames = preprocess(y)
    rms = np.sqrt(np.mean(frames**2, axis=1)+1e-12)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
    spec = np.abs(np.fft.rfft(frames, axis=1))+1e-12; power = spec**2
    den = np.sum(power, axis=1)+1e-12
    centroid = np.sum(power*FREQ[None, :], axis=1)/den
    bw = np.sqrt(np.sum(power*(FREQ[None, :]-centroid[:, None])**2, axis=1)/den)
    cum = np.cumsum(power, axis=1); total = cum[:, -1]
    roll = np.array([FREQ[min(np.searchsorted(cum[i], .85*total[i]), len(FREQ)-1)] for i in range(len(frames))])
    pitch = np.array([pitch_one(f)[0] for f in frames])
    voiced = pitch > 0
    mel = np.maximum(power@FB.T, 1e-12); logmel = np.log(mel)
    mfcc = dct(logmel, type=2, axis=1, norm='ortho')[:, :N_MFCC]
    vals = []; names = []
    for j in range(N_MFCC):
        vals += [float(np.mean(mfcc[:, j])), float(np.std(mfcc[:, j]))]
        names += [f'mfcc{j+1:02d}_mean', f'mfcc{j+1:02d}_std']
    for name, a in [('rms', rms), ('zcr', zcr), ('centroid_hz', centroid), ('bandwidth_hz', bw), ('rolloff85_hz', roll)]:
        vals += [float(np.mean(a)), float(np.std(a))]; names += [name+'_mean', name+'_std']
    pv = pitch[voiced]
    vals += [float(np.mean(pv)) if len(pv) else 0.0, float(np.std(pv)) if len(pv) else 0.0, float(np.mean(voiced))]
    names += ['f0_hz_mean', 'f0_hz_std', 'voiced_ratio']
    assert len(vals) == 39
    return np.asarray(vals, np.float32), np.asarray(names, dtype=object), logmel.astype(np.float32), len(y)/FS

# %% [markdown]
# ## 1) Akuisisi 6 block × 4 kelas

# %%
records = []
for block in range(1, N_BLOCKS+1):
    print(f'--- Acquisition block {block}/{N_BLOCKS} ---')
    for label in LABELS:
        p = acquire(label, block)
        if ACQUISITION_MODE == 'tts':
            subj = 'tts_' + BLOCK_PROFILE[block]['voice'].split()[1].lower()
            dev, dist = 'sapi_tts_simulated_mic', f"gain{BLOCK_PROFILE[block]['gain_db']:+.0f}dB"
        elif ACQUISITION_MODE == 'demo':
            subj, dev, dist = 'synthetic_demo', 'none', 'normal'
        else:
            subj, dev, dist = 'personal_user', 'local_mic', 'normal'
        records.append({'sample_id': p.stem, 'label': label, 'group_id': f'block_{block:02d}',
                        'subject_id': subj, 'device_id': dev, 'distance_tag': dist,
                        'sample_rate_hz': FS, 'path': p.relative_to(ROOT).as_posix()})
    if ACQUISITION_MODE == 'mic' and block < N_BLOCKS:
        print('Istirahat 5–10 detik / ubah posisi mikrofon sedikit sebelum block berikutnya.')
        time.sleep(2)

# %% [markdown]
# ## 2) Ekstraksi feature schema MP8 untuk seluruh sampel + simpan log-Mel

# %%
X = []; y = []; groups = []; ids = []; paths = []; durations = []; feature_names = None
for r in records:
    vec, names, logmel, dur = extract(ROOT/r['path'])
    if feature_names is None: feature_names = names
    else: assert np.array_equal(feature_names, names)
    X.append(vec); y.append(r['label']); groups.append(r['group_id']); ids.append(r['sample_id']); paths.append(r['path']); durations.append(dur)
    np.savez_compressed(logmel_dir/f"{r['sample_id']}.npz", log_mel=logmel, sample_id=r['sample_id'], label=r['label'])
    r['duration_s'] = round(dur, 4)                                   # durasi rekaman mentah (kode buku)
    r['active_duration_s'] = round(((logmel.shape[0]-1)*H + N)/FS, 4)   # durasi setelah VAD trim

X = np.vstack(X); y = np.asarray(y, dtype=object); groups = np.asarray(groups, dtype=object); ids = np.asarray(ids, dtype=object); paths = np.asarray(paths, dtype=object)
assert X.shape == (N_BLOCKS*len(LABELS), 39)
assert np.isfinite(X).all()
assert list(feature_names) == list(feat_cfg.get('feature_names', feature_names)), 'Urutan 39 fitur berbeda dari MP8'

# %% [markdown]
# ## 3) Grouped split: block 1–4 train, 5 validation, 6 test

# %%
train_mask = np.isin(groups, [f'block_{i:02d}' for i in range(1, 5)])
val_mask = groups == 'block_05'; test_mask = groups == 'block_06'
assert not np.any(train_mask & val_mask) and not np.any(train_mask & test_mask)
assert set(groups[train_mask]).isdisjoint(set(groups[val_mask]))
assert set(groups[train_mask]).isdisjoint(set(groups[test_mask]))


def ids_of(mask): return ids[mask].tolist()


split = {'strategy': 'grouped_by_acquisition_block', 'seed': SEED, 'train_ids': ids_of(train_mask), 'validation_ids': ids_of(val_mask), 'test_ids': ids_of(test_mask),
         'train_groups': sorted(set(groups[train_mask])), 'validation_groups': sorted(set(groups[val_mask])), 'test_groups': sorted(set(groups[test_mask]))}
(split_dir/'mp09_split.json').write_text(json.dumps(split, indent=2), encoding='utf-8')

# %% [markdown]
# ## 4) Metadata CSV dan dataset matrix

# %%
with (meta_dir/'mp09_metadata.csv').open('w', newline='', encoding='utf-8') as f:
    cols = ['sample_id', 'label', 'group_id', 'subject_id', 'device_id', 'distance_tag', 'sample_rate_hz', 'duration_s', 'path', 'split']
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in records:
        sid = r['sample_id']; sp = 'train' if sid in split['train_ids'] else ('validation' if sid in split['validation_ids'] else 'test')
        row = dict(r); row['split'] = sp; w.writerow({k: row[k] for k in cols})
np.savez_compressed(dataset_dir/'mp09_features_39d.npz', X=X, y=y, groups=groups, sample_ids=ids, paths=paths, feature_names=feature_names,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

# %% [markdown]
# ## 5) Config, metrik, visualisasi

# %%
cfg = {'sampling_rate_hz': FS, 'labels': LABELS, 'n_blocks': N_BLOCKS, 'samples_per_class': N_BLOCKS, 'total_samples': len(ids),
       'record_duration_s': DURATION_S, 'group_key': 'group_id', 'split_strategy': 'blocks_01_04_train__05_validation__06_test', 'seed': SEED,
       'feature_schema_source': 'config/feature_config.json', 'feature_dim': 39, 'augmentation': 'NONE_in_MP09__training_only_later',
       'acquisition_mode': ACQUISITION_MODE,
       'tts_block_profile': BLOCK_PROFILE if ACQUISITION_MODE == 'tts' else None}
(ROOT/'config/dataset_config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')


def class_counts(mask): return {lab: int(np.sum(y[mask] == lab)) for lab in LABELS}


metrics = {'n_samples': int(len(ids)), 'feature_dim': int(X.shape[1]), 'finite_features': bool(np.isfinite(X).all()),
           'class_counts_all': class_counts(np.ones(len(y), dtype=bool)), 'class_counts_train': class_counts(train_mask),
           'class_counts_validation': class_counts(val_mask), 'class_counts_test': class_counts(test_mask),
           'n_train': int(train_mask.sum()), 'n_validation': int(val_mask.sum()), 'n_test': int(test_mask.sum()),
           'group_overlap_train_val': sorted(set(groups[train_mask]) & set(groups[val_mask])),
           'group_overlap_train_test': sorted(set(groups[train_mask]) & set(groups[test_mask])),
           'record_duration_s': float(np.mean(durations)),
           'active_duration_after_vad_s': {k: float(fn([r['active_duration_s'] for r in records])) for k, fn in [('min', np.min), ('mean', np.mean), ('max', np.max)]},
           'logmel_frames': {k: int(fn([np.load(logmel_dir/f"{r['sample_id']}.npz")['log_mel'].shape[0] for r in records])) for k, fn in [('min', np.min), ('max', np.max)]},
           'acquisition_mode': ACQUISITION_MODE}
(metric_dir/'mp09_dataset_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(LABELS, [metrics['class_counts_all'][l] for l in LABELS]); ax.set_ylabel('samples'); ax.set_title('MP9 class distribution')
ax.grid(axis='y', alpha=.2); fig.tight_layout(); fig.savefig(fig_dir/'mp09_class_distribution.png', dpi=160); plt.close(fig)
fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(['train', 'validation', 'test'], [train_mask.sum(), val_mask.sum(), test_mask.sum()]); ax.set_ylabel('samples'); ax.set_title('MP9 grouped split')
ax.grid(axis='y', alpha=.2); fig.tight_layout(); fig.savefig(fig_dir/'mp09_split_distribution.png', dpi=160); plt.close(fig)

# %% [markdown]
# ## 6) Handoff ke MP10

# %%
status.update({'status': 'ready_for_mp10', 'last_completed_mini_project': 9,
               'mp10_dataset': 'features/dataset/mp09_features_39d.npz', 'mp10_split': 'splits/mp09_split.json',
               'dataset_config': 'config/dataset_config.json', 'dataset_metadata': 'data/metadata/mp09_metadata.csv', 'feature_config': 'config/feature_config.json'})
status.setdefault('completed_mini_projects', [])
for k in range(7, 10):
    if k not in status['completed_mini_projects']: status['completed_mini_projects'].append(k)
STATUS.write_text(json.dumps(status, indent=2), encoding='utf-8')

print('MP9 selesai:', X.shape, 'train/val/test =', train_mask.sum(), val_mask.sum(), test_mask.sum())
print('Class counts  :', metrics['class_counts_all'])
print('Group overlap :', metrics['group_overlap_train_val'], metrics['group_overlap_train_test'])
print('Durasi aktif  :', {k: round(v, 3) for k, v in metrics['active_duration_after_vad_s'].items()}, 's (rekaman', DURATION_S, 's)')
print('Frame log-Mel :', metrics['logmel_frames'])
print('status = ready_for_mp10')
