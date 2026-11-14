# %% [markdown]
# # Mini Project 16 — Integrasi End-to-End dan Acceptance Test Proyek Final
# Artefak beku MP15 dijalankan sebagai **satu sistem** (WAV → filter MP6 → preprocessing MP7 → feature MP8 →
# seleksi MP10 → model INT8 MP15 → aksi), konsistensinya diverifikasi terhadap prediksi MP13, latency diukur,
# acceptance report dibuat, lalu status serial ditutup menjadi `project_complete`.
#
# Adaptasi terhadap kode buku:
# * **Gate family-aware**: buku mewajibkan `classical_svm_int8`; MP15 running project ini (dengan policy buku) memilih
#   `classical_tree_int8`, sehingga gate menerima model klasik INT8 apa pun yang tercantum di manifest MP15 dan
#   `predict_int8` memilih jalur linear/tree sesuai `kind` di paket `mp15_classical_int8.npz`.
#   Kunci metrik MP15 `svm_int8` → `classical_int8` (dibaca dengan fallback).
# * `LIVE_MIC` dibaca dari environment `MP16_LIVE_MIC=1` (default False → replay acceptance) agar notebook tetap non-interaktif.
#   Live demo dijalankan di terminal: `set MP16_LIVE_MIC=1` lalu `python mini_project_16.py`.
# * Aksi (nyala=ON, mati=OFF, naik=level+, turun=level−) dengan batas 0..10 dan fail-safe ditambahkan di bagian replay (Langkah 5).

# %%
# Anggota Kelompok :
# 1. Christama Ezra Yudianto    (245150307111009)
# 2. Abdillah Fatah Suryanto    (245150300111008)
# 3. Ivang Arfandia             (245150307111003)
# 4. Jahzeel Erason Nicodemus   (245150300111002)
from pathlib import Path
import csv, json, time, hashlib, os
import numpy as np
import matplotlib
if "__file__" in globals():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import lfilter
from scipy.fft import dct

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
LIVE_MIC = os.environ.get('MP16_LIVE_MIC', '0') == '1'      # True pada Raspberry Pi/PC dengan mikrofon
LIVE_DURATION_S = 1.2
SEED = 2026
np.random.seed(SEED)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(65536), b''):
            h.update(block)
    return h.hexdigest()

# %% [markdown]
# ## 1. Gate serial MP15 → MP16

# %%
status_path = ROOT/'project_status.json'
status = json.loads(status_path.read_text(encoding='utf-8'))
if status.get('status') not in ('ready_for_mp16', 'project_complete'):
    raise RuntimeError(f"Status harus ready_for_mp16, ditemukan {status.get('status')!r}")

manifest = json.loads((ROOT/'deployment/mp15_manifest.json').read_text(encoding='utf-8'))
dep_cfg = json.loads((ROOT/'config/deployment_config.json').read_text(encoding='utf-8'))
rt_cfg = json.loads((ROOT/'config/realtime_architecture_config.json').read_text(encoding='utf-8'))
proj_cfg = json.loads((ROOT/'config/project_config.json').read_text(encoding='utf-8'))
pre_cfg = json.loads((ROOT/'config/preprocessing_config.json').read_text(encoding='utf-8'))
feat_cfg = json.loads((ROOT/'config/feature_config.json').read_text(encoding='utf-8'))
mp13 = json.loads((ROOT/'results/metrics/mp13_final_test_metrics.json').read_text(encoding='utf-8'))
mp14 = json.loads((ROOT/'results/metrics/mp14_realtime_metrics.json').read_text(encoding='utf-8'))
mp15 = json.loads((ROOT/'results/metrics/mp15_deployment_metrics.json').read_text(encoding='utf-8'))
split = json.loads((ROOT/'splits/mp09_split.json').read_text(encoding='utf-8'))
mp15_cls = mp15.get('classical_int8', mp15.get('svm_int8'))

if manifest['selected_model'] != dep_cfg['selected_deployment_model']:
    raise RuntimeError('Manifest MP15 dan deployment_config tidak konsisten.')
if dep_cfg.get('post_test_tuning_performed'):
    raise RuntimeError('Post-test tuning terdeteksi. Integrasi dihentikan.')
if not (manifest['selected_model'].startswith('classical_') and manifest['selected_model'].endswith('_int8')):
    raise RuntimeError('Script MP16 ini memakai jalur utama model klasik INT8 yang dipilih MP15.')
print('Gate OK | selected model =', manifest['selected_model'], '| family =', manifest.get('classical_family'))

# %% [markdown]
# ## 2. Front-end beku + model klasik INT8

# %%
FS = int(manifest['sample_rate_hz'])
LABELS = list(manifest['classes'])
filter_npz = np.load(ROOT/'features/intermediate/mp06_filter_final.npz', allow_pickle=True)
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

pkg = np.load(ROOT/'models/mp15_classical_int8.npz', allow_pickle=True)
KIND = str(pkg['kind']) if 'kind' in pkg else 'linear'
SEL = pkg['selected_indices'].astype(int)
MU = pkg['selected_mean'].astype(float)
SIG = pkg['selected_scale'].astype(float)
XQ_SCALE = float(pkg['input_quant_scale'][0])
CLASSES = pkg['classes'].astype(str).tolist()
assert CLASSES == LABELS
if KIND == 'linear':
    WQ = pkg['weight_q'].astype(np.int8)
    W_SCALE = pkg['weight_scales'].astype(float)
    BIAS = pkg['bias'].astype(float)
else:
    T_FEAT = pkg['tree_feature'].astype(int); T_THR = pkg['tree_threshold_q'].astype(int)
    T_LEFT = pkg['tree_left'].astype(int); T_RIGHT = pkg['tree_right'].astype(int); T_LEAF = pkg['tree_leaf_class'].astype(int)
print('Paket INT8:', KIND, '| input scale', XQ_SCALE, '| SHA-256 header =', sha256(ROOT/'firmware/mp15_classical_model_int8.h')[:16], '...')


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


def read_float(path):
    fs, y = wavfile.read(path)
    if fs != FS: raise RuntimeError(f'{path}: sample rate {fs}, seharusnya {FS}')
    if y.ndim > 1: y = y.mean(axis=1)
    if np.issubdtype(y.dtype, np.integer): y = y.astype(np.float64)/np.iinfo(y.dtype).max
    else: y = y.astype(np.float64)
    return y


def preprocess(y):
    y = lfilter(B, A, y)
    y = y-np.mean(y)
    y = y*(TARGET_PEAK/(np.max(np.abs(y))+1e-12))
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
    if len(y) < N: y = np.pad(y, (0, N-len(y)))
    starts = np.arange(0, len(y)-N+1, H)
    fr = np.array([y[s:s+N] for s in starts])
    fr = np.concatenate([fr[:, :1], fr[:, 1:]-PREEMPH*fr[:, :-1]], axis=1)
    return fr*np.hamming(N)[None, :]


def pitch_one(frame):
    x = frame-np.mean(frame); ac = np.correlate(x, x, mode='full')[len(x)-1:]
    if ac[0] <= 1e-12: return 0.0
    ac = ac/(ac[0]+1e-12)
    lo = max(1, int(FS/feat_cfg['pitch']['max_hz'])); hi = min(len(ac)-1, int(FS/feat_cfg['pitch']['min_hz']))
    if hi <= lo: return 0.0
    lag = int(np.argmax(ac[lo:hi+1])+lo); conf = float(ac[lag])
    if conf < float(feat_cfg['pitch']['confidence_threshold']): return 0.0
    return float(FS/lag)


def extract39(y):
    frames = preprocess(y)
    rms = np.sqrt(np.mean(frames**2, axis=1)+1e-12)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
    spec = np.abs(np.fft.rfft(frames, axis=1))+1e-12; power = spec**2
    den = np.sum(power, axis=1)+1e-12
    centroid = np.sum(power*FREQ[None, :], axis=1)/den
    bw = np.sqrt(np.sum(power*(FREQ[None, :]-centroid[:, None])**2, axis=1)/den)
    cum = np.cumsum(power, axis=1); total = cum[:, -1]
    roll = np.array([FREQ[min(np.searchsorted(cum[i], .85*total[i]), len(FREQ)-1)] for i in range(len(frames))])
    pitch = np.array([pitch_one(f) for f in frames]); voiced = pitch > 0
    mel = np.maximum(power@FB.T, 1e-12); logmel = np.log(mel)
    mfcc = dct(logmel, type=2, axis=1, norm='ortho')[:, :N_MFCC]
    vals = []
    for j in range(N_MFCC): vals += [float(np.mean(mfcc[:, j])), float(np.std(mfcc[:, j]))]
    for a in [rms, zcr, centroid, bw, roll]: vals += [float(np.mean(a)), float(np.std(a))]
    pv = pitch[voiced]
    vals += [float(np.mean(pv)) if len(pv) else 0.0, float(np.std(pv)) if len(pv) else 0.0, float(np.mean(voiced))]
    v = np.asarray(vals, dtype=np.float32)
    if v.shape != (39,) or not np.isfinite(v).all(): raise RuntimeError('Feature vector MP16 tidak valid.')
    return v


def predict_int8(v39):
    z = (v39[SEL]-MU)/SIG
    zq = np.clip(np.round(z/XQ_SCALE), -127, 127).astype(np.int8)
    if KIND == 'linear':
        acc = WQ.astype(np.int32)@zq.astype(np.int32)
        scores = acc.astype(float)*(W_SCALE*XQ_SCALE)+BIAS
        k = int(np.argmax(scores))
        return CLASSES[k], scores
    n = 0; path = []
    while T_FEAT[n] >= 0:
        path.append(n); n = T_LEFT[n] if int(zq[T_FEAT[n]]) <= T_THR[n] else T_RIGHT[n]
    scores = np.zeros(len(CLASSES)); scores[T_LEAF[n]] = 1.0      # tree: one-hot leaf (tanpa skor kontinu)
    return CLASSES[int(T_LEAF[n])], scores

# %% [markdown]
# ## 3. Replay acceptance pada WAV test beku (+ aksi console dengan batas & fail-safe)

# %%
meta = {}
with (ROOT/'data/metadata/mp09_metadata.csv').open(encoding='utf-8') as f:
    for r in csv.DictReader(f): meta[r['sample_id']] = r
mp13_pred = {}
with (ROOT/'results/predictions/mp13_test_predictions.csv').open(encoding='utf-8') as f:
    for r in csv.DictReader(f): mp13_pred[r['sample_id']] = r

LEVEL_MIN, LEVEL_MAX = 0, 10
device = {'power': False, 'level': 5}


def apply_action(cmd, dev):
    """nyala=ON, mati=OFF, naik=level+, turun=level-. Level hanya berubah saat ON; batas 0..10 (fail-safe: clamp)."""
    if cmd == 'nyala': dev['power'] = True
    elif cmd == 'mati': dev['power'] = False
    elif cmd == 'naik' and dev['power']: dev['level'] = min(LEVEL_MAX, dev['level']+1)
    elif cmd == 'turun' and dev['power']: dev['level'] = max(LEVEL_MIN, dev['level']-1)
    return dict(dev)


rows = []; feature_ms = []; infer_ms = []; total_ms = []
for sid in split['test_ids']:
    wav_path = ROOT/meta[sid]['path']
    y = read_float(wav_path)
    t0 = time.perf_counter_ns(); v = extract39(y); t1 = time.perf_counter_ns(); pred, scores = predict_int8(v); t2 = time.perf_counter_ns()
    fm = (t1-t0)/1e6; im = (t2-t1)/1e6; tm = (t2-t0)/1e6
    expected = meta[sid]['label']; frozen = mp13_pred[sid]['classical_prediction']
    st = apply_action(pred, device)
    rows.append({'sample_id': sid, 'expected': expected, 'mp13_classical_pred': frozen, 'mp16_int8_pred': pred, 'correct': pred == expected,
                 'agreement_mp13': pred == frozen, 'feature_ms': fm, 'inference_ms': im, 'total_ms': tm, 'audio_duration_ms': len(y)/FS*1000,
                 'action_power': st['power'], 'action_level': st['level']})
    feature_ms.append(fm); infer_ms.append(im); total_ms.append(tm)
    print(f"  {sid}: expected={expected:6s} mp13={frozen:6s} mp16={pred:6s} | {tm:6.2f} ms | aksi -> power={'ON' if st['power'] else 'OFF'} level={st['level']}")

# %% [markdown]
# ## 4. Live microphone demo (opsional, `MP16_LIVE_MIC=1`)

# %%
live_rows = []
if LIVE_MIC:
    import sounddevice as sd
    for expected in ['nyala', 'mati', 'naik', 'turun']:
        input(f'Ucapkan "{expected}" lalu tekan ENTER... ')
        y = sd.rec(int(LIVE_DURATION_S*FS), samplerate=FS, channels=1, dtype='float32'); sd.wait(); y = y[:, 0].astype(float)
        t0 = time.perf_counter_ns(); v = extract39(y); pred, scores = predict_int8(v); elapsed = (time.perf_counter_ns()-t0)/1e6
        live_rows.append({'expected': expected, 'predicted': pred, 'correct': pred == expected, 'pipeline_ms': elapsed})
        print(f'  {expected} -> {pred} ({elapsed:.2f} ms)')
else:
    print('LIVE_MIC=False -> replay acceptance saja. Untuk live demo: set MP16_LIVE_MIC=1 lalu jalankan skrip ini di terminal.')

# %% [markdown]
# ## 5. Acceptance metrics dan paket final

# %%
replay_accuracy = float(np.mean([r['correct'] for r in rows]))
agreement = float(np.mean([r['agreement_mp13'] for r in rows]))
rtf_p95 = float(np.percentile(total_ms, 95)/(LIVE_DURATION_S*1000))
stride_ms = float(rt_cfg['inference_stride_ms'])

checks = {
    'serial_gate_ready_for_mp16': True,
    'selected_model_matches_manifest': True,
    'post_test_tuning_false': not dep_cfg.get('post_test_tuning_performed', False),
    'mp15_numeric_agreement_test_ge_095': float(mp15_cls['agreement_test_vs_float']) >= 0.95,
    'mp14_overflow_is_0': int(mp14['overflow_count']) == 0,
    'replay_agreement_not_worse_than_mp15': agreement+1e-9 >= float(mp15_cls['agreement_test_vs_float']),
    'replay_features_are_finite': True,
    'p95_pipeline_faster_than_audio_duration': rtf_p95 < 1.0,
}
# Live microphone is informative, not a hard gate because acoustics and hardware vary.
all_required_pass = all(checks.values())

metrics = {
    'selected_model': manifest['selected_model'],
    'target_profile': dep_cfg['primary_target_profile'],
    'n_replay_samples': len(rows),
    'replay_accuracy': replay_accuracy,
    'replay_agreement_with_mp13': agreement,
    'feature_latency_ms': {'mean': float(np.mean(feature_ms)), 'p95': float(np.percentile(feature_ms, 95))},
    'inference_latency_ms': {'mean': float(np.mean(infer_ms)), 'p95': float(np.percentile(infer_ms, 95))},
    'pipeline_latency_ms': {'mean': float(np.mean(total_ms)), 'p95': float(np.percentile(total_ms, 95))},
    'pipeline_rtf_p95': rtf_p95,
    'realtime_stride_ms': stride_ms,
    'mp14_overflow_count': int(mp14['overflow_count']),
    'mp15_payload_bytes': int(mp15_cls['payload_bytes_estimate']),
    'live_microphone_enabled': LIVE_MIC,
    'live_demo': live_rows,
    'acceptance_checks': checks,
    'required_acceptance_pass': all_required_pass,
    'benchmark_platform': 'host PC (Windows, Python reference) - BUKAN angka ESP32/Raspberry Pi',
    'scientific_note': 'Replay test verifies end-to-end implementation consistency; it is not a new independent generalization estimate.'
}

out_pred = ROOT/'results/predictions/mp16_end_to_end_predictions.csv'; out_pred.parent.mkdir(parents=True, exist_ok=True)
with out_pred.open('w', newline='', encoding='utf-8') as f:
    fields = list(rows[0].keys()); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
save_json(ROOT/'results/metrics/mp16_acceptance_metrics.json', metrics)

syscfg = {
    'project': 'four_command_keyword_spotting_embedded',
    'classes': LABELS,
    'sample_rate_hz': FS,
    'input': 'mono_PCM16_or_float_equivalent',
    'dma_block_samples': int(manifest['dma_block_samples']),
    'frame_samples': int(manifest['feature_frame_samples']),
    'hop_samples': int(manifest['feature_hop_samples']),
    'inference_stride_frames': int(manifest['inference_stride_frames']),
    'selected_model': manifest['selected_model'],
    'selected_features': manifest['selected_feature_names'],
    'decision_map': {'nyala': 'LED/relay ON', 'mati': 'LED/relay OFF', 'naik': 'increase level', 'turun': 'decrease level'},
    'level_bounds': [LEVEL_MIN, LEVEL_MAX],
    'logging_fields': ['timestamp', 'predicted_class', 'score', 'pipeline_latency_ms', 'overflow_count'],
    'live_python_target': 'Raspberry Pi / PC Linux',
    'esp32_package': 'firmware/mp15_classical_model_int8.h',
    'no_post_test_tuning': True,
}
save_json(ROOT/'config/system_integration_config.json', syscfg)

# MCU constants and simple action mapping.
fw = ROOT/'firmware/mp16_project_config.h'; fw.parent.mkdir(parents=True, exist_ok=True)
fw.write_text(f'''// Generated by Mini Project 16
#pragma once
#define MP16_SAMPLE_RATE {FS}
#define MP16_DMA_BLOCK {manifest['dma_block_samples']}
#define MP16_FRAME_SAMPLES {manifest['feature_frame_samples']}
#define MP16_HOP_SAMPLES {manifest['feature_hop_samples']}
#define MP16_INFERENCE_STRIDE_FRAMES {manifest['inference_stride_frames']}
#define MP16_N_CLASSES {len(LABELS)}
#define MP16_LEVEL_MIN {LEVEL_MIN}
#define MP16_LEVEL_MAX {LEVEL_MAX}
// Class order: {', '.join(LABELS)}
''', encoding='utf-8')

final_manifest = {
    'project_complete': all_required_pass,
    'selected_model': manifest['selected_model'],
    'primary_runtime': 'Raspberry Pi/Python reference or ESP32/C++ port using frozen header',
    'model_package': 'models/mp15_classical_int8.npz',
    'model_c_header': 'firmware/mp15_classical_model_int8.h',
    'model_c_header_sha256': sha256(ROOT/'firmware/mp15_classical_model_int8.h'),
    'project_c_header': 'firmware/mp16_project_config.h',
    'system_config': 'config/system_integration_config.json',
    'acceptance_metrics': 'results/metrics/mp16_acceptance_metrics.json',
    'prediction_audit': 'results/predictions/mp16_end_to_end_predictions.csv',
    'frozen_test_metrics': 'results/metrics/mp13_final_test_metrics.json',
    'deployment_metrics': 'results/metrics/mp15_deployment_metrics.json',
    'source': ['mini_project_16.py', 'features/kws_frontend.py'],
    'post_test_tuning_performed': False,
}
save_json(ROOT/'deployment/mp16_final_project_manifest.json', final_manifest)

# %% [markdown]
# ## 6. Visualisasi

# %%
figdir = ROOT/'results/figures'; figdir.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.bar(['feature', 'inference', 'total'], [np.mean(feature_ms), np.mean(infer_ms), np.mean(total_ms)])
ax.set_ylabel('mean latency (ms)'); ax.set_title('MP16 end-to-end reference latency'); ax.grid(axis='y', alpha=.2)
fig.tight_layout(); fig.savefig(figdir/'mp16_latency_breakdown.png', dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 4.0))
labels = list(checks.keys()); vals = [1 if checks[k] else 0 for k in labels]
ax.barh(range(len(labels)), vals, color=['C2' if v else 'C3' for v in vals]); ax.set_yticks(range(len(labels))); ax.set_yticklabels([x.replace('_', ' ') for x in labels], fontsize=7)
ax.set_xlim(0, 1.05); ax.set_xlabel('PASS = 1'); ax.set_title('MP16 acceptance checklist'); ax.grid(axis='x', alpha=.2)
fig.tight_layout(); fig.savefig(figdir/'mp16_acceptance_summary.png', dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.4, 4.0))
x = np.arange(len(rows)); ax.plot(x, [r['total_ms'] for r in rows], marker='o', label='pipeline')
ax.axhline(LIVE_DURATION_S*1000, ls='--', label='audio duration'); ax.set_xticks(x); ax.set_xticklabels([r['sample_id'] for r in rows], rotation=25, ha='right', fontsize=8)
ax.set_ylabel('ms'); ax.set_yscale('log'); ax.set_title('MP16 replay processing time'); ax.legend(); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(figdir/'mp16_command_timeline.png', dpi=160); plt.close(fig)

# %% [markdown]
# ## 7. Status serial akhir

# %%
status.update({
    'status': 'project_complete',
    'last_completed_mini_project': 16,
    'mp16': {
        'system_config': 'config/system_integration_config.json',
        'acceptance_metrics': 'results/metrics/mp16_acceptance_metrics.json',
        'final_manifest': 'deployment/mp16_final_project_manifest.json',
        'prediction_audit': 'results/predictions/mp16_end_to_end_predictions.csv',
        'selected_model': manifest['selected_model'],
        'required_acceptance_pass': all_required_pass,
        'post_test_tuning_performed': False
    }
})
status.setdefault('completed_mini_projects', [])
if 16 not in status['completed_mini_projects']: status['completed_mini_projects'].append(16)
status_path.write_text(json.dumps(status, indent=2), encoding='utf-8')

print('MP16 selesai.')
print('selected model =', manifest['selected_model'])
print('replay accuracy =', replay_accuracy)
print('agreement with MP13 =', agreement)
print('p95 pipeline = %.3f ms, RTF = %.4f' % (metrics['pipeline_latency_ms']['p95'], rtf_p95))
for k, v in checks.items(): print(f"  [{'PASS' if v else 'FAIL'}] {k}")
print('acceptance =', 'PASS' if all_required_pass else 'FAIL')
print('status = project_complete')

# %%
if "__file__" not in globals():
    from IPython.display import Image, display
    for p in ['mp16_latency_breakdown.png', 'mp16_acceptance_summary.png', 'mp16_command_timeline.png']:
        display(Image(filename=str(figdir/p), width=600))
