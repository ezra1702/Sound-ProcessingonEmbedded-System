# %% [markdown]
# # Mini Project 14 — Menyusun dan Menguji Scheduler Streaming Real-Time
# Artefak MP13 diubah menjadi **kontrak arsitektur streaming** (DMA double-buffer 10 ms, frame 25 ms,
# hop 10 ms, context 96 frame = 975 ms, inference stride 100 ms) yang akan dipindahkan ke perangkat di MP15.
# Model, feature schema, dan split **tidak diubah**. Prediksi selama simulasi hanya *sanity check* aliran data.
# Status -> `ready_for_mp15`.
#
# Adaptasi terhadap kode buku (tanpa mengubah logika):
# * `project_config.json` running project memakai kunci `sample_rate_hz` (buku: `sampling_rate_hz`) -> dibaca dengan fallback.
# * ROOT dihitung dari lokasi skrip (folder `Mini Project 14`) agar bisa dijalankan dari mana pun / dari notebook.
# * Pemeriksaan status menerima `ready_for_mp14` atau `ready_for_mp15` (rerun idempoten).

# %%
# Anggota Kelompok :
# 1. Christama Ezra Yudianto    (245150307111009)
# 2. Abdillah Fatah Suryanto    (245150300111008)
# 3. Ivang Arfandia             (245150307111003)
# 4. Jahzeel Erason Nicodemus   (245150300111002)
from pathlib import Path
import json, time
import numpy as np
import matplotlib
if "__file__" in globals():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import lfilter, lfilter_zi
from scipy.fft import dct
import joblib
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
STATUS = ROOT/'project_status.json'
status = json.loads(STATUS.read_text(encoding='utf-8'))
if status.get('status') not in ('ready_for_mp14', 'ready_for_mp15'):
    raise RuntimeError(f"Status harus ready_for_mp14, ditemukan {status.get('status')!r}")
print('Status awal:', status['status'])

# %% [markdown]
# ## 1. Konfigurasi warisan MP1–MP13 dan kontrak timing

# %%
project_cfg = json.loads((ROOT/'config/project_config.json').read_text(encoding='utf-8'))
pre_cfg = json.loads((ROOT/'config/preprocessing_config.json').read_text(encoding='utf-8'))
feat_cfg = json.loads((ROOT/'config/feature_config.json').read_text(encoding='utf-8'))
red_cfg = json.loads((ROOT/'config/feature_reduction_config.json').read_text(encoding='utf-8'))
clf_cfg = json.loads((ROOT/'config/classifier_config.json').read_text(encoding='utf-8'))
dl_cfg = json.loads((ROOT/'config/deep_learning_config.json').read_text(encoding='utf-8'))
mp13 = json.loads((ROOT/'results/metrics/mp13_final_test_metrics.json').read_text(encoding='utf-8'))

fs = int(project_cfg.get('sampling_rate_hz', project_cfg.get('sample_rate_hz')))
frame_len = int(pre_cfg['frame_length_samples'])
hop_len = int(pre_cfg['hop_length_samples'])
frame_ms = 1000.0 * frame_len / fs
hop_ms = 1000.0 * hop_len / fs
n_mels = int(feat_cfg['mel']['n_mels'])
n_mfcc = int(feat_cfg['mfcc']['n_mfcc'])
context_frames = int(dl_cfg['input_shape_per_sample'][0])
context_ms = frame_ms + (context_frames - 1) * hop_ms
inference_stride_frames = 10   # 100 ms at the running-project hop
inference_stride_ms = inference_stride_frames * hop_ms

dma_block_samples = hop_len
buffer_period_ms = 1000.0 * dma_block_samples / fs
ring_capacity_blocks = 4
print(f'fs={fs} Hz | frame={frame_len} ({frame_ms} ms) | hop={hop_len} ({hop_ms} ms) | '
      f'context={context_frames} frame ({context_ms} ms) | stride={inference_stride_frames} frame ({inference_stride_ms} ms) | '
      f'DMA={dma_block_samples} sampel ({buffer_period_ms} ms)')

# %% [markdown]
# ## 2. Helper feature per-frame (streaming) + agregasi 39-D

# %%
def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0**(np.asarray(m) / 2595.0) - 1.0)


def make_mel_filterbank(n_fft, fs, n_mels=26, fmin=80.0, fmax=7600.0):
    n_bins = n_fft // 2 + 1
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / fs).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m-1], bins[m], bins[m+1]
        center = max(center, left + 1)
        right = max(right, center + 1)
        right = min(right, n_bins - 1)
        for k in range(left, min(center, n_bins)):
            fb[m-1, k] = (k - left) / max(1, center - left)
        for k in range(center, min(right + 1, n_bins)):
            fb[m-1, k] = (right - k) / max(1, right - center)
    return fb


freq = np.fft.rfftfreq(frame_len, d=1.0/fs)
fb = make_mel_filterbank(frame_len, fs, n_mels, float(feat_cfg['mel']['fmin_hz']), min(float(feat_cfg['mel']['fmax_hz']), 0.95*fs/2))
window = np.hamming(frame_len)
lag_min = max(1, int(fs / float(feat_cfg['pitch']['max_hz'])))
lag_max = min(frame_len - 2, int(fs / float(feat_cfg['pitch']['min_hz'])))


def frame_features(frame):
    x = np.asarray(frame, dtype=np.float64)
    x = x - np.mean(x)
    peak = np.max(np.abs(x)) + 1e-12
    x = x * min(1.0, 0.95 / peak)
    alpha = float(pre_cfg['pre_emphasis_alpha'])
    x = np.r_[x[0], x[1:] - alpha*x[:-1]]
    xw = x * window
    rms = float(np.sqrt(np.mean(xw*xw) + 1e-12))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(xw)))))
    X = np.fft.rfft(xw)
    power = np.abs(X)**2 + 1e-15
    psum = np.sum(power) + 1e-15
    centroid = float(np.sum(power * freq) / psum)
    bandwidth = float(np.sqrt(np.sum(power * (freq-centroid)**2) / psum))
    cum = np.cumsum(power)
    idx = int(np.searchsorted(cum, 0.85*cum[-1]))
    rolloff = float(freq[min(idx, len(freq)-1)])
    mel = np.maximum(power @ fb.T, 1e-12)
    logmel = np.log(mel).astype(np.float32)
    mfcc = dct(logmel, type=2, norm='ortho')[:n_mfcc].astype(np.float32)
    f0 = 0.0; conf = 0.0
    ac = np.correlate(xw, xw, mode='full')[frame_len-1:]
    if ac[0] > 1e-12:
        ac = ac/ac[0]
        search = ac[lag_min:lag_max+1]
        if len(search):
            lag = int(np.argmax(search)) + lag_min
            conf = float(ac[lag])
            if conf >= float(feat_cfg['pitch']['confidence_threshold']):
                f0 = float(fs/lag)
    return {'logmel': logmel, 'mfcc': mfcc, 'rms': rms, 'zcr': zcr, 'centroid': centroid,
            'bandwidth': bandwidth, 'rolloff': rolloff, 'f0': f0, 'voiced': f0 > 0}


def aggregate_39(hist):
    vals = []
    mfcc = np.stack([h['mfcc'] for h in hist])

    def ms(a, mask=None):
        a = np.asarray(a, dtype=np.float64)
        if mask is not None: a = a[np.asarray(mask, dtype=bool)]
        return (0.0, 0.0) if a.size == 0 else (float(a.mean()), float(a.std()))
    for j in range(n_mfcc): vals.extend(ms(mfcc[:, j]))
    for key in ['rms', 'zcr', 'centroid', 'bandwidth', 'rolloff']:
        vals.extend(ms([h[key] for h in hist]))
    voiced = np.array([h['voiced'] for h in hist], dtype=bool)
    vals.extend(ms([h['f0'] for h in hist], voiced))
    vals.append(float(voiced.mean()))
    return np.asarray(vals, dtype=np.float32)

# %% [markdown]
# ## 3. Muat kedua model kandidat (beku) + warm-up

# %%
class TinyKWSNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Linear(16, n_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


classic_model = joblib.load(ROOT/clf_cfg['model_file'])
ckpt = torch.load(ROOT/dl_cfg['model_file'], map_location='cpu', weights_only=False)
dl_classes = list(ckpt['classes']); mu = float(ckpt['normalization']['mean']); sigma = float(ckpt['normalization']['std'])
tiny = TinyKWSNet(len(dl_classes)); tiny.load_state_dict(ckpt['state_dict']); tiny.eval()
selected = np.asarray(red_cfg['selected_indices'], dtype=int)
scaler_mean = np.asarray(red_cfg['scaler_mean'], dtype=np.float32)
scaler_scale = np.asarray(red_cfg['scaler_scale'], dtype=np.float32)

# Warm-up inference runtimes.
with torch.no_grad():
    dummy = torch.zeros((1, 1, context_frames, n_mels), dtype=torch.float32)
    for _ in range(5): tiny(dummy)
for _ in range(5): classic_model.predict(np.zeros((1, len(selected)), dtype=np.float32))
print('Classical family:', clf_cfg['chosen_family'], '| CNN classes:', dl_classes, '| selected idx:', selected.tolist())

# %% [markdown]
# ## 4. Simulasi streaming DMA block-by-block (virtual CPU + ring buffer 4 blok)
# Input: blok training `nyala_b01.wav` (tidak memakai test untuk desain arsitektur), diulang 4x.

# %%
source = ROOT/'data/raw/mp09_dataset/nyala_b01.wav'  # training block; no reuse of test for architecture design
sr, audio = wavfile.read(source)
if sr != fs: raise RuntimeError(f'sampling rate {sr} != {fs}')
if audio.ndim > 1: audio = audio[:, 0]
if np.issubdtype(audio.dtype, np.integer):
    audio = audio.astype(np.float64)/max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
else: audio = audio.astype(np.float64)
# Repeat to obtain several inference opportunities without collecting new data.
audio = np.tile(audio, 4)

# Final filter from MP6 as a stateful streaming filter.
filt = np.load(ROOT/status['filter_final_coefficients'], allow_pickle=True)
b = np.asarray(filt['b'], dtype=np.float64); a = np.asarray(filt['a'], dtype=np.float64)
zi = lfilter_zi(b, a)*0.0

pcm = []; feature_hist = []
trace = []; classic_ms = []; cnn_ms = []; labels_trace = []
virtual_cpu_ready = 0.0
frame_counter = 0
n_infer = 0

for bi, start in enumerate(range(0, len(audio), dma_block_samples)):
    block = audio[start:start+dma_block_samples]
    if len(block) < dma_block_samples: break
    arrival_ms = bi*buffer_period_ms
    t0 = time.perf_counter()
    y, zi = lfilter(b, a, block, zi=zi)
    pcm.extend(y.tolist())
    if len(pcm) > frame_len: pcm = pcm[-frame_len:]
    did_infer = False; pred_c = ''; pred_d = ''
    if len(pcm) == frame_len:
        ff = frame_features(np.asarray(pcm))
        feature_hist.append(ff)
        if len(feature_hist) > context_frames: feature_hist = feature_hist[-context_frames:]
        frame_counter += 1
        if len(feature_hist) == context_frames and (frame_counter-context_frames) % inference_stride_frames == 0:
            did_infer = True; n_infer += 1
            full = aggregate_39(feature_hist)
            x8 = ((full-scaler_mean)/scaler_scale)[selected][None, :]
            tc = time.perf_counter(); pred_c = str(classic_model.predict(x8)[0]); classic_ms.append((time.perf_counter()-tc)*1000)
            lm = np.stack([h['logmel'] for h in feature_hist]).astype(np.float32)
            x = torch.from_numpy(((lm-mu)/sigma)[None, None, ...]).float()
            td = time.perf_counter()
            with torch.no_grad(): pred_d = dl_classes[int(tiny(x).argmax(1).item())]
            cnn_ms.append((time.perf_counter()-td)*1000)
            labels_trace.append((bi, pred_c, pred_d))
    compute_ms = (time.perf_counter()-t0)*1000
    start_ms = max(arrival_ms, virtual_cpu_ready)
    finish_ms = start_ms+compute_ms
    waiting_ms = start_ms-arrival_ms
    queue_blocks = int(np.floor(waiting_ms/buffer_period_ms+1e-12))
    overflow = queue_blocks >= ring_capacity_blocks
    virtual_cpu_ready = finish_ms
    trace.append([bi, arrival_ms, compute_ms, waiting_ms, queue_blocks, int(did_infer), int(overflow)])

trace = np.asarray(trace, dtype=np.float64)
print('Sanity-check label (block, classical, cnn):', labels_trace)

# %% [markdown]
# ## 5. Memory budget proxy, metrik, konfigurasi arsitektur, header C

# %%
# Memory proxy. MP15 will measure real target memory.
dma_bytes = 2*dma_block_samples*2
pcm_ring_bytes = frame_len*2
logmel_ring_bytes = context_frames*n_mels*4
classical_frame_state_bytes = context_frames*(13+7)*4
# Float32 activation upper bound: first conv output 8*T*F before pooling.
activation_proxy_bytes = 8*context_frames*n_mels*4
cnn_weight_bytes = int(mp13['resource_proxy']['tiny_cnn']['float32_parameter_bytes'])
classical_serialized = int(mp13['resource_proxy']['classical']['serialized_bytes'])

metrics = {
    'source_audio': str(source.relative_to(ROOT).as_posix()),
    'stream_duration_s': float(len(audio)/fs),
    'sample_rate_hz': fs,
    'dma_block_samples': dma_block_samples,
    'dma_block_ms': buffer_period_ms,
    'double_buffer': True,
    'ring_capacity_blocks': ring_capacity_blocks,
    'frame_samples': frame_len,
    'frame_ms': frame_ms,
    'hop_samples': hop_len,
    'hop_ms': hop_ms,
    'context_frames': context_frames,
    'context_ms': context_ms,
    'inference_stride_frames': inference_stride_frames,
    'inference_stride_ms': inference_stride_ms,
    'n_blocks': int(len(trace)),
    'n_inferences': int(n_infer),
    'mean_block_compute_ms': float(trace[:, 2].mean()),
    'p95_block_compute_ms': float(np.percentile(trace[:, 2], 95)),
    'max_block_compute_ms': float(trace[:, 2].max()),
    'max_waiting_ms': float(trace[:, 3].max()),
    'max_queue_blocks': int(trace[:, 4].max()),
    'overflow_count': int(trace[:, 6].sum()),
    'realtime_condition_pass': bool(trace[:, 6].sum() == 0 and np.percentile(trace[:, 2], 95) < buffer_period_ms),
    'classical_inference_ms_mean': float(np.mean(classic_ms) if classic_ms else 0),
    'classical_inference_ms_p95': float(np.percentile(classic_ms, 95) if classic_ms else 0),
    'tiny_cnn_inference_ms_mean': float(np.mean(cnn_ms) if cnn_ms else 0),
    'tiny_cnn_inference_ms_p95': float(np.percentile(cnn_ms, 95) if cnn_ms else 0),
    'memory_proxy_bytes': {
        'dma_double_buffer_int16': dma_bytes,
        'pcm_frame_ring_int16': pcm_ring_bytes,
        'logmel_context_float32': logmel_ring_bytes,
        'classical_frame_state_float32': classical_frame_state_bytes,
        'tiny_cnn_activation_float32_upper_proxy': activation_proxy_bytes,
        'tiny_cnn_weights_float32': cnn_weight_bytes,
        'classical_serialized_model': classical_serialized,
    },
    'sanity_labels': [list(map(str, t)) for t in labels_trace],
    'interpretation': 'Host-side scheduling and memory proxy only. Target-device latency, peak RAM, flash, and power must be measured in MP15.',
    'deployment_model_selected': False,
    'deployment_selection_deferred_to': 'MP15'
}

config = {
    'sample_rate_hz': fs,
    'input_interface': 'I2S_or_ADC_to_DMA',
    'dma_block_samples': dma_block_samples,
    'dma_buffer_count': 2,
    'dma_block_ms': buffer_period_ms,
    'pcm_ring_samples': frame_len,
    'feature_hop_samples': hop_len,
    'feature_frame_samples': frame_len,
    'feature_context_frames': context_frames,
    'feature_context_ms': context_ms,
    'inference_stride_frames': inference_stride_frames,
    'inference_stride_ms': inference_stride_ms,
    'ring_capacity_blocks': ring_capacity_blocks,
    'scheduler': 'DMA ISR only signals buffer-ready; DSP/feature/inference run outside ISR',
    'candidate_models': ['mp11_classical_baseline', 'mp12_tiny_cnn'],
    'final_model_selection': 'DEFERRED_TO_MP15',
    'overflow_policy': 'increment_counter_and_drop_oldest_unprocessed_block',
    'instrumentation': ['frame_counter', 'overflow_counter', 'timestamp_us', 'stage_latency_us']
}

(ROOT/'config').mkdir(exist_ok=True)
(ROOT/'results/metrics').mkdir(parents=True, exist_ok=True)
(ROOT/'results/figures').mkdir(parents=True, exist_ok=True)
(ROOT/'features/intermediate').mkdir(parents=True, exist_ok=True)
(ROOT/'firmware').mkdir(parents=True, exist_ok=True)
(ROOT/'config/realtime_architecture_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
(ROOT/'results/metrics/mp14_realtime_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
np.savez_compressed(ROOT/'features/intermediate/mp14_stream_trace.npz', trace=trace,
                    columns=np.array(['block', 'arrival_ms', 'compute_ms', 'waiting_ms', 'queue_blocks', 'did_infer', 'overflow'], dtype=object))

# Header constants for MP15 firmware integration.
header = (f'#pragma once\n// Auto-generated by Mini Project 14\n#define AUDIO_FS_HZ {fs}\n#define DMA_BLOCK_SAMPLES {dma_block_samples}\n'
          f'#define DMA_BUFFER_COUNT 2\n#define AUDIO_FRAME_SAMPLES {frame_len}\n#define FEATURE_HOP_SAMPLES {hop_len}\n'
          f'#define FEATURE_CONTEXT_FRAMES {context_frames}\n#define INFERENCE_STRIDE_FRAMES {inference_stride_frames}\n'
          f'#define AUDIO_RING_CAPACITY_BLOCKS {ring_capacity_blocks}\n')
(ROOT/'firmware/mp14_realtime_constants.h').write_text(header, encoding='utf-8')
print(header)

# %% [markdown]
# ## 6. Visualisasi: timing trace & memory budget

# %%
fig, ax = plt.subplots(figsize=(8.2, 4.2))
ax.plot(trace[:, 1], trace[:, 2], lw=1, label='compute per DMA block')
ax.axhline(buffer_period_ms, ls='--', lw=1, label='DMA block period')
ax.set_xlabel('Stream time (ms)'); ax.set_ylabel('Host compute time (ms)'); ax.grid(alpha=.2); ax.legend()
ax.set_title('MP14 host-side real-time scheduling trace'); fig.tight_layout(); fig.savefig(ROOT/'results/figures/mp14_timing_trace.png', dpi=170); plt.close(fig)

mem = metrics['memory_proxy_bytes']; names = list(mem.keys()); vals = np.array(list(mem.values()), dtype=float)/1024.0
fig, ax = plt.subplots(figsize=(8.0, 4.4)); ax.barh(np.arange(len(names)), vals); ax.set_yticks(np.arange(len(names)))
ax.set_yticklabels([n.replace('_', ' ') for n in names], fontsize=8); ax.set_xlabel('KiB'); ax.set_title('MP14 memory budget proxy')
ax.grid(axis='x', alpha=.2); fig.tight_layout()
fig.savefig(ROOT/'results/figures/mp14_memory_budget.png', dpi=170); plt.close(fig)

# %% [markdown]
# ## 7. Update project status + tabel validasi

# %%
status['status'] = 'ready_for_mp15'; status['last_completed_mini_project'] = 14
done = set(status.get('completed_mini_projects', [])); done.add(14); status['completed_mini_projects'] = sorted(done)
status['mp14'] = {
    'realtime_config': 'config/realtime_architecture_config.json',
    'stream_trace': 'features/intermediate/mp14_stream_trace.npz',
    'metrics': 'results/metrics/mp14_realtime_metrics.json',
    'firmware_constants': 'firmware/mp14_realtime_constants.h',
    'timing_figure': 'results/figures/mp14_timing_trace.png',
    'memory_figure': 'results/figures/mp14_memory_budget.png',
    'deployment_candidates': ['models/mp11_classical_baseline.joblib', 'models/mp12_tiny_cnn_state.pt'],
    'deployment_selection_deferred_to': 'MP15'
}
STATUS.write_text(json.dumps(status, indent=2), encoding='utf-8')

print('MP14 selesai')
print('blocks:', len(trace), 'inferences:', n_infer)
print('p95 block compute ms:', round(metrics['p95_block_compute_ms'], 4), 'period ms:', buffer_period_ms)
print('max queue blocks:', metrics['max_queue_blocks'], 'overflow:', metrics['overflow_count'])
print('classical infer mean ms:', round(metrics['classical_inference_ms_mean'], 4))
print('tiny cnn infer mean ms:', round(metrics['tiny_cnn_inference_ms_mean'], 4))
biggest = max(mem, key=mem.get)
print('komponen RAM proxy terbesar:', biggest, f'({mem[biggest]/1024:.2f} KiB)')
checks = [('Status awal', 'ready_for_mp14', 'ready_for_mp14 (diverifikasi di sel 1)'),
          ('DMA period', '10 ms', f'{buffer_period_ms} ms'),
          ('Context', '96 frame / 975 ms', f'{context_frames} frame / {context_ms} ms'),
          ('Inference stride', '100 ms', f'{inference_stride_ms} ms'),
          ('Overflow count', '0', str(metrics['overflow_count'])),
          ('Realtime condition', 'True', str(metrics['realtime_condition_pass'])),
          ('Model selection', 'False / deferred', f"{metrics['deployment_model_selected']} / {metrics['deployment_selection_deferred_to']}"),
          ('Header C', 'tersedia', str((ROOT/'firmware/mp14_realtime_constants.h').exists())),
          ('Status akhir', 'ready_for_mp15', status['status'])]
print(f"\n{'Pemeriksaan':20s} | {'Kriteria lulus':20s} | Hasil")
for c in checks: print(f'{c[0]:20s} | {c[1]:20s} | {c[2]}')
assert metrics['overflow_count'] == 0 and metrics['realtime_condition_pass']
print('status:', status['status'])

# %%
if "__file__" not in globals():
    from IPython.display import Image, display
    for p in ['mp14_timing_trace.png', 'mp14_memory_budget.png']:
        display(Image(filename=str(ROOT/'results/figures'/p), width=720))
