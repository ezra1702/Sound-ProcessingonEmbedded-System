# %% [markdown]
# ## FITUR TAMBAHAN MP14 — Stress Test Scheduler, Proyeksi MCU, Streaming KWS Kontinu & Ring Buffer C
# Modul mandiri setelah `mini_project_14.py`. Kontrak timing MP14 **tidak diubah**; model tidak dilatih ulang.
# Semua audio yang dipakai adalah blok **train+validation** (test tidak dipakai untuk desain arsitektur).
#
# 1. **Profil per-stage** (filter, feature/frame, classical, CNN) di host + **proyeksi ke MCU** (ESP32-S3 240 MHz)
#    berbasis hitungan MAC (asumsi dinyatakan eksplisit).
# 2. **CPU-load stress sweep**: compute diperlambat 1x–256x → kapan ring buffer 4 blok overflow? utilisasi, latensi p99, jitter.
# 3. **Inference-stride sweep** 50–500 ms: beban CPU vs latensi deteksi terburuk.
# 4. **Streaming KWS kontinu**: 20 ucapan train+val disambung dengan jeda derau; jendela 975 ms tiap 100 ms memakai
#    front-end beku (`kws_frontend.py`) + model klasik; dibandingkan *raw* vs *energy gate* (duty-cycling) vs
#    *gate + posterior smoothing + debounce/refractory* (materi Bab 14: deteksi event pada stream).
# 5. **Ring buffer C** (drop-oldest, kapasitas 4, critical-section macro) dikompilasi `gcc` & diuji terhadap model Python.
# 6. **Gantt chart** ISR / DSP / feature / inferensi pada 300 ms stream.
#
# Anggota Kelompok : Christama Ezra Yudianto (245150307111009), Abdillah Fatah Suryanto (245150300111008),
# Ivang Arfandia (245150307111003), Jahzeel Erason Nicodemus (245150300111002)

# %%
from pathlib import Path
import json, sys, time, shutil, subprocess
import numpy as np
import matplotlib
if "__file__" in globals():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT/'features'))
from kws_frontend import KWSFrontend  # noqa: E402

status = json.loads((ROOT/'project_status.json').read_text(encoding='utf-8'))
assert status['status'] in ('ready_for_mp15',), 'Jalankan mini_project_14.py terlebih dahulu'
rt = json.loads((ROOT/'config/realtime_architecture_config.json').read_text(encoding='utf-8'))
m14 = json.loads((ROOT/'results/metrics/mp14_realtime_metrics.json').read_text(encoding='utf-8'))
red = json.loads((ROOT/'config/feature_reduction_config.json').read_text(encoding='utf-8'))
clf_cfg = json.loads((ROOT/'config/classifier_config.json').read_text(encoding='utf-8'))
mp13 = json.loads((ROOT/'results/metrics/mp13_final_test_metrics.json').read_text(encoding='utf-8'))
split = json.loads((ROOT/'splits/mp09_split.json').read_text(encoding='utf-8'))
OUT_MET, OUT_FIG = ROOT/'results/metrics', ROOT/'results/figures'
fs = rt['sample_rate_hz']; BLK = rt['dma_block_samples']; PERIOD = rt['dma_block_ms']
CTX = rt['feature_context_frames']; STRIDE = rt['inference_stride_frames']; CAP = rt['ring_capacity_blocks']
fe = KWSFrontend(ROOT)
clf = joblib.load(ROOT/clf_cfg['model_file'])
SEL = np.asarray(red['selected_indices'], int); MU = np.asarray(red['scaler_mean']); SC = np.asarray(red['scaler_scale'])
rng = np.random.default_rng(2026)

# Fungsi streaming per-frame milik skrip utama (sel 1–2) dimuat ulang agar profil stage identik dengan MP14
src = (ROOT/'mini_project_14.py').read_text(encoding='utf-8')
ns = {'__name__': 'mp14_main_helpers', '__file__': str(ROOT/'mini_project_14.py')}
exec(src[:src.index('# ## 3. Muat kedua model')], ns)

# %% [markdown]
# ### 1. Profil per-stage di host + proyeksi MCU

# %%
import torch
from scipy.signal import lfilter
ck = torch.load(ROOT/'models/mp12_tiny_cnn_state.pt', map_location='cpu', weights_only=False)
exec(src[src.index('class TinyKWSNet'):src.index('classic_model = joblib.load')], ns)
cnn = ns['TinyKWSNet'](len(ck['classes'])); cnn.load_state_dict(ck['state_dict']); cnn.eval()
blk = rng.normal(scale=.05, size=BLK); frm = rng.normal(scale=.05, size=rt['feature_frame_samples'])
b, a = fe.b, fe.a


def bench(fn, n=300):
    for _ in range(10): fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); t.append((time.perf_counter()-t0)*1e3)
    return float(np.median(t)), float(np.percentile(t, 99))


hist = [ns['frame_features'](frm) for _ in range(CTX)]
x8 = np.zeros((1, len(SEL)))
lm = torch.zeros((1, 1, CTX, 26))
stage_host = {
    'iir_filter_per_block': bench(lambda: lfilter(b, a, blk)),
    'frame_features_full_per_hop': bench(lambda: ns['frame_features'](frm), 200),
    'aggregate39_per_inference': bench(lambda: ns['aggregate_39'](hist), 100),
    'classical_predict_per_inference': bench(lambda: clf.predict(x8)),
    'tiny_cnn_per_inference': bench(lambda: cnn(lm), 100),
}
print('Host median / p99 (ms):', {k: tuple(round(x, 4) for x in v) for k, v in stage_host.items()})

# Proyeksi MCU (asumsi eksplisit) — ESP32-S3 @240 MHz, FPU single-precision, ~4 siklus/MAC efektif (load+MAC+loop overhead)
MCU = {'name': 'ESP32-S3 (Xtensa LX7) @240MHz', 'f_hz': 240e6, 'cycles_per_mac_float': 4.0, 'cycles_per_mac_int8': 1.0}
fe_cost = json.loads((ROOT/'results/metrics/mp10_extension_selection_metrics.json').read_text(encoding='utf-8'))['front_end_cost']
mac = {
    'iir_filter_per_block': BLK*(len(b)+len(a)-1),
    'frame_features_full_per_hop': fe_cost['mac_per_frame_full39'],
    'frame_features_selected_per_hop': fe_cost['mac_per_frame_selected'],
    'classical_tree_per_inference': mp13['resource_proxy']['classical']['estimated_ops_per_sample'] + 2*len(SEL),
    'tiny_cnn_per_inference': mp13['resource_proxy']['tiny_cnn']['estimated_macs_per_inference'],
}
mcu_ms = {k: 1e3*v*MCU['cycles_per_mac_float']/MCU['f_hz'] for k, v in mac.items()}
mcu_ms['tiny_cnn_int8_per_inference'] = 1e3*mac['tiny_cnn_per_inference']*MCU['cycles_per_mac_int8']/MCU['f_hz']


def utilization(per_block_ms, per_infer_ms, stride_frames):
    return (per_block_ms + per_infer_ms/stride_frames)/PERIOD


pipelines = {
    'A: full39 front-end + tiny CNN float': (mcu_ms['iir_filter_per_block']+mcu_ms['frame_features_full_per_hop'], mcu_ms['tiny_cnn_per_inference']),
    'B: full39 front-end + tree': (mcu_ms['iir_filter_per_block']+mcu_ms['frame_features_full_per_hop'], mcu_ms['classical_tree_per_inference']),
    'C: selected front-end + tree (MP10 hemat)': (mcu_ms['iir_filter_per_block']+mcu_ms['frame_features_selected_per_hop'], mcu_ms['classical_tree_per_inference']),
    'D: log-mel only + tiny CNN INT8': (mcu_ms['iir_filter_per_block']+mcu_ms['frame_features_selected_per_hop'], mcu_ms['tiny_cnn_int8_per_inference']),
}
proj = {}
for k, (pb, pi) in pipelines.items():
    proj[k] = {'per_block_ms': pb, 'per_inference_ms': pi, 'worst_block_ms': pb+pi,
               'avg_cpu_utilization': utilization(pb, pi, STRIDE),
               'worst_block_fits_10ms': bool(pb+pi < PERIOD)}
    print(f"  {k:45s} worst-block={pb+pi:6.3f} ms  util={100*proj[k]['avg_cpu_utilization']:5.1f}%  fits={proj[k]['worst_block_fits_10ms']}")

# %% [markdown]
# ### 2. CPU-load stress sweep (virtual scheduler, ring buffer 4 blok)

# %%
T = np.load(ROOT/'features/intermediate/mp14_stream_trace.npz', allow_pickle=True)['trace']
comp0 = T[:, 2]


def schedule(comp, cap=CAP, period=PERIOD):
    """Virtual CPU: blok tiba tiap period; drop-oldest jika antrean >= cap."""
    ready = 0.0; lat = []; q = []; drops = 0
    for i, c in enumerate(comp):
        arr = i*period; st = max(arr, ready)
        qb = int(np.floor((st-arr)/period+1e-12))
        if qb >= cap:           # blok dibuang -> CPU tidak memprosesnya
            drops += 1; q.append(qb); continue
        ready = st+c; lat.append(ready-arr); q.append(qb)
    lat = np.asarray(lat)
    return {'overflow_drops': drops, 'max_queue_blocks': int(max(q)), 'utilization': float(np.sum(comp)/(len(comp)*period)),
            'latency_p50_ms': float(np.percentile(lat, 50)), 'latency_p99_ms': float(np.percentile(lat, 99)),
            'jitter_p99_minus_p50_ms': float(np.percentile(lat, 99)-np.percentile(lat, 50))}


stress = {}
for f in (1, 2, 4, 8, 16, 32, 64, 128, 256):
    stress[f] = schedule(comp0*f)
break_factor = next((f for f, r in stress.items() if r['overflow_drops'] > 0), None)
for f, r in stress.items():
    print(f"  slowdown x{f:<4d} util={100*r['utilization']:7.1f}%  maxQ={r['max_queue_blocks']:3d}  drops={r['overflow_drops']:4d}  p99 lat={r['latency_p99_ms']:9.2f} ms")
print('Faktor perlambatan pertama yang menyebabkan overflow:', break_factor)

# %% [markdown]
# ### 3. Inference-stride sweep (proyeksi MCU pipeline A & C)

# %%
stride_sweep = {}
for sf in (5, 10, 20, 50):
    row = {'stride_ms': sf*rt['feature_hop_samples']*1000/fs, 'inferences_per_s': 1000/(sf*10)}
    for k in ('A: full39 front-end + tiny CNN float', 'C: selected front-end + tree (MP10 hemat)'):
        pb, pi = pipelines[k]
        row[k[:1]+'_util'] = utilization(pb, pi, sf)
    row['worst_detection_latency_ms'] = row['stride_ms'] + pipelines['A: full39 front-end + tiny CNN float'][1]
    stride_sweep[sf] = row
    print(f"  stride {row['stride_ms']:5.0f} ms  inf/s={row['inferences_per_s']:5.1f}  utilA={100*row['A_util']:5.1f}%  utilC={100*row['C_util']:5.1f}%  worst latency={row['worst_detection_latency_ms']:.1f} ms")

# %% [markdown]
# ### 4. Streaming KWS kontinu: raw vs energy gate vs gate + smoothing + debounce

# %%
ids = split['train_ids'] + split['validation_ids']
order = rng.permutation(len(ids))
pieces, words = [], []
cursor = 0
noise_rms = []
for sid in ids:
    y = fe.read_wav(ROOT/'data/raw/mp09_dataset'/f'{sid}.wav'); noise_rms.append(np.sqrt(np.mean(y[:800]**2)))
gap_rms = float(np.median(noise_rms))
for i in order:
    sid = ids[i]
    y = fe.read_wav(ROOT/'data/raw/mp09_dataset'/f'{sid}.wav')
    gap = rng.normal(scale=gap_rms, size=int(fs*rng.uniform(0.6, 1.2)))
    pieces += [gap, y]; cursor += len(gap)
    _, (s0, s1) = fe.preprocess(y, return_bounds=True)
    words.append({'id': sid, 'label': sid.split('_b')[0], 'start_s': (cursor+s0)/fs, 'end_s': (cursor+s1)/fs})
    cursor += len(y)
pieces.append(rng.normal(scale=gap_rms, size=fs))
stream = np.concatenate(pieces)
WIN = int(round(rt['feature_context_ms']*fs/1000)); HOPW = STRIDE*rt['feature_hop_samples']
classes = list(clf.classes_)
print(f'Stream {len(stream)/fs:.1f} s, {len(words)} ucapan, jendela {WIN} sampel tiap {HOPW} sampel')

# energi per blok DMA untuk gate (noise floor = persentil-10 2 detik terakhir)
blk_rms = np.sqrt(np.mean(stream[:len(stream)//BLK*BLK].reshape(-1, BLK)**2, axis=1))
win_rows = []
for end in range(WIN, len(stream)+1, HOPW):
    bi = end//BLK
    nf = np.percentile(blk_rms[max(0, bi-200):bi], 10)+1e-9
    act = blk_rms[max(0, bi-60):bi].max()/nf      # puncak energi 600 ms terakhir vs noise floor
    recent = blk_rms[max(0, bi-20):bi].max()/nf   # puncak energi 200 ms terakhir (deteksi akhir ucapan)
    v, _ = fe.features(stream[end-WIN:end])
    P = clf.predict_proba(((v-MU)/SC)[SEL][None])[0]
    win_rows.append({'t': end/fs, 'gate_ratio': float(act), 'recent_ratio': float(recent), 'P': P})


def detect(rows, gate_db=None, smooth=1, thr=0.0, confirm=1, refractory_s=0.0):
    events, buf, last_lab, run, last_t = [], [], None, 0, -1e9
    for r in rows:
        if gate_db is not None and 20*np.log10(r['gate_ratio']) < gate_db:
            buf.clear(); last_lab, run = None, 0; continue
        buf.append(r['P']); buf[:] = buf[-smooth:]
        Pm = np.mean(buf, axis=0); lab = classes[int(Pm.argmax())]
        if Pm.max() < thr:
            last_lab, run = None, 0; continue
        run = run+1 if lab == last_lab else 1; last_lab = lab
        if run >= confirm and r['t']-last_t >= refractory_s:
            events.append((r['t'], lab)); last_t = r['t']; run = -10**6   # lock sampai label berubah
    return events


def detect_endpoint(rows, gate_db=12.0, end_db=6.0, refractory_s=1.0):
    """VAD-endpoint trigger: klasifikasi SATU kali saat ucapan baru selesai (energi 600 ms tinggi, 200 ms terakhir kembali
    ke noise floor) sehingga jendela 975 ms memuat kata utuh. Inferensi hanya dijalankan pada momen itu."""
    events, last_t, n_inf = [], -1e9, 0
    for r in rows:
        if 20*np.log10(r['gate_ratio']) >= gate_db and 20*np.log10(r['recent_ratio']) < end_db and r['t']-last_t >= refractory_s:
            n_inf += 1; events.append((r['t'], classes[int(np.argmax(r['P']))])); last_t = r['t']
    return events, n_inf


def score(events):
    used = set(); hit = sub = fa = 0
    for t, lab in events:
        match = [k for k, w in enumerate(words) if w['start_s']+0.15 <= t <= w['end_s']+1.0 and k not in used]
        if not match: fa += 1; continue
        k = match[0]; used.add(k)
        if words[k]['label'] == lab: hit += 1
        else: sub += 1
    dur_min = len(stream)/fs/60
    return {'events': len(events), 'hits': hit, 'substitutions': sub, 'false_alarms': fa, 'misses': len(words)-hit-sub,
            'detection_rate': hit/len(words), 'false_alarms_per_min': fa/dur_min}


GATE_DB = 12.0
configs = {
    'raw (tiap 100 ms, tanpa gate)': dict(),
    'energy gate 12 dB': dict(gate_db=GATE_DB, refractory_s=0.0),
    'gate + debounce (confirm 2, refractory 1 s)': dict(gate_db=GATE_DB, confirm=2, refractory_s=1.0),
    'gate + smoothing 3 + thr 0.6 + confirm 2 + refractory 1 s': dict(gate_db=GATE_DB, smooth=3, thr=0.6, confirm=2, refractory_s=1.0),
}
stream_res = {}
gate_open = np.mean([20*np.log10(r['gate_ratio']) >= GATE_DB for r in win_rows])
for k, cfg in configs.items():
    ev = detect(win_rows, **cfg); stream_res[k] = score(ev); stream_res[k]['config'] = cfg
    s = stream_res[k]
    print(f"  {k:58s} hit={s['hits']:2d}/{len(words)} sub={s['substitutions']:2d} FA={s['false_alarms']:3d} ({s['false_alarms_per_min']:.1f}/min)")
ev_end, n_inf_end = detect_endpoint(win_rows, GATE_DB)
k_end = 'VAD-endpoint trigger (1 inferensi per ucapan)'
stream_res[k_end] = score(ev_end); stream_res[k_end]['config'] = {'gate_db': GATE_DB, 'end_db': 6.0, 'refractory_s': 1.0}
stream_res[k_end]['inferences_run'] = n_inf_end; stream_res[k_end]['inference_saving_vs_raw'] = 1-n_inf_end/len(win_rows)
s = stream_res[k_end]
print(f"  {k_end:58s} hit={s['hits']:2d}/{len(words)} sub={s['substitutions']:2d} FA={s['false_alarms']:3d} ({s['false_alarms_per_min']:.1f}/min)  inferensi={n_inf_end}/{len(win_rows)}")
print(f'Duty-cycle: gate terbuka pada {100*gate_open:.1f}% jendela -> inferensi dihemat {100*(1-gate_open):.1f}%')

# %% [markdown]
# ### 5. Ring buffer C (drop-oldest) — compile & uji terhadap model Python

# %%
FW = ROOT/'firmware'; (FW/'build').mkdir(parents=True, exist_ok=True)
ring_h = r'''#pragma once
// MP14 FITUR TAMBAHAN: ring buffer blok DMA (SPSC) dengan kebijakan drop-oldest + overflow counter.
// ISR (producer) hanya memanggil ring_push(); task DSP (consumer) memanggil ring_pop().
// Pada MCU, CRIT_ENTER/CRIT_EXIT = portENTER_CRITICAL/portEXIT_CRITICAL (FreeRTOS) atau __disable_irq().
#include <stdint.h>
#include <string.h>
#include "mp14_realtime_constants.h"
#ifndef CRIT_ENTER
#define CRIT_ENTER()
#define CRIT_EXIT()
#endif
typedef struct {
    int16_t  data[AUDIO_RING_CAPACITY_BLOCKS][DMA_BLOCK_SAMPLES];
    uint32_t seq[AUDIO_RING_CAPACITY_BLOCKS];
    volatile uint32_t head, tail;      /* head: tulis berikut, tail: baca berikut (monoton) */
    volatile uint32_t overflow_count;
} block_ring_t;

static inline void ring_init(block_ring_t *r) { memset(r, 0, sizeof(*r)); }
static inline uint32_t ring_level(const block_ring_t *r) { return r->head - r->tail; }

/* Dipanggil dari ISR DMA half/full-complete. */
static inline void ring_push(block_ring_t *r, const int16_t *blk, uint32_t seq) {
    CRIT_ENTER();
    if (r->head - r->tail >= AUDIO_RING_CAPACITY_BLOCKS) { r->tail++; r->overflow_count++; } /* drop oldest */
    uint32_t i = r->head % AUDIO_RING_CAPACITY_BLOCKS;
    memcpy(r->data[i], blk, sizeof(r->data[i])); r->seq[i] = seq; r->head++;
    CRIT_EXIT();
}

/* Dipanggil dari task DSP; return 1 jika ada blok. */
static inline int ring_pop(block_ring_t *r, int16_t *out, uint32_t *seq) {
    CRIT_ENTER();
    if (r->head == r->tail) { CRIT_EXIT(); return 0; }
    uint32_t i = r->tail % AUDIO_RING_CAPACITY_BLOCKS;
    memcpy(out, r->data[i], sizeof(r->data[i])); *seq = r->seq[i]; r->tail++;
    CRIT_EXIT();
    return 1;
}
'''
(FW/'mp14_ring_buffer.h').write_text(ring_h, encoding='utf-8')

# skenario: tiap tick 1 blok tiba; consumer punya "busy ticks" setelah tiap pop (stall acak, termasuk stall panjang)
busy = rng.choice([0, 0, 0, 1, 2, 6, 9], size=200).astype(int)


def py_ring(busy, cap=CAP, n_ticks=260):
    q, ovf, out, wait, k = [], 0, [], 0, 0
    for tick in range(n_ticks):
        if len(q) >= cap: q.pop(0); ovf += 1
        q.append(tick)
        if wait > 0: wait -= 1; continue
        if q and k < len(busy):
            out.append(q.pop(0)); wait = int(busy[k]); k += 1
    return ovf, out


ovf_py, out_py = py_ring(busy)
test_c = '#include <stdio.h>\n#include "../mp14_ring_buffer.h"\n'
test_c += f'static const int BUSY[{len(busy)}] = {{{",".join(map(str, busy))}}};\n'
test_c += r'''int main(void){ static block_ring_t r; ring_init(&r); int16_t blk[DMA_BLOCK_SAMPLES], out[DMA_BLOCK_SAMPLES];
 uint32_t seq; int wait=0, k=0; const int NB=(int)(sizeof(BUSY)/sizeof(BUSY[0]));
 for(int tick=0; tick<260; ++tick){ for(int i=0;i<DMA_BLOCK_SAMPLES;++i) blk[i]=(int16_t)(tick*7+i);
   ring_push(&r, blk, (uint32_t)tick);
   if(wait>0){ --wait; continue; }
   if(k<NB && ring_pop(&r, out, &seq)){ if(out[3]!=(int16_t)(seq*7+3)){printf("DATA_ERR\n");return 2;} printf("%u ", seq); wait=BUSY[k++]; } }
 printf("\nOVF %u\n", r.overflow_count); return 0; }
'''
(FW/'build'/'test_mp14_ring.c').write_text(test_c, encoding='utf-8')
gcc = shutil.which('gcc') or (r'C:\msys64\ucrt64\bin\gcc.exe' if Path(r'C:\msys64\ucrt64\bin\gcc.exe').exists() else None)
ring_test = {'compiled': False}
if gcc:
    exe = FW/'build'/'test_mp14_ring.exe'
    env = dict(__import__('os').environ); env['PATH'] = str(Path(gcc).parent) + ';' + env.get('PATH', '')
    cp = subprocess.run([gcc, '-O2', '-std=c99', '-Wall', '-Wextra', '-Werror', '-o', str(exe), str(FW/'build'/'test_mp14_ring.c')],
                        capture_output=True, text=True, env=env)
    if cp.returncode == 0:
        run = subprocess.run([str(exe)], capture_output=True, text=True, env=env)
        lines = run.stdout.strip().splitlines()
        out_c = list(map(int, lines[0].split())); ovf_c = int(lines[1].split()[1])
        ring_test = {'compiled': True, 'gcc': gcc, 'consumed_blocks': len(out_c), 'overflow_c': ovf_c, 'overflow_python': ovf_py,
                     'sequence_match': out_c == out_py, 'overflow_match': ovf_c == ovf_py}
    else:
        ring_test['error'] = cp.stderr[-800:]
print('Ring buffer C vs Python:', ring_test)
if ring_test.get('compiled'):
    assert ring_test['sequence_match'] and ring_test['overflow_match']

# %% [markdown]
# ### 6. Simpan + visualisasi (stress, stride, stream, Gantt)

# %%
ext = {'host_stage_ms_median_p99': stage_host, 'mcu_assumptions': MCU, 'mac_counts': mac, 'mcu_projected_ms': mcu_ms,
       'pipeline_projection': proj, 'stress_sweep': {str(k): v for k, v in stress.items()}, 'first_overflow_slowdown_factor': break_factor,
       'stride_sweep': {str(k): v for k, v in stride_sweep.items()},
       'continuous_stream': {'source_ids': ids, 'duration_s': len(stream)/fs, 'n_words': len(words), 'gate_db': GATE_DB,
                             'gate_open_fraction': float(gate_open), 'results': stream_res,
                             'note': 'Integration test pada blok train+val (bukan estimasi generalisasi).'},
       'ring_buffer_c_test': ring_test, 'contract_changed': False, 'test_split_used': False}
(OUT_MET/'mp14_extension_realtime_metrics.json').write_text(json.dumps(ext, indent=2, default=float), encoding='utf-8')

fig = plt.figure(figsize=(18, 10)); gs = fig.add_gridspec(2, 3)
ax = fig.add_subplot(gs[0, 0]); fs_ = list(stress)
ax.semilogx(fs_, [100*stress[f]['utilization'] for f in fs_], 'o-', base=2, label='utilisasi CPU (%)')
ax.semilogx(fs_, [stress[f]['overflow_drops'] for f in fs_], 's--', base=2, label='blok dibuang (overflow)')
ax.axhline(100, color='k', lw=.8, ls=':'); ax.set_xlabel('faktor perlambatan compute'); ax.set_title('Stress sweep (ring 4 blok)'); ax.legend(); ax.grid(alpha=.2)
ax = fig.add_subplot(gs[0, 1]); names = list(proj)
ax.barh(names, [100*proj[k]['avg_cpu_utilization'] for k in names], color=['C3' if not proj[k]['worst_block_fits_10ms'] else 'C2' for k in names])
ax.set_xlabel('utilisasi CPU rata-rata (%) - proyeksi ESP32-S3'); ax.set_title('Proyeksi MCU (merah = worst block > 10 ms)'); ax.tick_params(axis='y', labelsize=8)
ax = fig.add_subplot(gs[0, 2]); sw = list(stride_sweep)
ax.plot([stride_sweep[s]['stride_ms'] for s in sw], [100*stride_sweep[s]['A_util'] for s in sw], 'o-', label='util pipeline A (%)')
ax.plot([stride_sweep[s]['stride_ms'] for s in sw], [100*stride_sweep[s]['C_util'] for s in sw], 'o-', label='util pipeline C (%)')
ax2 = ax.twinx(); ax2.plot([stride_sweep[s]['stride_ms'] for s in sw], [stride_sweep[s]['worst_detection_latency_ms'] for s in sw], 'k^--', label='latensi terburuk')
ax2.set_ylabel('latensi deteksi terburuk (ms)'); ax.set_xlabel('inference stride (ms)'); ax.set_title('Trade-off stride'); ax.legend(loc='upper center'); ax.grid(alpha=.2)
ax = fig.add_subplot(gs[1, :2]); tt = np.arange(len(stream))/fs
ax.plot(tt, stream, lw=.3, color='0.6')
for w in words: ax.axvspan(w['start_s'], w['end_s'], color='C0', alpha=.15); ax.text(w['start_s'], .9, w['label'], fontsize=7, color='C0')
for t, lab in ev_end: ax.axvline(t, color='C3', lw=1); ax.text(t, -.95, lab, fontsize=7, color='C3', rotation=90)
gr = np.array([20*np.log10(r['gate_ratio']) for r in win_rows]); ax.plot([r['t'] for r in win_rows], gr/max(gr.max(), 1), 'C2', lw=.8, label='gate ratio (dB, ternormalisasi)')
ax.set_ylim(-1, 1); ax.set_xlabel('waktu (s)'); ax.legend(loc='lower right', fontsize=7)
ax.set_title('Streaming KWS kontinu (biru = ucapan ground truth, merah = event VAD-endpoint trigger)')
ax = fig.add_subplot(gs[1, 2])
host = {'ISR (signal)': 0.01, 'IIR filter': mcu_ms['iir_filter_per_block'], 'feature/frame (full39)': mcu_ms['frame_features_full_per_hop'],
        'classical tree': mcu_ms['classical_tree_per_inference'], 'tiny CNN float': mcu_ms['tiny_cnn_per_inference']}
lanes = list(host); cpu_free = 0.0
for k in range(15):
    t0 = k*PERIOD
    ax.broken_barh([(t0, .4)], (4, .8), color='C0')                  # ISR (dilebarkan agar terlihat)
    st = max(t0, cpu_free)
    ax.broken_barh([(st, host['IIR filter'])], (3, .8), color='C1'); st += host['IIR filter']
    ax.broken_barh([(st, host['feature/frame (full39)'])], (2, .8), color='C2'); st += host['feature/frame (full39)']
    if k % STRIDE == STRIDE-1:
        ax.broken_barh([(st, max(host['classical tree'], .4))], (1, .8), color='C3')
        ax.broken_barh([(st, host['tiny CNN float'])], (0, .8), color='C4'); st += host['tiny CNN float']
        ax.annotate('CNN float melewati\nbatas blok -> antre', (st, .4), (st+3, 1.8), fontsize=7, arrowprops=dict(arrowstyle='->'))
    cpu_free = st
    ax.axvline(t0, color='k', lw=.3, ls=':')
ax.set_yticks([4.4, 3.4, 2.4, 1.4, .4], lanes); ax.set_xlabel('waktu (ms) - durasi proyeksi ESP32-S3 (pipeline A/B)'); ax.set_xlim(0, 150)
ax.set_title('Gantt 150 ms (garis titik = blok DMA 10 ms)'); ax.tick_params(axis='y', labelsize=8)
fig.suptitle('MP14 FITUR TAMBAHAN - stress test, proyeksi MCU, stride, streaming KWS kontinu, Gantt'); fig.tight_layout()
fig.savefig(OUT_FIG/'mp14_extension_realtime_analysis.png', dpi=130); plt.close(fig)
print('Fitur tambahan MP14 selesai -> results/metrics/mp14_extension_realtime_metrics.json, firmware/mp14_ring_buffer.h')

# %%
if "__file__" not in globals():
    from IPython.display import Image, display
    display(Image(filename=str(OUT_FIG/'mp14_extension_realtime_analysis.png'), width=1000))
