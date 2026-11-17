# %% [markdown]
# ## FITUR TAMBAHAN MP16 — Command State Machine, Penolakan Non-Keyword, Stream Demo, C/ESP32, Dashboard & Laporan Final
# Modul mandiri setelah `mini_project_16.py`. Model, fitur, split, dan scale **tidak diubah**.
# Ambang penolakan dikalibrasi **hanya dari train+validation** (ditetapkan sebelum melihat distraktor/test).
#
# 1. **Penolakan out-of-vocabulary** (classifier klasik 4 kelas tidak punya kelas "unknown"): jarak Mahalanobis (Ledoit-Wolf,
#    39-D z-score, fit train) dengan ambang = 1,25 × jarak maksimum train+val. Diuji pada 8 ucapan **non-keyword** (TTS: halo,
#    lampu, satu, tolong — kelompok tidak merekam kata non-keyword) + 4 distraktor non-speech (burst noise, tone 1 kHz, hum, tepukan).
# 2. **Command state machine** IDLE → COOLDOWN, FAULT (watchdog audio 500 ms → fail-safe OFF), aksi
#    nyala=ON, mati=OFF, naik=level+, turun=level− dengan batas 0..10; implementasi **Python & C identik** (diuji gcc).
# 3. **Demo stream kontinu** (semua 96 keyword 4 penutur + distraktor + simulasi DMA stall) memakai VAD-endpoint trigger MP14
#    (desain awal v1 vs desain final v2 setelah analisis kegagalan, keduanya dilaporkan),
#    menghasilkan **event log JSONL** (timestamp, kelas, jarak OOD, aksi, state, latency, SHA-256 model).
# 4. **Sketch ESP32 (Arduino/ESP-IDF I2S)** sebagai baseline porting + header FSM C.
# 5. **Dashboard Streamlit** (`dashboard_mp16.py`) diuji otomatis dengan `streamlit.testing.AppTest`.
# 6. **Laporan final HTML** (MP7–MP16, figure tertanam) + **paket final ZIP** ber-SHA-256.
#
# Anggota Kelompok : Christama Ezra Yudianto (245150307111009), Abdillah Fatah Suryanto (245150300111008),
# Ivang Arfandia (245150307111003), Jahzeel Erason Nicodemus (245150300111002)

# %%
from pathlib import Path
import json, sys, os, time, hashlib, shutil, subprocess, tempfile, zipfile, base64, datetime, warnings
import numpy as np
import matplotlib
if "__file__" in globals():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import lfilter
from sklearn.covariance import LedoitWolf

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT/'features'))
from kws_frontend import KWSFrontend  # noqa: E402

status = json.loads((ROOT/'project_status.json').read_text(encoding='utf-8'))
assert status['status'] == 'project_complete', 'Jalankan mini_project_16.py terlebih dahulu'
acc = json.loads((ROOT/'results/metrics/mp16_acceptance_metrics.json').read_text(encoding='utf-8'))
split = json.loads((ROOT/'splits/mp09_split.json').read_text(encoding='utf-8'))
pkg = np.load(ROOT/'models/mp15_classical_int8.npz', allow_pickle=True)
CLASSES = pkg['classes'].astype(str).tolist()
D39 = np.load(ROOT/'features/dataset/mp09_features_39d.npz', allow_pickle=True)
OUT_MET, OUT_FIG, FW = ROOT/'results/metrics', ROOT/'results/figures', ROOT/'firmware'
LOG_DIR = ROOT/'results/logs'; LOG_DIR.mkdir(parents=True, exist_ok=True); (FW/'build').mkdir(parents=True, exist_ok=True)
fe = KWSFrontend(ROOT); FS = fe.fs
rng = np.random.default_rng(2026)
MODEL_SHA = hashlib.sha256((ROOT/'firmware/mp15_classical_model_int8.h').read_bytes()).hexdigest()


def predict_int8(v39):
    z = (v39[pkg['selected_indices']]-pkg['selected_mean'])/pkg['selected_scale']
    zq = np.clip(np.round(z/pkg['input_quant_scale'][0]), -127, 127)
    if str(pkg['kind']) == 'tree':
        n = 0
        while pkg['tree_feature'][n] >= 0:
            n = pkg['tree_left'][n] if zq[pkg['tree_feature'][n]] <= pkg['tree_threshold_q'][n] else pkg['tree_right'][n]
        return int(pkg['tree_leaf_class'][n])
    return int(np.argmax((pkg['weight_q'].astype(int)@zq)*pkg['weight_scales']*pkg['input_quant_scale'][0]+pkg['bias']))

# %% [markdown]
# ### 1. Gate penolakan OOV (Mahalanobis Ledoit-Wolf, kalibrasi train+val saja)

# %%
X = D39['X'].astype(np.float64); trm = D39['train_mask'].astype(bool); vam = D39['val_mask'].astype(bool)
mu39 = X[trm].mean(0); sd39 = X[trm].std(0)+1e-9
lw = LedoitWolf().fit((X[trm]-mu39)/sd39)


def ood_distance(v39):
    return float(np.sqrt(lw.mahalanobis(((np.asarray(v39, float)-mu39)/sd39)[None])[0]))


d_trval = np.array([ood_distance(x) for x in X[trm | vam]])
OOD_THR = 1.25*float(d_trval.max())
print(f'Ambang OOD = 1.25 x max(train+val) = {OOD_THR:.2f} (train+val: {d_trval.min():.2f}..{d_trval.max():.2f})')

# %% [markdown]
# ### 2. Distraktor: ucapan non-keyword (TTS) + non-speech

# %%
DIS = ROOT/'data/raw/mp16_distractors'; DIS.mkdir(parents=True, exist_ok=True)
NONKW = {'halo': 'hah low', 'lampu': 'lahm poo', 'satu': 'sah too', 'tolong': 'toe long'}
VOICES = {'zira': 'Microsoft Zira Desktop', 'david': 'Microsoft David Desktop'}
need = [(w, vk) for w in NONKW for vk in VOICES if not (DIS/f'{w}_{vk}.wav').exists()]
if need and os.name == 'nt':
    tmp = Path(tempfile.mkdtemp(prefix='mp16_tts_'))
    lines = ["Add-Type -AssemblyName System.Speech",
             "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)"]
    for w, vk in need:
        lines += ["$s = New-Object System.Speech.Synthesis.SpeechSynthesizer", f"$s.SelectVoice('{VOICES[vk]}')",
                  f"$s.SetOutputToWaveFile('{tmp/(w+'_'+vk+'.wav')}', $fmt)", f"$s.Speak('{NONKW[w]}')", "$s.Dispose()"]
    ps1 = tmp/'tts.ps1'; ps1.write_text('\n'.join(lines), encoding='utf-8')
    try:
        subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(ps1)], check=True, capture_output=True, timeout=300)
        for w, vk in need:
            _, s = wavfile.read(tmp/f'{w}_{vk}.wav'); s = s.astype(float)/32768
            env = np.abs(s) > 0.02*np.abs(s).max(); i0, i1 = np.argmax(env), len(env)-np.argmax(env[::-1])
            s = 0.5*s[max(0, i0-80):i1+80]/np.abs(s).max()*10**(-3/20)
            y = np.zeros(int(1.2*FS)); on = int(0.2*FS); y[on:on+min(len(s), len(y)-on)] = s[:len(y)-on]
            # RNG terpisah: pembangkitan distraktor (sekali, lalu di-cache) tidak menggeser RNG stream -> hasil deterministik
            y += np.random.default_rng(99).normal(size=len(y))*np.sqrt(np.mean(s**2)/10**(30/10))
            wavfile.write(DIS/f'{w}_{vk}.wav', FS, np.round(np.clip(y, -1, 1)*32767).astype(np.int16))
    except Exception as e:  # noqa: BLE001
        print('TTS non-keyword gagal (dilewati):', e)
nonkw = {p.stem: fe.read_wav(p) for p in sorted(DIS.glob('*.wav'))}
t = np.arange(int(1.2*FS))/FS
env = np.zeros_like(t); env[int(.2*FS):int(.8*FS)] = np.hanning(int(.6*FS))
synth = {'noise_burst': 0.3*env*rng.normal(size=len(t)), 'tone_1khz': 0.3*env*np.sin(2*np.pi*1000*t),
         'hum_50hz': 0.3*env*(np.sin(2*np.pi*50*t)+.5*np.sin(2*np.pi*150*t)),
         'clap': sum(0.6*np.exp(-np.maximum(t-c, 0)*80)*(t >= c)*rng.normal(size=len(t)) for c in (.3, .55, .8))}
for k in synth: synth[k] = synth[k] + 0.002*rng.normal(size=len(t))
iso = []
for name, y in list(nonkw.items()) + list(synth.items()):
    v, _ = fe.features(y); d = ood_distance(v)
    iso.append({'name': name, 'type': 'non_keyword_speech' if name in nonkw else 'non_speech', 'pred_without_gate': CLASSES[predict_int8(v)],
                'ood_distance': d, 'rejected': d > OOD_THR})
kw_iso = []
for sid in split['train_ids']+split['validation_ids']+split['test_ids']:
    v, _ = fe.features(fe.read_wav(ROOT/'data/raw/mp09_dataset'/f'{sid}.wav')); d = ood_distance(v)
    kw_iso.append({'id': sid, 'split': 'test' if sid in split['test_ids'] else 'train/val', 'ood_distance': d, 'accepted': d <= OOD_THR})
rej_speech = np.mean([r['rejected'] for r in iso if r['type'] == 'non_keyword_speech']) if nonkw else float('nan')
rej_ns = np.mean([r['rejected'] for r in iso if r['type'] == 'non_speech'])
acc_test = np.mean([r['accepted'] for r in kw_iso if r['split'] == 'test'])
print(f'Isolated: keyword test diterima {acc_test:.2f} | non-keyword speech ditolak {rej_speech:.2f} | non-speech ditolak {rej_ns:.2f}')
for r in iso: print(f"  {r['name']:14s} tanpa gate -> {r['pred_without_gate']:6s} | D={r['ood_distance']:7.2f} | {'DITOLAK' if r['rejected'] else 'diterima'}")

# %% [markdown]
# ### 3. Command state machine (Python) — identik dengan `firmware/mp16_command_fsm.h`

# %%
IDLE, COOLDOWN, FAULT = 0, 1, 2
A_NONE, A_ON, A_OFF, A_UP, A_DOWN, A_REJECT, A_FAILSAFE, A_RECOVER, A_BOUND = range(9)
ACTION_NAME = ['NONE', 'ON', 'OFF', 'LEVEL_UP', 'LEVEL_DOWN', 'REJECT', 'FAILSAFE_OFF', 'RECOVER', 'IGNORED_BOUND']
STATE_NAME = ['IDLE', 'COOLDOWN', 'FAULT']
WD_TICKS, COOLDOWN_TICKS, LVL_MIN, LVL_MAX = 5, 8, 0, 10
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}


class CommandFSM:
    def __init__(self):
        self.state, self.power, self.level, self.cd, self.wd = IDLE, 0, 5, 0, 0

    def step(self, audio_ok, endpoint, cls, accepted):
        act = A_NONE
        if not audio_ok:
            self.wd += 1
            if self.wd >= WD_TICKS and self.state != FAULT:
                self.state, self.power, act = FAULT, 0, A_FAILSAFE
            return act
        self.wd = 0
        if self.state == FAULT:
            self.state = IDLE; return A_RECOVER
        if self.state == COOLDOWN:
            self.cd -= 1
            if self.cd <= 0: self.state = IDLE
            return act
        if endpoint:
            if not accepted: act = A_REJECT
            else:
                c = CLASSES[cls]
                if c == 'nyala': self.power, act = 1, A_ON
                elif c == 'mati': self.power, act = 0, A_OFF
                elif c == 'naik': act = A_UP if (self.power and self.level < LVL_MAX) else A_BOUND
                elif c == 'turun': act = A_DOWN if (self.power and self.level > LVL_MIN) else A_BOUND
                if act == A_UP: self.level += 1
                if act == A_DOWN: self.level -= 1
            self.state, self.cd = COOLDOWN, COOLDOWN_TICKS
        return act

# %% [markdown]
# ### 4. Demo stream kontinu (semua 96 keyword + distraktor + DMA stall) → event log JSONL

# %%
segs = [('kw', sid, sid.split('_b')[0].split('_')[-1], fe.read_wav(ROOT/'data/raw/mp09_dataset'/f'{sid}.wav'))
        for sid in split['train_ids']+split['validation_ids']+split['test_ids']]
segs += [('nonkw', k, None, y) for k, y in nonkw.items()] + [('nonspeech', k, None, y) for k, y in synth.items()]
segs = [segs[i] for i in rng.permutation(len(segs))]
gap_rms = float(np.median([np.sqrt(np.mean(s[3][:800]**2)) for s in segs if s[0] == 'kw']))
pieces, truth, cur = [], [], 0
# Derau jeda sebelum keyword = derau latar rekaman itu sendiri (mic/ruangan penutur yang sama; level antar penutur
# berbeda hingga ~12 dB). Distraktor sintetis tidak punya derau latar asli -> pakai median derau keyword.
for kind, name, lab, y in segs:
    g_rms = float(np.sqrt(np.mean(y[:800]**2))) if kind == 'kw' else gap_rms
    g = rng.normal(scale=g_rms, size=int(FS*rng.uniform(0.7, 1.3))); pieces += [g, y]; cur += len(g)
    if kind == 'kw': _, (s0, s1) = fe.preprocess(y, return_bounds=True)
    else:
        e = np.abs(y) > 0.1*np.abs(y).max(); s0, s1 = int(np.argmax(e)), int(len(e)-np.argmax(e[::-1]))
    truth.append({'kind': kind, 'name': name, 'label': lab, 'start_s': (cur+s0)/FS, 'end_s': (cur+s1)/FS}); cur += len(y)
pieces.append(rng.normal(scale=gap_rms, size=2*FS))
stream = np.concatenate(pieces)
# simulasi DMA stall 0,8 s di tengah stream (blok tidak datang) -> watchdog harus fail-safe lalu recover
stall_t0 = truth[len(truth)//2]['start_s'] - 0.9
stall = (stall_t0, stall_t0+0.8)
BLK = 160; HOPW = 1600
blk_rms = np.sqrt(np.mean(stream[:len(stream)//BLK*BLK].reshape(-1, BLK)**2, axis=1))
t_start = datetime.datetime(2026, 11, 15, 9, 0, 0)
# v1 = desain awal (jendela 975 ms = konteks MP14, endpoint: energi 200 ms < 6 dB di atas noise floor)
# v2 = desain final (jendela 1,2 s = durasi klip training MP9 = LIVE_DURATION_S buku MP16; endpoint juga terpicu
#      bila energi 200 ms terakhir turun >= 20 dB dari puncak 600 ms -> tahan terhadap lantai derau klip yang lebih tinggi)
ENDPOINT_CFG = {'v1_baseline': {'win_s': 0.975, 'rel_drop_db': None}, 'v2_final': {'win_s': 1.2, 'rel_drop_db': 20.0}}


def run_stream(win_s, rel_drop_db, log_file=None):
    WIN = int(win_s*FS); fsm = CommandFSM(); ticks, events = [], []
    LOG = log_file.open('w', encoding='utf-8') if log_file else None
    for end in range(WIN, len(stream)+1, HOPW):
        tt = end/FS; bi = end//BLK
        audio_ok = not (stall[0] <= tt < stall[1])
        nf = np.percentile(blk_rms[max(0, bi-200):bi], 10)+1e-9
        g600 = 20*np.log10(blk_rms[max(0, bi-60):bi].max()/nf); g200 = 20*np.log10(blk_rms[max(0, bi-20):bi].max()/nf)
        ended = g200 < 6.0 or (rel_drop_db is not None and g600-g200 >= rel_drop_db)
        endpoint = audio_ok and g600 >= 12.0 and ended
        cls, accepted, d, lat = -1, False, None, 0.0
        if endpoint and fsm.state == IDLE:
            t0 = time.perf_counter(); v, _ = fe.features(stream[end-WIN:end]); cls = predict_int8(v); d = ood_distance(v)
            accepted = d <= OOD_THR; lat = (time.perf_counter()-t0)*1e3
        act = fsm.step(audio_ok, endpoint, max(cls, 0), accepted)
        ticks.append((int(audio_ok), int(endpoint), max(cls, 0), int(accepted), act, fsm.state, fsm.power, fsm.level))
        if act != A_NONE:
            ev = {'timestamp': (t_start+datetime.timedelta(seconds=tt)).isoformat(timespec='milliseconds'), 't_stream_s': round(tt, 2),
                  'predicted_class': CLASSES[cls] if cls >= 0 else None, 'ood_distance': None if d is None else round(d, 3),
                  'accepted': bool(accepted), 'action': ACTION_NAME[act], 'state': STATE_NAME[fsm.state], 'power': 'ON' if fsm.power else 'OFF',
                  'level': fsm.level, 'pipeline_latency_ms': round(lat, 3), 'overflow_count': 0, 'model_sha256': MODEL_SHA[:16]}
            events.append(ev)
            if LOG: LOG.write(json.dumps(ev)+'\n')
    if LOG: LOG.close()
    return ticks, events


def score_stream(ticks, events):
    used = set(); seg_res = []
    for tr in truth:
        evs = [e for i, e in enumerate(events) if i not in used and e['predicted_class'] is not None and tr['start_s']+0.15 <= e['t_stream_s'] <= tr['end_s']+1.0]
        e = evs[0] if evs else None
        if e: used.add(events.index(e))
        if tr['kind'] == 'kw':
            out = 'hit' if (e and e['accepted'] and e['predicted_class'] == tr['label']) else ('false_reject' if (e and not e['accepted']) else
                  ('substitution' if e else 'miss'))
        else:
            out = 'correct_reject' if (e is None or not e['accepted']) else 'false_accept'
        seg_res.append({**tr, 'outcome': out, 'event': e})
    cnt = lambda k, o: sum(1 for s_ in seg_res if s_['kind'] == k and s_['outcome'] == o)  # noqa: E731
    sc = {'keywords': len([s_ for s_ in seg_res if s_['kind'] == 'kw']), 'kw_hit': cnt('kw', 'hit'), 'kw_false_reject': cnt('kw', 'false_reject'),
          'kw_substitution': cnt('kw', 'substitution'), 'kw_miss': cnt('kw', 'miss'),
          'nonkw_correct_reject': cnt('nonkw', 'correct_reject'), 'nonkw_false_accept': cnt('nonkw', 'false_accept'),
          'nonspeech_correct_reject': cnt('nonspeech', 'correct_reject'), 'nonspeech_false_accept': cnt('nonspeech', 'false_accept'),
          'failsafe_events': sum(e['action'] == 'FAILSAFE_OFF' for e in events), 'recover_events': sum(e['action'] == 'RECOVER' for e in events),
          'level_always_within_bounds': all(LVL_MIN <= t_[7] <= LVL_MAX for t_ in ticks), 'stream_duration_s': len(stream)/FS,
          'failed_segments': [f"{s_['name']}:{s_['outcome']}" for s_ in seg_res if s_['outcome'] not in ('hit', 'correct_reject')]}
    return sc, seg_res


stream_compare = {}
for k, cfg in ENDPOINT_CFG.items():
    tk_, ev_ = run_stream(cfg['win_s'], cfg['rel_drop_db'], LOG_DIR/'mp16_event_log.jsonl' if k == 'v2_final' else LOG_DIR/'mp16_event_log_v1_baseline.jsonl')
    sc_, sr_ = score_stream(tk_, ev_)
    stream_compare[k] = {**cfg, **sc_}
    print(f"  {k}: kw hit {sc_['kw_hit']}/{sc_['keywords']} | non-kw ditolak {sc_['nonkw_correct_reject']} | non-speech ditolak "
          f"{sc_['nonspeech_correct_reject']} | gagal: {sc_['failed_segments']}")
    if k == 'v2_final':
        ticks, events, stream_scores, seg_res = tk_, ev_, sc_, sr_
print('Stream (v2_final):', {k: v for k, v in stream_scores.items() if k != 'failed_segments'})

# %% [markdown]
# ### 5. FSM dalam C (header firmware) + uji ekuivalensi gcc + sketch ESP32

# %%
fsm_h = f'''// MP16 FITUR TAMBAHAN: command state machine (identik dengan CommandFSM Python). Dipanggil tiap inference tick (100 ms).
#pragma once
#include <stdint.h>
enum {{ FSM_IDLE = 0, FSM_COOLDOWN = 1, FSM_FAULT = 2 }};
enum {{ ACT_NONE, ACT_ON, ACT_OFF, ACT_LEVEL_UP, ACT_LEVEL_DOWN, ACT_REJECT, ACT_FAILSAFE_OFF, ACT_RECOVER, ACT_IGNORED_BOUND }};
#define FSM_WD_TICKS {WD_TICKS}
#define FSM_COOLDOWN_TICKS {COOLDOWN_TICKS}
#define FSM_LEVEL_MIN {LVL_MIN}
#define FSM_LEVEL_MAX {LVL_MAX}
/* urutan kelas: {', '.join(f'{i}={c}' for i, c in enumerate(CLASSES))} */
#define CLS_NYALA {CLS_IDX['nyala']}
#define CLS_MATI {CLS_IDX['mati']}
#define CLS_NAIK {CLS_IDX['naik']}
#define CLS_TURUN {CLS_IDX['turun']}
typedef struct {{ uint8_t state, power; int8_t level; int16_t cd, wd; }} cmd_fsm_t;
static inline void fsm_init(cmd_fsm_t *f) {{ f->state = FSM_IDLE; f->power = 0; f->level = 5; f->cd = 0; f->wd = 0; }}
static inline int fsm_step(cmd_fsm_t *f, int audio_ok, int endpoint, int cls, int accepted) {{
    int act = ACT_NONE;
    if (!audio_ok) {{
        f->wd++;
        if (f->wd >= FSM_WD_TICKS && f->state != FSM_FAULT) {{ f->state = FSM_FAULT; f->power = 0; act = ACT_FAILSAFE_OFF; }}
        return act;
    }}
    f->wd = 0;
    if (f->state == FSM_FAULT) {{ f->state = FSM_IDLE; return ACT_RECOVER; }}
    if (f->state == FSM_COOLDOWN) {{ if (--f->cd <= 0) f->state = FSM_IDLE; return act; }}
    if (endpoint) {{
        if (!accepted) act = ACT_REJECT;
        else if (cls == CLS_NYALA) {{ f->power = 1; act = ACT_ON; }}
        else if (cls == CLS_MATI) {{ f->power = 0; act = ACT_OFF; }}
        else if (cls == CLS_NAIK) {{ if (f->power && f->level < FSM_LEVEL_MAX) {{ f->level++; act = ACT_LEVEL_UP; }} else act = ACT_IGNORED_BOUND; }}
        else if (cls == CLS_TURUN) {{ if (f->power && f->level > FSM_LEVEL_MIN) {{ f->level--; act = ACT_LEVEL_DOWN; }} else act = ACT_IGNORED_BOUND; }}
        f->state = FSM_COOLDOWN; f->cd = FSM_COOLDOWN_TICKS;
    }}
    return act;
}}
'''
(FW/'mp16_command_fsm.h').write_text(fsm_h, encoding='utf-8')
# tambahkan tick sintetis agresif (batas level, fault beruntun) agar semua cabang teruji
extra = []
for i in range(60):
    extra.append((int(rng.random() > .08), int(rng.random() > .5), int(rng.integers(0, 4)), int(rng.random() > .2)))
inputs = [t_[:4] for t_ in ticks] + extra
ref = CommandFSM(); py_out = []
for a in inputs:
    act = ref.step(*a); py_out.append((act, ref.state, ref.power, ref.level))
arr = ',\n'.join('{%d,%d,%d,%d}' % a for a in inputs)
test_c = f'''#include <stdio.h>
#include "mp16_command_fsm.h"
static const int IN[{len(inputs)}][4] = {{
{arr}}};
int main(void){{ cmd_fsm_t f; fsm_init(&f);
  for (int i = 0; i < {len(inputs)}; ++i) {{ int a = fsm_step(&f, IN[i][0], IN[i][1], IN[i][2], IN[i][3]); printf("%d %d %d %d\\n", a, f.state, f.power, f.level); }}
  return 0; }}
'''
(FW/'build'/'test_mp16_fsm.c').write_text(test_c, encoding='utf-8')
gcc = shutil.which('gcc') or (r'C:\msys64\ucrt64\bin\gcc.exe' if Path(r'C:\msys64\ucrt64\bin\gcc.exe').exists() else None)
fsm_c = {'compiled': False}
if gcc:
    env = dict(os.environ); env['PATH'] = str(Path(gcc).parent)+os.pathsep+env.get('PATH', '')
    exe = FW/'build'/'test_mp16_fsm.exe'
    cp = subprocess.run([gcc, '-O2', '-std=c99', '-Wall', '-Wextra', '-Werror', '-I', str(FW), '-o', str(exe), str(FW/'build'/'test_mp16_fsm.c')],
                        capture_output=True, text=True, env=env)
    if cp.returncode == 0:
        out = [tuple(map(int, ln.split())) for ln in subprocess.run([str(exe)], capture_output=True, text=True, env=env).stdout.strip().splitlines()]
        fsm_c = {'compiled': True, 'n_ticks': len(out), 'trace_identical': out == py_out,
                 'actions_covered': sorted({ACTION_NAME[o[0]] for o in out})}
    else:
        fsm_c['error'] = cp.stderr[-600:]
    # header klasik + FSM + konstanta dikompilasi bersama (cek konflik simbol)
    (FW/'build'/'test_mp16_all.c').write_text('#include "mp14_realtime_constants.h"\n#include "mp14_ring_buffer.h"\n#include "mp15_classical_model_int8.h"\n'
                                               '#include "mp16_project_config.h"\n#include "mp16_command_fsm.h"\nint main(void){ float x[39]={0}; cmd_fsm_t f; fsm_init(&f);'
                                               ' return fsm_step(&f,1,1,mp15_classical_predict(x),1) < 0; }\n', encoding='utf-8')
    cp2 = subprocess.run([gcc, '-O2', '-std=c99', '-Wall', '-Wextra', '-Werror', '-I', str(FW), '-o', str(FW/'build'/'test_mp16_all.exe'),
                          str(FW/'build'/'test_mp16_all.c'), '-lm'], capture_output=True, text=True, env=env)
    fsm_c['all_headers_compile_together'] = cp2.returncode == 0
print('FSM C vs Python:', fsm_c)

sketch_dir = FW/'esp32_kws_mp16'; sketch_dir.mkdir(exist_ok=True)
(sketch_dir/'esp32_kws_mp16.ino').write_text(r'''// MP16 FITUR TAMBAHAN - baseline porting ESP32 (Arduino-ESP32 core 2.x, I2S MEMS mic mis. INMP441).
// Header beku disalin ke folder sketch: mp14_realtime_constants.h, mp14_ring_buffer.h, mp15_classical_model_int8.h,
// mp16_project_config.h, mp16_command_fsm.h. Front-end 39-D (filter MP6 -> preprocessing MP7 -> feature MP8) harus diport
// dari features/kws_frontend.py ke fungsi mp16_extract39() (CMSIS/ESP-DSP untuk FFT). Angka latency/RAM WAJIB diukur di perangkat.
#include <Arduino.h>
#include <driver/i2s.h>
#define CRIT_ENTER() portENTER_CRITICAL_ISR(&ring_mux)
#define CRIT_EXIT()  portEXIT_CRITICAL_ISR(&ring_mux)
static portMUX_TYPE ring_mux = portMUX_INITIALIZER_UNLOCKED;
#include "mp14_realtime_constants.h"
#include "mp14_ring_buffer.h"
#include "mp15_classical_model_int8.h"
#include "mp16_project_config.h"
#include "mp16_command_fsm.h"

#define PIN_LED 2
#define PIN_PWM 4
static block_ring_t ring;
static cmd_fsm_t fsm;
static int16_t window_pcm[(MP16_SAMPLE_RATE * 12) / 10];   // 1,2 s = jendela endpoint desain final v2
extern bool mp16_extract39(const int16_t *pcm, int n, float *x39, bool *endpoint);  // TODO: port kws_frontend.py

static void i2s_task(void *) {           // produser: baca 1 blok DMA (10 ms) -> ring buffer
  int16_t blk[DMA_BLOCK_SAMPLES]; size_t got; uint32_t seq = 0;
  for (;;) { i2s_read(I2S_NUM_0, blk, sizeof(blk), &got, portMAX_DELAY); ring_push(&ring, blk, seq++); }
}

void setup() {
  Serial.begin(115200); pinMode(PIN_LED, OUTPUT); ledcSetup(0, 5000, 8); ledcAttachPin(PIN_PWM, 0);
  i2s_config_t cfg = { (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX), AUDIO_FS_HZ, I2S_BITS_PER_SAMPLE_16BIT, I2S_CHANNEL_FMT_ONLY_LEFT,
                       I2S_COMM_FORMAT_STAND_I2S, 0, DMA_BUFFER_COUNT * 2, DMA_BLOCK_SAMPLES, false, false, 0 };
  i2s_pin_config_t pins = { .bck_io_num = 26, .ws_io_num = 25, .data_out_num = -1, .data_in_num = 33 };
  i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL); i2s_set_pin(I2S_NUM_0, &pins);
  ring_init(&ring); fsm_init(&fsm);
  xTaskCreatePinnedToCore(i2s_task, "i2s", 4096, NULL, 5, NULL, 0);
}

void loop() {                            // konsumen: DSP + inferensi tiap INFERENCE_STRIDE_FRAMES blok
  static int hop = 0, blocks_since_audio = 0; int16_t blk[DMA_BLOCK_SAMPLES]; uint32_t seq;
  if (!ring_pop(&ring, blk, &seq)) { vTaskDelay(1); if (++blocks_since_audio > 50) { fsm_step(&fsm, 0, 0, 0, 0); blocks_since_audio = 0; } return; }
  blocks_since_audio = 0;
  memmove(window_pcm, window_pcm + DMA_BLOCK_SAMPLES, sizeof(window_pcm) - sizeof(blk));
  memcpy(window_pcm + (sizeof(window_pcm)/2 - DMA_BLOCK_SAMPLES), blk, sizeof(blk));
  if (++hop < INFERENCE_STRIDE_FRAMES) return; hop = 0;
  uint32_t t0 = micros(); float x39[39]; bool endpoint = false;
  int cls = 0, accepted = 0;
  if (mp16_extract39(window_pcm, sizeof(window_pcm)/2, x39, &endpoint) && endpoint) { cls = mp15_classical_predict(x39); accepted = 1; /* + gate OOD */ }
  int act = fsm_step(&fsm, 1, endpoint, cls, accepted);
  digitalWrite(PIN_LED, fsm.power); ledcWrite(0, fsm.power ? fsm.level * 25 : 0);
  if (act != ACT_NONE) Serial.printf("{\"t_ms\":%lu,\"cls\":%d,\"act\":%d,\"power\":%d,\"level\":%d,\"lat_us\":%lu,\"ovf\":%lu}\n",
                                     millis(), cls, act, fsm.power, fsm.level, micros() - t0, (unsigned long)ring.overflow_count);
}
''', encoding='utf-8')

# %% [markdown]
# ### 6. Uji otomatis dashboard Streamlit (AppTest)

# %%
dash = {'file': 'dashboard_mp16.py'}
ext_tmp = {'stream_scores': stream_scores}
(OUT_MET/'mp16_extension_system_metrics.json').write_text(json.dumps(ext_tmp, indent=2), encoding='utf-8')   # agar dashboard punya data
try:
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT/'dashboard_mp16.py'), default_timeout=120).run()
    dash.update({'apptest_ran': True, 'exceptions': [str(e.value) for e in at.exception], 'n_metrics': len(at.metric),
                 'title': at.title[0].value if len(at.title) else None})
except Exception as e:  # noqa: BLE001
    dash.update({'apptest_ran': False, 'error': repr(e)[:300]})
print('Dashboard AppTest:', dash)

# %% [markdown]
# ### 7. Visualisasi, laporan HTML final, paket ZIP

# %%
fig = plt.figure(figsize=(18, 9)); gs = fig.add_gridspec(2, 3)
ax = fig.add_subplot(gs[0, 0])
ax.scatter([r['ood_distance'] for r in kw_iso if r['split'] != 'test'], [0]*sum(r['split'] != 'test' for r in kw_iso), label='keyword train/val', s=30)
ax.scatter([r['ood_distance'] for r in kw_iso if r['split'] == 'test'], [1]*sum(r['split'] == 'test' for r in kw_iso), label='keyword test', s=30, c='C2')
ax.scatter([r['ood_distance'] for r in iso if r['type'] == 'non_keyword_speech'], [2]*len(nonkw), label='non-keyword speech', s=30, c='C1')
ax.scatter([r['ood_distance'] for r in iso if r['type'] == 'non_speech'], [3]*len(synth), label='non-speech', s=30, c='C3')
ax.axvline(OOD_THR, color='k', ls='--', label=f'ambang {OOD_THR:.1f}'); ax.set_xscale('log'); ax.set_yticks(range(4), ['kw tr/val', 'kw test', 'non-kw', 'non-speech'])
ax.set_xlabel('jarak Mahalanobis (log)'); ax.set_title('Gate penolakan OOV (kalibrasi train+val)'); ax.legend(fontsize=7)
ax = fig.add_subplot(gs[0, 1:]); tt = np.arange(len(stream))/FS
ax.plot(tt, stream, lw=.25, color='.6')
colors = {'kw': 'C0', 'nonkw': 'C1', 'nonspeech': 'C3'}
for s in seg_res:
    ax.axvspan(s['start_s'], s['end_s'], color=colors[s['kind']], alpha=.18)
    ax.text(s['start_s'], .92, s['label'] or s['name'][:6], fontsize=6, color=colors[s['kind']], rotation=90, va='top')
for e in events:
    c = {'REJECT': 'C1', 'FAILSAFE_OFF': 'k', 'RECOVER': 'k'}.get(e['action'], 'C3')
    ax.axvline(e['t_stream_s'], color=c, lw=.9); ax.text(e['t_stream_s'], -.98, e['action'][:6], fontsize=6, rotation=90, color=c)
ax.axvspan(*stall, color='k', alpha=.25); ax.set_ylim(-1, 1); ax.set_xlabel('waktu (s)')
ax.set_title('Stream kontinu: biru=keyword, oranye=non-keyword, merah=non-speech, abu gelap=DMA stall; garis=event FSM')
ax = fig.add_subplot(gs[1, 0:2]); tk = np.arange(len(ticks))*0.1
ax.step(tk, [t_[6] for t_ in ticks], where='post', label='power (ON=1)')
ax.step(tk, [t_[7]/10 for t_ in ticks], where='post', label='level/10')
ax.step(tk, [t_[5]/2 for t_ in ticks], where='post', label='state/2 (0 IDLE, .5 COOLDOWN, 1 FAULT)', alpha=.6)
ax.set_xlabel('waktu (s)'); ax.set_title('Trajektori state machine'); ax.legend(fontsize=8); ax.grid(alpha=.2)
ax = fig.add_subplot(gs[1, 2])
labels_ = ['kw hit', 'kw false rej', 'kw subst', 'kw miss', 'non-kw rej', 'non-kw FA', 'non-sp rej', 'non-sp FA']
vals_ = [stream_scores[k] for k in ('kw_hit', 'kw_false_reject', 'kw_substitution', 'kw_miss', 'nonkw_correct_reject', 'nonkw_false_accept',
                                     'nonspeech_correct_reject', 'nonspeech_false_accept')]
ax.barh(labels_, vals_, color=['C2', 'C1', 'C3', 'C3', 'C2', 'C3', 'C2', 'C3']); ax.invert_yaxis(); ax.set_title('Hasil per segmen stream')
for i, v in enumerate(vals_): ax.text(v, i, f' {v}', va='center')
fig.suptitle(f"MP16 FITUR TAMBAHAN - FSM C==Python: {fsm_c.get('trace_identical')} | fail-safe {stream_scores['failsafe_events']}x, recover {stream_scores['recover_events']}x")
fig.tight_layout(); fig.savefig(OUT_FIG/'mp16_extension_system_analysis.png', dpi=130); plt.close(fig)


def jl(rel):
    p = ROOT/rel
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}


def img(rel, w=560):
    p = ROOT/rel
    return f'<img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" width="{w}">' if p.exists() else ''


m7, m8, m9 = jl('results/metrics/mp07_preprocessing_metrics.json'), jl('results/metrics/mp08_feature_metrics.json'), jl('results/metrics/mp09_dataset_metrics.json')
m10, m11, m12 = jl('config/feature_reduction_config.json'), jl('config/classifier_config.json'), jl('results/metrics/mp12_deep_learning_metrics.json')
m13, m14, m15 = jl('results/metrics/mp13_final_test_metrics.json'), jl('results/metrics/mp14_realtime_metrics.json'), jl('results/metrics/mp15_deployment_metrics.json')
m11m = jl('results/metrics/mp11_classifier_metrics.json'); dcfg = jl('config/dataset_config.json')
n_kw = stream_scores['keywords']
x14 = jl('results/metrics/mp14_extension_realtime_metrics.json')
rows_html = [
    ('MP07 Preprocessing', 'DC removal, normalisasi peak, VAD, spectral subtraction', f"aktif {m7.get('active_ratio', '-')}"),
    ('MP08 Feature', 'log-Mel 26, MFCC 13, spektral, pitch → vektor 39-D', f"dim {m8.get('sample_vector_dim', m8.get('feature_dim', 39))}"),
    ('MP09 Dataset', f"{m9.get('n_samples')} WAV ({dcfg.get('n_speakers', 1)} penutur × 6 block × 4 kelas), split per block {m9.get('n_train')}/{m9.get('n_validation')}/{m9.get('n_test')}",
     'rekaman suara asli 4 anggota kelompok' if dcfg.get('acquisition_mode') == 'group' else f"mode {dcfg.get('acquisition_mode')}"),
    ('MP10 Seleksi fitur', f"{m10.get('chosen_candidate', '-')}", ', '.join(m10.get('selected_feature_names', []))),
    ('MP11 Classifier klasik', f"{m11.get('chosen_family', '-')} {m11.get('chosen_params', '')}", f"val F1 {round(m11m.get('chosen_validation_macro_f1', float('nan')), 3)}"),
    ('MP12 Tiny CNN', f"params {m12.get('parameter_count', '-')}, MAC {m12.get('estimated_macs_per_inference', '-')}", f"val F1 {round(m12.get('validation_macro_f1', float('nan')), 3)}"),
    ('MP13 Frozen test', f"classical F1 {round(m13.get('classical', {}).get('macro_f1', 0), 3)} (n = {m13.get('n_test')})", f"CNN F1 {round(m13.get('tiny_cnn', {}).get('macro_f1', 0), 3)}"),
    ('MP14 Real-time', f"p95 blok {round(m14.get('p95_block_compute_ms', 0), 3)} ms / 10 ms", f"overflow {m14.get('overflow_count', '-')}"),
    ('MP15 Deployment', f"{m15.get('selection', {}).get('selected_model', '-')}", f"payload {m15.get('classical_int8', {}).get('payload_bytes_estimate', '-')} B, agreement {m15.get('classical_int8', {}).get('agreement_all_vs_float', '-')}"),
    ('MP16 Acceptance', 'replay test + aksi + FSM', f"{'PASS' if acc['required_acceptance_pass'] else 'FAIL'} | RTF {acc['pipeline_rtf_p95']:.4f}"),
]
html = f'''<!doctype html><html lang="id"><head><meta charset="utf-8"><title>Laporan Final KWS Embedded</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}td,th{{border:1px solid #ccc;padding:6px 8px;font-size:14px;text-align:left}}
th{{background:#eef}}.pass{{color:#080;font-weight:bold}}.fail{{color:#b00;font-weight:bold}}figure{{display:inline-block;margin:6px}}</style></head><body>
<h1>Laporan Final — Keyword Spotting Embedded (nyala / mati / naik / turun)</h1>
<p><b>Anggota Kelompok:</b> Christama Ezra Yudianto (245150307111009), Abdillah Fatah Suryanto (245150300111008),
Ivang Arfandia (245150307111003), Jahzeel Erason Nicodemus (245150300111002)</p>
<p>Dibangkitkan otomatis oleh <code>features/mp16_extension_system_integration.py</code>. Model deployment: <b>{acc['selected_model']}</b>
(SHA-256 header <code>{MODEL_SHA[:16]}…</code>). Status: <b>{status['status']}</b>. Semua latency adalah <b>host PC</b>, bukan ESP32/RPi.</p>
<h2>Ringkasan Mini Project 7–16</h2><table><tr><th>Tahap</th><th>Keputusan / konfigurasi</th><th>Hasil</th></tr>
{''.join(f'<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>' for a, b, c in rows_html)}</table>
<h2>Acceptance checklist MP16</h2><table><tr><th>Check</th><th>Hasil</th></tr>
{''.join(f'<tr><td>{k}</td><td class="{"pass" if v else "fail"}">{"PASS" if v else "FAIL"}</td></tr>' for k, v in acc['acceptance_checks'].items())}</table>
<h2>Fitur tambahan sistem (MP14–MP16)</h2><table><tr><th>Uji</th><th>Hasil</th></tr>
<tr><td>Streaming KWS VAD-endpoint (MP14, train+val)</td><td>{x14.get('continuous_stream', {}).get('results', {}).get('VAD-endpoint trigger (1 inferensi per ucapan)', {}).get('hits', '-')}/{x14.get('continuous_stream', {}).get('n_words', '-')} hit</td></tr>
<tr><td>Stream MP16 desain awal v1 (jendela 975 ms, endpoint absolut)</td><td>{stream_compare['v1_baseline']['kw_hit']}/{n_kw} hit; {len(stream_compare['v1_baseline']['failed_segments'])} segmen gagal</td></tr>
<tr><td>Stream MP16 desain final v2 (jendela 1,2 s, + drop relatif 20 dB) ({n_kw} keyword)</td><td>{stream_scores['kw_hit']}/{n_kw} hit, {stream_scores['kw_false_reject']} false-reject, {stream_scores['kw_substitution']} substitusi</td></tr>
<tr><td>Non-keyword speech ditolak</td><td>{stream_scores['nonkw_correct_reject']}/{len(nonkw)}</td></tr>
<tr><td>Non-speech ditolak</td><td>{stream_scores['nonspeech_correct_reject']}/{len(synth)}</td></tr>
<tr><td>Watchdog DMA stall</td><td>fail-safe {stream_scores['failsafe_events']}×, recover {stream_scores['recover_events']}×</td></tr>
<tr><td>FSM C == Python (gcc)</td><td>{fsm_c.get('trace_identical')}</td></tr></table>
<h2>Gambar utama</h2>
<figure>{img('results/figures/mp13_test_confusion_matrices.png')}<figcaption>MP13 confusion matrix test</figcaption></figure>
<figure>{img('results/figures/mp14_extension_realtime_analysis.png', 1000)}<figcaption>MP14 real-time</figcaption></figure>
<figure>{img('results/figures/mp15_extension_quantization_analysis.png', 1000)}<figcaption>MP15 kuantisasi</figcaption></figure>
<figure>{img('results/figures/mp16_extension_system_analysis.png', 1000)}<figcaption>MP16 sistem</figcaption></figure>
<h2>Catatan kejujuran ilmiah</h2><ul>
<li>Dataset MP9 = rekaman suara asli 4 anggota kelompok (96 WAV); split per acquisition block sehingga setiap penutur ada di
train, validation, dan test. Test berisi penutur yang <b>sudah dikenal</b> model (sesi berbeda), jadi akurasi bukan estimasi untuk penutur baru.</li>
<li>Test set ({m13.get('n_test')} sampel) dibuka sekali di MP13; interval kepercayaan masih lebar (Wilson 95% akurasi
{[round(v, 2) for v in m13.get('classical', {}).get('accuracy_wilson_95', [])]}).</li>
<li>Ucapan non-keyword untuk uji penolakan adalah TTS (kelompok tidak merekam kata non-keyword), sehingga laju penolakan belum mewakili suara manusia.</li>
<li>Latency/RAM perangkat (ESP32/RPi) belum diukur; sketch ESP32 disediakan sebagai baseline porting.</li></ul>
</body></html>'''
(ROOT/'deployment/mp16_final_report.html').write_text(html, encoding='utf-8')

final_files = ['mini_project_16.py', 'dashboard_mp16.py', 'features/kws_frontend.py', 'features/mp16_extension_system_integration.py',
               'project_status.json', 'deployment/mp15_manifest.json', 'deployment/mp16_final_project_manifest.json', 'deployment/mp16_final_report.html',
               'config/system_integration_config.json', 'config/deployment_config.json', 'config/realtime_architecture_config.json',
               'config/preprocessing_config.json', 'config/feature_config.json', 'config/project_config.json', 'config/filter_final_config.json',
               'models/mp15_classical_int8.npz', 'models/mp15_tiny_cnn_int8_torchscript.pt', 'features/intermediate/mp06_filter_final.npz',
               'results/metrics/mp13_final_test_metrics.json', 'results/metrics/mp15_deployment_metrics.json', 'results/metrics/mp16_acceptance_metrics.json',
               'results/predictions/mp16_end_to_end_predictions.csv', 'results/logs/mp16_event_log.jsonl']
final_files += [p.relative_to(ROOT).as_posix() for p in sorted(FW.glob('*.h'))] + ['firmware/esp32_kws_mp16/esp32_kws_mp16.ino']
final_files = [f for f in dict.fromkeys(final_files) if (ROOT/f).exists()]
sums = {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in final_files}
zpath = ROOT/'deployment/mp16_final_package.zip'
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in final_files: z.write(ROOT/f, f)
    z.writestr('SHA256SUMS.json', json.dumps(sums, indent=2))
with zipfile.ZipFile(zpath) as z:
    zip_ok = z.testzip() is None and all(hashlib.sha256(z.read(f)).hexdigest() == h for f, h in sums.items())
print(f'Paket final: {len(final_files)} file, {zpath.stat().st_size/1024:.1f} KiB, integritas = {zip_ok}')

ext = {'ood_gate': {'method': 'LedoitWolf Mahalanobis 39-D (fit train)', 'threshold': OOD_THR, 'calibration': 'train+val only',
                    'isolated_distractors': iso, 'isolated_keywords': kw_iso, 'keyword_test_acceptance': float(acc_test),
                    'nonkw_speech_rejection': float(rej_speech), 'non_speech_rejection': float(rej_ns)},
       'fsm': {'wd_ticks': WD_TICKS, 'cooldown_ticks': COOLDOWN_TICKS, 'level_bounds': [LVL_MIN, LVL_MAX], 'c_equivalence': fsm_c},
       'stream_scores': stream_scores, 'endpoint_design_comparison': stream_compare, 'stream_segments': [{k: v for k, v in s.items() if k != 'event'} for s in seg_res],
       'event_log': 'results/logs/mp16_event_log.jsonl', 'n_events': len(events), 'dashboard_test': dash,
       'esp32_sketch': 'firmware/esp32_kws_mp16/esp32_kws_mp16.ino', 'final_report': 'deployment/mp16_final_report.html',
       'final_package': {'zip': 'deployment/mp16_final_package.zip', 'n_files': len(final_files), 'integrity_ok': zip_ok},
       'model_changed': False, 'post_test_tuning_performed': False,
       'note': f'Stream memakai seluruh {n_kw} ucapan (termasuk test yang sudah dibuka MP13) -> uji integrasi sistem, bukan estimasi akurasi baru.'}
(OUT_MET/'mp16_extension_system_metrics.json').write_text(json.dumps(ext, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o)), encoding='utf-8')
print('Fitur tambahan MP16 selesai -> results/metrics/mp16_extension_system_metrics.json, deployment/mp16_final_report.html')

# %%
if "__file__" not in globals():
    from IPython.display import Image, display
    display(Image(filename=str(OUT_FIG/'mp16_extension_system_analysis.png'), width=1000))
