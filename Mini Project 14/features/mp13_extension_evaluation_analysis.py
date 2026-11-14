# %% [markdown]
# ## FITUR TAMBAHAN MP13 — Uncertainty, Uji Signifikansi, Kalibrasi, ROC/PR & Robustness Stress Test
# Modul mandiri setelah `mini_project_13.py`. **Tidak ada post-test tuning**: semua model dibaca
# beku (hash diverifikasi), hasil di sini tidak dipakai untuk memilih/mengubah model.
#
# Fitur tambahan (materi Bab 13: uncertainty, ROC/PR, calibration/Brier, distribusi kesalahan):
# 1. **Bootstrap 95% CI** macro-F1 & accuracy (sampel test di-resample).
# 2. **Uji McNemar exact** classical vs CNN (apakah perbedaan signifikan?).
# 3. **Kalibrasi**: Brier score, NLL, ECE + reliability diagram (CNN softmax & tree leaf-probability).
# 4. **ROC-AUC & PR-AP one-vs-rest** per kelas.
# 5. **Robustness stress test** pada WAV test: derau putih SNR 30→0 dB, gain −12/+6 dB, time-shift,
#    memakai front-end beku `kws_frontend.py` (MP9) → kurva degradasi per model.
# 6. **Evaluasi eksploratif** semua kandidat beku MP11 (arsip keluarga) & varian CNN MP12
#    (registry SHA-256) — dilaporkan terpisah, BUKAN dasar seleksi deployment.
#
# Anggota Kelompok : Christama Ezra Yudianto (245150307111009), Abdillah Fatah Suryanto (245150300111008),
# Ivang Arfandia (245150307111003), Jahzeel Erason Nicodemus (245150300111002)

# %%
from pathlib import Path
import json, hashlib, sys, warnings, importlib.util
from math import comb
import numpy as np
import matplotlib
if "__file__" in globals():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT/'features'))
from kws_frontend import KWSFrontend  # noqa: E402

status = json.loads((ROOT/'project_status.json').read_text(encoding='utf-8'))
assert status['status'] in ('ready_for_mp14',), 'Jalankan mini_project_13.py terlebih dahulu'
rep = json.loads((ROOT/'results/metrics/mp13_final_test_metrics.json').read_text(encoding='utf-8'))
clf_cfg = json.loads((ROOT/'config/classifier_config.json').read_text(encoding='utf-8'))
dl_cfg = json.loads((ROOT/'config/deep_learning_config.json').read_text(encoding='utf-8'))
red = json.loads((ROOT/'config/feature_reduction_config.json').read_text(encoding='utf-8'))
split = json.loads((ROOT/'splits/mp09_split.json').read_text(encoding='utf-8'))
classes = rep['classes']; c2i = {c: i for i, c in enumerate(classes)}
test_ids = split['test_ids']
OUT_MET = ROOT/'results/metrics'; OUT_FIG = ROOT/'results/figures'
fe = KWSFrontend(ROOT)
SEL = np.asarray(red['selected_indices'], int)
MU39 = np.asarray(red['scaler_mean']); SC39 = np.asarray(red['scaler_scale'])
rng = np.random.default_rng(2026)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# Arsitektur CNN (resmi + varian eksplorasi) dimuat dari skrip MP12 tambahan agar identik
spec = importlib.util.spec_from_file_location('mp12x', ROOT/'features/mp12_extension_cnn_analysis.py')
src12 = (ROOT/'features/mp12_extension_cnn_analysis.py').read_text(encoding='utf-8')
arch_ns = {'nn': nn, 'torch': torch, 'np': np, 'random': __import__('random'), 'F': 26, 'classes': classes}
for cls_name in ('TinyKWSNet', 'DSCNN', 'TinyKWSFreqPool'):
    s0 = src12.index(f'class {cls_name}(')
    s1 = src12.index('\n\n\n', s0)
    exec(src12[s0:s1], arch_ns)

# %% [markdown]
# ### Muat semua model beku + verifikasi hash

# %%
models = {}
official_hash = json.loads((ROOT/'results/metrics/mp13_pretest_artifact_hashes.json').read_text(encoding='utf-8'))
assert sha(ROOT/'models/mp11_classical_baseline.joblib') == official_hash['models/mp11_classical_baseline.joblib']
assert sha(ROOT/'models/mp12_tiny_cnn_state.pt') == official_hash['models/mp12_tiny_cnn_state.pt']
models['classical_' + clf_cfg['chosen_family'] + ' (resmi)'] = ('sk', joblib.load(ROOT/clf_cfg['model_file']))


def load_cnn(path, arch):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    net = arch_ns[arch](len(classes)); net.load_state_dict(ck['state_dict']); net.eval()
    return ('cnn', net, float(ck['normalization']['mean']), float(ck['normalization']['std']))


models['tiny_cnn (resmi)'] = load_cnn(ROOT/dl_cfg['model_file'], 'TinyKWSNet')
hash_ok = {}
mp11x = ROOT/'results/metrics/mp11_extension_classifier_metrics.json'
if mp11x.exists():
    for fam, info in json.loads(mp11x.read_text(encoding='utf-8'))['frozen_family_archive'].items():
        ok = sha(ROOT/info['file']) == info['sha256']; hash_ok[f'mp11_{fam}'] = ok
        if ok and fam != clf_cfg['chosen_family']:
            models[f'mp11_{fam} (eksplorasi)'] = ('sk', joblib.load(ROOT/info['file']))
reg = ROOT/'config/mp12_extension_candidates.json'
if reg.exists():
    for name, info in json.loads(reg.read_text(encoding='utf-8')).items():
        ok = sha(ROOT/info['file']) == info['sha256']; hash_ok[f'mp12_{name}'] = ok
        if ok:
            models[f'cnn_{name} (eksplorasi)'] = load_cnn(ROOT/info['file'], info['architecture'])
print('Hash eksplorasi valid:', hash_ok)
print('Model dievaluasi:', list(models))


def predict(m, v39_list, lm_list):
    """Return (pred labels, prob matrix or None)."""
    if m[0] == 'sk':
        Z = ((np.stack(v39_list)-MU39)/SC39)[:, SEL]
        pr = m[1].predict(Z).astype(str)
        P = None
        if hasattr(m[1], 'predict_proba'):
            P = np.zeros((len(Z), len(classes)))
            P[:, [c2i[c] for c in m[1].classes_]] = m[1].predict_proba(Z)
        return pr, P
    _, net, mu, sd = m
    Xl = torch.from_numpy(np.stack([(fe.fixed_logmel(l)-mu)/sd for l in lm_list])[:, None]).float()
    with torch.no_grad():
        P = torch.softmax(net(Xl), 1).numpy()
    return np.array([classes[i] for i in P.argmax(1)]), P

# %% [markdown]
# ### 1–4. Evaluasi test bersih: bootstrap CI, McNemar, kalibrasi, ROC/PR

# %%
wavs = {sid: fe.read_wav(ROOT/'data/raw/mp09_dataset'/f'{sid}.wav') for sid in test_ids}
y_true = np.array([sid.split('_b')[0] for sid in test_ids])
clean_feats = [fe.features(wavs[s]) for s in test_ids]
v_clean = [f[0] for f in clean_feats]; lm_clean = [f[1] for f in clean_feats]
results, preds, probs = {}, {}, {}
for name, m in models.items():
    pr, P = predict(m, v_clean, lm_clean)
    preds[name], probs[name] = pr, P
    bs_f1, bs_acc = [], []
    for _ in range(2000):
        b = rng.integers(0, len(y_true), len(y_true))
        bs_f1.append(f1_score(y_true[b], pr[b], average='macro', zero_division=0)); bs_acc.append(np.mean(y_true[b] == pr[b]))
    r = {'macro_f1': float(f1_score(y_true, pr, average='macro', zero_division=0)), 'accuracy': float(np.mean(pr == y_true)),
         'bootstrap95_macro_f1': [float(np.percentile(bs_f1, 2.5)), float(np.percentile(bs_f1, 97.5))],
         'bootstrap95_accuracy': [float(np.percentile(bs_acc, 2.5)), float(np.percentile(bs_acc, 97.5))],
         'predictions': pr.tolist()}
    if P is not None:
        Y = np.eye(len(classes))[[c2i[c] for c in y_true]]
        conf = P.max(1); corr = (pr == y_true).astype(float)
        bins = np.linspace(0, 1, 6); ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mk = (conf > lo) & (conf <= hi)
            if mk.any(): ece += mk.mean()*abs(conf[mk].mean()-corr[mk].mean())
        r.update({'brier': float(np.mean(np.sum((P-Y)**2, 1))), 'nll': float(-np.mean(np.log(np.clip((P*Y).sum(1), 1e-12, 1)))),
                  'ece_5bins': float(ece), 'mean_confidence': float(conf.mean())})
        try:
            r['roc_auc_ovr_macro'] = float(roc_auc_score(Y, P, average='macro'))
            r['pr_ap_ovr_macro'] = float(average_precision_score(Y, P, average='macro'))
        except ValueError:
            pass
    results[name] = r
    print(f"  {name:32s} F1={r['macro_f1']:.3f} CI95={np.round(r['bootstrap95_macro_f1'], 2).tolist()} "
          + (f"Brier={r['brier']:.3f} ECE={r['ece_5bins']:.3f}" if 'brier' in r else ''))
off_c = [k for k in models if k.startswith('classical_')][0]
# Konsistensi: macro-F1 resmi hasil re-ekstraksi dari WAV harus sama dengan laporan mini_project_13.py
consistency = {'classical_macro_f1_report': rep.get('classical', {}).get('macro_f1'),
               'classical_macro_f1_recomputed': results[off_c]['macro_f1'],
               'cnn_macro_f1_report': rep.get('tiny_cnn', {}).get('macro_f1'),
               'cnn_macro_f1_recomputed': results['tiny_cnn (resmi)']['macro_f1']}
print('Konsistensi dengan laporan resmi:', consistency)


def mcnemar_exact(a_ok, b_ok):
    b = int(np.sum(a_ok & ~b_ok)); c = int(np.sum(~a_ok & b_ok)); n = b+c
    p = 1.0 if n == 0 else min(1.0, 2*sum(comb(n, k) for k in range(0, min(b, c)+1))/2**n)
    return {'a_only_correct': b, 'b_only_correct': c, 'p_value_exact': float(p)}


mcn = mcnemar_exact(preds[off_c] == y_true, preds['tiny_cnn (resmi)'] == y_true)
print('McNemar classical vs CNN resmi:', mcn)

# %% [markdown]
# ### 5. Robustness stress test (derau, gain, time-shift) pada WAV test

# %%
def add_noise(x, snr):
    n = rng.normal(size=len(x)); return x + n*np.sqrt(np.mean(x**2)/(np.mean(n**2)*10**(snr/10)))


conditions = {'clean': lambda x: x}
for snr in (30, 20, 10, 5, 0):
    conditions[f'snr_{snr}dB'] = (lambda s: (lambda x: add_noise(x, s)))(snr)
conditions['gain_-12dB'] = lambda x: x*10**(-12/20)
conditions['gain_+6dB'] = lambda x: np.clip(x*2.0, -1, 1)
conditions['shift_+200ms'] = lambda x: np.r_[np.zeros(3200), x[:-3200]]
robust = {name: {} for name in models}
REPS = 5
for cname, fn in conditions.items():
    reps = 1 if cname in ('clean', 'gain_-12dB', 'gain_+6dB', 'shift_+200ms') else REPS
    acc = {name: [] for name in models}
    for _ in range(reps):
        feats = [fe.features(fn(wavs[s])) for s in test_ids]
        for name, m in models.items():
            pr, _ = predict(m, [f[0] for f in feats], [f[1] for f in feats])
            acc[name].append(np.mean(pr == y_true))
    for name in models:
        robust[name][cname] = float(np.mean(acc[name]))
print('Robustness (accuracy):')
for name in models:
    print(f'  {name:32s}', ' '.join(f'{k}={v:.2f}' for k, v in robust[name].items()))

# %% [markdown]
# ### 6. Simpan + visualisasi

# %%
ext = {"consistency_with_official_report": consistency, "clean_test_results": results, 'mcnemar_classical_vs_cnn_official': mcn, 'robustness_accuracy': robust,
       'hash_verification_exploratory': hash_ok, 'n_test': len(test_ids),
       'post_test_tuning_performed': False, 'used_for_model_selection': False,
       'note': ('Model eksplorasi dibekukan + di-hash sebelum MP13. Hasil eksplorasi bersifat informatif '
                '(rekomendasi untuk siklus pengembangan berikutnya dengan data baru), bukan pengganti kandidat resmi.')}
(OUT_MET/'mp13_extension_evaluation_metrics.json').write_text(json.dumps(ext, indent=2), encoding='utf-8')

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
names = list(models)
short = [n.replace(' (eksplorasi)', '*').replace(' (resmi)', ' [R]') for n in names]
f1s = [results[n]['macro_f1'] for n in names]
lo = [results[n]['macro_f1']-results[n]['bootstrap95_macro_f1'][0] for n in names]
hi = [results[n]['bootstrap95_macro_f1'][1]-results[n]['macro_f1'] for n in names]
ax[0].barh(short, f1s, xerr=[lo, hi], color=['C3' if '[R]' in s else 'C0' for s in short], capsize=3)
ax[0].set_xlim(0, 1.05); ax[0].set_title('Test macro-F1 + bootstrap 95% CI ([R]=resmi, *=eksplorasi)'); ax[0].tick_params(axis='y', labelsize=8)
conds = [c for c in conditions if c.startswith(('clean', 'snr'))]
for n, s in zip(names, short):
    ax[1].plot(range(len(conds)), [robust[n][c] for c in conds], 'o-' if '[R]' in s else '.--', lw=2.2 if '[R]' in s else 1, label=s)
ax[1].set_xticks(range(len(conds)), conds, rotation=25); ax[1].set_ylim(-0.05, 1.05)
ax[1].set_title('Robustness: accuracy vs derau putih (test WAV)'); ax[1].legend(fontsize=6)
for n, s in zip(names, short):
    P = probs[n]
    if P is None: continue
    conf = P.max(1); corr = (preds[n] == y_true)
    ax[2].scatter(conf, corr + rng.uniform(-0.04, 0.04, len(conf)), label=s, s=30)
ax[2].plot([0, 1], [0, 1], 'k:'); ax[2].set_xlabel('confidence (max prob)'); ax[2].set_ylabel('benar (1) / salah (0)')
ax[2].set_title('Reliability (per sampel, n=4)'); ax[2].legend(fontsize=6)
for a in ax: a.grid(alpha=.2)
fig.suptitle(f"MP13 FITUR TAMBAHAN - McNemar p={mcn['p_value_exact']:.2f} (n=4: perbedaan tidak dapat diklaim signifikan)")
fig.tight_layout(); fig.savefig(OUT_FIG/'mp13_extension_evaluation_analysis.png', dpi=140); plt.close(fig)
print('Fitur tambahan MP13 selesai -> results/metrics/mp13_extension_evaluation_metrics.json')

# %%
if "__file__" not in globals():
    from IPython.display import Image, display
    for p in ['mp13_test_confusion_matrices.png', 'mp13_test_metric_comparison.png', 'mp13_resource_proxy.png', 'mp13_extension_evaluation_analysis.png']:
        display(Image(filename=str(OUT_FIG/p), width=820))
