# %% [markdown]
# ## FITUR TAMBAHAN MP11 — Keluarga Model Tambahan, Robustness, Export C & Verifikasi gcc
# Modul mandiri setelah `mini_project_11.py`. Model beku MP11 **tidak diubah**, test **tidak dibuka**.
#
# Fitur tambahan (materi Bab 11: kernel SVM, hardware-aware, parameter tersimpan):
# 1. **Keluarga tambahan**: SVM RBF, Gaussian Naive Bayes, LDA classifier (protokol identik).
# 2. **Overfitting gap** (train-fit F1 vs GroupKFold F1) & **robustness terhadap derau fitur**
#    (σ = 0.25/0.5/1.0 pada fitur terstandar, GroupKFold TRAIN) — semua keluarga seri di F1=1.0,
#    sehingga robustness memberi informasi tambahan tentang margin keputusan.
# 3. **Arsip semua kandidat beku** (`models/mp11_family_models/*.joblib`, fit TRAIN saja) + SHA-256,
#    agar jalur SVM INT8 versi buku tetap dapat dibangun di MP15 tanpa retraining.
# 4. **Export model terpilih ke C** (`firmware/mp11_classifier_float.h`, tree → if/else, linear → dot product)
#    lalu **dikompilasi dengan gcc** dan diverifikasi identik dengan scikit-learn pada TRAIN+VALIDATION.
# 5. **Latency inferensi host** per keluarga (proxy).
#
# Anggota Kelompok : Christama Ezra Yudianto (245150307111009), Abdillah Fatah Suryanto (245150300111008),
# Ivang Arfandia (245150307111003), Jahzeel Erason Nicodemus (245150300111002)

# %%
from pathlib import Path
import json, hashlib, shutil, subprocess, time, warnings
import numpy as np
import matplotlib
if "__file__" in globals():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
SEED = 2026
status = json.loads((ROOT/'project_status.json').read_text(encoding='utf-8'))
assert status['status'] in ('ready_for_mp12',), 'Jalankan mini_project_11.py terlebih dahulu'
cfg = json.loads((ROOT/'config/classifier_config.json').read_text(encoding='utf-8'))
rep = json.loads((ROOT/'results/metrics/mp11_classifier_metrics.json').read_text(encoding='utf-8'))
D = np.load(ROOT/'features/dataset/mp10_features_final.npz', allow_pickle=True)
X = D['X'].astype(np.float64); y = D['y'].astype(str); g = D['groups'].astype(str)
tr = D['train_mask'].astype(bool); va = D['val_mask'].astype(bool)
Xtr, ytr, gtr, Xv, yv = X[tr], y[tr], g[tr], X[va], y[va]
classes = list(cfg['classes'])
OUT_MET = ROOT/'results/metrics'; OUT_FIG = ROOT/'results/figures'; FW = ROOT/'firmware'
FAM_DIR = ROOT/'models/mp11_family_models'
for d in (FW, FAM_DIR):
    d.mkdir(parents=True, exist_ok=True)
cv = GroupKFold(n_splits=min(4, len(np.unique(gtr))))

# rebuild model tuned MP11 dari best_params (fit TRAIN saja — sama persis dengan MP11)
exec_ns = {}
src = (ROOT/'mini_project_11.py').read_text(encoding='utf-8')
build_src = src[src.index('def build(name, p):'):src.index('# %% [markdown]\n# ## 3.')]
exec('from sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.linear_model import LogisticRegression\n'
     'from sklearn.svm import LinearSVC\nfrom sklearn.tree import DecisionTreeClassifier\n'
     'from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.neural_network import MLPClassifier\n'
     f'SEED={SEED}\n' + build_src, exec_ns)
build = exec_ns['build']


def fam_model(name):
    p = dict(rep['family_results'][name]['best_params'])
    if 'hidden_layer_sizes' in p:
        p['hidden_layer_sizes'] = tuple(p['hidden_layer_sizes'])
    return build(name, p)


def cv_f1(est, Xa, noise=0.0, reps=1):
    rng = np.random.default_rng(SEED)
    s = []
    for a, b in cv.split(Xa, ytr, gtr):
        m = clone(est).fit(Xa[a], ytr[a])
        for _ in range(reps):
            Xb = Xa[b] + (rng.normal(0, noise, Xa[b].shape) if noise else 0)
            s.append(f1_score(ytr[b], m.predict(Xb), average='macro'))
    return float(np.mean(s))

# %% [markdown]
# ### 1. Keluarga tambahan: SVM RBF, Gaussian NB, LDA (grid kecil, protokol identik MP11)

# %%
extra = {}
grids = {'svm_rbf': [dict(C=c, gamma='scale') for c in (0.3, 1.0, 3.0)],
         'gaussian_nb': [dict(var_smoothing=v) for v in (1e-9, 1e-3)],
         'lda': [dict(solver='svd')]}
mk = {'svm_rbf': lambda p: SVC(kernel='rbf', random_state=SEED, **p), 'gaussian_nb': lambda p: GaussianNB(**p),
      'lda': lambda p: LinearDiscriminantAnalysis(**p)}
for fam, grid in grids.items():
    trials = [(cv_f1(mk[fam](p), Xtr), p) for p in grid]
    best_cv, best_p = max(trials, key=lambda t: t[0])
    m = mk[fam](best_p).fit(Xtr, ytr)
    sv = int(m.support_vectors_.shape[0]) if fam == 'svm_rbf' else None
    extra[fam] = {'best_params': best_p, 'cv_macro_f1': best_cv,
                  'validation_macro_f1': float(f1_score(yv, m.predict(Xv), average='macro')),
                  'support_vectors': sv,
                  'ops_proxy': (sv*Xtr.shape[1]*3 if sv else (len(classes)*Xtr.shape[1]*4 if fam == 'gaussian_nb' else len(classes)*Xtr.shape[1]*2))}
    print(f"  {fam:12s} {best_p}  CV F1={best_cv:.3f}  val F1={extra[fam]['validation_macro_f1']:.3f}  ops~{extra[fam]['ops_proxy']}")

# %% [markdown]
# ### 2. Overfitting gap & robustness terhadap derau fitur (GroupKFold TRAIN)

# %%
fams = list(rep['family_results'].keys())
robust = {}
for fam in fams:
    est = fam_model(fam)
    train_fit = f1_score(ytr, clone(est).fit(Xtr, ytr).predict(Xtr), average='macro')
    robust[fam] = {'train_fit_f1': float(train_fit), 'cv_f1': rep['family_results'][fam]['cv_macro_f1_mean'],
                   'overfit_gap': float(train_fit - rep['family_results'][fam]['cv_macro_f1_mean']),
                   'cv_f1_feature_noise': {str(s): cv_f1(est, Xtr, noise=s, reps=20) for s in (0.25, 0.5, 1.0)}}
for fam in extra:
    est = mk[fam](extra[fam]['best_params'])
    robust[fam] = {'train_fit_f1': float(f1_score(ytr, clone(est).fit(Xtr, ytr).predict(Xtr), average='macro')),
                   'cv_f1': extra[fam]['cv_macro_f1'], 'overfit_gap': None,
                   'cv_f1_feature_noise': {str(s): cv_f1(est, Xtr, noise=s, reps=20) for s in (0.25, 0.5, 1.0)}}
    robust[fam]['overfit_gap'] = robust[fam]['train_fit_f1'] - robust[fam]['cv_f1']
for fam, r in robust.items():
    print(f"  {fam:12s} gap={r['overfit_gap']:+.3f}  noise sigma=0.25/0.5/1.0 -> " +
          ' / '.join(f"{v:.3f}" for v in r['cv_f1_feature_noise'].values()))

# %% [markdown]
# ### 3. Arsip semua kandidat beku (fit TRAIN saja) + SHA-256

# %%
archive = {}
for fam in fams:
    m = fam_model(fam).fit(Xtr, ytr)
    pred_same = np.array_equal(m.predict(Xv), np.asarray(rep['family_results'][fam]['validation_predictions']))
    p = FAM_DIR/f'mp11_{fam}.joblib'
    joblib.dump(m, p)
    archive[fam] = {'file': p.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                    'reproduces_mp11_validation_predictions': bool(pred_same), 'fit_scope': 'training_only',
                    'test_used': False}
print('Arsip kandidat:', {k: v['reproduces_mp11_validation_predictions'] for k, v in archive.items()})

# %% [markdown]
# ### 4. Export model terpilih ke C + kompilasi gcc + verifikasi identik (TRAIN+VALIDATION)

# %%
chosen = cfg['chosen_family']
model = joblib.load(ROOT/cfg['model_file'])
d = X.shape[1]
lines = ['#pragma once', '/* Auto-generated by Mini Project 11 - FITUR TAMBAHAN: classifier float32 C */',
         f'#define MP11_INPUT_DIM {d}', f'#define MP11_N_CLASSES {len(classes)}',
         '/* class order: ' + ', '.join(f'{i}={c}' for i, c in enumerate(classes)) + ' */']
if chosen == 'tree':
    t = model.tree_
    def emit(node, ind):
        if t.children_left[node] == -1:
            cls = int(np.argmax(t.value[node][0]))
            return [f'{ind}return {cls}; /* {model.classes_[cls]} */']
        out = [f'{ind}if (x[{t.feature[node]}] <= {t.threshold[node]:.9g}f) {{']
        out += emit(t.children_left[node], ind + '    ')
        out += [f'{ind}}} else {{'] + emit(t.children_right[node], ind + '    ') + [f'{ind}}}']
        return out
    lines += ['static inline int mp11_predict(const float *x) {'] + emit(0, '    ') + ['}']
elif chosen in ('logreg', 'linear_svm'):
    W = np.asarray(model.coef_, np.float32); b = np.asarray(model.intercept_, np.float32)
    lines += [f'static const float MP11_W[{W.size}] = {{ ' + ', '.join(f'{v:.9g}f' for v in W.ravel()) + ' };',
              f'static const float MP11_B[{b.size}] = {{ ' + ', '.join(f'{v:.9g}f' for v in b) + ' };',
              'static inline int mp11_predict(const float *x) {',
              '    int best = 0; float bs = -1e30f;',
              '    for (int k = 0; k < MP11_N_CLASSES; ++k) { float s = MP11_B[k];',
              '        for (int j = 0; j < MP11_INPUT_DIM; ++j) s += MP11_W[k*MP11_INPUT_DIM+j]*x[j];',
              '        if (s > bs) { bs = s; best = k; } }', '    return best;', '}']
else:
    lines += ['/* Export C otomatis hanya untuk tree/linear; family lain memakai joblib reference. */']
(FW/'mp11_classifier_float.h').write_text('\n'.join(lines) + '\n', encoding='utf-8')

c_check = {'compiled': False}
gcc = shutil.which('gcc')
idx_tv = np.where(tr | va)[0]
if gcc and chosen in ('tree', 'logreg', 'linear_svm'):
    build_dir = ROOT/'firmware/build'; build_dir.mkdir(exist_ok=True)
    vec = ', '.join(f'{v:.9g}f' for v in X[idx_tv].astype(np.float32).ravel())
    test_c = f"""#include <stdio.h>
#include "../mp11_classifier_float.h"
static const float XS[{len(idx_tv)*d}] = {{ {vec} }};
int main(void) {{ for (int i = 0; i < {len(idx_tv)}; ++i) printf("%d\\n", mp11_predict(&XS[i*{d}])); return 0; }}
"""
    (build_dir/'mp11_test.c').write_text(test_c, encoding='utf-8')
    exe = build_dir/'mp11_test.exe'
    subprocess.run([gcc, '-O2', '-std=c99', '-Wall', '-o', str(exe), str(build_dir/'mp11_test.c')], check=True, capture_output=True)
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout.split()
    c_pred = np.array([classes[int(v)] for v in out])
    py_pred = model.predict(X[idx_tv]).astype(str)
    c_check = {'compiled': True, 'compiler': subprocess.run([gcc, '--version'], capture_output=True, text=True).stdout.splitlines()[0],
               'n_vectors': int(len(idx_tv)), 'scope': 'train+validation (test tidak dibuka)',
               'agreement_c_vs_sklearn': float(np.mean(c_pred == py_pred)), 'header': 'firmware/mp11_classifier_float.h'}
print('Verifikasi C (gcc):', c_check)
if chosen == 'tree':
    print((ROOT/'models/mp11_tree_rules.txt').read_text(encoding='utf-8'))

# %% [markdown]
# ### 5. Latency inferensi host per keluarga + simpan + visualisasi

# %%
lat = {}
x1 = Xv[:1]
for fam in fams:
    m = joblib.load(ROOT/archive[fam]['file'])
    for _ in range(20): m.predict(x1)
    t0 = time.perf_counter()
    for _ in range(200): m.predict(x1)
    lat[fam] = (time.perf_counter()-t0)/200*1e3
ext = {'extra_families': extra, 'robustness_and_overfit': robust, 'frozen_family_archive': archive,
       'c_export_verification': c_check, 'host_latency_ms_per_sample': lat, 'chosen_family_mp11': chosen,
       'note': 'Analisis tambahan pada TRAIN/VALIDATION. Test set tidak dibuka. Pilihan MP11 tidak diubah.'}
(OUT_MET/'mp11_extension_classifier_metrics.json').write_text(json.dumps(ext, indent=2), encoding='utf-8')

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
allf = list(robust.keys())
for s, mkr in zip(('0.25', '0.5', '1.0'), 'os^'):
    ax[0].plot(allf, [robust[f]['cv_f1_feature_noise'][s] for f in allf], mkr+'-', label=f'σ={s}')
ax[0].set_title('Robustness: GroupKFold F1 dengan derau fitur'); ax[0].set_ylim(0, 1.05); ax[0].legend(); ax[0].tick_params(axis='x', rotation=35)
ax[1].bar(allf, [robust[f]['overfit_gap'] for f in allf]); ax[1].set_title('Overfitting gap (train-fit − CV)'); ax[1].tick_params(axis='x', rotation=35)
ax[2].bar(list(lat), list(lat.values())); ax[2].set_ylabel('ms/sampel'); ax[2].set_title('Latency host (proxy)'); ax[2].tick_params(axis='x', rotation=35)
for a in ax: a.grid(alpha=.2)
fig.suptitle(f'MP11 FITUR TAMBAHAN - model terpilih: {chosen} (C export agreement = {c_check.get("agreement_c_vs_sklearn", "n/a")})')
fig.tight_layout(); fig.savefig(OUT_FIG/'mp11_extension_classifier_analysis.png', dpi=150); plt.close(fig)
print('Fitur tambahan MP11 selesai -> results/metrics/mp11_extension_classifier_metrics.json')

# %%
if "__file__" not in globals():
    from IPython.display import Image, display
    for p in ['mp11_classifier_comparison.png', 'mp11_resource_tradeoff.png', 'mp11_validation_confusion_matrix.png', 'mp11_extension_classifier_analysis.png']:
        display(Image(filename=str(OUT_FIG/p), width=760))
