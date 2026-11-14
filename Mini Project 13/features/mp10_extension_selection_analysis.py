# %% [markdown]
# ## FITUR TAMBAHAN MP10 — Embedded Selection, Stabilitas, Kurva-k & Cost-Aware Front-End
# Modul mandiri setelah `mini_project_10.py`. **Semua analisis TRAIN-only + GroupKFold**;
# representasi beku MP10 tidak diubah (analisis pembanding saja).
#
# Fitur tambahan (materi Bab 10 — keluarga *embedded*, stabilitas, hardware-aware):
# 1. **Embedded methods**: L1-Logistic Regression & Random-Forest importance (seleksi di dalam fold).
# 2. **Stabilitas seleksi** antar fold (indeks Jaccard rata-rata) untuk MI, RFE, L1, RF.
# 3. **Kurva performa vs jumlah fitur k** (MI ranking di-fit ulang di tiap fold → tanpa leakage).
# 4. **Permutation importance** out-of-fold untuk fitur terpilih.
# 5. **Cost-aware front-end**: kelompok fitur yang benar-benar dibutuhkan subset terpilih →
#    estimasi penghematan MAC/frame memakai profil biaya MP8 (fitur tambahan MP8).
# 6. **Proyeksi LDA 2D** (fit TRAIN) sebagai visual separabilitas.
#
# Anggota Kelompok : Christama Ezra Yudianto (245150307111009), Abdillah Fatah Suryanto (245150300111008),
# Ivang Arfandia (245150307111003), Jahzeel Erason Nicodemus (245150300111002)

# %%
from pathlib import Path
import json
import warnings
import numpy as np
import matplotlib
if "__file__" in globals():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
SEED = 2026
status = json.loads((ROOT/'project_status.json').read_text(encoding='utf-8'))
assert status['status'] in ('ready_for_mp11',), 'Jalankan mini_project_10.py terlebih dahulu'
red = json.loads((ROOT/'config/feature_reduction_config.json').read_text(encoding='utf-8'))
D = np.load(ROOT/'features/dataset/mp09_features_39d.npz', allow_pickle=True)
X = D['X'].astype(np.float64); y = D['y'].astype(str); g = D['groups'].astype(str)
names = D['feature_names'].astype(str); tr = D['train_mask'].astype(bool)
Xtr, ytr, gtr = X[tr], y[tr], g[tr]
K = int(red['output_dim']) if red['selected_indices'] else 8
OUT_MET = ROOT/'results/metrics'; OUT_FIG = ROOT/'results/figures'
cv = GroupKFold(n_splits=len(np.unique(gtr)))
base = LogisticRegression(max_iter=3000, random_state=SEED)


def select(method, Z, yy, k):
    if method == 'mi':
        return np.sort(np.argsort(mutual_info_classif(Z, yy, random_state=SEED))[::-1][:k])
    if method == 'rfe':
        return np.where(RFE(clone(base), n_features_to_select=k, step=0.2).fit(Z, yy).support_)[0]
    if method == 'l1':
        m = LogisticRegression(penalty='l1', solver='saga', C=0.5, max_iter=20000, random_state=SEED).fit(Z, yy)
        return np.sort(np.argsort(np.abs(m.coef_).sum(0))[::-1][:k])
    if method == 'rf':
        m = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=SEED).fit(Z, yy)
        return np.sort(np.argsort(m.feature_importances_)[::-1][:k])
    raise KeyError(method)

# %% [markdown]
# ### 1–2. Embedded methods + stabilitas seleksi antar fold (Jaccard)

# %%
methods = ['mi', 'rfe', 'l1', 'rf']
res = {}
for meth in methods:
    sets, f1s = [], []
    for a, b in cv.split(Xtr, ytr, gtr):
        sc = StandardScaler().fit(Xtr[a])
        Za, Zb = sc.transform(Xtr[a]), sc.transform(Xtr[b])
        idx = select(meth, Za, ytr[a], K)
        sets.append(set(idx.tolist()))
        clf = clone(base).fit(Za[:, idx], ytr[a])
        f1s.append(f1_score(ytr[b], clf.predict(Zb[:, idx]), average='macro'))
    jac = [len(s1 & s2)/len(s1 | s2) for i, s1 in enumerate(sets) for s2 in sets[i+1:]]
    freq = np.zeros(X.shape[1])
    for s in sets:
        freq[list(s)] += 1
    res[meth] = {'cv_macro_f1_mean': float(np.mean(f1s)), 'cv_macro_f1_std': float(np.std(f1s)),
                 'jaccard_stability_mean': float(np.mean(jac)), 'k': K,
                 'always_selected': names[freq == len(sets)].tolist()}
    print(f"  {meth:4s} k={K}: CV F1={np.mean(f1s):.3f}±{np.std(f1s):.3f} | Jaccard stabilitas={np.mean(jac):.3f}")

# %% [markdown]
# ### 3. Kurva performa vs k (MI ranking di-fit di dalam fold)

# %%
ks = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 26, 39]
curve = []
for k in ks:
    f1s = []
    for a, b in cv.split(Xtr, ytr, gtr):
        sc = StandardScaler().fit(Xtr[a]); Za, Zb = sc.transform(Xtr[a]), sc.transform(Xtr[b])
        idx = select('mi', Za, ytr[a], k)
        f1s.append(f1_score(ytr[b], clone(base).fit(Za[:, idx], ytr[a]).predict(Zb[:, idx]), average='macro'))
    curve.append((k, float(np.mean(f1s)), float(np.std(f1s))))
k_min_best = min(k for k, m, _ in curve if m >= max(c[1] for c in curve) - 0.03)
print('Kurva k:', [(k, round(m, 3)) for k, m, _ in curve], '| k minimum dalam toleransi 0.03 =', k_min_best)

# %% [markdown]
# ### 4. Permutation importance out-of-fold (fitur terpilih MP10)

# %%
sel = np.asarray(red['selected_indices'], int) if red['selected_indices'] else np.arange(X.shape[1])
rng = np.random.default_rng(SEED)
drops = np.zeros(len(sel))
for a, b in cv.split(Xtr, ytr, gtr):
    sc = StandardScaler().fit(Xtr[a]); Za, Zb = sc.transform(Xtr[a])[:, sel], sc.transform(Xtr[b])[:, sel]
    clf = clone(base).fit(Za, ytr[a])
    base_f1 = f1_score(ytr[b], clf.predict(Zb), average='macro')
    for j in range(len(sel)):
        d = []
        for _ in range(20):
            Zp = Zb.copy(); Zp[:, j] = rng.permutation(Zp[:, j])
            d.append(base_f1 - f1_score(ytr[b], clf.predict(Zp), average='macro'))
        drops[j] += np.mean(d)/cv.get_n_splits()
perm = sorted([(str(names[i]), float(v)) for i, v in zip(sel, drops)], key=lambda t: -t[1])
print('Permutation importance (penurunan macro-F1):', [(n, round(v, 3)) for n, v in perm])

# %% [markdown]
# ### 5. Cost-aware front-end: MAC/frame yang dibutuhkan subset terpilih

# %%
def group_of(n):
    if n.startswith('mfcc'): return 'mfcc'
    if n.startswith('rms'): return 'rms'
    if n.startswith('zcr'): return 'zcr'
    if n.startswith(('centroid', 'bandwidth', 'rolloff')): return 'spectral_shape'
    return 'pitch'


mp8 = ROOT/'results/metrics/mp08_extension_feature_metrics.json'
mac = json.loads(mp8.read_text(encoding='utf-8'))['compute_cost_mac_per_frame'] if mp8.exists() else {
    'rms': 400, 'zcr': 400, 'fft_power_spectrum': 7602, 'centroid_bandwidth_rolloff': 804,
    'mel_filterbank_log': 332, 'dct_13_mfcc': 338, 'pitch_autocorrelation': 62400}
stage = {'rms': ['rms'], 'zcr': ['zcr'], 'spectral_shape': ['fft_power_spectrum', 'centroid_bandwidth_rolloff'],
         'mfcc': ['fft_power_spectrum', 'mel_filterbank_log', 'dct_13_mfcc'], 'pitch': ['pitch_autocorrelation']}


def cost_of(groups_needed):
    st = set(s for gname in groups_needed for s in stage[gname])
    return int(sum(mac[s] for s in st)), sorted(st)


full_cost, _ = cost_of({group_of(n) for n in names})
sel_groups = {group_of(names[i]) for i in sel}
sel_cost, sel_stages = cost_of(sel_groups)
cost = {'groups_needed_by_selection': sorted(sel_groups), 'stages_needed': sel_stages,
        'mac_per_frame_full39': full_cost, 'mac_per_frame_selected': sel_cost,
        'front_end_saving_pct': float(100*(1-sel_cost/full_cost)),
        'pitch_autocorrelation_skippable': 'pitch' not in sel_groups}
print('Front-end cost: full39 = %d MAC/frame, selected = %d MAC/frame (hemat %.1f%%)' %
      (full_cost, sel_cost, cost['front_end_saving_pct']), '| butuh:', sorted(sel_groups))

# %% [markdown]
# ### 6. Simpan + visualisasi

# %%
sc = StandardScaler().fit(Xtr)
lda = LinearDiscriminantAnalysis(n_components=2).fit(sc.transform(Xtr), ytr)
L2 = lda.transform(sc.transform(Xtr))
ext = {'embedded_and_filter_methods_train_cv': res, 'mi_k_curve': [{'k': k, 'mean': m, 'std': s} for k, m, s in curve],
       'k_min_within_0.03_of_best': int(k_min_best), 'permutation_importance_oof': [{'feature': n, 'macro_f1_drop': v} for n, v in perm],
       'front_end_cost': cost, 'frozen_mp10_choice': red['chosen_candidate'], 'scope': 'training_only_groupkfold',
       'validation_used': False, 'test_used': False}
(OUT_MET/'mp10_extension_selection_metrics.json').write_text(json.dumps(ext, indent=2), encoding='utf-8')

fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
ax[0, 0].bar(methods, [res[m]['cv_macro_f1_mean'] for m in methods], yerr=[res[m]['cv_macro_f1_std'] for m in methods], alpha=.8, label='CV macro-F1')
ax[0, 0].plot(methods, [res[m]['jaccard_stability_mean'] for m in methods], 'ko-', label='Jaccard stabilitas')
ax[0, 0].set_ylim(0, 1.1); ax[0, 0].set_title(f'Metode seleksi k={K}: performa & stabilitas'); ax[0, 0].legend()
kk, mm, ss = zip(*curve)
ax[0, 1].errorbar(kk, mm, yerr=ss, marker='o', capsize=3); ax[0, 1].axvline(K, ls='--', c='C3', label=f'MP10 k={K}')
ax[0, 1].set_xlabel('k fitur (MI)'); ax[0, 1].set_ylabel('GroupKFold macro-F1'); ax[0, 1].set_title('Kurva performa vs jumlah fitur'); ax[0, 1].legend()
pn, pv = zip(*perm)
ax[1, 0].barh(list(pn)[::-1], list(pv)[::-1]); ax[1, 0].set_title('Permutation importance OOF (fitur terpilih)'); ax[1, 0].tick_params(axis='y', labelsize=8)
for c in np.unique(ytr):
    m = ytr == c; ax[1, 1].scatter(L2[m, 0], L2[m, 1], s=60, label=c)
ax[1, 1].set_title('Proyeksi LDA 2D (TRAIN)'); ax[1, 1].set_xlabel('LD1'); ax[1, 1].set_ylabel('LD2'); ax[1, 1].legend()
for a in ax.ravel():
    a.grid(alpha=.2)
fig.suptitle(f"MP10 FITUR TAMBAHAN - front-end hemat {cost['front_end_saving_pct']:.0f}% MAC/frame dengan subset {red['chosen_candidate']}")
fig.tight_layout(); fig.savefig(OUT_FIG/'mp10_extension_selection_analysis.png', dpi=150); plt.close(fig)
print('Fitur tambahan MP10 selesai -> results/metrics/mp10_extension_selection_metrics.json')

# %%
if "__file__" not in globals():
    from IPython.display import Image, display
    for p in ['mp10_representation_tradeoff.png', 'mp10_feature_ranking.png', 'mp10_pca_projection.png', 'mp10_extension_selection_analysis.png']:
        display(Image(filename=str(OUT_FIG/p), width=760))
