# %% [markdown]
# ## FITUR TAMBAHAN MP9 — Dataset Quality Audit, Reproducibility & Leakage Guard
# Modul mandiri, dijalankan SETELAH `mini_project_09.py`. Split & dataset **tidak diubah**.
#
# Fitur tambahan (materi Bab 9: metadata, reproducibility checklist, data leakage):
# 1. **Audit kualitas per sampel**: peak, clipping, DC offset, RMS dBFS, estimasi SNR,
#    durasi aktif, flag otomatis (terlalu pelan / clipping / terpotong di tepi rekaman).
# 2. **Reproducibility manifest**: SHA-256 setiap WAV + config, versi library, seed.
# 3. **Leakage guard**: duplikat eksak (hash) dan *near-duplicate* (cosine similarity 39-D
#    terstandar dengan scaler TRAIN) antar split.
# 4. **Front-end beku reusable** `features/kws_frontend.py` + self-test bit-exact terhadap dataset MP9.
# 5. **Galeri log-Mel kelas × penutur** + kualitas per block dan per penutur (4 anggota kelompok).
# 6. **Preview separabilitas training-only**: Fisher score per fitur (hanya TRAIN, tanpa validation/test).
#
# Anggota Kelompok : Christama Ezra Yudianto (245150307111009), Abdillah Fatah Suryanto (245150300111008),
# Ivang Arfandia (245150307111003), Jahzeel Erason Nicodemus (245150300111002)

# %%
from pathlib import Path
import csv, json, hashlib, platform, sys
import numpy as np
import matplotlib
if "__file__" in globals():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT/'features'))
from kws_frontend import KWSFrontend  # noqa: E402

status = json.loads((ROOT/'project_status.json').read_text(encoding='utf-8'))
assert status['status'] in ('ready_for_mp10',), 'Jalankan mini_project_09.py terlebih dahulu'
D = np.load(ROOT/'features/dataset/mp09_features_39d.npz', allow_pickle=True)
X = D['X'].astype(np.float64); y = D['y'].astype(str); groups = D['groups'].astype(str)
ids = D['sample_ids'].astype(str); names = D['feature_names'].astype(str)
train_mask = D['train_mask'].astype(bool); val_mask = D['val_mask'].astype(bool); test_mask = D['test_mask'].astype(bool)
meta = list(csv.DictReader((ROOT/'data/metadata/mp09_metadata.csv').open(encoding='utf-8')))
OUT_MET = ROOT/'results/metrics'; OUT_FIG = ROOT/'results/figures'
fe = KWSFrontend(ROOT)
labels = fe.labels


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 16), b''):
            h.update(blk)
    return h.hexdigest()

# %% [markdown]
# ### 1. Audit kualitas per sampel

# %%
audit = []
for r in meta:
    x = fe.read_wav(ROOT/r['path'])
    _, (s0, s1) = fe.preprocess(x, return_bounds=True)
    seg = np.zeros(len(x), bool); seg[s0:s1] = True
    p_act = np.mean(x[seg]**2) if seg.any() else 0.0
    p_bg = np.mean(x[~seg]**2) if (~seg).any() else 1e-12
    row = {
        'sample_id': r['sample_id'], 'label': r['label'], 'group_id': r['group_id'], 'subject_id': r.get('subject_id', ''), 'split': r['split'],
        'peak': float(np.max(np.abs(x))), 'clipping_samples': int(np.sum(np.abs(x) >= 0.999)),
        'dc_offset': float(np.mean(x)), 'rms_dbfs': float(20*np.log10(np.sqrt(np.mean(x**2))+1e-12)),
        'snr_est_db': float(10*np.log10(max(p_act-p_bg, 1e-12)/(p_bg+1e-12))),
        'active_start_s': s0/fe.fs, 'active_end_s': s1/fe.fs, 'active_duration_s': (s1-s0)/fe.fs,
    }
    flags = []
    if row['peak'] < 0.05: flags.append('too_quiet')
    if row['clipping_samples'] > 0: flags.append('clipping')
    if s0 <= fe.H or s1 >= len(x)-fe.H: flags.append('word_touches_record_edge')
    if row['snr_est_db'] < 10: flags.append('low_snr')
    if abs(row['dc_offset']) > 0.02: flags.append('dc_offset')
    row['flags'] = ';'.join(flags) if flags else 'ok'
    audit.append(row)
with (OUT_MET/'mp09_extension_quality_audit.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(audit[0].keys())); w.writeheader(); w.writerows(audit)
n_flag = sum(r['flags'] != 'ok' for r in audit)
print(f'Audit kualitas: {len(audit)} sampel, {n_flag} ber-flag')
print('  SNR estimasi (dB): min=%.1f mean=%.1f max=%.1f' % tuple(f([r['snr_est_db'] for r in audit]) for f in (np.min, np.mean, np.max)))
print('  Peak: min=%.3f max=%.3f | clipping total=%d' % (min(r['peak'] for r in audit), max(r['peak'] for r in audit), sum(r['clipping_samples'] for r in audit)))

# %% [markdown]
# ### 2. Reproducibility manifest (SHA-256 + versi library + seed)

# %%
import scipy, sklearn, matplotlib as mpl  # noqa: E401
files = sorted((ROOT/'data/raw/mp09_dataset').glob('*.wav')) + [ROOT/'config'/c for c in
        ['project_config.json', 'preprocessing_config.json', 'feature_config.json', 'dataset_config.json']] + \
        [ROOT/'splits/mp09_split.json', ROOT/'features/intermediate/mp06_filter_final.npz', ROOT/'features/dataset/mp09_features_39d.npz']
repro = {
    'python': platform.python_version(), 'platform': platform.platform(),
    'library_versions': {'numpy': np.__version__, 'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__,
                         'matplotlib': mpl.__version__},
    'seed': json.loads((ROOT/'config/dataset_config.json').read_text(encoding='utf-8'))['seed'],
    'acquisition_mode': json.loads((ROOT/'config/dataset_config.json').read_text(encoding='utf-8')).get('acquisition_mode'),
    'sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in files},
}
(OUT_MET/'mp09_extension_reproducibility_manifest.json').write_text(json.dumps(repro, indent=2), encoding='utf-8')
print('Manifest SHA-256:', len(repro['sha256']), 'file')

# %% [markdown]
# ### 3. Leakage guard: duplikat eksak & near-duplicate antar split

# %%
wav_hash = {r['sample_id']: repro['sha256'][r['path']] for r in meta}
dup_exact = [(a, b) for i, a in enumerate(ids) for b in ids[i+1:] if wav_hash[a] == wav_hash[b]]
mu = X[train_mask].mean(0); sd = X[train_mask].std(0)+1e-9      # scaler TRAIN-only
Zs = (X-mu)/sd
Zn = Zs/np.linalg.norm(Zs, axis=1, keepdims=True)
S = Zn @ Zn.T
split_of = np.where(train_mask, 'train', np.where(val_mask, 'validation', 'test'))
cross = []
for i in range(len(ids)):
    for j in range(i+1, len(ids)):
        if split_of[i] != split_of[j]:
            cross.append((float(S[i, j]), ids[i], ids[j]))
cross.sort(reverse=True)
NEAR_DUP = 0.995
near_dups = [c for c in cross if c[0] >= NEAR_DUP]
leak = {'exact_duplicate_pairs': dup_exact, 'near_duplicate_threshold_cosine': NEAR_DUP,
        'near_duplicate_cross_split_pairs': [{'cos': c[0], 'a': c[1], 'b': c[2]} for c in near_dups],
        'top5_most_similar_cross_split': [{'cos': c[0], 'a': c[1], 'b': c[2]} for c in cross[:5]],
        'scaler_fit_scope': 'training_only', 'leakage_detected': bool(dup_exact or near_dups)}
print('Duplikat eksak:', len(dup_exact), '| near-duplicate lintas split:', len(near_dups),
      '| cos tertinggi lintas split = %.3f' % cross[0][0])

# %% [markdown]
# ### 4. Self-test front-end beku (kws_frontend.py) terhadap dataset MP9

# %%
st = fe.selftest()
print('Self-test kws_frontend:', st)
assert st['bit_exact'], 'Front-end reusable tidak identik dengan MP9!'

# %% [markdown]
# ### 5. Fisher score training-only (preview separabilitas, tanpa validation/test)

# %%
Xtr, ytr = X[train_mask], y[train_mask]
overall = Xtr.mean(0)
num = sum((ytr == c).sum()*(Xtr[ytr == c].mean(0)-overall)**2 for c in labels)
den = sum((ytr == c).sum()*Xtr[ytr == c].var(0) for c in labels)+1e-12
fisher = num/den
order = np.argsort(fisher)[::-1]
print('Top-8 Fisher score (TRAIN):', [(str(names[i]), round(float(fisher[i]), 2)) for i in order[:8]])

ext = {'quality_audit_csv': 'results/metrics/mp09_extension_quality_audit.csv',
       'n_flagged_samples': n_flag, 'flag_summary': {f: sum(f in r['flags'] for r in audit) for f in
                                                    ['too_quiet', 'clipping', 'word_touches_record_edge', 'low_snr', 'dc_offset']},
       'snr_est_db': {'min': min(r['snr_est_db'] for r in audit), 'mean': float(np.mean([r['snr_est_db'] for r in audit])),
                      'max': max(r['snr_est_db'] for r in audit)},
       'snr_est_db_per_subject': {s: {'min': float(min(v)), 'median': float(np.median(v)), 'max': float(max(v))}
                                  for s in sorted({r['subject_id'] for r in audit})
                                  for v in [[r['snr_est_db'] for r in audit if r['subject_id'] == s]]},
       'leakage_guard': leak, 'frontend_selftest': st,
       'fisher_score_train_only_top10': [{'feature': str(names[i]), 'score': float(fisher[i])} for i in order[:10]],
       'reproducibility_manifest': 'results/metrics/mp09_extension_reproducibility_manifest.json'}
(OUT_MET/'mp09_extension_dataset_metrics.json').write_text(json.dumps(ext, indent=2), encoding='utf-8')

# %% [markdown]
# ### 6. Visualisasi: galeri log-Mel (kelas × penutur), kualitas per block & penutur, similarity, Fisher score

# %%
subjects = sorted({r['subject_id'] for r in meta}, key=lambda s: [r['subject_id'] for r in meta].index(s))
fig, axes = plt.subplots(len(labels), len(subjects), figsize=(3.2*len(subjects)+1, 8), sharey=True, squeeze=False)
for i, lab in enumerate(labels):
    for j, sj in enumerate(subjects):
        r = next(r for r in meta if r['label'] == lab and r['subject_id'] == sj and r['group_id'] == 'block_01')
        lm = np.load(ROOT/'features/logmel/mp09'/f"{r['sample_id']}.npz")['log_mel']
        ax = axes[i, j]; ax.imshow(lm.T, aspect='auto', origin='lower', cmap='magma')
        ax.set_title(f'{lab} · {sj} (block 01)', fontsize=8); ax.set_xticks([])
        if j == 0: ax.set_ylabel('Mel bin')
fig.suptitle('MP9 FITUR TAMBAHAN - galeri log-Mel (baris = kelas, kolom = penutur; block 01 / train)')
fig.tight_layout(); fig.savefig(OUT_FIG/'mp09_extension_logmel_gallery.png', dpi=130); plt.close(fig)

fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
blk = [f'block_{b:02d}' for b in range(1, 7)]
for lab in labels:
    ax[0].plot(range(1, 7), [np.mean([r['snr_est_db'] for r in audit if r['group_id'] == g and r['label'] == lab]) for g in blk], 'o-', label=lab)
ax[0].set_xlabel('Acquisition block'); ax[0].set_ylabel('SNR estimasi rata-rata penutur (dB)'); ax[0].set_title('Kualitas per block'); ax[0].legend(); ax[0].grid(alpha=.2)
ax[3].boxplot([[r['snr_est_db'] for r in audit if r['subject_id'] == sj] for sj in subjects]); ax[3].set_xticks(range(1, len(subjects)+1), subjects)
ax[3].set_ylabel('SNR estimasi (dB)'); ax[3].set_title('Kualitas per penutur (24 sampel/penutur)'); ax[3].grid(axis='y', alpha=.2)
im = ax[1].imshow(S, vmin=-1, vmax=1, cmap='coolwarm'); fig.colorbar(im, ax=ax[1], fraction=.046)
ax[1].set_title('Cosine similarity 39-D (scaler TRAIN)'); ax[1].set_xlabel('sample idx'); ax[1].set_ylabel('sample idx')
k = 12
ax[2].barh(names[order[:k]][::-1], fisher[order[:k]][::-1]); ax[2].set_title('Fisher score TRAIN-only (top 12)'); ax[2].grid(axis='x', alpha=.2)
ax[2].tick_params(axis='y', labelsize=8)
fig.tight_layout(); fig.savefig(OUT_FIG/'mp09_extension_dataset_audit.png', dpi=150); plt.close(fig)
print('Fitur tambahan MP9 selesai -> results/metrics/mp09_extension_dataset_metrics.json')

# %%
if "__file__" not in globals():
    from IPython.display import Image, display
    for p in ['mp09_class_distribution.png', 'mp09_split_distribution.png', 'mp09_extension_logmel_gallery.png', 'mp09_extension_dataset_audit.png']:
        display(Image(filename=str(OUT_FIG/p), width=820))
