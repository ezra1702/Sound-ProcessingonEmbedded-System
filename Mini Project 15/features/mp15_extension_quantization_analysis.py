# %% [markdown]
# ## FITUR TAMBAHAN MP15 — Verifikasi C INT8, Jalur SVM Buku, Bit-Width Sweep, Margin, Per-Channel vs Per-Tensor, FP16 & Paket ZIP
# Modul mandiri setelah `mini_project_15.py`. **Model/scale terpilih tidak diubah** (artefak dibaca beku);
# semua analisis bersifat verifikasi & dokumentasi (materi Bab 15: PTQ, symmetric/asymmetric, per-channel,
# numeric equivalence, packaging reproducible).
#
# 1. **Equivalence test C vs Python**: header `mp15_classical_model_int8.h` dikompilasi `gcc` dan dijalankan pada seluruh vektor 39-D dataset (96).
# 2. **Jalur INT8 buku untuk linear SVM** (arsip MP11, hash diverifikasi) → header referensi + uji C — membuktikan kode buku tetap
#    berjalan walau SVM bukan model terpilih.
# 3. **Bit-width sweep** 8→2 bit (model terpilih + SVM buku), skala input *per-tensor* vs *per-feature*, diuji pada seluruh
#    dataset (96) + 2000 sampel ter-jitter dari train (resolusi agreement lebih halus).
# 4. **Margin keputusan** model terpilih setelah kuantisasi (tree: jarak ke threshold dalam LSB; linear: selisih skor top-1 − top-2).
# 5. **CNN**: SQNR bobot per-tensor vs per-channel, SQNR logit INT8 vs float, FP16 (agreement & error).
# 6. **Footprint terkompilasi** (`size` pada object `-Os`) + **paket deployment** (salinan file manifest + SHA-256 + ZIP terverifikasi).
#
# Anggota Kelompok : Christama Ezra Yudianto (245150307111009), Abdillah Fatah Suryanto (245150300111008),
# Ivang Arfandia (245150307111003), Jahzeel Erason Nicodemus (245150300111002)

# %%
from pathlib import Path
import json, hashlib, shutil, subprocess, os, zipfile, warnings
import numpy as np
import matplotlib
if "__file__" in globals():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
status = json.loads((ROOT/'project_status.json').read_text(encoding='utf-8'))
assert status['status'] in ('ready_for_mp16',), 'Jalankan mini_project_15.py terlebih dahulu'
dep = json.loads((ROOT/'config/deployment_config.json').read_text(encoding='utf-8'))
rep = json.loads((ROOT/'results/metrics/mp15_deployment_metrics.json').read_text(encoding='utf-8'))
manifest = json.loads((ROOT/'deployment/mp15_manifest.json').read_text(encoding='utf-8'))
red = json.loads((ROOT/'config/feature_reduction_config.json').read_text(encoding='utf-8'))
clf_cfg = json.loads((ROOT/'config/classifier_config.json').read_text(encoding='utf-8'))
split = json.loads((ROOT/'splits/mp09_split.json').read_text(encoding='utf-8'))
OUT_MET, OUT_FIG, FW = ROOT/'results/metrics', ROOT/'results/figures', ROOT/'firmware'
(FW/'build').mkdir(parents=True, exist_ok=True)
classes = list(clf_cfg['classes'])
D39 = np.load(ROOT/'features/dataset/mp09_features_39d.npz', allow_pickle=True)
X39 = D39['X'].astype(np.float64); train_mask = D39['train_mask'].astype(bool)
SEL = np.asarray(red['selected_indices'], int); MU = np.asarray(red['scaler_mean']); SC = np.asarray(red['scaler_scale'])
Z8 = ((X39-MU)/SC)[:, SEL]
pkg = np.load(ROOT/'models/mp15_classical_int8.npz', allow_pickle=True)
x_scale = float(pkg['input_quant_scale'][0])
rng = np.random.default_rng(2026)
gcc = shutil.which('gcc') or (r'C:\msys64\ucrt64\bin\gcc.exe' if Path(r'C:\msys64\ucrt64\bin\gcc.exe').exists() else None)
ENV = dict(os.environ)
if gcc: ENV['PATH'] = str(Path(gcc).parent) + os.pathsep + ENV.get('PATH', '')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_c(src_name, code):
    """Compile & run harness C; return list of int predictions or None."""
    if not gcc: return None, 'gcc tidak ditemukan'
    src = FW/'build'/src_name; exe = src.with_suffix('.exe')
    src.write_text(code, encoding='utf-8')
    cp = subprocess.run([gcc, '-O2', '-std=c99', '-Wall', '-Wextra', '-Werror', '-I', str(FW), '-o', str(exe), str(src), '-lm'],
                        capture_output=True, text=True, env=ENV)
    if cp.returncode: return None, cp.stderr[-600:]
    out = subprocess.run([str(exe)], capture_output=True, text=True, env=ENV).stdout.split()
    return list(map(int, out)), 'ok'


def cf(v):
    t = f'{float(v):.9g}'
    return (t if any(ch in t for ch in '.eEn') else t + '.0') + 'f'


def c_matrix(name, M):
    rows = ',\n'.join('{' + ', '.join(cf(v) for v in r) + '}' for r in M)
    return f'static const float {name}[{len(M)}][{M.shape[1]}] = {{\n{rows}}};\n'

# %% [markdown]
# ### 1. Equivalence C vs Python — model terpilih (header MP15)

# %%
clf = joblib.load(ROOT/clf_cfg['model_file'])
float_idx = np.array([classes.index(str(c)) for c in clf.predict(Z8)])


def tree_int8_idx(z, scale, thr_q, bits=8):
    qmax = 2**(bits-1)-1
    zq = np.clip(np.round(z/scale), -qmax, qmax)
    n = 0; f = pkg['tree_feature']
    while f[n] >= 0:
        n = pkg['tree_left'][n] if zq[f[n]] <= thr_q[n] else pkg['tree_right'][n]
    return int(pkg['tree_leaf_class'][n])


kind = str(pkg['kind'])
if kind == 'tree':
    py_idx = np.array([tree_int8_idx(z, x_scale, pkg['tree_threshold_q']) for z in Z8])
else:
    Wq = pkg['weight_q'].astype(np.int32); Ws = pkg['weight_scales']; bb = pkg['bias']
    py_idx = np.array([int(np.argmax((Wq @ np.clip(np.round(z/x_scale), -127, 127)).astype(float)*Ws*x_scale+bb)) for z in Z8])
harness = ('#include <stdio.h>\n#include "mp15_classical_model_int8.h"\n' + c_matrix('X', X39) +
           'int main(void){ for(int i=0;i<%d;++i) printf("%%d ", mp15_classical_predict(X[i])); return 0; }\n' % len(X39))
c_idx, msg = run_c('test_mp15_classical.c', harness)
c_equiv = {'compiled': c_idx is not None, 'message': msg}
if c_idx is not None:
    c_equiv.update({'c_vs_python_int8_agreement': float(np.mean(np.array(c_idx) == py_idx)),
                    'c_vs_float_agreement': float(np.mean(np.array(c_idx) == float_idx)), 'n': len(c_idx)})
print('C (header terpilih) vs Python:', c_equiv)

# %% [markdown]
# ### 2. Jalur INT8 buku untuk linear SVM (arsip MP11, hash diverifikasi)

# %%
arch = json.loads((ROOT/'results/metrics/mp11_extension_classifier_metrics.json').read_text(encoding='utf-8'))['frozen_family_archive']['linear_svm']
assert sha(ROOT/arch['file']) == arch['sha256'], 'Hash arsip linear SVM berubah!'
svm = joblib.load(ROOT/arch['file'])
svm_float = np.array([classes.index(str(c)) for c in svm.predict(Z8)])
W = svm.coef_.astype(np.float64); b = svm.intercept_.astype(np.float64)
svm_cls_idx = [classes.index(str(c)) for c in svm.classes_]


def linear_int8_idx(z, scale, W, b, cls_idx, bits=8, per_feature=None):
    """INT8 (atau bits) symmetric: bobot per-kelas, input per-tensor (scale) atau per-feature."""
    qmax = 2**(bits-1)-1
    Wsc = np.maximum(np.max(np.abs(W), axis=1)/qmax, 1e-8); Wq_ = np.clip(np.round(W/Wsc[:, None]), -qmax, qmax)
    if per_feature is not None:
        Wf = W*per_feature[None, :]; Wsc = np.maximum(np.max(np.abs(Wf), axis=1)/qmax, 1e-8)
        Wq_ = np.clip(np.round(Wf/Wsc[:, None]), -qmax, qmax)
        zq = np.clip(np.round(z/per_feature), -qmax, qmax); s = (Wq_@zq)*Wsc + b
    else:
        zq = np.clip(np.round(z/scale), -qmax, qmax); s = (Wq_@zq)*Wsc*scale + b
    return cls_idx[int(np.argmax(s))]


def svm_int8_idx(z, scale, bits=8, per_feature=None):
    qmax = 2**(bits-1)-1
    Wsc = np.maximum(np.max(np.abs(W), axis=1)/qmax, 1e-8); Wq_ = np.clip(np.round(W/Wsc[:, None]), -qmax, qmax)
    if per_feature is not None:   # skala input per-feature dilebur ke bobot sebelum kuantisasi
        Wf = W*per_feature[None, :]; Wsc = np.maximum(np.max(np.abs(Wf), axis=1)/qmax, 1e-8)
        Wq_ = np.clip(np.round(Wf/Wsc[:, None]), -qmax, qmax)
        zq = np.clip(np.round(z/per_feature), -qmax, qmax); s = (Wq_@zq)*Wsc + b
    else:
        zq = np.clip(np.round(z/scale), -qmax, qmax); s = (Wq_@zq)*Wsc*scale + b
    return svm_cls_idx[int(np.argmax(s))]


svm_q = np.array([svm_int8_idx(z, x_scale) for z in Z8])
Wsc8 = np.maximum(np.max(np.abs(W), axis=1)/127, 1e-8); Wq8 = np.clip(np.round(W/Wsc8[:, None]), -127, 127).astype(int)
svm_h = ['// MP15 FITUR TAMBAHAN: referensi jalur INT8 buku (linear SVM arsip MP11) - BUKAN model terpilih\n#pragma once\n#include <stdint.h>\n#include <math.h>\n',
         f'static const float SVM_IN_SCALE = {cf(x_scale)};\n',
         'static const int32_t SVM_IDX[8] = {' + ', '.join(map(str, SEL)) + '};\n',
         'static const float SVM_MEAN[8] = {' + ', '.join(cf(v) for v in MU[SEL]) + '};\n',
         'static const float SVM_SCALE[8] = {' + ', '.join(cf(v) for v in SC[SEL]) + '};\n',
         f'static const int8_t SVM_WQ[{Wq8.size}] = {{' + ', '.join(map(str, Wq8.ravel())) + '};\n',
         'static const float SVM_WS[4] = {' + ', '.join(cf(v) for v in Wsc8) + '};\n',
         'static const float SVM_B[4] = {' + ', '.join(cf(v) for v in b) + '};\n',
         'static const uint8_t SVM_CLASS[4] = {' + ', '.join(map(str, svm_cls_idx)) + '};\n',
         '''static inline int svm_int8_predict(const float *x39){ int8_t zq[8]; int best=0; float bs=-1e30f;
  for(int i=0;i<8;++i){ float q=roundf(((x39[SVM_IDX[i]]-SVM_MEAN[i])/SVM_SCALE[i])/SVM_IN_SCALE); zq[i]=(int8_t)(q>127.f?127:(q<-127.f?-127:q)); }
  for(int c=0;c<4;++c){ int32_t acc=0; for(int i=0;i<8;++i) acc+=(int32_t)SVM_WQ[c*8+i]*zq[i];
    float s=(float)acc*SVM_WS[c]*SVM_IN_SCALE+SVM_B[c]; if(s>bs){bs=s;best=c;} }
  return SVM_CLASS[best]; }
''']
(FW/'mp15_svm_int8_reference.h').write_text(''.join(svm_h), encoding='utf-8')
c_svm, msg2 = run_c('test_mp15_svm.c', '#include <stdio.h>\n#include "mp15_svm_int8_reference.h"\n' + c_matrix('X', X39) +
                    'int main(void){ for(int i=0;i<%d;++i) printf("%%d ", svm_int8_predict(X[i])); return 0; }\n' % len(X39))
svm_path = {'archive_hash_verified': True, 'int8_vs_float_agreement_all': float(np.mean(svm_q == svm_float)),
            'c_compiled': c_svm is not None, 'note': 'Jalur buku (SVM) didemonstrasikan; model deployment tetap ' + dep['selected_deployment_model']}
if c_svm is not None:
    svm_path['c_vs_python_agreement'] = float(np.mean(np.array(c_svm) == svm_q))
print('Jalur SVM INT8 buku:', svm_path)

# %% [markdown]
# ### 3. Bit-width sweep + skala per-tensor vs per-feature (seluruh dataset + 2000 sampel jitter)

# %%
Ztr = Z8[train_mask]
Zjit = Ztr[rng.integers(0, len(Ztr), 2000)] + rng.normal(scale=0.35, size=(2000, Z8.shape[1]))
pf_scale = np.maximum(np.max(np.abs(Ztr), axis=0), 1e-8)   # per-feature max (training-only)
thr_f = pkg['tree_threshold_float'] if kind == 'tree' else None
if kind != 'tree':   # model terpilih linear (mis. logreg): bobot float dari model beku
    W_sel = np.asarray(clf.coef_, np.float64); b_sel = np.asarray(clf.intercept_, np.float64)
    sel_cls_idx = [classes.index(str(c)) for c in clf.classes_]
sweep = {}
for bits in (8, 6, 5, 4, 3, 2):
    qmax = 2**(bits-1)-1
    s_t = np.max(np.abs(Ztr))/qmax; s_f = pf_scale/qmax
    row = {}
    for name, Z, ref_tree, ref_svm in (('all', Z8, float_idx, svm_float),
                                      ('jitter2000', Zjit, np.array([classes.index(str(c)) for c in clf.predict(Zjit)]),
                                       np.array([classes.index(str(c)) for c in svm.predict(Zjit)]))):
        if kind == 'tree':
            thr_t = np.floor(thr_f/s_t)
            pt = np.array([tree_int8_idx(z, s_t, thr_t, bits) for z in Z])
            # per-feature: tiap node pakai skala fiturnya sendiri
            pfq = []
            for z in Z:
                zq = np.clip(np.round(z/s_f), -qmax, qmax); n = 0; f = pkg['tree_feature']
                while f[n] >= 0:
                    n = pkg['tree_left'][n] if zq[f[n]] <= np.floor(thr_f[n]/s_f[f[n]]) else pkg['tree_right'][n]
                pfq.append(int(pkg['tree_leaf_class'][n]))
            row[f'tree_per_tensor_{name}'] = float(np.mean(pt == ref_tree))
            row[f'tree_per_feature_{name}'] = float(np.mean(np.array(pfq) == ref_tree))
        else:
            row[f'selected_per_tensor_{name}'] = float(np.mean(np.array([linear_int8_idx(z, s_t, W_sel, b_sel, sel_cls_idx, bits) for z in Z]) == ref_tree))
            row[f'selected_per_feature_{name}'] = float(np.mean(np.array([linear_int8_idx(z, None, W_sel, b_sel, sel_cls_idx, bits, s_f) for z in Z]) == ref_tree))
        row[f'svm_per_tensor_{name}'] = float(np.mean(np.array([svm_int8_idx(z, s_t, bits) for z in Z]) == ref_svm))
        row[f'svm_per_feature_{name}'] = float(np.mean(np.array([svm_int8_idx(z, None, bits, s_f) for z in Z]) == ref_svm))
    sweep[bits] = row
    print(f'  {bits} bit:', {k: round(v, 3) for k, v in row.items() if 'jitter' in k})

# %% [markdown]
# ### 4. Margin keputusan model terpilih setelah kuantisasi (seluruh dataset)

# %%
margins = []
if kind == 'tree':
    for z in Z8:
        zq = np.clip(np.round(z/x_scale), -127, 127); n = 0; f = pkg['tree_feature']; mn = 1e9
        while f[n] >= 0:
            mn = min(mn, abs(zq[f[n]] - (pkg['tree_threshold_q'][n] + 0.5)))   # batas efektif di antara t_q dan t_q+1
            n = pkg['tree_left'][n] if zq[f[n]] <= pkg['tree_threshold_q'][n] else pkg['tree_right'][n]
        margins.append(float(mn))
    margins = np.array(margins); margin_unit, margin_small = 'LSB input ke threshold', 3.0
    print(f'Margin minimum = {margins.min():.1f} LSB ({margins.min()*x_scale:.4f} z-unit); sampel dengan margin < 3 LSB: {int(np.sum(margins < 3))}/{len(margins)}')
else:
    # linear INT8: skor = (Wq·zq)·Ws·x_scale + b ; margin = skor top-1 − top-2 (satuan logit)
    for z in Z8:
        zq = np.clip(np.round(z/x_scale), -127, 127)
        sc_ = np.sort((Wq @ zq).astype(float)*Ws*x_scale + bb)[::-1]
        margins.append(float(sc_[0]-sc_[1]))
    margins = np.array(margins); margin_unit, margin_small = 'logit (skor top-1 - top-2)', 0.5
    print(f'Margin skor minimum = {margins.min():.3f} logit; sampel dengan margin < {margin_small}: {int(np.sum(margins < margin_small))}/{len(margins)}')

# %% [markdown]
# ### 5. CNN: per-tensor vs per-channel, SQNR logit INT8, FP16

# %%
def sqnr(ref, q):
    return float(10*np.log10(np.sum(ref**2)/(np.sum((ref-q)**2)+1e-30)))


ck = torch.load(ROOT/'models/mp12_tiny_cnn_state.pt', map_location='cpu', weights_only=False)
sd = ck['state_dict']; mu, sig = float(ck['normalization']['mean']), float(ck['normalization']['std'])
w_rows = {}
for key in ('features.0.weight', 'features.4.weight', 'classifier.weight'):
    w = sd[key].numpy().astype(np.float64)
    pt = np.round(w/(np.abs(w).max()/127))*(np.abs(w).max()/127)
    ch = np.abs(w.reshape(w.shape[0], -1)).max(1).reshape((-1,)+(1,)*(w.ndim-1))/127
    pc = np.round(w/ch)*ch
    w_rows[key] = {'per_tensor_sqnr_db': sqnr(w, pt), 'per_channel_sqnr_db': sqnr(w, pc), 'fp16_sqnr_db': sqnr(w, w.astype(np.float16).astype(np.float64))}
    print(f'  {key:20s} per-tensor {w_rows[key]["per_tensor_sqnr_db"]:.1f} dB | per-channel {w_rows[key]["per_channel_sqnr_db"]:.1f} dB | FP16 {w_rows[key]["fp16_sqnr_db"]:.1f} dB')

src = (ROOT/'mini_project_15.py').read_text(encoding='utf-8')
ns = {'nn': nn, 'torch': torch}
exec(src[src.index('class FloatTinyKWS'):src.index('class QuantTinyKWS')], ns)
fnet = ns['FloatTinyKWS'](len(classes)); fnet.load_state_dict(sd); fnet.eval()
qnet = torch.jit.load(str(ROOT/'models/mp15_tiny_cnn_int8_torchscript.pt')); qnet.eval()
h16 = ns['FloatTinyKWS'](len(classes)); h16.load_state_dict(sd); h16 = h16.half().eval()
ids = split['train_ids'] + split['validation_ids'] + split['test_ids']
Lf, Lq, Lh = [], [], []
with torch.no_grad():
    for sid in ids:
        x = np.load(ROOT/'features/logmel/mp09'/f'{sid}.npz')['log_mel'].astype(np.float32)
        x = x[:96] if len(x) >= 96 else np.pad(x, ((0, 96-len(x)), (0, 0)), constant_values=float(x.min()))
        xt = torch.from_numpy(((x-mu)/sig)[None, None]).float()
        Lf.append(fnet(xt).numpy()[0]); Lq.append(qnet(xt).numpy()[0]); Lh.append(h16(xt.half()).float().numpy()[0])
Lf, Lq, Lh = map(np.array, (Lf, Lq, Lh))
cnn_q = {'logit_sqnr_int8_db': sqnr(Lf, Lq), 'logit_sqnr_fp16_db': sqnr(Lf, Lh),
         'agreement_int8': float(np.mean(Lf.argmax(1) == Lq.argmax(1))), 'agreement_fp16': float(np.mean(Lf.argmax(1) == Lh.argmax(1))),
         'max_abs_logit_err_int8': float(np.abs(Lf-Lq).max()), 'max_abs_logit_err_fp16': float(np.abs(Lf-Lh).max()),
         'min_top2_logit_gap_float': float(np.min(np.sort(Lf, 1)[:, -1]-np.sort(Lf, 1)[:, -2]))}
print('CNN logit:', {k: round(v, 4) for k, v in cnn_q.items()})

# %% [markdown]
# ### 6. Footprint terkompilasi + paket deployment ZIP ber-SHA-256

# %%
footprint = {}
size_exe = shutil.which('size', path=ENV['PATH'])
if gcc and size_exe:
    for hdr, fn in (('mp15_classical_model_int8.h', 'mp15_classical_predict'), ('mp15_svm_int8_reference.h', 'svm_int8_predict')):
        c = FW/'build'/f'fp_{hdr[:-2]}.c'
        c.write_text(f'#include "{hdr}"\nint entry(const float *x){{ return {fn}(x); }}\n', encoding='utf-8')
        o = c.with_suffix('.o')
        cp = subprocess.run([gcc, '-Os', '-c', '-std=c99', '-I', str(FW), '-o', str(o), str(c)], capture_output=True, text=True, env=ENV)
        if cp.returncode == 0:
            out = subprocess.run([size_exe, '-A', str(o)], capture_output=True, text=True, env=ENV).stdout
            secs = {ln.split()[0]: int(ln.split()[1]) for ln in out.splitlines() if ln.startswith('.') and len(ln.split()) >= 2}
            footprint[hdr] = {'text_bytes': secs.get('.text', 0), 'rodata_bytes': sum(v for k, v in secs.items() if k.startswith('.rdata') or k.startswith('.rodata')),
                              'note': 'x86-64 -Os (host); ukuran data konstan identik di MCU, ukuran kode berbeda per ISA'}
print('Footprint:', footprint)

PKG = ROOT/'deployment/mp15_package'
if PKG.exists(): shutil.rmtree(PKG)
files = list(dict.fromkeys(manifest['files_required'] + ['deployment/mp15_manifest.json', 'results/metrics/mp15_deployment_metrics.json',
                                                         'models/mp15_tiny_cnn_int8_torchscript.pt', 'features/kws_frontend.py']))
pkg_manifest = {}
for f in files:
    dst = PKG/f; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT/f, dst)
    pkg_manifest[f] = {'sha256': sha(ROOT/f), 'bytes': (ROOT/f).stat().st_size}
(PKG/'SHA256SUMS.json').write_text(json.dumps(pkg_manifest, indent=2), encoding='utf-8')
zpath = ROOT/'deployment/mp15_deployment_package.zip'
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.rglob('*')):
        if p.is_file(): z.write(p, p.relative_to(PKG).as_posix())
with zipfile.ZipFile(zpath) as z:
    zip_ok = z.testzip() is None and all(hashlib.sha256(z.read(f)).hexdigest() == v['sha256'] for f, v in pkg_manifest.items())
print(f'Paket: {len(files)} file, ZIP {zpath.stat().st_size} B, integritas SHA-256 = {zip_ok}')

# %% [markdown]
# ### 7. Simpan + visualisasi

# %%
ext = {'c_equivalence_selected_model': c_equiv, 'book_svm_int8_path': svm_path, 'bit_width_sweep': {str(k): v for k, v in sweep.items()},
       'decision_margin': {'model_kind': kind, 'unit': margin_unit, 'small_margin_threshold': margin_small,
                           'min': float(margins.min()), 'n_below_threshold': int(np.sum(margins < margin_small)), 'n': int(len(margins)),
                           'per_sample': margins.tolist()},
       'cnn_weight_quant': w_rows, 'cnn_logit_equivalence': cnn_q, 'compiled_footprint': footprint,
       'package': {'zip': 'deployment/mp15_deployment_package.zip', 'zip_sha256': sha(zpath), 'files': pkg_manifest, 'integrity_ok': zip_ok},
       'selected_model_unchanged': dep['selected_deployment_model'], 'post_test_tuning_performed': False}
(OUT_MET/'mp15_extension_quantization_metrics.json').write_text(json.dumps(ext, indent=2), encoding='utf-8')

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
bits = list(sweep)
for k, st in (('tree_per_tensor_jitter2000', 'o-'), ('tree_per_feature_jitter2000', 'o--'), ('selected_per_tensor_jitter2000', 'o-'), ('selected_per_feature_jitter2000', 'o--'), ('svm_per_tensor_jitter2000', 's-'), ('svm_per_feature_jitter2000', 's--')):
    if k in sweep[8]: ax[0].plot(bits, [sweep[b_][k] for b_ in bits], st, label=k.replace('_jitter2000', ''))
ax[0].invert_xaxis(); ax[0].set_xlabel('bit'); ax[0].set_ylabel('agreement vs float (2000 sampel jitter)'); ax[0].set_title('Bit-width sweep'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.2)
keys = list(w_rows); xx = np.arange(len(keys))
ax[1].bar(xx-.25, [w_rows[k]['per_tensor_sqnr_db'] for k in keys], .25, label='INT8 per-tensor')
ax[1].bar(xx, [w_rows[k]['per_channel_sqnr_db'] for k in keys], .25, label='INT8 per-channel')
ax[1].bar(xx+.25, [w_rows[k]['fp16_sqnr_db'] for k in keys], .25, label='FP16')
ax[1].set_xticks(xx, ['conv1', 'conv2', 'fc']); ax[1].set_ylabel('SQNR bobot (dB)'); ax[1].set_title(f"CNN: logit SQNR INT8={cnn_q['logit_sqnr_int8_db']:.1f} dB"); ax[1].legend(fontsize=8); ax[1].grid(axis='y', alpha=.2)
if len(margins):
    ax[2].bar(range(len(margins)), margins, color=['C3' if m < margin_small else 'C2' for m in margins]); ax[2].axhline(margin_small, color='k', ls=':')
    ax[2].set_xlabel(f'sampel ({len(margins)})'); ax[2].set_ylabel(f'margin ({margin_unit})')
    ax[2].set_title(f"Margin keputusan {dep['selected_deployment_model']} (merah < {margin_small})")
fig.suptitle(f"MP15 FITUR TAMBAHAN - C equivalence {c_equiv.get('c_vs_python_int8_agreement')} | paket ZIP terverifikasi={zip_ok}")
fig.tight_layout(); fig.savefig(OUT_FIG/'mp15_extension_quantization_analysis.png', dpi=140); plt.close(fig)
print('Fitur tambahan MP15 selesai -> results/metrics/mp15_extension_quantization_metrics.json')

# %%
if "__file__" not in globals():
    from IPython.display import Image, display
    display(Image(filename=str(OUT_FIG/'mp15_extension_quantization_analysis.png'), width=1000))
