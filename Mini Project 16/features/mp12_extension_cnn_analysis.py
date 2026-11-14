# %% [markdown]
# ## FITUR TAMBAHAN MP12 — Diagnosis Underfitting, DS-CNN, Seed Stability, ONNX & Memory Profile
# Modul mandiri setelah `mini_project_12.py`. CNN resmi MP12 **tidak diubah**. Test **tidak dimuat**.
#
# Fitur tambahan (materi Bab 12: depthwise-separable CNN, overfitting/early stopping, resource budget):
# 1. **Diagnosis underfitting** CNN resmi: jumlah langkah optimizer, train-set F1, tren loss.
# 2. **Varian eksplorasi** dengan protokol identik (norm/augment TRAIN-only, checkpoint VALIDATION):
#    a) TinyKWSNet *long-schedule* (arsitektur sama, 300 epoch, OneCycle LR),
#    b) **DS-CNN** (depthwise-separable) — biaya ∝ H·W·(K²·M + M·N) vs H·W·K²·M·N,
#    c) **TinyKWS-FreqPool**: conv identik, pooling hanya sumbu waktu (jadwal training resmi 60 epoch).
#    Model dibekukan + SHA-256 **sebelum** MP13; dievaluasi di MP13 sebagai eksplorasi.
# 3. **Seed stability** konfigurasi resmi (3 seed) → variasi validation F1.
# 4. **Export ONNX** + verifikasi onnxruntime vs PyTorch (portabilitas runtime).
# 5. **Profil memori aktivasi per layer** (peak RAM proxy / tensor arena) TinyKWSNet vs DS-CNN.
# 6. **Saliency map (gradient × input)** pada validation → wilayah waktu-frekuensi yang dipakai CNN.
#
# Anggota Kelompok : Christama Ezra Yudianto (245150307111009), Abdillah Fatah Suryanto (245150300111008),
# Ivang Arfandia (245150307111003), Jahzeel Erason Nicodemus (245150300111002)

# %%
from pathlib import Path
import json, copy, hashlib, random, warnings
import numpy as np
import matplotlib
if "__file__" in globals():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
status = json.loads((ROOT/'project_status.json').read_text(encoding='utf-8'))
assert status['status'] in ('ready_for_mp13',), 'Jalankan mini_project_12.py terlebih dahulu'
dl = json.loads((ROOT/'config/deep_learning_config.json').read_text(encoding='utf-8'))
met = json.loads((ROOT/'results/metrics/mp12_deep_learning_metrics.json').read_text(encoding='utf-8'))
split = json.loads((ROOT/'splits/mp09_split.json').read_text(encoding='utf-8'))
classes = dl['classes']; c2i = {c: i for i, c in enumerate(classes)}
T, F = dl['input_shape_per_sample']
OUT_MET = ROOT/'results/metrics'; OUT_FIG = ROOT/'results/figures'; MD = ROOT/'models'
torch.set_num_threads(1)


def load(ids):
    xs, ys = [], []
    for sid in ids:
        d = np.load(ROOT/'features/logmel/mp09'/f'{sid}.npz', allow_pickle=True)
        x = d['log_mel'].astype(np.float32)
        x = x[:T] if x.shape[0] >= T else np.pad(x, ((0, T-x.shape[0]), (0, 0)), constant_values=float(x.min()))
        xs.append(x); ys.append(c2i[str(d['label'])])
    return np.stack(xs), np.asarray(ys)


Xtr, ytr = load(split['train_ids']); Xv, yv = load(split['validation_ids'])   # test TIDAK dimuat
mu, sd = float(Xtr.mean()), float(Xtr.std()+1e-6)
Xtr = (Xtr-mu)/sd; Xv = (Xv-mu)/sd


class TinyKWSNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(8, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Linear(16, n)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class DSCNN(nn.Module):
    """Depthwise-separable KWS: conv standar kecil lalu 2 blok depthwise(3x3)+pointwise(1x1)."""
    def __init__(self, n, c=16):
        super().__init__()
        def ds(ci, co):
            return [nn.Conv2d(ci, ci, 3, padding=1, groups=ci, bias=False), nn.BatchNorm2d(ci), nn.ReLU(),
                    nn.Conv2d(ci, co, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU()]
        self.features = nn.Sequential(nn.Conv2d(1, c, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(),
                                      *ds(c, c), *ds(c, c), nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Linear(c, n)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class TinyKWSFreqPool(nn.Module):
    """Conv identik TinyKWSNet, tetapi pooling akhir hanya pada sumbu WAKTU sehingga informasi
    frekuensi (pola formant) tetap sampai ke classifier: 16 kanal x 13 Mel-bin -> Linear."""
    def __init__(self, n):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(8, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
                                      nn.AdaptiveAvgPool2d((1, None)))
        self.classifier = nn.Linear(16*(F//2), n)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def make(cls, seed=2026):
    """Seed SEBELUM inisialisasi bobot agar hasil reprodusibel."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return cls(len(classes))


def augment(xb, rng):
    xb = xb*rng.uniform(0.95, 1.05) + rng.normal(0, 0.015, xb.shape).astype(np.float32)
    for x in xb:
        if rng.random() < 0.5:
            t0 = rng.integers(0, T-6); x[t0:t0+4] = 0
        if rng.random() < 0.5:
            f0 = rng.integers(0, F-4); x[:, f0:f0+3] = 0
    return xb


def train(net, epochs, lr, seed, onecycle=False, patience=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = int(np.ceil(len(ytr)/8))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs*steps_per_epoch) if onecycle else None
    crit = nn.CrossEntropyLoss(); best = (-1, 1e9, None, 0); hist = []; bad = 0
    Xvt = torch.from_numpy(Xv[:, None]).float(); yvt = torch.from_numpy(yv)
    for ep in range(1, epochs+1):
        net.train(); perm = rng.permutation(len(ytr)); tl = 0
        for i in range(0, len(perm), 8):
            b = perm[i:i+8]
            xb = torch.from_numpy(augment(Xtr[b].copy(), rng)[:, None]).float()
            opt.zero_grad(); loss = crit(net(xb), torch.from_numpy(ytr[b])); loss.backward(); opt.step()
            if sch: sch.step()
            tl += float(loss)*len(b)
        net.eval()
        with torch.no_grad():
            lo = net(Xvt); vl = float(crit(lo, yvt)); vf = f1_score(yv, lo.argmax(1).numpy(), average='macro', zero_division=0)
        hist.append((tl/len(ytr), vl, vf))
        if vf > best[0]+1e-9 or (abs(vf-best[0]) < 1e-9 and vl < best[1]):
            best = (vf, vl, copy.deepcopy(net.state_dict()), ep); bad = 0
        else:
            bad += 1
        if patience and bad >= patience and ep >= 20:
            break
    net.load_state_dict(best[2]); net.eval()
    return {'val_f1': float(best[0]), 'val_loss': float(best[1]), 'best_epoch': int(best[3]),
            'epochs_run': len(hist), 'optimizer_steps': len(hist)*steps_per_epoch, 'history': hist}


def train_f1(net):
    with torch.no_grad():
        return float(f1_score(ytr, net(torch.from_numpy(Xtr[:, None]).float()).argmax(1).numpy(), average='macro'))

# %% [markdown]
# ### 1. Diagnosis underfitting CNN resmi

# %%
official = TinyKWSNet(len(classes))
ck = torch.load(ROOT/dl['model_file'], map_location='cpu')
official.load_state_dict(ck['state_dict']); official.eval()
h = met['history']
diag = {'epochs_run': len(h['train_loss']), 'optimizer_steps': len(h['train_loss'])*int(np.ceil(len(ytr)/8)),
        'final_train_loss': h['train_loss'][-1], 'chance_level_loss_ln4': float(np.log(len(classes))),
        'train_set_macro_f1': train_f1(official), 'validation_macro_f1': met['validation_macro_f1'],
        'diagnosis': None}
diag['diagnosis'] = ('UNDERFITTING: train loss masih dekat ln(4) dan train-F1 rendah; '
                     'jumlah langkah optimizer terlalu sedikit (16 sampel / batch 8 = 2 langkah per epoch).'
                     if diag['final_train_loss'] > 0.8*diag['chance_level_loss_ln4'] else 'Konvergen')
print('Diagnosis CNN resmi:', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in diag.items()})

# %% [markdown]
# ### 2. Varian eksplorasi: TinyKWSNet long-schedule & DS-CNN (checkpoint dari VALIDATION)

# %%
variants = {}
net_long = make(TinyKWSNet)
variants['tinykws_long'] = {'net': net_long, **train(net_long, 300, 3e-3, seed=2026, onecycle=True)}
net_ds = make(DSCNN)
variants['dscnn'] = {'net': net_ds, **train(net_ds, 300, 3e-3, seed=2026, onecycle=True)}
# jadwal training RESMI (60 epoch, lr 1e-3, patience 12) agar perbandingan arsitektur adil
net_fp = make(TinyKWSFreqPool)
variants['tinykws_freqpool'] = {'net': net_fp, **train(net_fp, 60, 1e-3, seed=2026, patience=12)}


def macs_of(net, shape=(1, 1, T, F)):
    total = 0
    hooks = []
    def hk(m, i, o):
        nonlocal total
        if isinstance(m, nn.Conv2d):
            total += int(o.numel()*m.in_channels//m.groups*m.kernel_size[0]*m.kernel_size[1])
        elif isinstance(m, nn.Linear):
            total += int(m.in_features*m.out_features)
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hk))
    with torch.no_grad():
        net(torch.zeros(shape))
    for hh in hooks: hh.remove()
    return total


cand_cfg = {}
for name, v in variants.items():
    net = v['net']
    p = MD/f'mp12_ext_{name}.pt'
    torch.save({'state_dict': net.state_dict(), 'classes': classes, 'normalization': {'mean': mu, 'std': sd},
                'architecture': type(net).__name__}, p)
    v.update({'params': int(sum(q.numel() for q in net.parameters())), 'macs': macs_of(net), 'train_f1': train_f1(net)})
    cand_cfg[name] = {'file': p.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                      'architecture': type(net).__name__, 'val_macro_f1': v['val_f1'], 'best_epoch': v['best_epoch'],
                      'params': v['params'], 'macs': v['macs'], 'normalization_fit_scope': 'training_only',
                      'checkpoint_selection_scope': 'validation_only', 'test_files_loaded': False,
                      'role': 'exploratory_candidate_frozen_before_mp13 (bukan kandidat deployment resmi)'}
    print(f"  {name:13s} val F1={v['val_f1']:.3f} (epoch {v['best_epoch']}) train F1={v['train_f1']:.3f} "
          f"params={v['params']} MACs={v['macs']:,}")
official_macs = macs_of(official)
(ROOT/'config/mp12_extension_candidates.json').write_text(json.dumps(cand_cfg, indent=2), encoding='utf-8')

# %% [markdown]
# ### 3. Seed stability konfigurasi resmi (60 epoch, lr 1e-3, patience 12)

# %%
seed_res = []
for s in (2026, 7, 42):
    r = train(make(TinyKWSNet, s), 60, 1e-3, seed=s, patience=12)
    seed_res.append(r['val_f1'])
print('Seed stability (resmi): val F1 =', [round(v, 3) for v in seed_res], '| mean = %.3f ± %.3f' % (np.mean(seed_res), np.std(seed_res)))

# %% [markdown]
# ### 4. Export ONNX + verifikasi onnxruntime (model resmi & varian terbaik)

# %%
onnx_res = {}
try:
    import onnxruntime as ort
    for name, net in [('official_tinykws', official), ('best_variant', max(variants.values(), key=lambda v: (v['val_f1'], v['train_f1']))['net'])]:
        p = MD/f'mp12_{name}.onnx'
        torch.onnx.export(net, torch.zeros(1, 1, T, F), str(p), input_names=['logmel'], output_names=['logits'],
                          opset_version=17, dynamo=False)
        sess = ort.InferenceSession(str(p), providers=['CPUExecutionProvider'])
        xv = Xv[:, None].astype(np.float32)
        o_ort = np.concatenate([sess.run(None, {'logmel': xv[i:i+1]})[0] for i in range(len(xv))])
        with torch.no_grad():
            o_pt = net(torch.from_numpy(xv)).numpy()
        onnx_res[name] = {'file': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
                          'max_abs_logit_diff': float(np.abs(o_ort-o_pt).max()),
                          'argmax_agreement': float(np.mean(o_ort.argmax(1) == o_pt.argmax(1)))}
except Exception as e:  # pragma: no cover
    onnx_res = {'error': repr(e)}
print('ONNX:', onnx_res)

# %% [markdown]
# ### 5. Profil memori aktivasi per layer (float32) — tensor arena proxy

# %%
def act_profile(net):
    rows = []
    hooks = [m.register_forward_hook(lambda m, i, o, n=n: rows.append((n, type(m).__name__, int(o.numel())*4)))
             for n, m in net.named_modules() if isinstance(m, (nn.Conv2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Linear))]
    with torch.no_grad():
        net(torch.zeros(1, 1, T, F))
    for hh in hooks: hh.remove()
    inp = T*F*4
    peak = max(inp + rows[0][2], max(rows[i][2]+rows[i+1][2] for i in range(len(rows)-1)))  # in+out ping-pong
    return rows, peak


prof = {}
for name, net in [('official_tinykws', official), ('dscnn', variants['dscnn']['net']), ('tinykws_freqpool', variants['tinykws_freqpool']['net'])]:
    rows, peak = act_profile(net)
    prof[name] = {'layers': [{'layer': n, 'type': t, 'out_bytes': b} for n, t, b in rows], 'peak_pingpong_bytes': int(peak)}
    print(f'  {name:16s} peak activation (ping-pong) = {peak/1024:.1f} KiB')

# %% [markdown]
# ### 6. Saliency (gradient × input) pada validation untuk varian terbaik

# %%
best_name = max(variants, key=lambda k: (variants[k]['val_f1'], variants[k]['train_f1']))
bnet = variants[best_name]['net']
xin = torch.from_numpy(Xv[:, None]).float().requires_grad_(True)
out = bnet(xin)
out.gather(1, torch.from_numpy(yv)[:, None]).sum().backward()
sal = (xin.grad*xin).abs().detach().numpy()[:, 0]

ext = {'official_underfitting_diagnosis': diag,
       'exploratory_variants': {k: {kk: vv for kk, vv in v.items() if kk not in ('net', 'history')} for k, v in variants.items()},
       'official_macs_hook_count': official_macs, 'seed_stability_official_val_f1': seed_res,
       'onnx_export': onnx_res, 'activation_memory_profile': prof, 'saliency_model': best_name,
       'candidate_registry': 'config/mp12_extension_candidates.json', 'test_files_loaded': False,
       'note': 'Varian eksplorasi dipilih via validation; CNN resmi MP12 tetap kandidat deployment.'}
(OUT_MET/'mp12_extension_cnn_metrics.json').write_text(json.dumps(ext, indent=2), encoding='utf-8')

fig, ax = plt.subplots(2, 3, figsize=(17, 8.5))
for k, v in variants.items():
    hh = np.array(v['history']); ax[0, 0].plot(hh[:, 0], label=f'{k} train'); ax[0, 0].plot(hh[:, 1], '--', label=f'{k} val')
ax[0, 0].plot(h['train_loss'], 'k', lw=2, label='resmi train'); ax[0, 0].axhline(np.log(4), c='gray', ls=':', label='ln 4 (chance)')
ax[0, 0].set_ylim(0, 2); ax[0, 0].set_title('Loss: resmi (60 ep) vs long-schedule'); ax[0, 0].legend(fontsize=7); ax[0, 0].set_xlabel('epoch')
names = ['resmi'] + list(variants)
ax[0, 1].bar(names, [met['validation_macro_f1']] + [variants[k]['val_f1'] for k in variants], alpha=.8, label='val F1')
ax[0, 1].plot(names, [diag['train_set_macro_f1']] + [variants[k]['train_f1'] for k in variants], 'rs', label='train F1')
ax[0, 1].tick_params(axis='x', rotation=20)
ax[0, 1].scatter(['resmi']*3, seed_res, c='k', zorder=3, label='resmi 3 seed'); ax[0, 1].legend()
ax[0, 1].set_ylim(0, 1.05); ax[0, 1].set_title('Validation macro-F1')
pm = [met['parameter_count']] + [variants[k]['params'] for k in variants]
mc = [official_macs] + [variants[k]['macs'] for k in variants]
nb = len(names)
ax[0, 2].bar(np.arange(nb)-0.2, pm, 0.4, label='params'); ax2 = ax[0, 2].twinx(); ax2.bar(np.arange(nb)+0.2, mc, 0.4, color='C1', label='MACs')
ax[0, 2].set_xticks(range(nb), names, rotation=20); ax[0, 2].set_title('Params (biru) vs MACs (oranye)')
for i in range(3):
    ax[1, i].imshow(sal[i].T, aspect='auto', origin='lower', cmap='hot')
    ax[1, i].set_title(f'Saliency {best_name}: {classes[yv[i]]} (val)'); ax[1, i].set_xlabel('frame'); ax[1, i].set_ylabel('Mel')
for a in ax[0]: a.grid(alpha=.2)
fig.suptitle('MP12 FITUR TAMBAHAN - diagnosis underfitting, DS-CNN, saliency')
fig.tight_layout(); fig.savefig(OUT_FIG/'mp12_extension_cnn_analysis.png', dpi=140); plt.close(fig)
print('Fitur tambahan MP12 selesai -> results/metrics/mp12_extension_cnn_metrics.json')

# %%
if "__file__" not in globals():
    from IPython.display import Image, display
    for p in ['mp12_learning_curve.png', 'mp12_validation_confusion_matrix.png', 'mp12_resource_proxy.png', 'mp12_extension_cnn_analysis.png']:
        display(Image(filename=str(OUT_FIG/p), width=760))
