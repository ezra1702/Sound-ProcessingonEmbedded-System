# %% [markdown]
# ## FITUR TAMBAHAN MP7 — Preprocessing Quality, Alternatif Metode & Kesiapan Firmware
# Modul mandiri, dijalankan SETELAH `mini_project_07.py`. Tidak mengubah artefak wajib
# (preprocessing_config.json, frame NPZ, status) sehingga kontrak MP8 tetap utuh.
#
# Fitur tambahan (diambil dari materi Bab 7 yang belum ada di tugas):
# 1. **Perbandingan normalisasi** peak vs RMS vs tanpa normalisasi (crest factor, headroom, clipping risk).
# 2. **Wiener filter** (decision-directed, Ephraim–Malah sederhana) vs spectral subtraction:
#    a-posteriori SNR, segmental SNR, dan *musical-noise proxy*.
# 3. **VAD dual-feature (energi + ZCR + hangover)** vs VAD RMS adaptif → agreement & retensi ucapan.
# 4. **Speech-to-noise ratio & noise floor** sebelum/sesudah preprocessing memakai label VAD.
# 5. **Clipping check sebelum scaling** pada audio input.
# 6. **Simulasi fixed-point Q15** DC-blocker + pre-emphasis + window (siap MCU) dan export
#    `firmware/mp07_preprocessing_q15.h` (window Hamming Q15 + konstanta).
#
# Anggota Kelompok : Christama Ezra Yudianto (245150307111009), Abdillah Fatah Suryanto (245150300111008),
# Ivang Arfandia (245150307111003), Jahzeel Erason Nicodemus (245150300111002)

# %%
from pathlib import Path
import json
import numpy as np
import matplotlib
if "__file__" in globals():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, istft, welch, lfilter

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
status = json.loads((ROOT/'project_status.json').read_text(encoding='utf-8'))
assert status['status'] in ('ready_for_mp08',), 'Jalankan mini_project_07.py terlebih dahulu'
cfg = json.loads((ROOT/'config/preprocessing_config.json').read_text(encoding='utf-8'))
fs, raw = wavfile.read(ROOT/status['mp07_input_audio'])
x = raw.astype(np.float64) / 32768.0 if np.issubdtype(raw.dtype, np.integer) else raw.astype(np.float64)
N, H = cfg['frame_length_samples'], cfg['hop_length_samples']
OUT_MET = ROOT/'results/metrics'; OUT_FIG = ROOT/'results/figures'; FW = ROOT/'firmware'
for d in (OUT_MET, OUT_FIG, FW):
    d.mkdir(parents=True, exist_ok=True)


def frame_signal(sig, n=N, h=H):
    st = np.arange(0, max(1, len(sig) - n + 1), h)
    return np.stack([np.pad(sig[s:s+n], (0, max(0, n - len(sig[s:s+n])))) for s in st]), st


# %% [markdown]
# ### 1. Clipping check sebelum scaling + perbandingan metode normalisasi

# %%
pre_clip = int(np.sum(np.abs(x) >= 0.999))
x_dc = x - x.mean()
peak_norm = 0.95 * x_dc / (np.max(np.abs(x_dc)) + 1e-12)
target_rms_dbfs = -20.0
g_rms = 10**(target_rms_dbfs/20) / (np.sqrt(np.mean(x_dc**2)) + 1e-12)
rms_norm = x_dc * g_rms
norm_cmp = {}
for name, s in [('none', x_dc), ('peak_0.95', peak_norm), ('rms_-20dBFS', rms_norm)]:
    pk = float(np.max(np.abs(s))); r = float(np.sqrt(np.mean(s**2)))
    norm_cmp[name] = {
        'peak': pk, 'rms': r,
        'rms_dbfs': float(20*np.log10(r + 1e-12)),
        'crest_factor_db': float(20*np.log10(pk/(r+1e-12))),
        'headroom_db': float(-20*np.log10(pk + 1e-12)),
        'would_clip_samples': int(np.sum(np.abs(s) >= 1.0)),
    }
print('Clipping sebelum scaling:', pre_clip, 'sampel')
for k, v in norm_cmp.items():
    print(f"  {k:12s} peak={v['peak']:.3f} rms={v['rms_dbfs']:.1f} dBFS crest={v['crest_factor_db']:.1f} dB clip={v['would_clip_samples']}")

# %% [markdown]
# ### 2. VAD dual-feature (energi + ZCR + hangover) vs VAD RMS adaptif MP7

# %%
fr, st = frame_signal(peak_norm)
rms = np.sqrt(np.mean(fr**2, axis=1) + 1e-12)
zcr = np.mean(np.abs(np.diff(np.signbit(fr), axis=1)), axis=1)
vad_mp7 = rms >= cfg['vad']['threshold']
e_db = 20*np.log10(rms + 1e-12)
noise_db = np.percentile(e_db, 10)
speech_db = np.percentile(e_db, 95)
thr_hi = noise_db + 0.45*(speech_db - noise_db)      # onset kuat
thr_lo = noise_db + 0.20*(speech_db - noise_db)      # fricative lemah dibantu ZCR
zcr_thr = float(np.percentile(zcr, 70))
cand = (e_db >= thr_hi) | ((e_db >= thr_lo) & (zcr >= zcr_thr))
# hangover 3 frame (30 ms) agar ekor kata tidak terpotong
hang = 3
vad_dual = cand.copy()
cnt = 0
for i in range(len(cand)):
    if cand[i]:
        cnt = hang
    elif cnt > 0:
        vad_dual[i] = True; cnt -= 1
agree = float(np.mean(vad_dual == vad_mp7))
energy = fr**2
speech_energy_mp7 = float(energy[vad_mp7].sum() / energy.sum())
speech_energy_dual = float(energy[vad_dual].sum() / energy.sum())
print(f'VAD agreement MP7 vs dual = {agree:.3f}; retensi energi MP7={speech_energy_mp7:.3f}, dual={speech_energy_dual:.3f}')

# %% [markdown]
# ### 3. Wiener filter (decision-directed) vs spectral subtraction — uji derau terkontrol
# Audio MP6 hampir bebas derau aditif (frame diam ≈ −80 dBFS, sisanya ekor reverb), sehingga
# denoiser tidak punya derau untuk dikurangi. Karena itu dilakukan **uji terkontrol**: derau putih
# ditambahkan pada SNR 0/5/10/20 dB ke sinyal bersih, lalu SNR output dihitung terhadap referensi
# bersih (gain-matched). Ini juga menguji klaim materi: denoising tidak selalu menguntungkan.

# %%
nper = min(512, N); nov = nper//2
rng = np.random.default_rng(2026)


def _stft(sig):
    return stft(sig, fs=fs, window='hann', nperseg=nper, noverlap=nov, boundary='zeros')[2]


def _istft(Zc, n):
    return istft(Zc, fs=fs, window='hann', nperseg=nper, noverlap=nov)[1][:n]


def noise_psd_est(Z):
    m2 = np.abs(Z)**2
    cols = np.argsort(m2.mean(axis=0))[:max(1, int(0.2*m2.shape[1]))]
    return np.median(m2[:, cols], axis=1, keepdims=True) + 1e-15


def denoise_ss(sig, beta=0.6, floor=0.25):
    Z = _stft(sig); nmag = np.sqrt(noise_psd_est(Z)); mag = np.abs(Z)
    return _istft(np.maximum(mag - beta*nmag, floor*nmag)*np.exp(1j*np.angle(Z)), len(sig))


def denoise_wiener_dd(sig, alpha_dd=0.98, xi_min_db=-25):
    Z = _stft(sig); m2 = np.abs(Z)**2; npsd = noise_psd_est(Z)
    gamma = m2/npsd; G = np.ones_like(m2); prev_s2 = np.zeros(m2.shape[0])
    for k in range(m2.shape[1]):
        xi = alpha_dd*prev_s2/npsd[:, 0] + (1-alpha_dd)*np.maximum(gamma[:, k]-1, 0)
        xi = np.maximum(xi, 10**(xi_min_db/10))
        G[:, k] = xi/(1+xi)
        prev_s2 = (G[:, k]**2)*m2[:, k]
    return _istft(np.sqrt(m2)*G*np.exp(1j*np.angle(Z)), len(sig))


def snr_vs_ref(ref, est):
    g = float(np.dot(ref, est)/(np.dot(est, est)+1e-15))   # gain matching (least squares)
    e = ref - g*est
    return float(10*np.log10(np.sum(ref**2)/(np.sum(e**2)+1e-15)))


def musical_noise_proxy(sig, vad_mask):
    """Std log-spektrum pada frame non-speech: makin tinggi → makin banyak 'musical tone' terisolasi."""
    frs, _ = frame_signal(sig)
    m = min(len(frs), len(vad_mask))
    nf = frs[:m][~vad_mask[:m]]
    if len(nf) == 0:
        return float('nan')
    P = np.abs(np.fft.rfft(nf*np.hamming(N), axis=1))**2 + 1e-15
    return float(np.mean(np.std(10*np.log10(P), axis=1)))


clean = peak_norm
noise = rng.normal(size=len(clean))
denoise_cmp = {}
for snr_in in [0, 5, 10, 20]:
    nz = noise*np.sqrt(np.mean(clean**2)/(np.mean(noise**2)*10**(snr_in/10)))
    noisy = clean + nz
    y_ss = denoise_ss(noisy); y_wf = denoise_wiener_dd(noisy)
    denoise_cmp[f'snr_in_{snr_in}dB'] = {
        'input_snr_db': snr_vs_ref(clean, noisy),
        'spectral_subtraction_snr_db': snr_vs_ref(clean, y_ss),
        'wiener_dd_snr_db': snr_vs_ref(clean, y_wf),
        'musical_noise_proxy_db': {'noisy': musical_noise_proxy(noisy, vad_mp7),
                                   'spectral_subtraction': musical_noise_proxy(y_ss, vad_mp7),
                                   'wiener_dd': musical_noise_proxy(y_wf, vad_mp7)},
    }
    if snr_in == 5:
        demo = (noisy, y_ss, y_wf)
for k, v in denoise_cmp.items():
    print(f"  {k:13s} in={v['input_snr_db']:6.2f} dB | SS={v['spectral_subtraction_snr_db']:6.2f} dB | "
          f"Wiener={v['wiener_dd_snr_db']:6.2f} dB")
denoise_cmp['clean_mp6_noise_floor_note'] = ('Pada audio MP6 asli noise estimate sangat kecil (≈-107 dB per kolom STFT), '
                                             'sehingga MP7 secara efektif hanya melakukan trimming + normalisasi.')
# %% [markdown]
# ### 4. Simulasi fixed-point Q15: DC-blocker + pre-emphasis + Hamming window
# Pada MCU tanpa FPU, operasi per sampel dilakukan dalam int16/int32 (Q15). Simulasi ini
# mengukur error kuantisasi terhadap referensi float64.

# %%
def q15(v):
    return np.clip(np.round(np.asarray(v)*32768.0), -32768, 32767).astype(np.int32)


R_DC = 0.995  # DC blocker y[n] = x[n]-x[n-1] + R*y[n-1]
alpha = cfg['pre_emphasis_alpha']
win_q15 = q15(np.hamming(N))
xin_q15 = q15(peak_norm*0.5)  # headroom 6 dB untuk fixed-point
# referensi float
y_dc_f = lfilter([1, -1], [1, -R_DC], peak_norm*0.5)
y_pe_f = np.r_[y_dc_f[0], y_dc_f[1:] - alpha*y_dc_f[:-1]]
# fixed-point
Rq, Aq = int(round(R_DC*32768)), int(round(alpha*32768))
y_dc_q = np.zeros_like(xin_q15); prev_x = 0; prev_y = 0
for n_, xv in enumerate(xin_q15):
    acc = (int(xv) - prev_x) * 32768 + Rq * prev_y          # Q30
    yv = int(np.clip((acc + (1 << 14)) >> 15, -32768, 32767))
    y_dc_q[n_] = yv; prev_x = int(xv); prev_y = yv
y_pe_q = np.empty_like(y_dc_q)
y_pe_q[0] = y_dc_q[0]
acc = y_dc_q[1:].astype(np.int64)*32768 - Aq*y_dc_q[:-1].astype(np.int64)
y_pe_q[1:] = np.clip((acc + (1 << 14)) >> 15, -32768, 32767)
err = y_pe_f - y_pe_q/32768.0
q15_sqnr = float(10*np.log10(np.mean(y_pe_f**2)/(np.mean(err**2)+1e-20)))
# window Q15 per frame
fr_f, _ = frame_signal(y_pe_f); fr_q, _ = frame_signal(y_pe_q.astype(np.float64))
w_q = (fr_q*win_q15[None, :]/32768.0)/32768.0
w_f = fr_f*np.hamming(N)[None, :]
win_sqnr = float(10*np.log10(np.mean(w_f**2)/(np.mean((w_f-w_q)**2)+1e-20)))
print(f'Q15 DC-blocker+pre-emphasis SQNR = {q15_sqnr:.1f} dB ; setelah window Q15 = {win_sqnr:.1f} dB')

vals = ', '.join(str(int(v)) for v in win_q15)
header = f"""#pragma once
/* Auto-generated by Mini Project 7 - FITUR TAMBAHAN (Q15 preprocessing) */
#include <stdint.h>
#define MP07_FS_HZ {fs}
#define MP07_FRAME_SAMPLES {N}
#define MP07_HOP_SAMPLES {H}
#define MP07_DC_BLOCK_R_Q15 {Rq}      /* R = {R_DC} */
#define MP07_PREEMPH_ALPHA_Q15 {Aq}  /* alpha = {alpha} */
#define MP07_VAD_RMS_THRESHOLD_Q15 {int(round(cfg['vad']['threshold']*32768))}
/* y[n] = x[n] - x[n-1] + R*y[n-1]  ;  p[n] = y[n] - alpha*y[n-1]  (Q15, saturasi int16) */
static const int16_t MP07_HAMMING_Q15[{N}] = {{ {vals} }};
"""
(FW/'mp07_preprocessing_q15.h').write_text(header, encoding='utf-8')

# %% [markdown]
# ### 5. Simpan metrik + visualisasi 2x2

# %%
ext = {
    'pre_scaling_clipping_samples': pre_clip,
    'normalization_comparison': norm_cmp,
    'vad_comparison': {'agreement_mp7_vs_dual': agree,
                       'active_ratio_mp7': float(vad_mp7.mean()),
                       'active_ratio_dual': float(vad_dual.mean()),
                       'speech_energy_retained_mp7': speech_energy_mp7,
                       'speech_energy_retained_dual': speech_energy_dual,
                       'dual_params': {'thr_hi_db': float(thr_hi), 'thr_lo_db': float(thr_lo),
                                       'zcr_thr': zcr_thr, 'hangover_frames': hang}},
    'denoise_comparison': denoise_cmp,
    'fixed_point_q15': {'dc_block_preemph_sqnr_db': q15_sqnr, 'after_window_sqnr_db': win_sqnr,
                        'header': 'firmware/mp07_preprocessing_q15.h'},
    'note': 'Analisis tambahan; artefak wajib MP7 tidak diubah.'
}
(OUT_MET/'mp07_extension_preprocessing_metrics.json').write_text(json.dumps(ext, indent=2), encoding='utf-8')

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
tt_ = np.arange(len(rms))*H/fs
ax[0, 0].plot(tt_, e_db, lw=.9, label='energi (dB)')
ax[0, 0].fill_between(tt_, e_db.min(), e_db.max(), where=vad_mp7, alpha=.15, label='VAD MP7')
ax[0, 0].fill_between(tt_, e_db.min(), e_db.min()+(e_db.max()-e_db.min())*0.15, where=vad_dual, color='C3', alpha=.6, label='VAD dual')
ax[0, 0].axhline(thr_hi, ls='--', c='C3', lw=.8); ax[0, 0].axhline(thr_lo, ls=':', c='C3', lw=.8)
ax[0, 0].set_title(f'VAD RMS vs dual (energi+ZCR), agreement={agree:.2f}'); ax[0, 0].set_xlabel('s'); ax[0, 0].legend(fontsize=7)
keys = [k for k in denoise_cmp if k.startswith('snr_in')]
xin = [denoise_cmp[k]['input_snr_db'] for k in keys]
ax[0, 1].plot(xin, xin, 'k:', label='tanpa denoise')
ax[0, 1].plot(xin, [denoise_cmp[k]['spectral_subtraction_snr_db'] for k in keys], 'o-', label='spectral subtraction')
ax[0, 1].plot(xin, [denoise_cmp[k]['wiener_dd_snr_db'] for k in keys], 's-', label='Wiener decision-directed')
ax[0, 1].set_title('Uji derau terkontrol: SNR output vs input'); ax[0, 1].set_xlabel('SNR input (dB)')
ax[0, 1].set_ylabel('SNR output (dB)'); ax[0, 1].legend(fontsize=8)
names = list(norm_cmp)
ax[1, 0].bar(names, [norm_cmp[n]['crest_factor_db'] for n in names], label='crest factor (dB)')
ax[1, 0].bar(names, [norm_cmp[n]['headroom_db'] for n in names], alpha=.6, label='headroom (dB)')
ax[1, 0].set_title('Perbandingan normalisasi'); ax[1, 0].legend()
seg = min(len(y_pe_f), int(0.03*fs)); i0 = int(np.argmax(np.abs(y_pe_f)))
sl = slice(max(0, i0-seg//2), max(0, i0-seg//2)+seg)
ax[1, 1].plot(y_pe_f[sl], label='float64', lw=1)
ax[1, 1].plot(y_pe_q[sl]/32768.0, '--', label=f'Q15 (SQNR {q15_sqnr:.1f} dB)', lw=1)
ax[1, 1].set_title('DC-blocker + pre-emphasis: float vs Q15'); ax[1, 1].legend()
for a in ax.ravel():
    a.grid(alpha=.2)
fig.suptitle('Mini Project 7 - FITUR TAMBAHAN: kualitas preprocessing & kesiapan firmware')
fig.tight_layout(); fig.savefig(OUT_FIG/'mp07_extension_preprocessing_analysis.png', dpi=160); plt.close(fig)
print('Fitur tambahan MP7 selesai ->', 'results/metrics/mp07_extension_preprocessing_metrics.json')

# %%
if "__file__" not in globals():  # tampilkan gambar hanya di notebook
    from IPython.display import Image, display
    for p in ['mp07_vad.png', 'mp07_preprocessing_overview.png', 'mp07_extension_preprocessing_analysis.png']:
        display(Image(filename=str(OUT_FIG/p), width=760))
