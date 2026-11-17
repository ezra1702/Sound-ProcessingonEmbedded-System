# %% [markdown]
# ## FITUR TAMBAHAN MP8 — Extended Features, Biaya Komputasi & Tabel Fitur untuk Firmware
# Modul mandiri, dijalankan SETELAH `mini_project_08.py`. Kontrak 39-D **tidak diubah**;
# fitur tambahan disimpan terpisah sebagai eksplorasi.
#
# Fitur tambahan (materi Bab 8 yang tidak diminta tugas):
# 1. **Delta & delta-delta MFCC** (regresi lebar 2) — dinamika spektral.
# 2. **Spectral flux, spectral flatness, energy contour (dB)** — flux memerlukan state frame sebelumnya.
# 3. **Pitch YIN (CMNDF)** sebagai pembanding pitch autokorelasi (validasi silang F0).
# 4. **Profiling biaya komputasi per kelompok fitur** (µs/frame di host) → dasar seleksi hardware-aware MP10.
# 5. **Uji presisi float32 vs float64** untuk MFCC (simulasi MCU ber-FPU single precision).
# 6. **Sanity check filterbank & DCT** (cakupan filter, orthonormalitas) +
#    export `firmware/mp08_feature_tables.h` (Mel filterbank sparse + matriks DCT float32).
#
# Anggota Kelompok : Christama Ezra Yudianto (245150307111009), Abdillah Fatah Suryanto (245150300111008),
# Ivang Arfandia (245150307111003), Jahzeel Erason Nicodemus (245150300111002)

# %%
from pathlib import Path
import json
import time
import numpy as np
import matplotlib
if "__file__" in globals():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import dct

ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
status = json.loads((ROOT/'project_status.json').read_text(encoding='utf-8'))
assert status['status'] in ('ready_for_mp09',), 'Jalankan mini_project_08.py terlebih dahulu'
fcfg = json.loads((ROOT/'config/feature_config.json').read_text(encoding='utf-8'))
Z = np.load(ROOT/status['mp08_input_frames'], allow_pickle=True)
frames = Z['frames'].astype(np.float64); fs = int(Z['fs']); N = frames.shape[1]
FF = np.load(ROOT/status['frame_features'], allow_pickle=True)
mfcc = FF['mfcc'].astype(np.float64); log_mel = FF['log_mel'].astype(np.float64)
fb = FF['mel_filterbank'].astype(np.float64); pitch_ac = FF['pitch_hz'].astype(np.float64)
OUT_MET = ROOT/'results/metrics'; OUT_FIG = ROOT/'results/figures'; FW = ROOT/'firmware'
for d in (OUT_MET, OUT_FIG, FW):
    d.mkdir(parents=True, exist_ok=True)
freq = np.fft.rfftfreq(N, 1/fs)
power = np.abs(np.fft.rfft(frames, axis=1))**2 + 1e-15

# %% [markdown]
# ### 1–2. Delta/delta-delta MFCC, spectral flux, flatness, energy contour

# %%
def deltas(feat, width=2):
    """Regresi delta standar (HTK): d_t = sum_n n (c_{t+n}-c_{t-n}) / (2 sum n^2)."""
    T = len(feat)
    pad = np.pad(feat, ((width, width), (0, 0)), mode='edge')
    den = 2*sum(n*n for n in range(1, width+1))
    return np.stack([sum(n*(pad[t+width+n]-pad[t+width-n]) for n in range(1, width+1))/den for t in range(T)])


d_mfcc = deltas(mfcc); dd_mfcc = deltas(d_mfcc)
mag = np.sqrt(power)
mag_n = mag/(mag.sum(axis=1, keepdims=True)+1e-12)
flux = np.r_[0.0, np.sqrt(np.sum(np.diff(mag_n, axis=0).clip(min=0)**2, axis=1))]
flatness = np.exp(np.mean(np.log(power), axis=1))/np.mean(power, axis=1)
energy_db = 10*np.log10(np.mean(frames**2, axis=1)+1e-12)
extended = {
    'delta_mfcc_mean': d_mfcc.mean(axis=0).tolist(), 'delta_mfcc_std': d_mfcc.std(axis=0).tolist(),
    'delta2_mfcc_mean': dd_mfcc.mean(axis=0).tolist(), 'delta2_mfcc_std': dd_mfcc.std(axis=0).tolist(),
    'spectral_flux_mean': float(flux.mean()), 'spectral_flux_std': float(flux.std()),
    'spectral_flatness_mean': float(flatness.mean()), 'spectral_flatness_std': float(flatness.std()),
    'energy_db_range': float(energy_db.max()-energy_db.min()),
    'extended_vector_dim_if_used': 39 + 13*4 + 4 + 1,
}
np.savez(ROOT/'features/intermediate/mp08_extended_frame_features.npz',
         delta_mfcc=d_mfcc.astype(np.float32), delta2_mfcc=dd_mfcc.astype(np.float32),
         spectral_flux=flux.astype(np.float32), spectral_flatness=flatness.astype(np.float32),
         energy_db=energy_db.astype(np.float32))
print('Delta/delta-delta MFCC:', d_mfcc.shape, dd_mfcc.shape, '| flux mean =', round(flux.mean(), 4),
      '| flatness mean =', round(float(flatness.mean()), 4))

# %% [markdown]
# ### 3. Pitch YIN (cumulative mean normalized difference) vs autokorelasi MP8

# %%
def yin_pitch(fr, fmin=80.0, fmax=350.0, thr=0.30, fallback=0.50):
    """YIN. Frame MP7 sudah ber-window Hamming (amplitudo tepi turun), sehingga threshold
    dilonggarkan (0.30) dan dipakai fallback minimum global bila CMNDF < 0.50."""
    W = len(fr)//2
    tau_min, tau_max = int(fs/fmax), min(W-1, int(fs/fmin))
    d = np.array([np.sum((fr[:W]-fr[t:t+W])**2) for t in range(tau_max+1)])
    cmnd = np.ones_like(d)
    cmnd[1:] = d[1:]*np.arange(1, len(d))/(np.cumsum(d[1:])+1e-12)
    cand = np.where(cmnd[tau_min:tau_max] < thr)[0]
    if len(cand) == 0:
        t_best = int(np.argmin(cmnd[tau_min:tau_max])) + tau_min
        if cmnd[t_best] >= fallback:
            return 0.0
        cand = np.array([t_best - tau_min])
    t = cand[0] + tau_min
    while t+1 < tau_max and cmnd[t+1] < cmnd[t]:
        t += 1
    # interpolasi parabola untuk resolusi sub-sampel
    if 1 <= t < len(cmnd)-1:
        a, b, c = cmnd[t-1], cmnd[t], cmnd[t+1]
        den = a - 2*b + c
        t = t + 0.5*(a - c)/den if abs(den) > 1e-12 else t
    return float(fs/t)


rms = np.sqrt(np.mean(frames**2, axis=1))
pitch_yin = np.array([yin_pitch(f) if r > np.percentile(rms, 25) else 0.0 for f, r in zip(frames, rms)])
both = (pitch_yin > 0) & (pitch_ac > 0)
pitch_cmp = {
    'voiced_ratio_autocorr': float(np.mean(pitch_ac > 0)),
    'voiced_ratio_yin': float(np.mean(pitch_yin > 0)),
    'frames_both_voiced': int(both.sum()),
    'median_f0_autocorr_hz': float(np.median(pitch_ac[pitch_ac > 0])) if np.any(pitch_ac > 0) else 0.0,
    'median_f0_yin_hz': float(np.median(pitch_yin[pitch_yin > 0])) if np.any(pitch_yin > 0) else 0.0,
    'mean_abs_diff_hz_on_both_voiced': float(np.mean(np.abs(pitch_ac[both]-pitch_yin[both]))) if both.any() else None,
    'gross_error_rate_20pct': float(np.mean(np.abs(pitch_ac[both]-pitch_yin[both])/pitch_yin[both] > 0.2)) if both.any() else None,
}
print('Pitch autocorr vs YIN:', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in pitch_cmp.items()})

# %% [markdown]
# ### 4. Profiling biaya komputasi per kelompok fitur (host, us per frame)

# %%
fr0 = frames[len(frames)//2]
win_lag = (int(fs/350), min(N-2, int(fs/80)))


def t_us(fn, reps=300):
    for _ in range(20):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter()-t0)/reps*1e6


P0 = np.abs(np.fft.rfft(fr0))**2 + 1e-15
cost = {
    'rms': t_us(lambda: np.sqrt(np.mean(fr0**2))),
    'zcr': t_us(lambda: np.mean(np.abs(np.diff(np.signbit(fr0))))),
    'fft_power_spectrum': t_us(lambda: np.abs(np.fft.rfft(fr0))**2),
    'centroid_bandwidth_rolloff': t_us(lambda: (np.sum(P0*freq)/P0.sum(), np.cumsum(P0))),
    'mel_filterbank_log': t_us(lambda: np.log(np.maximum(P0 @ fb.T, 1e-12))),
    'dct_13_mfcc': t_us(lambda: dct(np.log(P0 @ fb.T), type=2, norm='ortho')[:13]),
    'pitch_autocorrelation': t_us(lambda: np.correlate(fr0, fr0, mode='full')[N-1:]),
    'pitch_yin': t_us(lambda: yin_pitch(fr0), reps=40),
}
# MAC estimate (platform independent)
nnz_fb = int(np.count_nonzero(fb))
macs = {
    'rms': N, 'zcr': N, 'fft_power_spectrum': int(N*np.log2(512)*2 + (N//2+1)*2),
    'centroid_bandwidth_rolloff': 4*(N//2+1), 'mel_filterbank_log': nnz_fb,
    'dct_13_mfcc': 26*13, 'pitch_autocorrelation': int(N*(win_lag[1]-win_lag[0]+1)),
    'pitch_yin': int((N//2)*(win_lag[1]+1)),
}
for k in cost:
    print(f'  {k:28s} {cost[k]:8.2f} us/frame (host)  ~{macs[k]:>7d} MAC/frame')

# %% [markdown]
# ### 5. Presisi float32 vs float64 MFCC (simulasi MCU single-precision FPU)

# %%
f32 = frames.astype(np.float32)
P32 = (np.abs(np.fft.rfft(f32, axis=1)).astype(np.float32)**2 + np.float32(1e-15))
lm32 = np.log(np.maximum(P32 @ fb.astype(np.float32).T, np.float32(1e-12)))
mf32 = dct(lm32, type=2, axis=1, norm='ortho')[:, :13]
mfcc_err = np.abs(mf32.astype(np.float64)-mfcc)
prec = {'mfcc_max_abs_err': float(mfcc_err.max()), 'mfcc_mean_abs_err': float(mfcc_err.mean()),
        'mfcc_sqnr_db': float(10*np.log10(np.mean(mfcc**2)/(np.mean(mfcc_err**2)+1e-30)))}
print('float32 vs float64 MFCC:', prec)

# %% [markdown]
# ### 6. Sanity check filterbank & DCT + export tabel fitur ke header C

# %%
coverage = fb.sum(axis=0)
nonempty = int(np.sum(fb.sum(axis=1) > 0))
Dm = dct(np.eye(26), type=2, axis=0, norm='ortho')[:13]       # 13x26
ortho_err = float(np.abs(Dm @ Dm.T - np.eye(13)).max())
fb_check = {'n_filters_nonempty': nonempty, 'n_filters': int(fb.shape[0]),
            'nonzero_weights': nnz_fb, 'dense_weights': int(fb.size),
            'sparsity_saving_pct': float(100*(1-nnz_fb/fb.size)),
            'dct_orthonormal_max_err': ortho_err,
            'bins_without_coverage_in_band': int(np.sum((coverage == 0) & (freq >= fcfg['mel']['fmin_hz']) & (freq <= fcfg['mel']['fmax_hz'])))}
print('Filterbank check:', fb_check)

# sparse representation: per filter (start_bin, length) + weights
starts, lens, weights = [], [], []
for m in range(fb.shape[0]):
    nz = np.nonzero(fb[m])[0]
    starts.append(int(nz[0])); lens.append(int(len(nz))); weights.extend(fb[m, nz[0]:nz[0]+len(nz)].tolist())
fmt = lambda arr: ', '.join(f'{v:#.8g}f' for v in arr)
hdr = f"""#pragma once
/* Auto-generated by Mini Project 8 - FITUR TAMBAHAN: tabel fitur untuk firmware */
#include <stdint.h>
#define MP08_FS_HZ {fs}
#define MP08_FRAME_SAMPLES {N}
#define MP08_N_FFT_BINS {N//2+1}
#define MP08_N_MELS {fb.shape[0]}
#define MP08_N_MFCC 13
#define MP08_MEL_NNZ {len(weights)}
/* Mel filter m memakai bin [start[m], start[m]+len[m]) dengan bobot berurutan di MP08_MEL_W */
static const uint16_t MP08_MEL_START[{len(starts)}] = {{ {', '.join(map(str, starts))} }};
static const uint16_t MP08_MEL_LEN[{len(lens)}] = {{ {', '.join(map(str, lens))} }};
static const float MP08_MEL_W[{len(weights)}] = {{ {fmt(weights)} }};
/* DCT-II orthonormal 13x26 (row-major): mfcc[k] = sum_m DCT[k][m] * log_mel[m] */
static const float MP08_DCT[{Dm.size}] = {{ {fmt(Dm.ravel())} }};
/* Flash tabel: {4*len(weights) + 4*Dm.size + 4*len(starts)} byte (vs dense {4*fb.size + 4*Dm.size} byte) */
"""
(FW/'mp08_feature_tables.h').write_text(hdr, encoding='utf-8')

# %% [markdown]
# ### 7. Simpan metrik + visualisasi

# %%
ext = {'extended_features': extended, 'pitch_validation': pitch_cmp,
       'compute_cost_us_per_frame_host': cost, 'compute_cost_mac_per_frame': macs,
       'float32_precision': prec, 'filterbank_dct_check': fb_check,
       'firmware_header': 'firmware/mp08_feature_tables.h',
       'note': 'Kontrak 39-D tidak berubah. Biaya host hanya proxy relatif.'}
(OUT_MET/'mp08_extension_feature_metrics.json').write_text(json.dumps(ext, indent=2), encoding='utf-8')

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].imshow(np.vstack([mfcc.T, d_mfcc.T, dd_mfcc.T]), aspect='auto', origin='lower', cmap='magma')
ax[0, 0].axhline(12.5, c='w', lw=.8); ax[0, 0].axhline(25.5, c='w', lw=.8)
ax[0, 0].set_title('MFCC | Δ | ΔΔ (baris 0-12 | 13-25 | 26-38)'); ax[0, 0].set_xlabel('Frame')
ax[0, 1].plot(flux/flux.max(), label='spectral flux (norm)'); ax[0, 1].plot(flatness, label='spectral flatness')
ax[0, 1].plot((energy_db-energy_db.min())/(np.ptp(energy_db)+1e-9), label='energy contour (norm)')
ax[0, 1].set_title('Fitur dinamis tambahan'); ax[0, 1].legend(fontsize=8); ax[0, 1].set_xlabel('Frame')
ax[1, 0].plot(np.where(pitch_ac > 0, pitch_ac, np.nan), 'o-', ms=3, label='autokorelasi (MP8)')
ax[1, 0].plot(np.where(pitch_yin > 0, pitch_yin, np.nan), 's--', ms=3, label='YIN (tambahan)')
ax[1, 0].set_title('Validasi silang F0'); ax[1, 0].set_ylabel('Hz'); ax[1, 0].legend()
ks = list(cost)
ax[1, 1].barh(ks, [cost[k] for k in ks]); ax[1, 1].set_xscale('log')
ax[1, 1].set_title('Biaya host per kelompok fitur (us/frame, log)')
for a in ax.ravel():
    a.grid(alpha=.2)
fig.suptitle('Mini Project 8 - FITUR TAMBAHAN: extended features, validasi pitch, biaya komputasi')
fig.tight_layout(); fig.savefig(OUT_FIG/'mp08_extension_feature_analysis.png', dpi=160); plt.close(fig)
print('Fitur tambahan MP8 selesai -> results/metrics/mp08_extension_feature_metrics.json')

# %%
if "__file__" not in globals():
    from IPython.display import Image, display
    for p in ['mp08_logmel_mfcc.png', 'mp08_scalar_features.png', 'mp08_extension_feature_analysis.png']:
        display(Image(filename=str(OUT_FIG/p), width=760))
