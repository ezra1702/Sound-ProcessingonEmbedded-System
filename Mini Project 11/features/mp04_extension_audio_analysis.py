"""
mp04_extension_audio_analysis.py
=================================
Fitur Tambahan Mini Project 4 - Advanced Acoustic Channel & Room Metrics
Disalin dari: mini_project_04.py (bagian FITUR TAMBAHAN, baris 189-333)

File ini adalah modul mandiri (standalone) yang membaca artefak hasil
mini_project_04.py (RIR .npz, WAV channel, spectral_config) lalu
menghitung metrik akustik ruangan lanjutan.

Jalankan SETELAH mini_project_04.py selesai dieksekusi.

Anggota Kelompok :
1. Christama Ezra Yudianto    (245150307111009)
2. Abdillah Fatah Suryanto    (245150300111008)
3. Ivang Arfandia             (245150307111003)
4. Jahzeel Erason Nicodemus   (245150300111002)

Fitur Tambahan yang Dihitung:
1. Clarity Index (C50)       : Rasio energi suara awal (0-50 ms) vs reverberasi
                               akhir (> 50 ms) — standar ISO 3382-1 Speech Clarity
2. Direct-to-Reverberant Ratio (DRR): Rasio energi lintasan langsung vs pantulan (dB)
3. Early Decay Time (EDT)    : Estimasi peluruhan awal 0 s.d. -10 dB,
                               diextrapolasi ke -60 dB
4. Comb Filtering & Spectral Coloration: Kedalaman notch/puncak pada |H(f)|
                               di rentang vokal 300-3400 Hz
5. Cross-Correlation & Temporal Smearing: Koefisien korelasi dan time lag
                               antara sinyal clean vs channel
6. Visualisasi 3 Subplot     :
   - Early Reflections & C50 window (zoom 0-80 ms)
   - EDC regression: EDT vs RT60
   - Channel frequency response & comb filtering notch
"""

from pathlib import Path
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve

# ------------------------------------------------------------
# SETUP PATH MANDIRI (relatif terhadap Mini Project 04/)
# ------------------------------------------------------------
ROOT         = Path(__file__).resolve().parent.parent   # naik ke Mini Project 04/
INPUT_WAV    = ROOT / "data/raw/nyala_001.wav"
CHANNEL_WAV  = ROOT / "data/processed/mp04_channel/nyala_001_channel.wav"
RIR_NPZ      = ROOT / "features/intermediate/mp04_synthetic_rir.npz"
SPECTRAL_PATH = ROOT / "config/spectral_config.json"
CONFIG_PATH  = ROOT / "config/channel_config.json"
OUT_MET      = ROOT / "results/metrics"
OUT_FIG      = ROOT / "results/figures"

for p in [OUT_MET, OUT_FIG]:
    p.mkdir(parents=True, exist_ok=True)

# Validasi artefak tersedia
for fp in [INPUT_WAV, CHANNEL_WAV, RIR_NPZ, SPECTRAL_PATH]:
    if not fp.exists():
        raise FileNotFoundError(
            f"Artefak tidak ditemukan: {fp}\n"
            "Pastikan mini_project_04.py sudah dijalankan terlebih dahulu."
        )

# Baca spectral config
with open(SPECTRAL_PATH, "r", encoding="utf-8") as f:
    spectral = json.load(f)
fs = int(spectral["sample_rate_hz"])

# Baca sinyal bersih (clean)
_, raw_x = wavfile.read(INPUT_WAV)
if raw_x.ndim > 1:
    raw_x = raw_x.mean(axis=1)
if np.issubdtype(raw_x.dtype, np.integer):
    scale = float(max(abs(np.iinfo(raw_x.dtype).min), np.iinfo(raw_x.dtype).max))
    x = raw_x.astype(np.float32) / scale
else:
    x = raw_x.astype(np.float32)
x = np.clip(x, -1.0, 1.0)

# Baca sinyal channel (setelah konvolusi RIR)
_, raw_y = wavfile.read(CHANNEL_WAV)
if raw_y.ndim > 1:
    raw_y = raw_y.mean(axis=1)
y = raw_y.astype(np.float32) / 32767.0
y = np.clip(y, -1.0, 1.0)

# Muat artefak RIR dari .npz
data_npz   = np.load(RIR_NPZ)
rir        = data_npz["rir"].astype(np.float32)
time_ir    = data_npz["time_s"]
freqs      = data_npz["freqs_hz"]
mag_db     = data_npz["magnitude_db"]
edc_db     = data_npz["edc_db"]

# Baca channel_config untuk mendapatkan parameter RIR
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    channel_config = json.load(f)
reflection_delays_ms = channel_config["reflection_delays_ms"]
rt60_target_s        = channel_config["rt60_target_s"]

# Hitung RT60 dari slope -5 sampai -35 dB
mask_rt60 = (edc_db <= -5.0) & (edc_db >= -35.0)
if np.count_nonzero(mask_rt60) >= 10:
    slope, intercept = np.polyfit(time_ir[mask_rt60], edc_db[mask_rt60], 1)
    rt60_est = float(-60.0 / slope) if slope < 0 else float("nan")
else:
    slope = float("nan")
    intercept = float("nan")
    rt60_est = float("nan")

# Hitung FFT respons channel
nfft_ir = max(8192, 2 ** int(np.ceil(np.log2(len(rir)))))
H = np.fft.rfft(rir, n=nfft_ir)

# ============================================================
# FITUR TAMBAHAN - ADVANCED ACOUSTIC CHANNEL & ROOM METRICS
# ============================================================
print("\n" + "=" * 58)
print("FITUR TAMBAHAN - ADVANCED ACOUSTIC CHANNEL & ROOM METRICS")
print("=" * 58)

# 1. CLARITY INDEX C50 (SPEECH CLARITY - ISO 3382-1)
idx_50ms = int(round(0.050 * fs))
energy_early_50ms = float(np.sum(rir[:idx_50ms] ** 2))
energy_late_50ms  = float(np.sum(rir[idx_50ms:] ** 2) + 1e-12)
c50_db = float(10.0 * np.log10(energy_early_50ms / energy_late_50ms))

# 2. DIRECT-TO-REVERBERANT RATIO (DRR)
# Direct sound: t < 5 ms (sebelum pantulan pertama pada 16 ms)
idx_direct   = int(round(0.005 * fs))
energy_direct = float(np.sum(rir[:idx_direct] ** 2))
energy_reverb = float(np.sum(rir[idx_direct:] ** 2) + 1e-12)
drr_db = float(10.0 * np.log10(energy_direct / energy_reverb))

# 3. EARLY DECAY TIME (EDT) - 0 dB sampai -10 dB
mask_edt = (edc_db <= 0.0) & (edc_db >= -10.0)
if np.count_nonzero(mask_edt) >= 5:
    slope_edt, intercept_edt = np.polyfit(time_ir[mask_edt], edc_db[mask_edt], 1)
    edt_est = float(-60.0 / slope_edt) if slope_edt < 0 else float("nan")
else:
    slope_edt   = float("nan")
    intercept_edt = float("nan")
    edt_est     = float("nan")

# 4. COMB FILTERING & SPECTRAL COLORATION (Pita Vokal 300-3400 Hz)
mask_speech      = (freqs >= 300) & (freqs <= 3400)
peak_to_notch_db = float(np.max(mag_db[mask_speech]) - np.min(mag_db[mask_speech]))
h_linear         = np.abs(H[mask_speech]) + 1e-12
spectral_flatness = float(np.exp(np.mean(np.log(h_linear))) / np.mean(h_linear))

# 5. KORELASI SILANG & TEMPORAL SMEARING (CLEAN VS CHANNEL)
n_corr   = min(len(x), len(y))
x_center = x[:n_corr] - np.mean(x[:n_corr])
y_center = y[:n_corr] - np.mean(y[:n_corr])
denom    = (np.std(x[:n_corr]) * np.std(y[:n_corr]) * n_corr) + 1e-12
xcorr    = np.correlate(x_center, y_center, mode='full') / denom
lags     = np.arange(-n_corr + 1, n_corr)
best_idx = np.argmax(xcorr)
max_correlation = float(xcorr[best_idx])
time_lag_ms     = float(lags[best_idx] / fs * 1000.0)

# 6. SIMPAN METRIK FITUR TAMBAHAN KE JSON
ext_metrics = {
    "feature_extension": "Advanced Acoustic Channel & Room Metrics",
    "speech_clarity_c50_db": c50_db,
    "direct_to_reverberant_ratio_drr_db": drr_db,
    "early_decay_time_edt_s": edt_est,
    "rt60_estimated_s": rt60_est,
    "comb_filtering_analysis": {
        "speech_band_hz": [300, 3400],
        "peak_to_notch_depth_db": peak_to_notch_db,
        "spectral_flatness": spectral_flatness
    },
    "correlation_and_smearing": {
        "max_cross_correlation": max_correlation,
        "alignment_lag_ms": time_lag_ms
    }
}

ext_met_path = OUT_MET / "mp04_extension_channel_metrics.json"
with open(ext_met_path, "w", encoding="utf-8") as f:
    json.dump(ext_metrics, f, indent=2)

# 7. VISUALISASI FITUR TAMBAHAN (3 SUBPLOT DETAIL)
fig_ext, ax_ext = plt.subplots(3, 1, figsize=(9, 8.5))

# Subplot 1: Pola Refleksi Awal & Jendela C50 (Zoom 0-80 ms)
mask_early = time_ir <= 0.080
ax_ext[0].plot(time_ir[mask_early] * 1000.0, rir[mask_early],
               color="steelblue", linewidth=1.0, label="RIR (0-80 ms)")
ax_ext[0].axvline(50, color="crimson", linestyle="--", linewidth=1.2, label="Batas C50 (50 ms)")
for d_ms in reflection_delays_ms:
    ax_ext[0].axvline(d_ms, color="darkorange", linestyle=":", linewidth=1.0, alpha=0.8,
                      label=f"Refleksi {d_ms} ms" if d_ms == reflection_delays_ms[0] else "")
ax_ext[0].set_title("Extension 1: Pola Kedatangan Refleksi Awal (Early Reflections) & Ambang Batas C50")
ax_ext[0].set_xlabel("Waktu (ms)")
ax_ext[0].set_ylabel("Amplitudo")
ax_ext[0].legend(loc="upper right", fontsize=8)
ax_ext[0].grid(alpha=0.25)

# Subplot 2: Analisis Regresi EDC (EDT vs RT60)
ax_ext[1].plot(time_ir, edc_db, color="darkgreen", linewidth=1.0, label="Energy Decay Curve (EDC)")
if not np.isnan(slope_edt):
    t_edt_plot = np.array([0, 0.15])
    ax_ext[1].plot(t_edt_plot, slope_edt * t_edt_plot + intercept_edt,
                   color="darkorange", linestyle="--",
                   label=f"Fit EDT (0 s.d. -10 dB) = {edt_est:.3f} s")
if not np.isnan(slope):
    t_rt_plot = np.array([0, 0.35])
    ax_ext[1].plot(t_rt_plot, slope * t_rt_plot + intercept,
                   color="red", linestyle=":",
                   label=f"Fit RT60 (-5 s.d. -35 dB) = {rt60_est:.3f} s")
ax_ext[1].axhline(0,   color="gray",       linestyle="--", linewidth=0.6)
ax_ext[1].axhline(-10, color="darkorange", linestyle="--", linewidth=0.7, alpha=0.7)
ax_ext[1].axhline(-35, color="red",        linestyle="--", linewidth=0.7, alpha=0.7)
ax_ext[1].set_ylim(-65, 3)
ax_ext[1].set_title("Extension 2: Perbandingan Peluruhan Awal (EDT) vs Waktu Reverberasi Penuh (RT60)")
ax_ext[1].set_xlabel("Waktu (s)")
ax_ext[1].set_ylabel("Energi (dB)")
ax_ext[1].legend(loc="upper right", fontsize=8)
ax_ext[1].grid(alpha=0.25)

# Subplot 3: Respons Frekuensi & Comb Filtering pada Rentang Vokal
ax_ext[2].plot(freqs, mag_db, color="purple", linewidth=0.9, label="|H(f)| Respons Kanal")
ax_ext[2].axvspan(300, 3400, color="lightyellow", alpha=0.5, label="Pita Vokal Utama (300-3400 Hz)")
ax_ext[2].set_xlim(0, min(6000, fs / 2))
ax_ext[2].set_ylim(-45, 5)
ax_ext[2].set_title(f"Extension 3: Analisis Comb Filtering | Peak-to-Notch Depth = {peak_to_notch_db:.2f} dB")
ax_ext[2].set_xlabel("Frekuensi (Hz)")
ax_ext[2].set_ylabel("Magnitudo (dB)")
ax_ext[2].legend(loc="lower right", fontsize=8)
ax_ext[2].grid(alpha=0.25)

fig_ext.tight_layout()
ext_fig_path = OUT_FIG / "mp04_extension_room_acoustics.png"
fig_ext.savefig(ext_fig_path, dpi=180)
plt.close(fig_ext)

# 8. OUTPUT TERMINAL FITUR TAMBAHAN
print("\n--- HASIL FITUR TAMBAHAN ---")
print("Speech Clarity (C50)        :", round(c50_db, 2), "dB")
print("Direct-to-Reverb Ratio (DRR):", round(drr_db, 2), "dB")
print("Early Decay Time (EDT)      :", round(edt_est, 3), "s")
print("Peak-to-Notch Ripple (Voice):", round(peak_to_notch_db, 2), "dB")
print("Spectral Flatness           :", round(spectral_flatness, 4))
print("Max Cross-Correlation       :", round(max_correlation, 4))
print("Extension Metrics File      :", ext_met_path)
print("Extension Figure File       :", ext_fig_path)
print("=" * 58)
