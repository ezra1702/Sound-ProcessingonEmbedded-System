"""
mp02_extension_audio_analysis.py
=================================
Fitur Tambahan Mini Project 2 - Advanced Sampling & Quantization Quality Analysis
Disalin dari: mini_project_02.py (bagian FITUR TAMBAHAN, baris 185-317)

File ini adalah modul mandiri (standalone) yang membaca file audio referensi
dan melakukan evaluasi mendalam terhadap kualitas sinyal hasil kuantisasi 8-bit
serta resampling 8 kHz.

Jalankan SETELAH mini_project_02.py selesai dieksekusi atau sebagai modul pengujian mandiri.

Anggota Kelompok :
1. Christama Ezra Yudianto    (245150307111009)
2. Abdillah Fatah Suryanto    (245150300111008)
3. Ivang Arfandia             (245150307111003)
4. Jahzeel Erason Nicodemus   (245150300111002)

Fitur Tambahan yang Dihitung:
1. PSNR (Peak Signal-to-Noise Ratio) untuk Kuantisasi 8-bit & Resampling 8 kHz
2. Analisis Galat Kuantisasi Lengkap (RMSE, MAE, Max Absolute Error)
3. High-Frequency Energy Loss (> 4 kHz): Mengukur persentase energi akustik yang hilang akibat downsampling ke 8 kHz
4. Analisis Bitrate Transmisi Audio untuk Embedded Stream (kbps)
5. Visualisasi Detail: Sinyal Galat Domain Waktu, Histogram PDF Error, dan Distribusi Kumulatif Energi
"""

from pathlib import Path
import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly

# ------------------------------------------------------------
# SETUP PATH MANDIRI (relatif terhadap Mini Project 02/)
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config/project_config.json"
FIG = ROOT / "results/figures"
MET = ROOT / "results/metrics"

for p in [FIG, MET]:
    p.mkdir(parents=True, exist_ok=True)

# Baca konfigurasi audio proyek
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

input_wav = ROOT / config.get("reference_file", "data/raw/nyala_001.wav")
if not input_wav.exists():
    raise FileNotFoundError(f"File audio acuan tidak ditemukan di: {input_wav}")

fs, raw = wavfile.read(input_wav)
if raw.ndim > 1:
    raw = raw.mean(axis=1)

# Konversi input PCM menjadi float [-1.0, 1.0]
if raw.dtype == np.uint8:
    x = (raw.astype(np.float32) - 128.0) / 128.0
elif np.issubdtype(raw.dtype, np.integer):
    scale = float(max(abs(np.iinfo(raw.dtype).min), np.iinfo(raw.dtype).max))
    x = raw.astype(np.float32) / scale
else:
    x = raw.astype(np.float32)
x = np.clip(x, -1.0, 1.0)

# ------------------------------------------------------------
# PEMROSESAN DASAR (Resampling & Kuantisasi untuk bahan evaluasi)
# ------------------------------------------------------------
target_fs = 8000
g = math.gcd(int(fs), target_fs)
x_8k = resample_poly(x, target_fs // g, int(fs) // g)

def quantize_u8(signal):
    return np.round((np.clip(signal, -1, 1) + 1.0) * 127.5).astype(np.uint8)

def dequantize_u8(q):
    return q.astype(np.float32) / 127.5 - 1.0

q_16k8 = quantize_u8(x)
x_q8 = dequantize_u8(q_16k8)

def snr_db(reference, test):
    n = min(len(reference), len(test))
    reference = reference[:n]
    test = test[:n]
    noise = reference - test
    ps = np.mean(reference ** 2) + 1e-12
    pn = np.mean(noise ** 2) + 1e-12
    return float(10.0 * np.log10(ps / pn))

quant_snr = snr_db(x, x_q8)

# Resample 8 kHz kembali ke fs asli untuk perbandingan numerik roundtrip
g2 = math.gcd(target_fs, int(fs))
x_8k_back = resample_poly(x_8k, int(fs) // g2, target_fs // g2)
resample_snr = snr_db(x, x_8k_back)

# ============================================================
# FITUR TAMBAHAN - ADVANCED SAMPLING & QUANTIZATION ANALYSIS
# ============================================================
print("\n" + "=" * 55)
print("FITUR TAMBAHAN - ADVANCED AUDIO QUALITY & ERROR ANALYSIS")
print("=" * 55)

# 1. METRIK ERROR KUANTISASI (RMSE, MAE, MAX ERROR, PSNR)
err_q8 = x - x_q8
mse_q8 = float(np.mean(err_q8 ** 2))
rmse_q8 = float(np.sqrt(mse_q8))
mae_q8 = float(np.mean(np.abs(err_q8)))
max_err_q8 = float(np.max(np.abs(err_q8)))
# PSNR dengan rentang peak sinyal ternormalisasi = 1.0
psnr_q8 = float(10.0 * np.log10(1.0 / (mse_q8 + 1e-12)))

# 2. METRIK ERROR RESAMPLING 8 kHz (ROUNDTRIP)
n_len = min(len(x), len(x_8k_back))
err_8k = x[:n_len] - x_8k_back[:n_len]
mse_8k = float(np.mean(err_8k ** 2))
rmse_8k = float(np.sqrt(mse_8k))
psnr_8k = float(10.0 * np.log10(1.0 / (mse_8k + 1e-12)))

# 3. HIGH-FREQUENCY ENERGY LOSS (> 4 kHz) AKIBAT DOWNSAMPLING
n_fft = len(x)
w_fft = np.hanning(n_fft)
X_fft = np.abs(np.fft.rfft(x * w_fft))
freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
hf_mask = freqs > (target_fs / 2.0)  # Frekuensi di atas Nyquist 8 kHz (4 kHz)
total_energy = float(np.sum(X_fft ** 2) + 1e-12)
hf_energy = float(np.sum(X_fft[hf_mask] ** 2))
hf_loss_pct = float(100.0 * (hf_energy / total_energy))

# 4. ANALISIS BITRATE TRANSMISI EMBEDDED (kbps)
bitrates_kbps = {
    "16k16_baseline": int(16000 * 16 / 1000),
    "8k16": int(8000 * 16 / 1000),
    "16k8": int(16000 * 8 / 1000),
    "8k8": int(8000 * 8 / 1000)
}

# 5. SIMPAN METRIK FITUR TAMBAHAN KE JSON
extension_metrics = {
    "feature_extension": "Advanced Sampling & Quantization Quality Analysis",
    "quantization_8bit_analysis": {
        "snr_db": quant_snr,
        "psnr_db": psnr_q8,
        "mse": mse_q8,
        "rmse": rmse_q8,
        "mae": mae_q8,
        "max_absolute_error": max_err_q8
    },
    "resampling_8k_analysis": {
        "roundtrip_snr_db": resample_snr,
        "psnr_db": psnr_8k,
        "rmse": rmse_8k,
        "hf_cutoff_hz": target_fs / 2.0,
        "hf_energy_loss_percent": hf_loss_pct
    },
    "transmission_bitrate_kbps": bitrates_kbps,
    "bandwidth_comparison": {
        "16kHz_bandwidth_hz": int(fs / 2),
        "8kHz_bandwidth_hz": int(target_fs / 2),
        "bandwidth_reduction_percent": 50.0
    }
}

extension_metrics_path = MET / "mp02_extension_audio_metrics.json"
with open(extension_metrics_path, "w", encoding="utf-8") as f:
    json.dump(extension_metrics, f, indent=2)

# 6. VISUALISASI FITUR TAMBAHAN (3 SUBPLOT DETAIL)
fig_ext, ax_ext = plt.subplots(3, 1, figsize=(9, 8))

# Subplot 1: Galat Kuantisasi 8-bit pada domain waktu
n_plot = min(int(0.04 * fs), len(err_q8))
t_plot = np.arange(n_plot) / fs * 1000.0
ax_ext[0].plot(t_plot, err_q8[:n_plot], color="crimson", linewidth=0.9, label="Quantization Error e[n] = x - x_q8")
ax_ext[0].axhline(0, color="black", linestyle="--", linewidth=0.7, alpha=0.6)
ax_ext[0].set_title("Extension 1: Sinyal Galat Kuantisasi 8-bit (Domain Waktu)")
ax_ext[0].set_xlabel("Waktu (ms)")
ax_ext[0].set_ylabel("Error Amplitudo")
ax_ext[0].legend(loc="upper right")
ax_ext[0].grid(alpha=0.25)

# Subplot 2: Histogram Distribusi Galat Kuantisasi (Uji Keseragaman Noise)
ax_ext[1].hist(err_q8, bins=60, color="navy", alpha=0.75, edgecolor="black", density=True)
ax_ext[1].set_title("Extension 2: Histogram Distribusi Probabilitas Galat Kuantisasi (PDF)")
ax_ext[1].set_xlabel("Nilai Error Kuantisasi")
ax_ext[1].set_ylabel("Densitas Probabilitas")
ax_ext[1].grid(alpha=0.25)

# Subplot 3: Kurva Kumulatif Energi Frekuensi & Cutoff 4 kHz
cum_energy = np.cumsum(X_fft ** 2)
cum_energy_norm = (cum_energy / cum_energy[-1]) * 100.0
ax_ext[2].plot(freqs, cum_energy_norm, color="forestgreen", linewidth=1.2, label="Kumulatif Energi Spektrum")
ax_ext[2].axvline(4000, color="red", linestyle="--", linewidth=1.0, label="Batas Nyquist 8 kHz (4 kHz Cutoff)")
ax_ext[2].set_title("Extension 3: Kurva Distribusi Kumulatif Energi Spektral")
ax_ext[2].set_xlabel("Frekuensi (Hz)")
ax_ext[2].set_ylabel("Kumulatif Energi (%)")
ax_ext[2].set_xlim(0, fs / 2)
ax_ext[2].legend(loc="lower right")
ax_ext[2].grid(alpha=0.25)

fig_ext.tight_layout()
extension_figure_path = FIG / "mp02_extension_error_analysis.png"
fig_ext.savefig(extension_figure_path, dpi=160)
plt.close(fig_ext)

# 7. OUTPUT TERMINAL FITUR TAMBAHAN
print("\n--- HASIL FITUR TAMBAHAN ---")
print("RMSE Kuantisasi 8-bit        :", round(rmse_q8, 6))
print("MAE Kuantisasi 8-bit         :", round(mae_q8, 6))
print("Max Error Kuantisasi 8-bit   :", round(max_err_q8, 6))
print("PSNR Kuantisasi 8-bit        :", round(psnr_q8, 2), "dB")
print("PSNR Resampling 8 kHz        :", round(psnr_8k, 2), "dB")
print("Kehilangan Energi HF (>4 kHz):", round(hf_loss_pct, 2), "%")
print("Bitrate Baseline (16k/16)    :", bitrates_kbps["16k16_baseline"], "kbps")
print("Bitrate Minimum (8k/8)       :", bitrates_kbps["8k8"], "kbps")
print("Extension Metrics            :", extension_metrics_path)
print("Extension Figure             :", extension_figure_path)
print("=" * 55)
