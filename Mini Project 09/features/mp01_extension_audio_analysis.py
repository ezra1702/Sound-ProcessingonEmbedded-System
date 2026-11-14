"""
mp01_extension_audio_analysis.py
=================================
Fitur Tambahan Mini Project 1 - Audio Signal Analysis
Disalin dari: mini_project_01.py (bagian FITUR TAMBAHAN, baris 104-224)

File ini adalah modul mandiri (standalone) yang membaca hasil rekaman
yang sudah disimpan oleh mini_project_01.py, lalu menghitung dan
menyimpan metrik analisis sinyal audio lanjutan.

Jalankan SETELAH mini_project_01.py selesai dieksekusi.

Anggota Kelompok :
1. Christama Ezra Yudianto    (245150307111009)
2. Abdillah Fatah Suryanto    (245150300111008)
3. Ivang Arfandia             (245150307111003)
4. Jahzeel Erason Nicodemus   (245150300111002)

Fitur yang dianalisis:
1. RMS dalam dBFS           : Tingkat energi rata-rata sinyal audio
2. Peak Amplitude dalam dBFS : Amplitudo maksimum sinyal audio
3. Zero Crossing Rate (ZCR)  : Frekuensi perubahan tanda positif/negatif
4. Fast Fourier Transform    : Transformasi sinyal ke domain frekuensi
5. Spectral Centroid         : Pusat distribusi energi frekuensi
6. Dominant Frequency        : Frekuensi dengan magnitude terbesar
7. Signal Energy             : Total energi numerik sinyal audio
8. Frequency Spectrum        : Distribusi magnitude vs frekuensi (visualisasi)
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# ------------------------------------------------------------
# SETUP PATH (berdiri sendiri, membaca hasil MP01)
# ------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent.parent   # naik ke Mini Project 01/
WAV_PATH    = ROOT / "data/raw/nyala_001.wav"
CONFIG_PATH = ROOT / "config/project_config.json"
FIG_DIR     = ROOT / "results/figures"
MET_DIR     = ROOT / "results/metrics"

for d in [FIG_DIR, MET_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Baca konfigurasi proyek
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

fs = config["sample_rate_hz"]

# Baca file WAV hasil rekaman Mini Project 1
sample_rate, raw = wavfile.read(WAV_PATH)
if raw.ndim > 1:
    raw = raw[:, 0]

# Konversi PCM-16 -> float32 [-1, 1]
x = raw.astype(np.float32) / 32767.0
x = np.clip(x, -1.0, 1.0)

# Hitung peak & rms (dibutuhkan fitur 1 & 2)
peak           = float(np.max(np.abs(x)))
rms            = float(np.sqrt(np.mean(x ** 2)))
clipping_ratio = float(np.mean(np.abs(x) >= 0.99))

print("FITUR TAMBAHAN - AUDIO SIGNAL ANALYSIS")

# ============================================================
# FITUR 1 - RMS DALAM dBFS
# ============================================================
if rms > 0:
    rms_dbfs = float(20 * np.log10(rms))
else:
    rms_dbfs = -np.inf

# ============================================================
# FITUR 2 - PEAK AMPLITUDE DALAM dBFS
# ============================================================
if peak > 0:
    peak_dbfs = float(20 * np.log10(peak))
else:
    peak_dbfs = -np.inf

# ============================================================
# FITUR 3 - ZERO CROSSING RATE
# ============================================================
zero_crossings = np.sum(np.abs(np.diff(np.signbit(x))))
zcr            = float(zero_crossings / max(len(x) - 1, 1))

# ============================================================
# FITUR 4 - FAST FOURIER TRANSFORM (FFT)
# ============================================================
N          = len(x)
window     = np.hanning(N)
x_windowed = x * window
fft_result = np.fft.rfft(x_windowed)

frequencies = np.fft.rfftfreq(N, d=1 / fs)
magnitude   = np.abs(fft_result)

# ============================================================
# FITUR 5 - DOMINANT FREQUENCY
# ============================================================
if len(magnitude) > 1:
    dominant_index = (np.argmax(magnitude[1:]) + 1)
else:
    dominant_index = 0
dominant_frequency_hz = float(frequencies[dominant_index])

# ============================================================
# FITUR 6 - SPECTRAL CENTROID
# ============================================================
magnitude_sum = np.sum(magnitude)
if magnitude_sum > 0:
    spectral_centroid_hz = float(np.sum(frequencies * magnitude) / magnitude_sum)
else:
    spectral_centroid_hz = 0.0

# ============================================================
# FITUR 7 - SIGNAL ENERGY
# ============================================================
signal_energy = float(np.sum(x ** 2))

# ============================================================
# SIMPAN METRICS FITUR TAMBAHAN
# ============================================================
extension_metrics = {
    "feature_extension": "Audio Signal Analysis",
    "sample_rate_hz": fs,
    "duration_s": float(len(x) / fs),
    "num_samples": int(N),
    "peak_normalized": peak,
    "peak_dbfs": peak_dbfs,
    "rms_normalized": rms,
    "rms_dbfs": rms_dbfs,
    "zero_crossing_rate": zcr,
    "zero_crossings": int(zero_crossings),
    "spectral_centroid_hz": spectral_centroid_hz,
    "dominant_frequency_hz": dominant_frequency_hz,
    "signal_energy": signal_energy,
    "clipping_ratio": clipping_ratio
}

extension_metrics_path = MET_DIR / "mp01_extension_audio_metrics.json"
with open(extension_metrics_path, "w", encoding="utf-8") as f:
    json.dump(extension_metrics, f, indent=2)

# ============================================================
# FITUR 8 - VISUALISASI FREQUENCY SPECTRUM
# ============================================================
if np.max(magnitude) > 0:
    magnitude_db = 20 * np.log10(magnitude / np.max(magnitude) + 1e-12)
else:
    magnitude_db = np.zeros_like(magnitude)

plt.figure(figsize=(9, 4))
plt.plot(frequencies, magnitude_db, linewidth=0.8)
plt.xlabel("Frekuensi (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Mini Project 1 Extension - Frequency Spectrum")
plt.xlim(0, fs / 2)
plt.grid(alpha=0.25)
plt.tight_layout()

extension_figure_path = FIG_DIR / "mp01_extension_frequency_spectrum.png"
plt.savefig(extension_figure_path, dpi=160)
plt.close()

# ============================================================
# OUTPUT TERMINAL
# ============================================================
print("\n--- HASIL FITUR TAMBAHAN ---")
print("RMS dBFS           :", round(rms_dbfs, 2))
print("Peak dBFS          :", round(peak_dbfs, 2))
print("Zero Crossing Rate :", round(zcr, 6))
print("Spectral Centroid  :", round(spectral_centroid_hz, 2), "Hz")
print("Dominant Frequency :", round(dominant_frequency_hz, 2), "Hz")
print("Signal Energy      :", round(signal_energy, 4))
print("Extension Metrics  :", extension_metrics_path)
print("Extension Figure   :", extension_figure_path)
