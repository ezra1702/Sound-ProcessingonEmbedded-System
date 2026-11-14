from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile
# Anggota Kelompok : 
# 1. Christama Ezra Yudianto    (245150307111009)
# 2. Abdillah Fatah Suryanto    (245150300111008)
# 3. Ivang Arfandia             (245150307111003)
# 4. Jahzeel Erason Nicodemus   (245150300111002)


# ------------------------------------------------------------
# MINI PROJECT 1 - INISIALISASI PROYEK KEYWORD SPOTTING
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DIRS = [
 "config", "data/raw", "data/processed", "features",
 "models", "results/figures", "results/metrics", "firmware"
]
for folder in DIRS:
 (ROOT / folder).mkdir(parents=True, exist_ok=True)
CONFIG = {
 "project_name": "embedded_keyword_spotting",
 "labels": ["nyala", "mati", "naik", "turun"],
 "sample_rate_hz": 16000,
 "channels": 1,
 "bit_depth": 16,
 "record_duration_s": 2.0,
 "reference_file": "data/raw/nyala_001.wav"
}
config_path = ROOT / "config/project_config.json"
with open(config_path, "w", encoding="utf-8") as f:
 json.dump(CONFIG, f, indent=2)
fs = CONFIG["sample_rate_hz"]
duration = CONFIG["record_duration_s"]
output_wav = ROOT / CONFIG["reference_file"]
print("Mini Project 1 - Keyword Spotting Embedded")
print("Katakan kata: NYALA")
input("Tekan ENTER ketika siap merekam...")

# sounddevice menghasilkan float32 pada rentang sekitar -1 sampai +1
recording = sd.rec(
 int(duration * fs), samplerate=fs, channels=1, dtype="float32"
)
sd.wait()
x = recording[:, 0]

# Pemeriksaan kualitas dasar sebelum konversi ke PCM-16
peak = float(np.max(np.abs(x)))
rms = float(np.sqrt(np.mean(x ** 2)))
clipping_ratio = float(np.mean(np.abs(x) >= 0.99))

# Konversi float32 [-1,1] ke PCM signed 16-bit
x_pcm16 = np.int16(np.clip(x, -1.0, 1.0) * 32767)
wavfile.write(output_wav, fs, x_pcm16)
metadata = {
 "file": str(output_wav.relative_to(ROOT)),
 "sample_rate_hz": fs,
 "channels": 1,
 "bit_depth": 16,
 "duration_s": len(x) / fs,
 "num_samples": int(len(x)),
 "peak_normalized": peak,
 "rms_normalized": rms,
 "clipping_ratio": clipping_ratio,
 "quality_check": "PASS" if peak > 0.02 and clipping_ratio == 0.0 else "CHECK"
}
metadata_path = ROOT / "results/metrics/mp01_audio_metadata.json"
with open(metadata_path, "w", encoding="utf-8") as f:
 json.dump(metadata, f, indent=2)
 
# Waveform untuk inspeksi visual
axis_t = np.arange(len(x)) / fs
plt.figure(figsize=(9, 3))
plt.plot(axis_t, x, linewidth=0.8)
plt.xlabel("Waktu (s)")
plt.ylabel("Amplitudo ternormalisasi")
plt.title("Mini Project 1 - Waveform Kata 'nyala'")
plt.grid(alpha=0.25)
plt.tight_layout()
figure_path = ROOT / "results/figures/mp01_waveform.png"
plt.savefig(figure_path, dpi=160)
plt.close()
status = {
 "last_completed_mini_project": 1,
 "next_mini_project": 2,
 "next_input": CONFIG["reference_file"],
 "config": "config/project_config.json",
 "status": "ready_for_mp02"
}
with open(ROOT / "project_status.json", "w", encoding="utf-8") as f:
 json.dump(status, f, indent=2)
print("\nSelesai.")
print("Audio :", output_wav)
print("Metadata :", metadata_path)
print("Waveform :", figure_path)
print("Peak :", round(peak, 4))
print("RMS :", round(rms, 4))
print("Clipping :", clipping_ratio)
print("Check :", metadata["quality_check"])

# ============================================================
# FITUR TAMBAHAN YANG KAMI GUNAKAN
# ============================================================
# Fitur tambahan ini ditambahkan setelah proses original, Mini Project 1 selesai.

# Fitur yang digunakan:
# 1. RMS dalam dBFS : Mengukur tingkat energi rata-rata sinyal audio
# 2. Peak Amplitude dalam dBFS : Mengukur amplitudo maksimum sinyal audio
# 3. Zero Crossing Rate (ZCR) : Mengukur frekuensi perubahan tanda positif dan negatif pada sinyal
# 4. Fast Fourier Transform (FFT) : Mengubah sinyal dari domain waktu ke domain frekuensi
# 5. Spectral Centroid : Mengukur pusat distribusi energi frekuensi pada sinyal
# 6. Dominant Frequency : Menentukan frekuensi dengan magnitude terbesar pada spektrum
# 7. Signal Energy : Mengukur total energi numerik yang terdapat pada sinyal audio
# 8. Frequency Spectrum : Menampilkan distribusi magnitude sinyal terhadap frekuensi

# Kode original Mini Project 1 tetap dipertahankan.
# Fitur tambahan hanya menggunakan data audio yang sudah
# direkam oleh proses original.
# ============================================================


print("FITUR TAMBAHAN - AUDIO SIGNAL ANALYSIS")
# FITUR 1 - RMS DALAM dBFS
if rms > 0:
    rms_dbfs = float(
        20 * np.log10(rms)
    )
else:
    rms_dbfs = -np.inf

# FITUR 2 - PEAK AMPLITUDE DALAM dBFS
if peak > 0:
    peak_dbfs = float(
        20 * np.log10(peak)
    )
else:
    peak_dbfs = -np.inf

# FITUR 3 - ZERO CROSSING RATE
zero_crossings = np.sum(np.abs(np.diff(np.signbit(x))))
zcr = float(zero_crossings /max(len(x) - 1, 1))

# FITUR 4 - FAST FOURIER TRANSFORM (FFT)
N = len(x)
window = np.hanning(N)
x_windowed = x * window
fft_result = np.fft.rfft(x_windowed)

frequencies = np.fft.rfftfreq(N,d=1 / fs)
magnitude = np.abs(fft_result)

# FITUR 5 - DOMINANT FREQUENCY
if len(magnitude) > 1:
    dominant_index = (np.argmax(magnitude[1:]) + 1)
else:
    dominant_index = 0
dominant_frequency_hz = float(frequencies[dominant_index])

# FITUR 6 - SPECTRAL CENTROID
magnitude_sum = np.sum(magnitude)
if magnitude_sum > 0:
    spectral_centroid_hz = float(np.sum(frequencies * magnitude) / magnitude_sum)
else:
    spectral_centroid_hz = 0.0

# FITUR 7 - SIGNAL ENERGY
signal_energy = float(np.sum(x ** 2))

# SIMPAN METRICS FITUR TAMBAHAN
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
    "spectral_centroid_hz":spectral_centroid_hz,
    "dominant_frequency_hz":dominant_frequency_hz,
    "signal_energy":signal_energy,
    "clipping_ratio": clipping_ratio
}


extension_metrics_path = (ROOT /"results/metrics/""mp01_extension_audio_metrics.json")


with open(extension_metrics_path,"w",encoding="utf-8") as f:
    json.dump(extension_metrics,f,indent=2)


# VISUALISASI FREQUENCY SPECTRUM
if np.max(magnitude) > 0:
    magnitude_db = (20 * np.log10( magnitude / np.max(magnitude) +1e-12))
else:
    magnitude_db = np.zeros_like(magnitude)
plt.figure(figsize=(9, 4))
plt.plot(frequencies,magnitude_db,linewidth=0.8)
plt.xlabel("Frekuensi (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Mini Project 1 Extension - Frequency Spectrum")
plt.xlim(0, fs / 2)
plt.grid(alpha=0.25)
plt.tight_layout()
extension_figure_path = (ROOT /"results/figures/""mp01_extension_frequency_spectrum.png")
plt.savefig(extension_figure_path,dpi=160)
plt.close()

# OUTPUT FITUR TAMBAHAN
print("\n--- HASIL FITUR TAMBAHAN ---")
print("RMS dBFS :",round(rms_dbfs, 2))
print("Peak dBFS :",round(peak_dbfs, 2))
print("Zero Crossing Rate :",round(zcr, 6))
print("Spectral Centroid :",round(spectral_centroid_hz, 2),"Hz")
print("Dominant Frequency :",round(dominant_frequency_hz,2),"Hz")
print("Signal Energy :",round(signal_energy,4))
print("Extension Metrics :",extension_metrics_path)
print("Extension Figure :",extension_figure_path)