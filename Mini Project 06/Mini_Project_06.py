from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import (lfilter, firwin, remez, iirdesign, iirnotch, freqz, welch,
                          group_delay, tf2sos, sosfilt)
# Anggota Kelompok :
# 1. Christama Ezra Yudianto    (245150307111009)
# 2. Abdillah Fatah Suryanto    (245150300111008)
# 3. Ivang Arfandia             (245150307111003)
# 4. Jahzeel Erason Nicodemus   (245150300111002)

ROOT = Path(r"C:\Users\Erick W\OneDrive\Documents\Python\Mini Project 06")
STATUS_PATH = ROOT / "project_status.json"
integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

with open(STATUS_PATH, "r", encoding="utf-8") as f:
    status = json.load(f)
assert status.get("status") in ("ready_for_mp06", "ready_for_mp07"), "Jalankan Mini Project 5 terlebih dahulu."
input_wav = ROOT / status["filter_input_hum"]
fs, x_pcm = wavfile.read(input_wav)
if x_pcm.ndim > 1:
    x_pcm = x_pcm[:, 0]
if np.issubdtype(x_pcm.dtype, np.integer):
    x = x_pcm.astype(np.float64) / np.iinfo(x_pcm.dtype).max
else:
    x = x_pcm.astype(np.float64)
    
spec = {
    "sample_rate_hz": int(fs),
    "target_interference_hz": 50.0,
    "hum_attenuation_min_db": 25.0,
    "speech_band_hz": [150.0, 4000.0],
    "speech_spectral_distortion_max_db_rms": 1.5,
    "additional_delay_max_ms": 10.0,
    "multiplies_per_sample_target": 30,
    "selection_rule": "meet quality first, then minimize distortion + compute cost + delay",
}
config_dir = ROOT / "config"
feat_dir = ROOT / "features/intermediate"
out_audio = ROOT / "data/processed/mp06_design"
out_met = ROOT / "results/metrics"
out_fig = ROOT / "results/figures"
firmware = ROOT / "firmware"
for d in [config_dir, feat_dir, out_audio, out_met, out_fig, firmware]:
    d.mkdir(parents=True, exist_ok=True)
with open(config_dir / "filter_design_spec.json", "w", encoding="utf-8") as f:
    json.dump(spec, f, indent=2)
    
candidates = {}
# A. FIR window: baseline terkontrol
b = firwin(161, 115.0, pass_zero=False, window="hamming", fs=fs)
candidates["fir_hamming_hp"] = {"b": b, "a": np.array([1.0]), "type": "FIR", "delay_ms": (len(b)-1)/2/fs*1000}
# B. FIR Parks-McClellan: stopband 0-80 Hz, passband >=150 Hz
b = remez(161, [0, 80, 150, fs/2], [0, 1], weight=[10, 1], fs=fs, maxiter=1000)
candidates["fir_remez_hp"] = {"b": b, "a": np.array([1.0]), "type": "FIR", "delay_ms": (len(b)-1)/2/fs*1000}
# C. IIR Elliptic high-pass sesuai spesifikasi formal
b, a = iirdesign(wp=150, ws=80, gpass=1, gstop=40, fs=fs, ftype="ellip", output="ba")
candidates["iir_ellip_hp"] = {"b": b, "a": a, "type": "IIR", "delay_ms": 0.0}
# D-E-F. Notch terlokalisasi dengan tiga Q untuk kebutuhan hum 50 Hz
for q in [20.0, 30.0, 40.0]:
    b, a = iirnotch(50.0, Q=q, fs=fs)
    candidates[f"iir_notch_q{int(q)}"] = {"b": b, "a": a, "type": "IIR", "delay_ms": 0.0}
    
f_psd, P_in = welch(x, fs=fs, nperseg=min(4096, len(x)))

def band_power(P, f, lo, hi):
    m = (f >= lo) & (f <= hi)
    return float(integrate(P[m], f[m])) if np.any(m) else 0.0

def speech_distortion_db(y):
    f, P_y = welch(y, fs=fs, nperseg=min(4096, len(y)))
    m = (f >= spec["speech_band_hz"][0]) & (f <= spec["speech_band_hz"][1])
    in_db = 10*np.log10(P_in[m] + 1e-18)
    y_db = 10*np.log10(P_y[m] + 1e-18)
    return float(np.sqrt(np.mean((y_db - in_db)**2)))

p_hum_in = band_power(P_in, f_psd, 45, 55)
metrics = {}
outputs = {}
for name, c in candidates.items():
    y = lfilter(c["b"], c["a"], x)
    outputs[name] = y
    f_y, P_y = welch(y, fs=fs, nperseg=min(4096, len(y)))
    p_hum_y = band_power(P_y, f_y, 45, 55)
    attenuation_db = 10*np.log10((p_hum_in + 1e-18)/(p_hum_y + 1e-18))
    dist_db = speech_distortion_db(y)
    # Direct-form estimate: numerator + feedback multiplies per sample
    mult = len(c["b"]) + max(0, len(c["a"])-1)
    if c["type"] == "FIR":
        delay_ms = float(c["delay_ms"])
    else:
        gd_f = np.linspace(spec["speech_band_hz"][0], spec["speech_band_hz"][1], 256)
        _, gd_samples = group_delay((c["b"], c["a"]), w=gd_f, fs=fs)
        gd_samples = gd_samples[np.isfinite(gd_samples)]
        delay_ms = float(np.percentile(np.abs(gd_samples), 95) / fs * 1000) if len(gd_samples) else 0.0
    meets = (
        attenuation_db >= spec["hum_attenuation_min_db"] and
        dist_db <= spec["speech_spectral_distortion_max_db_rms"] and
        delay_ms <= spec["additional_delay_max_ms"]
    )
    # quality dominates; compute cost is secondary
    score = dist_db + 0.01*mult + 0.05*delay_ms - 0.005*attenuation_db
    metrics[name] = {
        "hum_attenuation_db": float(attenuation_db),
        "speech_spectral_distortion_db_rms": float(dist_db),
        "estimated_multiplies_per_sample": int(mult),
        "additional_delay_ms": delay_ms,
        "meets_quality_requirements": bool(meets),
        "selection_score": float(score),
        "b_len": int(len(c["b"])),
        "a_len": int(len(c["a"])),
    }

eligible = [n for n in candidates if metrics[n]["meets_quality_requirements"]]
if eligible:
    selected = min(eligible, key=lambda n: metrics[n]["selection_score"])
else:
    # Fallback pedagogis: pilih skor terbaik dan tandai bahwa requirement belum terpenuhi
    selected = min(candidates, key=lambda n: metrics[n]["selection_score"])
final = candidates[selected]
y_final = outputs[selected]
peak = np.max(np.abs(y_final))
if peak > 0.999:
    y_final = y_final / peak * 0.98
    
wav_out = out_audio / "nyala_001_filter_final.wav"
wavfile.write(wav_out, fs, np.int16(np.clip(y_final, -1, 1) * 32767))

npz_all = {}
for name, c in candidates.items():
    npz_all[f"{name}_b"] = np.asarray(c["b"], dtype=np.float64)
    npz_all[f"{name}_a"] = np.asarray(c["a"], dtype=np.float64)
np.savez(feat_dir / "mp06_filter_candidates.npz", **npz_all)
np.savez(feat_dir / "mp06_filter_final.npz",
         b=np.asarray(final["b"], dtype=np.float64),
         a=np.asarray(final["a"], dtype=np.float64),
         sample_rate_hz=np.array([fs]),
         selected_name=np.array([selected]))

N = min(len(x), int(0.25*fs))
test_x = x[:N]
test_y = lfilter(final["b"], final["a"], test_x)
np.savez(feat_dir / "mp06_filter_test_vector.npz",
         input=test_x.astype(np.float32),
         output=test_y.astype(np.float32),
         sample_rate_hz=np.array([fs]))

b = np.asarray(final["b"], dtype=np.float32)
a = np.asarray(final["a"], dtype=np.float32)

def c_array(name, arr):
    vals = ", ".join(f"{v:.9g}f" for v in arr)
    return f"static const float {name}[{len(arr)}] = {{ {vals} }};"

header = "\n".join([
    "#pragma once",
    "/* Auto-generated by Mini Project 6 */",
    f"#define AUDIO_FS_HZ {fs}",
    f"#define FILTER_B_LEN {len(b)}",
    f"#define FILTER_A_LEN {len(a)}",
    c_array("FILTER_B", b),
    c_array("FILTER_A", a),
    "/* Keep state between audio blocks; do not reset per frame. */",
])
with open(firmware / "mp06_filter_coefficients.h", "w", encoding="utf-8") as f:
    f.write(header + "\n")
    
final_config = {
    "selected_filter": selected,
    "sample_rate_hz": int(fs),
    "b_length": int(len(final["b"])),
    "a_length": int(len(final["a"])),
    "input_audio": str(input_wav.relative_to(ROOT)),
    "output_audio": str(wav_out.relative_to(ROOT)),
    "coefficients": "features/intermediate/mp06_filter_final.npz",
    "firmware_header": "firmware/mp06_filter_coefficients.h",
    "test_vector": "features/intermediate/mp06_filter_test_vector.npz",
}
with open(config_dir / "filter_final_config.json", "w", encoding="utf-8") as f:
    json.dump(final_config, f, indent=2)
report = {"spec": spec, "selected_filter": selected, "candidates": metrics}
with open(out_met / "mp06_filter_design_metrics.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
    
fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
for name, c in candidates.items():
    f_r, H = freqz(c["b"], c["a"], worN=8192, fs=fs)
    ax.plot(f_r, 20*np.log10(np.abs(H)+1e-12), label=name, linewidth=1.0)
ax.axvline(50, linestyle="--", linewidth=0.8)
ax.set_xlim(0, 700); ax.set_ylim(-90, 5)
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Magnitude (dB)")
ax.set_title("Mini Project 6 - candidate filter responses")
ax.grid(alpha=0.2); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(out_fig / "mp06_filter_candidates.png", dpi=180); plt.close()

f_y, P_y = welch(y_final, fs=fs, nperseg=min(4096, len(y_final)))
fig, ax = plt.subplots(2, 1, figsize=(9, 7))
show = min(len(x), int(0.2*fs)); t = np.arange(show)/fs
ax[0].plot(t, x[:show], label="input + hum", linewidth=0.8)
ax[0].plot(t, y_final[:show], label=f"final: {selected}", linewidth=0.8)
ax[0].set_xlabel("Time (s)"); ax[0].set_title("Waveform"); ax[0].legend(); ax[0].grid(alpha=0.2)
ax[1].plot(f_psd, 10*np.log10(P_in+1e-18), label="input")
ax[1].plot(f_y, 10*np.log10(P_y+1e-18), label="final")
ax[1].set_xlim(0, 1000); ax[1].set_xlabel("Frequency (Hz)"); ax[1].set_ylabel("PSD (dB/Hz)")
ax[1].set_title("PSD before / after final filter"); ax[1].legend(); ax[1].grid(alpha=0.2)
plt.tight_layout(); plt.savefig(out_fig / "mp06_filter_final_effect.png", dpi=180); plt.close()