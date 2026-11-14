# Mini Project 08 — Ekstraksi Fitur Audio (log-Mel, MFCC, spektral, temporal, pitch)

**Anggota Kelompok**
| Nama | NIM |
|---|---|
| Christama Ezra Yudianto | 245150307111009 |
| Abdillah Fatah Suryanto | 245150300111008 |
| Ivang Arfandia | 245150307111003 |
| Jahzeel Erason Nicodemus | 245150300111002 |

## Tujuan
Dari frame MP7 diekstraksi fitur per-frame (**log-Mel 26 band, MFCC 13, RMS, ZCR, centroid, bandwidth, roll-off 85%,
pitch autokorelasi**) lalu diagregasi (mean/std, F0 hanya frame voiced, + voiced ratio) menjadi **vektor sampel 39-D**.
Kontrak 39-D ini dibekukan dan dipakai MP9–MP16. Status → `ready_for_mp09`.

## Cara menjalankan
```bash
python run_all_mini_projects.py --only 8
# atau
cd "Mini Project 08" && python mini_project_08.py && python features/mp08_extension_feature_analysis.py
```

## File penting
| File | Isi |
|---|---|
| `mini_project_08.py` | Kode wajib |
| `features/mp08_extension_feature_analysis.py` | **FITUR TAMBAHAN** |
| `config/feature_config.json` | Skema fitur (+ `feature_names` 39 nama, ditambahkan) |
| `features/intermediate/mp08_frame_features.npz` | Fitur per-frame |
| `features/intermediate/mp08_sample_feature_vector.npz` | Vektor 39-D referensi |
| `results/figures/mp08_logmel_mfcc.png`, `mp08_scalar_features.png` | Gambar wajib |

## Hasil (catat hasil)
| Besaran | Nilai |
|---|---|
| Jumlah frame | 60 |
| Bentuk log-Mel / MFCC | (60, 26) / (60, 13) |
| Dimensi vektor sampel | 39 |
| Voiced ratio | 0,633 |
| Mean / std F0 (voiced) | 178,7 Hz / 40,4 Hz |
| NaN / Inf | 0 |

## FITUR TAMBAHAN
1. **Delta & delta-delta MFCC, spectral flux, spectral flatness, rentang energi** (vektor 96-D opsional; kontrak 39-D tidak diubah).
2. **Validasi pitch dengan YIN** (ambang 0,30, fallback 0,50): median F0 YIN 169,6 Hz vs autokorelasi 168,4 Hz, selisih rata-rata 0,27 Hz, gross error 0%.
3. **Biaya komputasi per fitur** (µs host & MAC/frame): FFT 7.602 MAC, mel 332, DCT 338, **pitch autokorelasi 62.400 MAC** (stage termahal → dipakai MP10 untuk analisis hemat daya).
4. **Presisi float32** MFCC: SQNR 142 dB → aman untuk MCU dengan FPU single precision.
5. **Sparsity filterbank** 93,6% (332 bobot non-nol dari 5.226) + header tabel konstanta `firmware/mp08_feature_tables.h`.
6. Output: `results/metrics/mp08_extension_feature_metrics.json`, `features/intermediate/mp08_extended_frame_features.npz`,
   `results/figures/mp08_extension_feature_analysis.png`.

## Adaptasi
- `feature_names` (39 nama berurutan) ditambahkan ke `feature_config.json` agar MP10 dapat melaporkan nama fitur terpilih.
- ROOT portabel (tanpa path absolut).
