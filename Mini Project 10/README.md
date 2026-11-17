# Mini Project 10 — Reduksi & Seleksi Fitur (training-only)

**Anggota Kelompok**
| Nama | NIM |
|---|---|
| Christama Ezra Yudianto | 245150307111009 |
| Abdillah Fatah Suryanto | 245150300111008 |
| Ivang Arfandia | 245150307111003 |
| Jahzeel Erason Nicodemus | 245150300111002 |

## Tujuan
Memilih representasi fitur paling ringkas dari vektor 39-D **hanya dengan data train** (64 sampel, 4 penutur × block 1–4).
StandardScaler di-fit pada train, dan CV memakai GroupKFold per block (4 fold).

Analisis yang dipakai: variansi, korelasi, mutual information, RFE, PCA, dan LDA. Kandidat dibandingkan dengan CV macro-F1.
Policy pemilihan (hardware-aware): ambil kandidat **seleksi fitur** (bukan proyeksi) terkecil yang masih dalam toleransi 0,03 dari skor terbaik.
Validation dan test tidak dipakai. Status → `ready_for_mp11`.

## Cara menjalankan
```bash
python run_all_mini_projects.py --only 10
```

## Hasil (catat hasil)
| Kandidat | Dim | CV macro-F1 (mean ± std) |
|---|---|---|
| raw39 | 39 | 0,921 ± 0,052 |
| mi8 | 8 | 0,905 ± 0,071 |
| mi12 | 12 | 0,904 ± 0,055 |
| **rfe8 (terpilih)** | **8** | **0,931 ± 0,069** |
| rfe12 | 12 | 0,952 ± 0,028 |
| pca6 | 6 | 0,868 ± 0,006 |
| lda3 | 3 | 0,952 ± 0,028 |

Skor terbaik adalah 0,952 (rfe12 dan lda3). Kandidat seleksi yang masih dalam toleransi (≥ 0,922) adalah rfe8 dan rfe12.
Dari keduanya dipilih yang terkecil, yaitu **rfe8**.

lda3 tidak ikut karena termasuk proyeksi: di MCU proyeksi tetap membutuhkan seluruh 39 fitur ditambah perkalian matriks,
sedangkan seleksi cukup menghitung 8 fitur.

**8 fitur terpilih:** `mfcc01_mean, mfcc03_mean, mfcc04_mean, mfcc04_std, mfcc06_std, mfcc07_mean, mfcc11_std, rms_mean`.

Tidak ada fitur near-zero-variance dan tidak ada pasangan redundan |r| ≥ ambang. MI tertinggi: `mfcc04_std` (0,68), `mfcc01_mean` (0,61), `centroid_hz_mean` (0,54).

Artefak: `config/feature_reduction_config.json` (indeks, nama, mean/scale scaler) dan `features/dataset/mp10_features_final.npz` (96 × 8).

## FITUR TAMBAHAN
1. **Metode seleksi embedded/filter tambahan** (CV train, k = 8):

   | Metode | CV-F1 | Stabilitas antar-fold (Jaccard) |
   |---|---|---|
   | MI | 0,905 | 0,605 |
   | RFE | 0,931 | 0,414 |
   | L1-logistic | 0,915 | 0,696 (paling stabil) |
   | RF importance | 0,873 | 0,635 |

   Fitur yang selalu terpilih di semua fold: `mfcc01_mean` dan `mfcc04_std` (konsisten di keempat metode).
2. **Kurva k** (jumlah fitur MI vs CV-F1): k = 8 → 0,905, k = 16 → 0,953, puncak di k = 20–26 (0,984).
   k minimum dalam toleransi 0,03 dari puncak adalah **20**, sehingga dengan data 4 penutur informasi tersebar di lebih banyak fitur.
   Pilihan resmi tetap rfe8 (kontrak buku).
3. **Permutation importance out-of-fold**: `mfcc04_std` (penurunan F1 0,237) > `mfcc01_mean` (0,224) > `mfcc04_mean` (0,137).
4. **Analisis biaya front-end (hardware-aware)**: fitur terpilih hanya butuh MFCC + RMS.
   - Biaya front-end turun dari **72.276 → 8.672 MAC/frame (hemat 88,0%)**.
   - Stage pitch autokorelasi dapat dilewati di MCU (dipakai lagi di proyeksi MCU MP14).
5. Output: `results/metrics/mp10_extension_selection_metrics.json`, `results/figures/mp10_extension_selection_analysis.png`.
