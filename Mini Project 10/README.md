# Mini Project 10 — Reduksi & Seleksi Fitur (training-only)

**Anggota Kelompok**
| Nama | NIM |
|---|---|
| Christama Ezra Yudianto | 245150307111009 |
| Abdillah Fatah Suryanto | 245150300111008 |
| Ivang Arfandia | 245150307111003 |
| Jahzeel Erason Nicodemus | 245150300111002 |

## Tujuan
Memilih representasi fitur paling ringkas dari vektor 39-D **hanya memakai data train** (StandardScaler fit train,
GroupKFold per block): analisis variansi, korelasi, mutual information, RFE, PCA, LDA → kandidat dibandingkan dengan
CV macro-F1, lalu dipilih yang terbaik (tie → dimensi terkecil). Validation & test tidak dipakai. Status → `ready_for_mp11`.

## Cara menjalankan
```bash
python run_all_mini_projects.py --only 10
```

## Hasil (catat hasil)
| Kandidat | Dim | CV macro-F1 (mean ± std) |
|---|---|---|
| raw39 | 39 | 0,750 ± 0,144 |
| **mi8 (terpilih)** | **8** | **1,000 ± 0,000** |
| mi12 | 12 | 0,917 ± 0,144 |
| rfe8 | 8 | 0,833 ± 0,167 |
| rfe12 | 12 | 0,833 ± 0,167 |
| pca6 | 6 | 0,917 ± 0,144 |
| lda3 | 3 | 0,667 ± 0,236 |

**8 fitur terpilih:** `mfcc02_mean, mfcc02_std, mfcc04_std, mfcc05_mean, zcr_mean, zcr_std, centroid_hz_mean, centroid_hz_std`.
Tidak ada fitur near-zero-variance; 5 pasangan redundan |r| ≥ ambang.
Artefak: `config/feature_reduction_config.json` (indeks, nama, mean/scale scaler), `features/dataset/mp10_features_final.npz`.

## FITUR TAMBAHAN
1. **Metode seleksi embedded/filter tambahan** (CV train): MI 1,00 · RFE 0,833 · L1-logistic (solver saga) 0,677 · Random-Forest importance 0,75;
   **stabilitas antar-fold (Jaccard)** MI 0,689 / RFE 0,418 / L1 0,425 / RF 0,414 → MI paling stabil.
2. **Kurva k** (jumlah fitur MI vs CV-F1): k terbaik = 8 (k minimum dalam toleransi 0,03).
3. **Permutation importance out-of-fold**: fitur paling penting `mfcc04_std`.
4. **Analisis biaya front-end (hardware-aware)**: fitur terpilih tidak butuh pitch → biaya front-end turun dari **72.276 → 9.476 MAC/frame (hemat 86,9%)**; stage pitch autokorelasi dapat dilewati di MCU (dipakai lagi di proyeksi MCU MP14).
5. Output: `results/metrics/mp10_extension_selection_metrics.json`, `results/figures/mp10_extension_selection_analysis.png`.
