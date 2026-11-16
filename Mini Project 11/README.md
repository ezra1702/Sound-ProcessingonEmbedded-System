# Mini Project 11 — Classifier Klasik (training CV → seleksi family di validation)

**Anggota Kelompok**
| Nama | NIM |
|---|---|
| Christama Ezra Yudianto | 245150307111009 |
| Abdillah Fatah Suryanto | 245150300111008 |
| Ivang Arfandia | 245150307111003 |
| Jahzeel Erason Nicodemus | 245150300111002 |

## Tujuan
Melatih 6 family pada 8 fitur MP10 (RFE-8): kNN, logistic regression, linear SVM, decision tree, random forest, dan MLP.
- Hyperparameter dipilih dengan GroupKFold **di train** (64 sampel, block 1–4).
- Family dipilih di **validation** (16 sampel, block 5): toleransi 0,03 macro-F1, dan bila seri dipilih biaya operasi terkecil.
- Test tidak disentuh.

Status → `ready_for_mp12`.

## Cara menjalankan
```bash
python run_all_mini_projects.py --only 11
```

## Hasil (catat hasil)
| Family | Hyperparameter | CV-F1 train | Val macro-F1 | Ops/sampel |
|---|---|---|---|---|
| kNN | k = 3, uniform | 0,984 | 0,937 | 1.536 |
| **Logistic regression — terpilih** | **C = 1** | **1,000** | **0,937** | **64** |
| Linear SVM | C = 10 | 0,984 | 0,803 | 64 |
| Decision tree | max_depth = 4 | 0,858 | 0,799 | 4 |
| Random forest | 50 pohon, depth 4 | 0,938 | 0,803 | 199 |
| MLP | 16 hidden, α = 0,001 | 1,000 | 0,873 | 384 |

Hanya kNN dan logreg yang masuk toleransi (shortlist). Dipilih **logistic regression** karena operasinya 24× lebih sedikit
(64 ops = 32 perkalian + 32 penjumlahan untuk 8 fitur × 4 kelas, dengan 36 parameter).

Confusion matrix validation (urutan mati, naik, nyala, turun): `[[3,1,0,0],[0,4,0,0],[0,0,4,0],[0,0,0,4]]`.
Satu-satunya kesalahan: `ivang_mati_b05` diprediksi *naik* (akurasi 15/16).

> Catatan: dibandingkan versi awal (data TTS), decision tree tidak lagi unggul. Pada data 4 penutur asli, tree overfit
> (train-F1 1,00 vs CV 0,858, gap 0,14), sedangkan logreg tidak memiliki gap (CV 1,00).
> Buku mengasumsikan linear SVM di MP15/MP16; logreg juga linear (`coef_`, `intercept_`), jadi jalur INT8 linear dipakai (lihat MP15).

## FITUR TAMBAHAN
1. **Family tambahan**: SVM-RBF (CV 0,970, val **1,00**, tetapi butuh 56 support vector / 1.344 ops), Gaussian NB (0,952 / 0,873), LDA (0,953 / 0,873).
   Hanya informatif; pilihan resmi tetap sesuai kontrak MP11.
2. **Overfit gap & robustness terhadap derau fitur** (σ = 0,25/0,5/1,0 pada z-score, CV-F1):
   - logreg 0,969 / 0,885 / 0,697, paling tahan bersama SVM-RBF.
   - tree paling rapuh (0,827 / 0,753 / 0,554) dan gap overfit terbesar (0,142).
3. **Arsip beku semua family** `models/mp11_family_models/*.joblib` + SHA-256.
   Dipakai untuk evaluasi eksploratif MP13 dan jalur SVM buku di MP15; semua mereproduksi prediksi validation MP11.
4. **Export C model float** `firmware/mp11_classifier_float.h` (linear → dot product), dikompilasi gcc.
   Agreement C vs sklearn **1,00** pada 80 vektor train+val (test tidak dibuka).
5. **Latency host per sampel** (ms, bervariasi antar-run): logreg 0,052 · linear SVM 0,050 · tree 0,063 · MLP 0,134 · kNN 0,739 · RF 2,907.
6. Output: `results/metrics/mp11_extension_classifier_metrics.json`, `results/figures/mp11_extension_classifier_analysis.png`.
