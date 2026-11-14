# Mini Project 11 — Classifier Klasik (training CV → seleksi family di validation)

**Anggota Kelompok**
| Nama | NIM |
|---|---|
| Christama Ezra Yudianto | 245150307111009 |
| Abdillah Fatah Suryanto | 245150300111008 |
| Ivang Arfandia | 245150307111003 |
| Jahzeel Erason Nicodemus | 245150300111002 |

## Tujuan
Melatih 6 family (kNN, logistic regression, linear SVM, decision tree, random forest, MLP) pada 8 fitur MP10.
Hyperparameter dipilih dengan GroupKFold **di train**, family dipilih di **validation** (toleransi 0,03 macro-F1, tie →
biaya operasi terkecil). Test tidak disentuh. Status → `ready_for_mp12`.

## Cara menjalankan
```bash
python run_all_mini_projects.py --only 11
```

## Hasil (catat hasil)
| Family | CV-F1 train | Val macro-F1 |
|---|---|---|
| kNN | 1,000 | 1,000 |
| Logistic regression | 1,000 | 1,000 |
| Linear SVM | 1,000 | 1,000 |
| **Decision tree (max_depth=2) — terpilih** | **1,000** | **1,000** |
| Random forest | 0,833 | 1,000 |
| MLP | 1,000 | 1,000 |

Semua family masuk toleransi → dipilih **decision tree depth 2** karena operasi per inferensi paling sedikit (2 perbandingan).
Aturan pohon (`models/mp11_tree_rules.txt`, feature_2 = `mfcc04_std`, feature_7 = `centroid_hz_std`):
```
feature_2 <= 0.04 : feature_7 <= -0.39 -> nyala | else -> turun
feature_2 >  0.04 : feature_7 <=  0.07 -> naik  | else -> mati
```
Confusion matrix validation: diagonal sempurna (4/4).

> Konsekuensi: buku mengasumsikan linear SVM di MP15/MP16. Karena family terpilih (dengan policy buku) adalah tree,
> MP15 dan MP16 dibuat **family-aware** (lihat README MP15/MP16).

## FITUR TAMBAHAN
1. **Family tambahan**: SVM-RBF (CV 1,00), Gaussian NB (0,917), LDA (0,490).
2. **Overfit gap & robustness derau fitur** (σ = 0,25/0,5/1,0 pada z-score): tree paling rapuh (0,870/0,754/0,571), MLP paling tahan (0,988/0,933/0,788).
3. **Arsip beku semua family** `models/mp11_family_models/*.joblib` + SHA-256 (dipakai evaluasi eksploratif MP13 dan jalur SVM buku di MP15).
4. **Export C model float** `firmware/mp11_classifier_float.h`, dikompilasi gcc → agreement C vs sklearn **1,00** (20 vektor train+val).
5. **Latency host per sampel** (ms, bervariasi antar-run): tree 0,061 · logreg 0,051 · linear SVM 0,051 · MLP 0,131 · kNN 0,716 · RF 1,824.
6. Output: `results/metrics/mp11_extension_classifier_metrics.json`, `results/figures/mp11_extension_classifier_analysis.png`.
