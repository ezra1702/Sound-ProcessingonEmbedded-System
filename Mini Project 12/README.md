# Mini Project 12 — Tiny CNN pada log-Mel (training/validation only)

**Anggota Kelompok**
| Nama | NIM |
|---|---|
| Christama Ezra Yudianto | 245150307111009 |
| Abdillah Fatah Suryanto | 245150300111008 |
| Ivang Arfandia | 245150307111003 |
| Jahzeel Erason Nicodemus | 245150300111002 |

## Tujuan
Melatih **TinyKWSNet** (2 conv + BN + ReLU, MaxPool, Global Average Pool, Linear) pada log-Mel 96×26 (crop/pad),
normalisasi & augmentasi train-only, checkpoint dipilih dari validation, early stopping. Test tidak dimuat.
Status → `ready_for_mp13`.

## Cara menjalankan
```bash
python run_all_mini_projects.py --only 12      # deterministik (seed 2026)
```

## Hasil (catat hasil)
| Besaran | Nilai |
|---|---|
| Data | train 64 / validation 16 (test 16 tidak dimuat) |
| Parameter | 1.340 (5.360 B float32 / ≈1.340 B INT8) |
| MAC per inferensi | 898.624 |
| Epoch terbaik / dijalankan | 25 / 37 (early stopping) |
| **Validation macro-F1 / accuracy** | **0,514 / 0,50** |

**Interpretasi jujur: model resmi underfitting.**
- Dengan 64 sampel dan batch 8 hanya ada 8 langkah optimizer per epoch. Early stopping berhenti di 296 langkah total.
- Train loss akhir 1,28 masih dekat ln 4 = 1,386, dan train-F1 hanya 0,387.
- Model resmi tetap diteruskan sebagai kandidat (sesuai protokol buku). Pemilihan deployment ditentukan di MP15.

## FITUR TAMBAHAN
1. **Diagnosis underfitting** (langkah optimizer, train loss vs chance, train-F1), ditulis otomatis ke metrics.
2. **Varian eksplorasi** (dipilih via validation saja, test tidak dimuat):

   | Varian | Val F1 | Train F1 | Param | MAC |
   |---|---|---|---|---|
   | tinykws_long (300 epoch, OneCycle) | 0,693 | 0,841 | 1.340 | 898.624 |
   | DS-CNN (depthwise separable) | **0,937** | 1,000 | 1.172 | 589.120 |
   | tinykws_freqpool (pool hanya sumbu waktu) | 0,879 | 1,000 | 2.108 | 899.392 |

   Varian dibekukan di `models/mp12_ext_*.pt` beserta registry SHA-256 `config/mp12_extension_candidates.json`, lalu dievaluasi secara eksploratif di MP13.
   Hasilnya: dengan langkah training yang cukup, CNN mampu belajar. DS-CNN menyamai classifier klasik di validation dengan MAC 34% lebih sedikit dari model resmi.
3. **Stabilitas seed** model resmi: val F1 0,421 / 0,437 / 0,384. Konsisten rendah, jadi penyebabnya underfitting, bukan keberuntungan seed.
4. **Export ONNX** model resmi & varian terbaik: selisih logit ≈ 0, agreement 1,00 (onnxruntime).
5. **Profil memori aktivasi ping-pong**: puncak 97,5 KiB (resmi) / 78,0 KiB (DS-CNN).
6. **Saliency map** (gradien input) pada varian DS-CNN.
7. Output: `results/metrics/mp12_extension_cnn_metrics.json`, `results/figures/mp12_extension_cnn_analysis.png`.
