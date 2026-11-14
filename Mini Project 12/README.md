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
| Parameter | 1.340 (5.360 B float32 / ≈1.340 B INT8) |
| MAC per inferensi | 898.624 |
| Epoch terbaik / dijalankan | 41 / 53 (early stopping) |
| **Validation macro-F1 / accuracy** | **0,375 / 0,50** |

**Interpretasi jujur:** model resmi **underfitting** — dengan 16 sampel dan batch 8 hanya ada 2 langkah optimizer per epoch
(106 langkah total); train loss akhir 1,15 masih dekat ln 4 = 1,386, train-F1 hanya 0,571. Model resmi tetap diteruskan
sebagai kandidat (sesuai protokol buku); pemilihan deployment ditentukan di MP15.

## FITUR TAMBAHAN
1. **Diagnosis underfitting** (langkah optimizer, train loss vs chance, train-F1).
2. **Varian eksplorasi (dipilih via validation saja, test tidak dimuat)**:

   | Varian | Val F1 | Train F1 | Param | MAC |
   |---|---|---|---|---|
   | tinykws_long (300 epoch, OneCycle) | 0,667 | 0,365 | 1.340 | 898.624 |
   | DS-CNN (depthwise separable) | 1,000 | 0,604 | 1.172 | 589.120 |
   | tinykws_freqpool (pool hanya sumbu waktu) | 1,000 | 0,743 | 2.108 | 899.392 |

   Dibekukan di `models/mp12_ext_*.pt` + registry SHA-256 `config/mp12_extension_candidates.json` (dievaluasi eksploratif di MP13).
3. **Stabilitas seed** model resmi: val F1 0,375 / 0,417 / 0,125 → sangat sensitif seed (tanda data terlalu sedikit).
4. **Export ONNX** model resmi & varian terbaik: selisih logit 0,0, agreement 1,00 (onnxruntime).
5. **Profil memori aktivasi ping-pong**: puncak 97,5 KiB (resmi) / 78,0 KiB (DS-CNN).
6. **Saliency map** (gradien input) pada varian freqpool.
7. Output: `results/metrics/mp12_extension_cnn_metrics.json`, `results/figures/mp12_extension_cnn_analysis.png`.
