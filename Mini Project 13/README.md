# Mini Project 13 — Membuka Test Set Sekali & Membekukan Laporan Evaluasi

**Anggota Kelompok**
| Nama | NIM |
|---|---|
| Christama Ezra Yudianto | 245150307111009 |
| Abdillah Fatah Suryanto | 245150300111008 |
| Ivang Arfandia | 245150307111003 |
| Jahzeel Erason Nicodemus | 245150300111002 |

## Tujuan
Mengevaluasi kedua kandidat beku — **classical (decision tree MP11)** dan **Tiny CNN (MP12)** — pada test set block 6
**satu kali**, tanpa mengubah model. SHA-256 model/config dihitung sebelum & sesudah test (audit trail). Kedua kandidat
diteruskan; keputusan deployment ditunda ke MP15. Status → `ready_for_mp14`.

## Cara menjalankan
```bash
python run_all_mini_projects.py --only 13
# menjalankan ulang langsung di folder ini -> "reproduction run": hash & metrik diverifikasi identik
cd "Mini Project 13" && python mini_project_13.py
```

## Hasil frozen test (catat hasil)
| Metrik | Classical (tree) | Tiny CNN |
|---|---|---|
| Accuracy | **1,00** | 0,50 |
| Balanced accuracy | 1,00 | 0,50 |
| Macro precision / recall | 1,00 / 1,00 | 0,25 / 0,50 |
| **Macro-F1** | **1,00** | **0,333** |
| Wilson 95% (accuracy) | [0,51; 1,00] | [0,15; 0,85] |

Prediksi per sampel (`results/predictions/mp13_test_predictions.csv`): classical benar 4/4; CNN salah pada `naik_b06`→nyala dan
`turun_b06`→mati. **Hash valid** (model/config tidak berubah selama test), `post_test_tuning_performed = false`.
Interpretasi: dengan n = 4, interval kepercayaan sangat lebar — hasil ini konsistensi, bukan bukti generalisasi.

| Pemeriksaan | Hasil |
|---|---|
| Pre-test gate (status, hash, split) | VALID |
| Test dibuka sekali | ✔ (rerun = reproduction run, hasil identik) |
| Hash sebelum = sesudah | ✔ |
| Post-test tuning | False |
| Status akhir | `ready_for_mp14` |

## FITUR TAMBAHAN (`features/mp13_extension_evaluation_analysis.py`)
Semua model dibaca beku dan **hash diverifikasi**; hasil tidak dipakai untuk memilih/mengubah model.
1. **Bootstrap 95% CI** macro-F1 (2.000 resample): tree [1,00; 1,00], CNN [0,00; 1,00].
2. **Uji McNemar exact** classical vs CNN: tree-benar-saja = 2, CNN-benar-saja = 0, **p = 0,5** → perbedaan tidak dapat diklaim signifikan pada n = 4.
3. **Kalibrasi**: Brier / ECE — tree 0,000 / 0,000; CNN 0,632 / 0,165 (softmax CNN hampir rata ≈0,33 → kurang percaya diri). ROC-AUC OvR CNN 0,833.
4. **Robustness stress test** (derau putih ditambahkan ke WAV test, front-end beku `kws_frontend.py`, 5 ulangan):

   | Model | bersih | SNR 30 | SNR 20 | SNR 10 | SNR 5 | SNR 0 | gain −12 dB / +6 dB | shift 200 ms |
   |---|---|---|---|---|---|---|---|---|
   | **tree [resmi]** | 1,00 | 1,00 | 0,65 | 0,00 | 0,30 | 0,25 | 1,00 / 1,00 | 1,00 |
   | tiny CNN [resmi] | 0,50 | 0,50 | 0,45 | 0,30 | 0,25 | 0,25 | 0,50 / 0,50 | 0,50 |
   | linear SVM* | 1,00 | 1,00 | 0,75 | 0,25 | 0,25 | 0,30 | 1,00 / 1,00 | 1,00 |
   | MLP* | 1,00 | 1,00 | 0,75 | 0,25 | 0,25 | 0,25 | 1,00 / 1,00 | 1,00 |
   | CNN freq-pool* | 1,00 | 1,00 | **1,00** | **0,85** | **0,60** | 0,25 | 1,00 / 1,00 | 1,00 |

   \* = eksplorasi. Temuan: model klasik **runtuh di bawah SNR 20 dB** (fitur MFCC/centroid tergeser derau), sedangkan varian
   CNN freq-pool paling tahan derau → rekomendasi siklus berikutnya: augmentasi derau saat training / data baru.
   Invariant terhadap gain dan time-shift karena normalisasi peak + VAD.
5. **Evaluasi eksploratif semua kandidat beku** (arsip MP11 & varian MP12, hash valid 9/9): klasik semua F1 1,00; CNN long 0,667, DS-CNN 0,667, freq-pool 1,00 — dilaporkan terpisah, **bukan** dasar seleksi.
6. Konsistensi: F1 hasil re-ekstraksi dari WAV = laporan resmi (1,00 & 0,333).
7. Output: `results/metrics/mp13_extension_evaluation_metrics.json`, `results/figures/mp13_extension_evaluation_analysis.png`.

## Adaptasi
- Lokasi skrip di root folder MP (buku: `scripts/`); hash memakai path relatif POSIX agar portabel.
- Mode **reproduction run**: bila MP13 sudah pernah dijalankan pada folder yang sama, rerun memverifikasi hash & metrik identik (test tidak dianggap "dibuka ulang" untuk tuning).
