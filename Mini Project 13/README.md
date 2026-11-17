# Mini Project 13 — Membuka Test Set Sekali & Membekukan Laporan Evaluasi

**Anggota Kelompok**
| Nama | NIM |
|---|---|
| Christama Ezra Yudianto | 245150307111009 |
| Abdillah Fatah Suryanto | 245150300111008 |
| Ivang Arfandia | 245150307111003 |
| Jahzeel Erason Nicodemus | 245150300111002 |

## Tujuan
Mengevaluasi kedua kandidat beku pada test set block 6 (16 sampel = 4 penutur × 4 kata) **satu kali**, tanpa mengubah model.
- Kandidat 1: **classical (logistic regression MP11)**.
- Kandidat 2: **Tiny CNN (MP12)**.

SHA-256 model/config dihitung sebelum & sesudah test (audit trail). Kedua kandidat diteruskan, dan keputusan deployment
ditunda ke MP15. Status → `ready_for_mp14`.

## Cara menjalankan
```bash
python run_all_mini_projects.py --only 13
# menjalankan ulang pada folder yang sama -> "reproduction run": hash & metrik diverifikasi identik
python run_all_mini_projects.py --only 13 --no-copy
```

## Hasil frozen test (catat hasil)
| Metrik | Classical (logreg) | Tiny CNN |
|---|---|---|
| Accuracy | **0,9375 (15/16)** | 0,5625 (9/16) |
| Balanced accuracy | 0,9375 | 0,5625 |
| Macro precision / recall | 0,950 / 0,9375 | 0,567 / 0,5625 |
| **Macro-F1** | **0,937** | **0,560** |
| Wilson 95% (accuracy) | [0,72; 0,99] | [0,33; 0,77] |

Per kelas (classical): mati, nyala F1 1,00; naik 0,889; turun 0,857.
Confusion matrix classical (mati, naik, nyala, turun): `[[4,0,0,0],[0,4,0,0],[0,0,4,0],[0,1,0,3]]`.

Prediksi per sampel ada di `results/predictions/mp13_test_predictions.csv`:
- Classical salah 1×, yaitu `ezra_turun_b06` → *naik* (margin 1,01). Prediksi dengan margin terkecil yang masih benar: `abdillah_mati_b06` (0,16).
- CNN salah 7×, dengan softmax maksimum hanya 0,27–0,40, artinya model hampir menebak.

**Hash valid** (model/config tidak berubah selama test), dan `post_test_tuning_performed = false`.

**Interpretasi:**
- Test berisi penutur yang sama dengan train (sesi/block berbeda). Angka ini mengukur generalisasi ke **sesi baru** penutur yang dikenal, bukan ke penutur baru.
- Dengan n = 16, interval kepercayaan masih lebar.

| Pemeriksaan | Hasil |
|---|---|
| Pre-test gate (status, hash, split) | VALID |
| Test dibuka sekali | ✔ (rerun = reproduction run, hasil identik) |
| Hash sebelum = sesudah | ✔ |
| Post-test tuning | False |
| Status akhir | `ready_for_mp14` |

## FITUR TAMBAHAN (`features/mp13_extension_evaluation_analysis.py`)
Semua model dibaca dalam keadaan beku dan **hash diverifikasi**. Hasilnya tidak dipakai untuk memilih atau mengubah model.
1. **Bootstrap 95% CI** macro-F1 (2.000 resample): logreg [0,73; 1,00], CNN [0,27; 0,77]. Interval hanya overlap tipis di 0,73–0,77.
2. **Uji McNemar exact** classical vs CNN:
   - Kasus yang hanya benar di logreg = 6, yang hanya benar di CNN = 0.
   - **p = 0,031**, jadi logreg lebih baik secara signifikan (α = 0,05) pada test ini.
3. **Kalibrasi** (Brier / NLL / ECE 5-bin):
   - logreg 0,134 / 0,261 / 0,128, dengan ROC-AUC OvR **0,995** dan PR-AP 0,988.
   - CNN 0,703 / 1,299 / 0,252, dengan AUC 0,755. Softmax CNN hampir rata (rata-rata keyakinan 0,31), jadi CNN kurang percaya diri.
4. **Robustness stress test**: derau putih ditambahkan ke WAV test, fitur dihitung ulang dengan front-end beku `kws_frontend.py`, 5 ulangan.

   | Model | bersih | SNR 30 | SNR 20 | SNR 10 | SNR 5 | SNR 0 | gain −12 dB / +6 dB | shift 200 ms |
   |---|---|---|---|---|---|---|---|---|
   | **logreg [resmi]** | 0,94 | 0,96 | 0,79 | 0,69 | 0,59 | 0,23 | 0,94 / 0,94 | 0,94 |
   | tiny CNN [resmi] | 0,56 | 0,51 | 0,38 | 0,33 | 0,35 | 0,30 | 0,56 / 0,56 | 0,56 |
   | kNN* | 1,00 | 1,00 | 0,76 | 0,63 | 0,41 | 0,14 | 1,00 / 1,00 | 1,00 |
   | linear SVM* | 0,94 | 1,00 | **0,93** | 0,69 | 0,53 | 0,31 | 0,94 / 0,94 | 0,94 |
   | DS-CNN* | 0,88 | 0,93 | 0,86 | 0,53 | 0,33 | 0,24 | 0,88 / 0,88 | 0,88 |

   \* = eksplorasi. Temuan:
   - Model klasik masih ≥ 0,79 pada SNR 20 dB, tetapi menurun tajam di bawah 10 dB karena fitur MFCC tergeser derau.
     Rekomendasi siklus berikutnya: augmentasi derau saat training.
   - Semua model invariant terhadap gain dan time-shift, berkat normalisasi peak + VAD.
5. **Evaluasi eksploratif semua kandidat beku** (arsip MP11 & varian MP12, hash valid 9/9), dilaporkan terpisah dan **bukan** dasar seleksi:
   - Klasik: kNN 1,00, linear SVM 0,937, MLP 0,937, RF 0,874, tree 0,803.
   - CNN: DS-CNN 0,874, freq-pool 0,867, long 0,493.
6. **Konsistensi**: F1 hasil re-ekstraksi dari WAV sama dengan laporan resmi (0,937 & 0,560).
7. Output: `results/metrics/mp13_extension_evaluation_metrics.json`, `results/figures/mp13_extension_evaluation_analysis.png`.

## Adaptasi
- Lokasi skrip di root folder MP (buku: `scripts/`); hash memakai path relatif POSIX agar portabel.
- Mode **reproduction run**: bila MP13 sudah pernah dijalankan pada folder yang sama, rerun memverifikasi hash & metrik identik (test tidak dianggap "dibuka ulang" untuk tuning).
