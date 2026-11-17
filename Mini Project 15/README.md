# Mini Project 15 — Kuantisasi, Verifikasi, dan Pembekuan Paket Deployment

**Anggota Kelompok**
| Nama | NIM |
|---|---|
| Christama Ezra Yudianto | 245150307111009 |
| Abdillah Fatah Suryanto | 245150300111008 |
| Ivang Arfandia | 245150307111003 |
| Jahzeel Erason Nicodemus | 245150300111002 |

## Tujuan
Mengkuantisasi kedua kandidat ke INT8 (kalibrasi **training-only**), memverifikasi kesetaraannya terhadap float, dan menjalankan benchmark host.
Lalu **satu model deployment dipilih** dengan policy buku:
1. Kandidat harus dalam 0,03 macro-F1 dari frozen-test terbaik.
2. Dari kandidat itu, pilih payload terkecil.
3. Bila masih seri, pilih p95 latency terendah.

Tidak ada retraining atau post-test tuning. Status → `ready_for_mp16`.

## Adaptasi utama: family-aware
Kode buku mengasumsikan model klasik = **linear SVM** (`svm.coef_`). Pada dataset 4 penutur, MP11 memilih **logistic regression**.
Logreg juga linear (`coef_`, `intercept_`), sehingga `mini_project_15.py` memakai jalur buku:
- Input z-score dikuantisasi per-tensor dengan skala dari train (`x_scale = 0,0292`).
- Bobot dikuantisasi INT8 per kelas (4 skala).
- Akumulasi dilakukan dengan `int32`, dan bias float ditambahkan di akhir.
- Hasilnya `classical_logreg_int8`.

Jalur **tree INT8** (threshold integer `t_q = floor(t / x_scale)`) tetap ada di kode dan otomatis dipakai bila MP11 memilih tree.

Header `firmware/mp15_classical_model_int8.h` berisi tabel dan fungsi `mp15_classical_predict(const float *x39)` yang siap dipakai MP16/ESP32.

## Cara menjalankan
```bash
python run_all_mini_projects.py --only 15
```

## Hasil (catat hasil)
| Besaran | Classical logreg INT8 | Tiny CNN INT8 (fbgemm, eager PTQ) |
|---|---|---|
| Frozen-test macro-F1 (MP13, float) | 0,937 | 0,560 |
| Agreement vs float (semua 96 / test 16) | **1,00 / 1,00** | 0,958 / 0,9375 |
| Payload | **164 B** (32 bobot INT8 + skala, bias, mean/scale & indeks 8 fitur) | 1.340 B bobot (TorchScript 28,7 KB) |
| p95 latency host INT8 (float) | 0,015 ms (0,075 ms) | 0,54 ms (0,26 ms) |
| Kalibrasi | training-only | training-only |

CNN INT8 mengubah 4 dari 96 prediksi (`abdillah_mati_b01`, `jahzeel_mati_b01`, `ezra_naik_b03`, `ezra_turun_b06`), karena logit CNN float sangat berdekatan.

**Model terpilih: `classical_logreg_int8`.**
- CNN di luar toleransi (0,560 vs 0,937), sehingga hanya model klasik yang masuk.
- Profil `MCU_first`, dengan fallback `tiny_cnn_int8`.
- `config/deployment_config.json` mencatat `post_test_tuning_performed = false`.

| Pemeriksaan | Kriteria | Hasil |
|---|---|---|
| Status awal | ready_for_mp15 | ✔ |
| Kalibrasi CNN / input klasik | training-only | ✔ / ✔ |
| Post-test tuning | False | False |
| Agreement klasik | ideal 1,00 | 1,00 |
| Agreement CNN | ≥ 0,95 | 0,958 |
| C header | tersedia | ✔ |
| Deployment manifest | tersedia, semua file ada | ✔ (`deployment/mp15_manifest.json`) |
| Model selection | tercatat + policy | classical_logreg_int8 |
| Status akhir | ready_for_mp16 | ✔ |

## FITUR TAMBAHAN (`features/mp15_extension_quantization_analysis.py`)
1. **Equivalence C vs Python**: header dikompilasi gcc lalu dijalankan pada 96 vektor 39-D mentah. Hasilnya **C = Python INT8 = float (1,00)**.
2. **Jalur INT8 buku untuk linear SVM** dari arsip MP11 (hash diverifikasi): agreement INT8 vs float 1,00 dan C vs Python 1,00. Header: `firmware/mp15_svm_int8_reference.h`.
3. **Bit-width sweep**: agreement vs float pada 96 sampel asli dan 2.000 sampel train ter-jitter.

   | Bit | 8 | 6 | 5 | 4 | 3 | 2 |
   |---|---|---|---|---|---|---|
   | logreg per-tensor (96 / jitter) | 1,00 / 0,998 | 1,00 / 0,992 | 1,00 / 0,987 | 0,990 / 0,975 | 0,948 / 0,919 | 0,417 / 0,459 |
   | logreg per-feature (96 / jitter) | 1,00 / 0,997 | 1,00 / 0,997 | 1,00 / 0,986 | 0,990 / 0,976 | 0,948 / 0,943 | 0,708 / 0,690 |
   | SVM per-tensor (96 / jitter) | 1,00 / 0,998 | 0,990 / 0,991 | 1,00 / 0,983 | 0,990 / 0,962 | 0,938 / 0,882 | 0,438 / 0,464 |

   Kesimpulannya, model linear aman sampai 4–5 bit. 8 bit memberi margin yang nyaman.
4. **Margin keputusan** (selisih logit top-1 − top-2) pada 96 sampel: minimum 0,137 logit, dengan 2 sampel di bawah 0,5.
   Agreement INT8 tetap 1,00, karena galat kuantisasi jauh lebih kecil dari margin tersebut.
5. **CNN**:
   - SQNR bobot per-tensor vs per-channel: conv1 47,7 vs 48,7 dB, conv2 45,7 vs 46,8 dB, fc 47,8 vs 47,9 dB. FP16 ≈ 73–74 dB.
   - SQNR logit INT8 26,9 dB, FP16 57,0 dB. Agreement INT8 0,958, FP16 1,00.
   - Gap top-2 logit float minimum hanya 0,0002, artinya CNN memang tidak yakin, sehingga galat kecil pun bisa membalik prediksi.
6. **Footprint terkompilasi** (`gcc -Os`, `size`, host x86-64): model logreg `.text` 240 B + `.rdata` 240 B; SVM referensi 256 B + 272 B.
7. **Paket deployment**: `deployment/mp15_package/` + `SHA256SUMS.json` + `deployment/mp15_deployment_package.zip` (13 file, integritas SHA-256 terverifikasi).
8. Output: `results/metrics/mp15_extension_quantization_metrics.json`, `results/figures/mp15_extension_quantization_analysis.png`.

## Adaptasi lain
- `torch.load(..., weights_only=False)` eksplisit (default PyTorch 2.6 berubah).
- Literal float C dijamin valid (mis. `4200.0f`, bukan `4200f`).
