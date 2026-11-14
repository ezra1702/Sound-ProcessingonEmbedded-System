# Mini Project 15 — Kuantisasi, Verifikasi, dan Pembekuan Paket Deployment

**Anggota Kelompok**
| Nama | NIM |
|---|---|
| Christama Ezra Yudianto | 245150307111009 |
| Abdillah Fatah Suryanto | 245150300111008 |
| Ivang Arfandia | 245150307111003 |
| Jahzeel Erason Nicodemus | 245150300111002 |

## Tujuan
Mengkuantisasi kedua kandidat ke INT8 (kalibrasi **training-only**), memverifikasi kesetaraan terhadap float, benchmark host,
dan **memilih satu model deployment** dengan policy buku: kandidat dalam 0,03 macro-F1 frozen-test terbaik → payload terkecil →
p95 latency terendah. Tanpa retraining / post-test tuning. Status → `ready_for_mp16`.

## Adaptasi utama: family-aware
Kode buku mengasumsikan model klasik = **linear SVM** (`svm.coef_`). Di proyek ini MP11 memilih **decision tree depth 2**,
sehingga `mini_project_15.py` memilih jalur sesuai family:
- *linear* (`coef_`) → jalur buku apa adanya → `classical_svm_int8`;
- *tree* (`tree_`) → **jalur tree INT8**: input dikuantisasi dengan skala per-tensor training-only (`x_scale = max|z_train| / 127`),
  threshold node dipetakan ke integer `t_q = floor(t / x_scale)` sehingga `z <= t` menjadi `z_q <= t_q` (murni integer) → `classical_tree_int8`.

Header `firmware/mp15_classical_model_int8.h` berisi tabel + fungsi `mp15_classical_predict(const float *x39)` siap pakai MP16/ESP32.

## Cara menjalankan
```bash
python run_all_mini_projects.py --only 15
```

## Hasil (catat hasil)
| Besaran | Classical tree INT8 | Tiny CNN INT8 (fbgemm, eager PTQ) |
|---|---|---|
| Frozen-test macro-F1 (MP13) | 1,00 | 0,333 |
| Agreement vs float (24 / test) | **1,00 / 1,00** | **1,00 / 1,00** |
| Payload | **149 B** (7 node) | 1.340 B bobot (TorchScript 28,7 KB) |
| p95 latency host INT8 | 0,008 ms | 0,54 ms |
| Kalibrasi | training-only | training-only |

Threshold tree: node 0 `z[2] <= +0,0450 → z_q[2] <= 2`; node 1 `z[7] <= −0,3910 → z_q[7] <= −26`; node 4 `z[7] <= +0,0704 → z_q[7] <= 4` (`x_scale = 0,01513`).

**Model terpilih: `classical_tree_int8`** — CNN di luar toleransi (0,333 vs 1,00), sehingga hanya tree yang masuk; profil `MCU_first`,
fallback `tiny_cnn_int8`. `config/deployment_config.json`: `post_test_tuning_performed = false`.

| Pemeriksaan | Kriteria | Hasil |
|---|---|---|
| Status awal | ready_for_mp15 | ✔ |
| Kalibrasi CNN / input klasik | training-only | ✔ / ✔ |
| Post-test tuning | False | False |
| Agreement klasik | ideal 1,00 | 1,00 |
| Agreement CNN | ideal 1,00 | 1,00 |
| C header | tersedia | ✔ |
| Deployment manifest | tersedia, semua file ada | ✔ (`deployment/mp15_manifest.json`) |
| Model selection | tercatat + policy | classical_tree_int8 |
| Status akhir | ready_for_mp16 | ✔ |

## FITUR TAMBAHAN (`features/mp15_extension_quantization_analysis.py`)
1. **Equivalence C vs Python**: header dikompilasi gcc dan dijalankan pada 24 vektor 39-D mentah → **C = Python INT8 = float (1,00)**.
2. **Jalur INT8 buku untuk linear SVM** dari arsip MP11 (hash diverifikasi): agreement INT8 vs float 1,00, C vs Python 1,00 → `firmware/mp15_svm_int8_reference.h`.
3. **Bit-width sweep** (agreement vs float pada 2.000 sampel train ter-jitter):

   | Bit | 8 | 6 | 5 | 4 | 3 | 2 |
   |---|---|---|---|---|---|---|
   | tree per-tensor | 0,998 | 0,991 | 0,988 | 0,968 | 0,900 | 0,574 |
   | tree per-feature | 0,998 | 0,992 | 0,992 | 0,975 | 0,907 | 0,614 |
   | SVM per-tensor | 1,000 | 0,998 | 0,998 | 0,994 | 0,975 | 0,812 |
4. **Margin keputusan tree**: margin minimum 7,5 LSB (0,11 satuan z) — tidak ada sampel di dekat batas.
5. **CNN**: SQNR bobot per-tensor vs per-channel conv1 47,6 vs 49,2 dB, conv2 45,6 vs 46,7 dB, fc 47,9 vs 49,5 dB; FP16 ≈74 dB.
   SQNR logit INT8 28,7 dB / FP16 62,5 dB; agreement keduanya 1,00 (gap top-2 logit float minimum 0,023 → CNN memang tidak yakin).
6. **Footprint terkompilasi** (`gcc -Os`, `size`): model tree `.text` 208 B + `.rdata` 240 B; SVM referensi 256 B + 272 B.
7. **Paket deployment** `deployment/mp15_package/` + `SHA256SUMS.json` + `deployment/mp15_deployment_package.zip` (13 file, integritas SHA-256 terverifikasi).
8. Output: `results/metrics/mp15_extension_quantization_metrics.json`, `results/figures/mp15_extension_quantization_analysis.png`.

## Adaptasi lain
- `torch.load(..., weights_only=False)` eksplisit (default PyTorch 2.6 berubah).
- Literal float C dijamin valid (mis. `4200.0f`, bukan `4200f`).
