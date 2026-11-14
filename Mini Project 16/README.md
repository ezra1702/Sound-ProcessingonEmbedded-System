# Mini Project 16 — Integrasi End-to-End dan Acceptance Test Proyek Final

**Anggota Kelompok**
| Nama | NIM |
|---|---|
| Christama Ezra Yudianto | 245150307111009 |
| Abdillah Fatah Suryanto | 245150300111008 |
| Ivang Arfandia | 245150307111003 |
| Jahzeel Erason Nicodemus | 245150300111002 |

## Tujuan
Menjalankan artefak beku MP15 sebagai **satu sistem** (WAV/mikrofon → filter MP6 → preprocessing MP7 → fitur MP8 → seleksi MP10 →
tree INT8 MP15 → aksi), memverifikasi konsistensi terhadap MP13, mengukur latency, membuat acceptance report, menutup status
menjadi `project_complete`.

## Cara menjalankan
```bash
python run_all_mini_projects.py --only 16            # replay acceptance + fitur tambahan (notebook)
cd "Mini Project 16"
python mini_project_16.py                             # replay acceptance saja
set MP16_LIVE_MIC=1 && python mini_project_16.py      # LIVE demo mikrofon (Windows cmd; PowerShell: $env:MP16_LIVE_MIC=1)
streamlit run dashboard_mp16.py                       # dashboard monitoring (FITUR TAMBAHAN)
```
Mapping aksi: **nyala = ON, mati = OFF, naik = level+1, turun = level−1** (level 0..10, hanya berubah saat ON).

## Hasil replay acceptance (catat hasil)
| Sampel test | Expected | MP13 | MP16 INT8 | Aksi |
|---|---|---|---|---|
| nyala_b06 | nyala | nyala | nyala | ON, level 5 |
| mati_b06 | mati | mati | mati | OFF |
| naik_b06 | naik | naik | naik | (OFF → level tidak berubah) |
| turun_b06 | turun | turun | turun | (OFF → level tidak berubah) |

| Besaran (host PC, bukan ESP32/RPi) | Nilai |
|---|---|
| Replay accuracy / agreement dengan MP13 | **1,00 / 1,00** |
| Latency fitur mean / p95 | 3,80 / 4,62 ms |
| Latency inferensi INT8 mean | 0,034 ms |
| Pipeline p95 / RTF p95 (vs klip 1,2 s) | 4,66 ms / **0,0039** |
| Payload model | 149 B |

| Checkpoint | Hasil |
|---|---|
| serial gate ready_for_mp16 | PASS |
| model sama dengan manifest MP15 | PASS |
| post_test_tuning = False | PASS |
| agreement numerik MP15 ≥ 0,95 | PASS (1,00) |
| overflow MP14 = 0 | PASS |
| agreement replay ≥ MP15 | PASS |
| fitur finite | PASS |
| p95 pipeline < durasi audio | PASS |
| **Acceptance** | **PASS → `project_complete`** |
| Live demo mikrofon | belum dijalankan (mode replay); jalankan dengan `MP16_LIVE_MIC=1` dan catat hasilnya |

Artefak: `config/system_integration_config.json`, `firmware/mp16_project_config.h`, `results/metrics/mp16_acceptance_metrics.json`,
`results/predictions/mp16_end_to_end_predictions.csv`, `deployment/mp16_final_project_manifest.json`, gambar `mp16_*.png`.

## FITUR TAMBAHAN (`features/mp16_extension_system_integration.py`, `dashboard_mp16.py`)
1. **Penolakan out-of-vocabulary** — tree tidak punya kelas "unknown", sehingga setiap suara keras akan menjadi perintah.
   Ditambahkan gate jarak Mahalanobis (Ledoit-Wolf, 39-D, fit train) dengan ambang **1,25 × jarak maks train+val = 9,03**
   (ditetapkan sebelum melihat distraktor). Uji terisolasi: keyword test diterima 4/4, **non-keyword speech** (TTS: halo, lampu,
   satu, tolong × 2 suara) ditolak **6/8**, **non-speech** (burst noise, tone 1 kHz, hum, tepukan) ditolak **4/4**.
2. **Command state machine** IDLE/COOLDOWN/FAULT: cooldown 0,8 s, batas level 0..10 (`IGNORED_BOUND`), **watchdog audio 500 ms →
   fail-safe OFF** lalu RECOVER. Implementasi Python dan C (`firmware/mp16_command_fsm.h`) **identik** pada 855 tick (gcc, semua 9 jenis aksi tercakup);
   semua header firmware dikompilasi bersama tanpa konflik.
3. **Demo stream kontinu 80,6 s** (24 keyword + 8 non-keyword + 4 non-speech + simulasi DMA stall 0,8 s) dengan **event log JSONL**
   (`results/logs/mp16_event_log.jsonl`: timestamp, kelas, jarak OOD, aksi, state, power, level, latency, SHA-256 model):

   | Desain endpoint | Keyword hit | Non-kw ditolak | Non-speech ditolak | Segmen gagal |
   |---|---|---|---|---|
   | v1 awal (jendela 975 ms, endpoint absolut 6 dB) | 22/24 | 6/8 | 4/4 | turun_b06 (kata terpotong), nyala_b04 (endpoint tak terpicu), 2 non-kw lolos |
   | **v2 final** (jendela 1,2 s, + drop relatif 20 dB) | **24/24** | 6/8 | 4/4 | tolong_zira, lampu_zira lolos |

   Analisis kegagalan v1: (a) endpoint terlambat sehingga jendela 975 ms memotong awal kata; (b) lantai derau di dalam klip (hum + pink
   noise) sedikit lebih tinggi dari derau jeda sehingga energi 200 ms tertahan ≈6,4 dB, tepat di atas ambang 6 dB. Perbaikan v2: jendela
   **1,2 s = durasi klip training = `LIVE_DURATION_S` buku**, dan endpoint juga terpicu bila energi turun ≥ 20 dB dari puncak ucapan.
   Kedua versi dilaporkan (`endpoint_design_comparison`). Watchdog: fail-safe 1×, recover 1×; level selalu dalam batas.
4. **Sketch ESP32** `firmware/esp32_kws_mp16/esp32_kws_mp16.ino` (I2S MEMS mic → ring buffer ISR-safe → fitur → `mp15_classical_predict`
   → FSM → LED/PWM, log JSON serial) sebagai baseline porting. *Front-end 39-D masih perlu diport ke C (`mp16_extract39`)*; angka perangkat wajib diukur di hardware.
5. **Dashboard Streamlit** `dashboard_mp16.py` (status, checklist, metrik, event log, upload WAV → prediksi pipeline beku), diuji otomatis
   dengan `streamlit.testing.AppTest`: 0 exception.
6. **Laporan final HTML** `deployment/mp16_final_report.html` (ringkasan MP7–MP16, checklist, gambar tertanam, catatan kejujuran ilmiah).
7. **Paket final** `deployment/mp16_final_package.zip` (35 file: skrip, config, model, header, sketch, metrik, log + `SHA256SUMS.json`, integritas terverifikasi).
8. Output: `results/metrics/mp16_extension_system_metrics.json`, `results/figures/mp16_extension_system_analysis.png`.

## Adaptasi terhadap kode buku
- **Gate family-aware**: buku mewajibkan `classical_svm_int8`; di sini menerima model klasik INT8 apa pun dari manifest MP15
  (terpilih `classical_tree_int8`). `predict_int8` memilih jalur linear/tree dari `kind` paket. Kunci metrik `svm_int8` → `classical_int8` (fallback).
- `LIVE_MIC` lewat environment `MP16_LIVE_MIC` agar notebook tetap non-interaktif.
- Aksi console dengan batas level & fail-safe ditambahkan di replay (Langkah 5 buku).

## Catatan kejujuran
Replay test memverifikasi konsistensi implementasi, **bukan** estimasi generalisasi baru. Stream demo memakai seluruh 24 ucapan
(termasuk test yang sudah dibuka di MP13) → uji integrasi sistem. Semua latency adalah host PC. Dataset adalah TTS tersimulasi (lihat README MP09).
