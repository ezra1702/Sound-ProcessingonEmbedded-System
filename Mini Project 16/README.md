# Mini Project 16 — Integrasi End-to-End dan Acceptance Test Proyek Final

**Anggota Kelompok**
| Nama | NIM |
|---|---|
| Christama Ezra Yudianto | 245150307111009 |
| Abdillah Fatah Suryanto | 245150300111008 |
| Ivang Arfandia | 245150307111003 |
| Jahzeel Erason Nicodemus | 245150300111002 |

## Tujuan
Menjalankan artefak beku MP15 sebagai **satu sistem**: WAV/mikrofon → filter MP6 → preprocessing MP7 → fitur MP8 → seleksi MP10 →
logreg INT8 MP15 → aksi. Lalu:
- memverifikasi konsistensi terhadap MP13,
- mengukur latency,
- membuat acceptance report,
- menutup status menjadi `project_complete`.

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
Replay memakai 16 WAV test (block 6, 4 penutur) melalui pipeline INT8 penuh. Detail per sampel ada di `results/predictions/mp16_end_to_end_predictions.csv`.

| Penutur | nyala | mati | naik | turun |
|---|---|---|---|---|
| Ezra | ✔ | ✔ | ✔ | ✘ (→ naik) |
| Abdillah | ✔ | ✔ | ✔ | ✔ |
| Ivang | ✔ | ✔ | ✔ | ✔ |
| Jahzeel | ✔ | ✔ | ✔ | ✔ |

Satu-satunya kesalahan, `ezra_turun_b06` → naik, **sama persis** dengan prediksi MP13 (agreement 16/16).
Urutan replay nyala → mati → naik → turun per penutur menghasilkan aksi ON → OFF, lalu naik/turun diabaikan karena lampu OFF.

| Besaran (host PC, bukan ESP32/RPi) | Nilai |
|---|---|
| Replay accuracy / agreement dengan MP13 | **0,9375 / 1,00** |
| Latency fitur mean / p95 | 2,95 / 4,15 ms |
| Latency inferensi INT8 mean | 0,032 ms |
| Pipeline p95 / RTF p95 (vs klip 1,2 s) | 4,18 ms / **0,0035** |
| Payload model | 164 B |

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
1. **Penolakan out-of-vocabulary**: classifier 4 kelas tidak punya kelas "unknown", sehingga setiap suara keras akan menjadi perintah.
   - Ditambahkan gate jarak Mahalanobis (Ledoit-Wolf, 39-D, fit train) dengan ambang **1,25 × jarak maks train+val = 11,81**.
     Ambang ini ditetapkan sebelum melihat distraktor.
   - Hasil uji terisolasi:
     - Keyword test diterima **16/16**.
     - **Non-speech** (burst noise, tone 1 kHz, hum, tepukan) ditolak **4/4**.
     - **Non-keyword speech** ditolak hanya **4/8**. Yang ditolak: halo_david, satu_david, satu_zira, tolong_zira.
   - Catatan: non-keyword adalah TTS (halo, lampu, satu, tolong × 2 suara), karena kelompok tidak merekam kata lain.
   - Dibanding versi awal, ambangnya lebih longgar (11,8 vs 9,0), karena variasi 4 penutur asli membuat sebaran keyword lebih lebar.
     Penolakan non-keyword yang andal butuh kelas "unknown" yang dilatih dengan rekaman kata lain.
2. **Command state machine** IDLE/COOLDOWN/FAULT:
   - Cooldown 0,8 s dan batas level 0..10 (`IGNORED_BOUND`).
   - **Watchdog audio 500 ms → fail-safe OFF**, lalu RECOVER.
   - Implementasi Python dan C (`firmware/mp16_command_fsm.h`) **identik** pada 2.463 tick (gcc, semua 9 jenis aksi tercakup).
   - Semua header firmware dikompilasi bersama tanpa konflik.
3. **Demo stream kontinu 241 s**: 96 keyword dari 4 penutur + 8 non-keyword + 4 non-speech + simulasi DMA stall 0,8 s.
   - Derau jeda mengikuti derau latar rekaman masing-masing penutur.
   - Setiap kejadian dicatat di **event log JSONL** (`results/logs/mp16_event_log.jsonl`): timestamp, kelas, jarak OOD, aksi, state, power, level, latency, dan SHA-256 model.

   | Desain endpoint | Keyword hit | Miss / false-reject / substitusi | Non-kw ditolak | Non-speech ditolak |
   |---|---|---|---|---|
   | v1 awal (jendela 975 ms, endpoint absolut 6 dB) | 66/96 | 26 / 3 / 1 | 4/8 | 4/4 |
   | **v2 final** (jendela 1,2 s, + drop relatif 20 dB) | **92/96** | 2 / 0 / 2 | 3/8 | 4/4 |

   **Analisis kegagalan v1:**
   - 16 dari 29 miss/false-reject berasal dari Ezra, penutur dengan lantai derau paling rendah (≈ −68 dBFS). Pada rekamannya, syarat
     endpoint absolut "energi 200 ms < noise floor + 6 dB" jarang terpenuhi tepat waktu, sehingga jendela memotong kata atau endpoint tidak terpicu.
   - Perbaikan v2: jendela diperpanjang menjadi **1,2 s**, sama dengan durasi klip training dan `LIVE_DURATION_S` buku.
     Selain itu, endpoint juga terpicu bila energi turun ≥ 20 dB dari puncak ucapan.

   **Sisa kegagalan v2:**
   - `ezra_turun_b06` → naik, sama dengan kesalahan MP13.
   - `ivang_mati_b05` → naik, sama dengan kesalahan validation MP11.
   - 2 miss Jahzeel (`nyala_b03`, `naik_b04`).
   - 5 non-keyword TTS lolos.

   Kedua versi dilaporkan di `endpoint_design_comparison`. Watchdog: fail-safe 1×, recover 1×, dan level selalu dalam batas.
4. **Sketch ESP32** `firmware/esp32_kws_mp16/esp32_kws_mp16.ino` sebagai baseline porting.
   - Alurnya: I2S MEMS mic → ring buffer ISR-safe → fitur → `mp15_classical_predict` → FSM → LED/PWM, dengan log JSON serial.
   - *Front-end 39-D masih perlu diport ke C (`mp16_extract39`)*, dan angka perangkat wajib diukur di hardware.
5. **Dashboard Streamlit** `dashboard_mp16.py` (status, checklist, metrik, event log, upload WAV → prediksi pipeline beku).
   Diuji otomatis dengan `streamlit.testing.AppTest`: 0 exception.
6. **Laporan final HTML** `deployment/mp16_final_report.html`: ringkasan MP7–MP16, checklist, gambar tertanam, dan catatan kejujuran ilmiah.
7. **Paket final** `deployment/mp16_final_package.zip`: 35 file (skrip, config, model, header, sketch, metrik, log) + `SHA256SUMS.json`, integritas terverifikasi.
8. Output: `results/metrics/mp16_extension_system_metrics.json`, `results/figures/mp16_extension_system_analysis.png`.

## Adaptasi terhadap kode buku
- **Gate family-aware**: buku mewajibkan `classical_svm_int8`. Di sini gate menerima model klasik INT8 apa pun dari manifest MP15 (terpilih `classical_logreg_int8`).
  - `predict_int8` memilih jalur linear/tree dari `kind` paket.
  - Kunci metrik `svm_int8` menjadi `classical_int8` (dengan fallback).
- **Nama file dataset** `<penutur>_<kata>_b0N`: label diparse dari bagian sebelum `_b`.
- `LIVE_MIC` dibaca lewat environment `MP16_LIVE_MIC` agar notebook tetap non-interaktif.
- Aksi console dengan batas level & fail-safe ditambahkan di replay (Langkah 5 buku).

## Catatan kejujuran
- Replay test memverifikasi konsistensi implementasi, **bukan** estimasi generalisasi baru.
- Stream demo memakai seluruh 96 ucapan, termasuk test yang sudah dibuka di MP13. Jadi ini uji integrasi sistem.
- Distraktor non-keyword adalah TTS.
- Semua latency diukur di host PC.
- Dataset keyword adalah rekaman asli 4 anggota kelompok (lihat README MP09).
