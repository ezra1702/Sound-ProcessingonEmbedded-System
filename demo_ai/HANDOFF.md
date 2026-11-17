# Handoff: Siklus 2 (Deep Learning + Mastekkom). Status per 17 Nov 2026

Catatan ini untuk melanjutkan pekerjaan di laptop lain (macOS). Folder `Mini Project 07–16` **tidak diubah** oleh siklus 2.

## Tujuan (permintaan kelompok)
Demo minggu depan: sistem bekerja dengan baik di embedded (ESP32/ESP32-S3), tanpa presentasi. Fitur di atas MP16:
1. **KWS deep learning**: DS-CNN INT8, kelas `nyala, mati, naik, turun, diam, lain`, augmentasi berat.
2. **Kenali siapa yang bicara**: embedding penutur + prototipe; "kunci suara" (hanya penutur dikenal).
3. **Visualisasi AI di web**: spectrogram log-Mel + CAM, bar probabilitas, perbandingan DS-CNN vs logreg MP15.
4. **Daftarkan pengguna baru**: few-shot, 3× tiap kata, prototipe disimpan di NVS (tanpa melatih ulang).
5. **Mastekkom**: asisten suara (seperti "Jarvis") yang membalas dengan suara **Ivang** (rekaman asli per frasa, `demo_web/mastekkom/`).
   Wajib tetap sesuai matkul: pemrosesan suara + AI berjalan **di ESP32**; Mastekkom hanya lapisan keluaran di web.

## Yang SUDAH selesai
| Bagian | File | Status |
|---|---|---|
| Front-end, konstanta, loader 96 klip | `common.py` | selesai |
| Dataset + augmentasi | `build_dataset.py` → `artifacts/dataset_*.npz` | selesai (train 17196, val 1248, test 1212) |
| Model DS-CNN multi-tugas | `model.py` | selesai |
| Training + pemilihan lebar 32/48/64 (val saja) | `train.py` | ditulis; **run pertama belum selesai di Windows (CPU)** |
| INT8 + referensi numpy bit-exact | `int8_ref.py` | selesai |
| Engine C (front-end FFT/mel/CMN + INT8 + CAM + penutur) | `firmware_src/kws_engine.{c,h}`, `test_engine.c` | selesai; **diverifikasi bit-exact vs numpy (gcc)** dengan model acak |
| Ekspor header + verifikasi gcc + web_page.h | `export_c.py` | selesai (smoke test lulus) |
| Eksperimen LOSO few-shot + ambang penutur | `fewshot.py` | ditulis, belum dijalankan |
| Evaluasi akhir test (sekali) + banding klasik | `evaluate.py` | ditulis, belum dijalankan |
| Firmware ESP32 | `esp32_kws_ai/` | **compile OK** untuk `esp32:esp32:esp32s3` (1,04 MB, RAM 121 KB) dan `esp32:esp32:esp32` |
| Web demo (panel AI, penutur, enrol, Mastekkom, simulasi) | `../demo_web/index.html` | selesai, diuji Playwright (tanpa error, HP 390 px ok) |
| Board tiruan PC dengan model DL | `../demo_web/mock_board.py` (`--model dl` default) | patch sudah masuk, **belum dites** (butuh artefak model) |
| Frasa Mastekkom | `../demo_web/mastekkom/phrases.json` | selesai; **clip suara Ivang belum direkam** |

## Urutan menjalankan (setelah clone)
```bash
cd demo_ai
python build_dataset.py      # ±3 menit; TTS diambil dari artifacts/tts (sudah di-commit, tak perlu Windows)
python train.py              # ±45 menit CPU. Di Mac M-series: tambahkan dukungan 'mps' di _pick_device() agar cepat
python fewshot.py            # LOSO 4 fold → artifacts/fewshot_report.json (ambang penutur, proto_tau)
python export_c.py           # INT8 + header C + gcc check (butuh gcc/clang: di mac `cc` → ubah shutil.which('gcc') bila perlu)
python evaluate.py           # SEKALI SAJA → artifacts/eval_report.json + figures/
python ../demo_web/mock_board.py --mic   # uji di laptop: http://localhost:8080
```
Urutan penting: `export_c.py` membaca `fewshot_report.json` (ambang), `evaluate.py` dan `mock_board.py` membaca `export_report.json` + `int8_params.npz`.

## Sisa pekerjaan
1. **Jalankan pipeline di atas** dan cek angka (train_report → width terpilih).
2. Perbaikan kecil yang tertunda di `demo_web/index.html`: tambah `nama_ezra/nama_abdillah/nama_ivang/nama_jahzeel` ke objek `PHRASE_FALLBACK`
   (sekarang teks fallback tertulis "nama_ivang").
3. **Mastekkom suara Ivang** (dengan persetujuan Ivang):
   - Kloning suara AI ditolak: model open-source belum mendukung bahasa Indonesia, dan GPU Windows (RTX 5060) tidak cocok dengan PyTorch terpasang.
   - Rencana: Ivang merekam setiap frasa di `phrases.json` menjadi `demo_web/mastekkom/clips/<id>.wav` (16 kHz mono, potong hening).
   - `kata_nyala/mati/naik/turun` bisa diambil dari rekaman dataset Ivang.
   - Buat halaman perekam kecil (atau pakai web perekam MP09).
   - `mock_board.py` perlu menyajikan `/mastekkom/*` (sekarang hanya `/` dan `/ws`).
   - Firmware perlu menyajikan clip dari LittleFS.
4. Firmware: sajikan `/mastekkom/*` dari LittleFS; uji di board nyata (INMP441: SCK/WS/SD lihat `config.h`, L/R ke GND).
   Atur `MIC_SHIFT`, `GATE_DB` lewat Serial Monitor.
5. Update `demo_web/PROTOKOL.md`:
   - Pesan baru: `set{lock}`, `enroll_start{name}`, `enroll_cancel`, `user_delete{name}`; balasan `enroll{state: prompt|retry|done|cancelled}`, `users{list}`, `settings{lock}`.
   - Field event baru: `probs`, `speaker`, `speaker_cos`, `speaker_scores`, `cam_*`, `spec_*`, `classical`, `lat_fe_ms`, `lat_nn_ms`.
   - Aksi baru: `REJECT_SPEAKER`.
6. README `demo_ai/README.md` dengan catatan kejujuran:
   - kata "lain" = TTS en-US (bukan manusia);
   - block 6 sudah dibuka di siklus 1 (MP13) tapi tidak pernah dilatih;
   - hanya 4 penutur, jadi open-set penutur lemah;
   - latency mock = PC, bukan ESP32.

## Batasan yang harus dijaga
- Jangan upgrade PyTorch global (reproduksi MP12/MP13).
- Repo PUBLIK: jangan taruh rahasia (secret Apps Script) di halaman publik.
- Commit/push hanya bila diminta. Branch kerja: `dataset-kelompok-mp07-16`.
