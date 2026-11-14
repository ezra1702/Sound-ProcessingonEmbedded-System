# Running Project — Keyword Spotting Embedded (nyala / mati / naik / turun)

Mata kuliah **Teknik Pemrosesan Suara dan AI pada Sistem Embedded** — Mini Project 1–16 (dokumen ini fokus pada **MP07–MP16**).

| Nama | NIM |
|---|---|
| Christama Ezra Yudianto | 245150307111009 |
| Abdillah Fatah Suryanto | 245150300111008 |
| Ivang Arfandia | 245150307111003 |
| Jahzeel Erason Nicodemus | 245150300111002 |

Sistem akhir: mikrofon → filter IIR MP6 → preprocessing (DC, normalisasi, VAD, spectral subtraction, pre-emphasis) →
fitur 39-D → 8 fitur terpilih → **decision tree INT8** (149 byte) → state machine aksi (ON/OFF/level 0..10, fail-safe).

---

## 1. Struktur folder

```
audio_ai_embedded_project/
├── run_all_mini_projects.py      # ORKESTRATOR: handoff artefak antar-folder + bangun & eksekusi notebook MP07..MP16
├── README.md                     # dokumen ini
├── Mini Project 01 … 06/         # tahap awal (rekaman, sampling, FFT, kanal, filter)
├── Mini Project 07 … 16/         # tiap folder = salinan kumulatif artefak MP sebelumnya + kode MP itu sendiri
│   ├── mini_project_NN.py        #   kode WAJIB (adaptasi kode buku, format sel "# %%")
│   ├── Mini_Project_NN.ipynb     #   notebook TEREKSEKUSI (output + gambar) = wajib + fitur tambahan
│   ├── README.md                 #   penjelasan, hasil, tabel validasi, fitur tambahan MP itu
│   ├── features/mpNN_extension_*.py  # FITUR TAMBAHAN (mandiri, dijalankan setelah skrip wajib)
│   ├── config/  data/  features/  firmware/  models/  results/  splits/  deployment/
│   └── project_status.json       #   status serial (ready_for_mpXX → project_complete)
└── *.pdf                         # soal mini project & materi bab
```

Setiap folder **dapat dijalankan berdiri sendiri** (semua path relatif terhadap folder itu). Status serial
`project_status.json` menjaga urutan: MP N menolak berjalan jika MP N−1 belum selesai.

## 2. Cara menjalankan

Kebutuhan: Python 3.11+ dengan `numpy scipy matplotlib scikit-learn joblib torch onnx onnxruntime nbformat jupyter_client ipykernel`
(opsional: `streamlit` untuk dashboard, `sounddevice` untuk mikrofon, `gcc` untuk uji header C).

```bash
# Semua MP07..MP16 berurutan (salin artefak N-1 -> N, bangkitkan & eksekusi notebook)
python run_all_mini_projects.py

# Satu mini project saja / mulai dari MP tertentu / tanpa menyalin artefak
python run_all_mini_projects.py --only 13
python run_all_mini_projects.py --start 10
python run_all_mini_projects.py --only 14 --no-copy

# Tanpa notebook: jalankan skrip langsung di foldernya
cd "Mini Project 16"
python mini_project_16.py
python features/mp16_extension_system_integration.py
streamlit run dashboard_mp16.py            # dashboard (FITUR TAMBAHAN)
set MP16_LIVE_MIC=1 && python mini_project_16.py   # demo mikrofon live (Windows cmd)
```

## 3. Ringkasan hasil MP07–MP16

| MP | Topik | Keputusan / hasil utama |
|---|---|---|
| 07 | Preprocessing | 2,35 s → 0,615 s aktif (ratio 0,258), 60 frame, tanpa clipping |
| 08 | Fitur | log-Mel 26, MFCC 13, spektral, temporal, pitch → **vektor 39-D** |
| 09 | Dataset | 24 WAV (6 block × 4 kelas), split grup 16/4/4 tanpa overlap |
| 10 | Seleksi fitur | **MI-8** (CV-F1 1,00): mfcc02 mean/std, mfcc04 std, mfcc05 mean, zcr mean/std, centroid mean/std |
| 11 | Classifier klasik | 6 family val-F1 1,00 → **decision tree depth 2** (ops paling sedikit) |
| 12 | Tiny CNN | 1.340 param, 898k MAC, **val-F1 0,375 (underfitting, didiagnosis)** |
| 13 | Frozen test (sekali) | classical **F1 1,00** (Wilson 95% acc [0,51; 1,00]); CNN F1 0,333; tanpa post-test tuning |
| 14 | Scheduler real-time | DMA 10 ms, context 96 frame / 975 ms, stride 100 ms, **overflow 0**, realtime PASS |
| 15 | Kuantisasi & paket | **classical_tree_int8 terpilih** (payload 149 B, agreement 1,00); CNN INT8 agreement 1,00 |
| 16 | Integrasi & acceptance | replay test = MP13 (agreement 1,00), RTF ≪ 1, **acceptance PASS**, status `project_complete` |

Detail angka, tabel validasi, dan gambar ada di README & notebook masing-masing folder.

## 4. Fitur tambahan (di luar tugas wajib)

| MP | Fitur tambahan |
|---|---|
| 07 | Perbandingan normalisasi; VAD energi+ZCR+hangover; uji denoise terkontrol SS vs Wiener-DD; DC-block+pre-emphasis **Q15** + header C |
| 08 | Delta/delta², flux, flatness; validasi pitch **YIN**; biaya per fitur (µs & MAC); presisi float32; sparsity filterbank + header tabel C |
| 09 | Mode akuisisi mic/tts/demo; modul front-end beku `kws_frontend.py` (bit-exact); audit kualitas; manifest SHA-256; leakage guard; Fisher score; galeri log-Mel |
| 10 | L1/RF/RFE vs MI + stabilitas Jaccard; kurva k; permutation importance; **analisis biaya front-end** (hemat 86,9% MAC) |
| 11 | SVM-RBF/GNB/LDA; robustness derau fitur; arsip semua family + SHA-256; **export C float diuji gcc**; latency host |
| 12 | Diagnosis underfitting; varian DS-CNN & freq-pool (val-F1 1,00); stabilitas seed; **ONNX**; profil memori aktivasi; saliency |
| 13 | Bootstrap CI; **uji McNemar exact**; Brier/NLL/ECE; ROC-AUC/PR-AP; **robustness stress test** (SNR 30→0 dB, gain, shift); evaluasi eksploratif model beku ber-hash |
| 14 | Profil per-stage; **proyeksi MCU ESP32-S3**; CPU stress sweep (overflow mulai ×32); stride sweep; **streaming KWS kontinu VAD-endpoint (20/20, 0 false alarm, hemat 95% inferensi)**; ring buffer C diuji gcc; Gantt |
| 15 | **C INT8 == Python == float** (gcc); jalur SVM INT8 buku dari arsip; bit-width sweep 8→2 bit; margin LSB; per-channel vs per-tensor; FP16; footprint terkompilasi; **paket ZIP SHA-256** |
| 16 | Gate penolakan **non-keyword** (Mahalanobis LW: non-kw speech ditolak 6/8, non-speech 4/4); **command state machine** (cooldown, batas level, watchdog fail-safe) Python == C; stream demo 80 s (**24/24 keyword**, desain v1 vs v2 + analisis kegagalan) + **event log JSONL**; sketch **ESP32**; **dashboard Streamlit** (diuji AppTest); laporan final HTML; paket final ZIP |

## 5. Adaptasi penting terhadap kode buku

1. **Path portabel** — skrip tidak memakai path absolut; `ROOT` dihitung dari lokasi skrip / cwd notebook.
2. **Rerun idempoten** — pemeriksaan status menerima status sekarang *atau* status berikutnya, sehingga notebook bisa dijalankan ulang.
3. **Model terpilih = decision tree, bukan linear SVM** — MP11 (policy buku: toleransi 0,03 → ops terkecil) memilih tree.
   Kode buku MP15/MP16 mengasumsikan `svm.coef_`, maka dibuat **family-aware**: jalur tree INT8 (threshold integer
   `t_q = floor(t / x_scale)`), nama model `classical_tree_int8`, gate MP16 menerima model klasik INT8 apa pun dari manifest.
   Jalur SVM INT8 buku tetap didemonstrasikan di fitur tambahan MP15 (arsip linear SVM MP11, hash diverifikasi).
4. **MP14**: kunci `sampling_rate_hz` di buku → `sample_rate_hz` di `project_config.json` (dibaca dengan fallback).
5. **MP13**: buku menaruh skrip di `scripts/`; di sini konsisten di root folder MP. Mode *reproduction run* memverifikasi hash & metrik identik bila MP13 dijalankan ulang (test tetap dianggap dibuka sekali).
6. **PyTorch 2.6** — `torch.load(..., weights_only=False)` eksplisit untuk checkpoint berisi metadata.

## 6. Catatan kejujuran ilmiah (penting dibaca)

- **Dataset MP9 adalah ucapan TTS (Windows SAPI, suara Zira & David) dengan simulasi kanal per block**, karena pengerjaan
  otomatis tidak memiliki kesempatan merekam suara anggota. Akurasi 1,00 berlaku untuk data ini dan **bukan** estimasi
  generalisasi ke penutur manusia. Untuk laporan akhir disarankan merekam ulang:
  ```bash
  del "Mini Project 09\data\raw\mp09_dataset\*.wav"
  set MP09_MODE=mic
  python run_all_mini_projects.py --only 9
  python run_all_mini_projects.py --start 10
  ```
- Test set hanya 4 sampel → interval kepercayaan lebar (Wilson 95% akurasi [0,51; 1,00]); uji McNemar classical vs CNN p = 0,5 (tidak signifikan).
- Semua latency adalah **host PC (Windows, Python)**. Angka ESP32 di MP14 adalah *proyeksi berbasis MAC*, bukan pengukuran. Sketch ESP32
  disediakan sebagai baseline porting; front-end 39-D masih perlu diport ke C (lihat `firmware/esp32_kws_mp16/`).
- Uji stream MP14/MP16 memakai data yang sudah dilihat model (train/val, dan test yang sudah dibuka MP13) → **uji integrasi sistem**, bukan estimasi akurasi baru.
