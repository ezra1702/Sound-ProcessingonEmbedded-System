# Running Project — Keyword Spotting Embedded (nyala / mati / naik / turun)

Mata kuliah **Teknik Pemrosesan Suara dan AI pada Sistem Embedded** — Mini Project 1–16 (dokumen ini fokus pada **MP07–MP16**).

| Nama | NIM |
|---|---|
| Christama Ezra Yudianto | 245150307111009 |
| Abdillah Fatah Suryanto | 245150300111008 |
| Ivang Arfandia | 245150307111003 |
| Jahzeel Erason Nicodemus | 245150300111002 |

Sistem akhir: mikrofon → filter IIR MP6 → preprocessing (DC, normalisasi, VAD, spectral subtraction, pre-emphasis) →
fitur 39-D → 8 fitur terpilih (RFE-8) → **logistic regression INT8** (payload 164 byte) → state machine aksi (ON/OFF/level 0..10, fail-safe).

Dataset: **rekaman suara asli 4 anggota kelompok**. Ada 4 penutur × 6 block × 4 kata = **96 WAV**. Rekaman dibuat lewat web
perekam (`Perekam_Dataset_MP09.html`), yang memeriksa kata secara otomatis dan menyimpan hasilnya ke Google Drive. File lalu disalin ke `dataset_rekaman/`.

---

## 1. Struktur folder

```
audio_ai_embedded_project/
├── run_all_mini_projects.py      # ORKESTRATOR: handoff artefak antar-folder + bangun & eksekusi notebook MP07..MP16
├── README.md                     # dokumen ini
├── dataset_rekaman/<Nama>/block_0N/<kata>_b0N.wav   # 96 WAV asli (sumber MP09)
├── Perekam_Dataset_MP09.html     # web perekam dataset (di-host: christama-ezra-audio.vercel.app)
├── drive_upload_apps_script.gs   # Apps Script penyimpan rekaman ke Google Drive
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
| 09 | Dataset | **96 WAV** (4 penutur × 6 block × 4 kelas). Split per block **64 / 16 / 16** (block 1–4 / 5 / 6), setiap penutur ada di semua split, tanpa overlap grup |
| 10 | Seleksi fitur | **RFE-8** (CV-F1 0,931). Skor terbaik 0,952, tetapi dipilih dimensi terkecil yang masih dalam toleransi 0,03. Fitur: mfcc01/03/04/07 mean, mfcc04/06/11 std, rms mean |
| 11 | Classifier klasik | shortlist knn & logreg (val-F1 0,937) → **logistic regression C=1** (64 ops, CV-F1 1,00) |
| 12 | Tiny CNN | 1.340 param, 898k MAC, **val-F1 0,514 (underfitting, sudah didiagnosis)** |
| 13 | Frozen test (sekali) | classical **F1 0,937 / akurasi 15/16** (Wilson 95% [0,72; 0,99]); CNN F1 0,560; McNemar p = 0,031; tanpa tuning sesudah test |
| 14 | Scheduler real-time | DMA 10 ms, context 96 frame / 975 ms, stride 100 ms, **overflow 0**, realtime PASS |
| 15 | Kuantisasi & paket | **classical_logreg_int8 terpilih** (payload 164 B, agreement INT8 vs float 1,00, C == Python); CNN INT8 agreement 0,958 |
| 16 | Integrasi & acceptance | replay test = MP13 (akurasi 0,9375, agreement 1,00), RTF p95 0,0035, **acceptance PASS**, status `project_complete` |

Detail angka, tabel validasi, dan gambar ada di README & notebook masing-masing folder.

## 4. Fitur tambahan (di luar tugas wajib)

| MP | Fitur tambahan |
|---|---|
| 07 | Perbandingan normalisasi; VAD energi+ZCR+hangover; uji denoise terkontrol SS vs Wiener-DD; DC-block+pre-emphasis **Q15** + header C |
| 08 | Delta/delta², flux, flatness; validasi pitch **YIN**; biaya per fitur (µs & MAC); presisi float32; sparsity filterbank + header tabel C |
| 09 | Mode akuisisi group/mic/tts/demo; modul front-end beku `kws_frontend.py` (bit-exact); audit kualitas **per penutur**; manifest SHA-256; leakage guard; Fisher score; galeri log-Mel kelas × penutur |
| 10 | L1/RF/RFE vs MI + stabilitas Jaccard; kurva k; permutation importance; **analisis biaya front-end** (hemat 88,0% MAC, pitch bisa dilewati) |
| 11 | SVM-RBF/GNB/LDA; robustness terhadap derau fitur; arsip semua family + SHA-256; **export C float diuji gcc** (agreement 1,00); latency host |
| 12 | Diagnosis underfitting; varian DS-CNN (val-F1 0,937) & freq-pool (0,879); stabilitas seed; **ONNX**; profil memori aktivasi; saliency |
| 13 | Bootstrap CI; **uji McNemar exact**; Brier/NLL/ECE; ROC-AUC/PR-AP (logreg AUC 0,995); **robustness stress test** (SNR 30→0 dB, gain, shift); evaluasi eksploratif model beku ber-hash |
| 14 | Profil per-stage; **proyeksi MCU ESP32-S3**; CPU stress sweep (overflow mulai ×32); stride sweep; ring buffer C diuji gcc; Gantt. **Streaming KWS kontinu 80 ucapan 4 penutur**: VAD-endpoint 57/80 hit, 0 false alarm, hemat 96% inferensi, sweep ambang train+val |
| 15 | **C INT8 == Python == float** (gcc, n = 96); jalur SVM INT8 buku dari arsip; bit-width sweep 8→2 bit (≥ 0,99 sampai 4 bit); margin keputusan logit; per-feature vs per-tensor; FP16; footprint terkompilasi; **paket ZIP SHA-256** |
| 16 | Gate penolakan **non-keyword** (Mahalanobis LW): keyword test diterima 16/16, non-keyword speech ditolak 4/8, non-speech 4/4. **Command state machine** (cooldown, batas level, watchdog fail-safe), Python == C. Stream demo 241 s: **92/96 keyword** pada desain v2 (v1: 66/96), plus analisis kegagalan dan **event log JSONL**. Sketch **ESP32**; **dashboard Streamlit** (diuji AppTest); laporan final HTML; paket final ZIP |

## 5. Adaptasi penting terhadap kode buku

1. **Path portabel**: skrip tidak memakai path absolut. `ROOT` dihitung dari lokasi skrip atau cwd notebook.
2. **Rerun idempoten**: pemeriksaan status menerima status sekarang *atau* status berikutnya, sehingga notebook bisa dijalankan ulang.
3. **Dataset 96, bukan 24**:
   - Buku meminta 1 penutur × 24 WAV. Kelompok merekam **4 penutur × 24 = 96 WAV** agar model tidak hanya mengenal satu suara.
   - Unit split tetap **acquisition block** seperti buku: block 1–4 train, block 5 validation, block 6 test. Jadi setiap penutur muncul di ketiga split.
   - `subject_id` dicatat di metadata dan dipakai untuk analisis per penutur.
   - Nama file menjadi `<penutur>_<kata>_b0N.wav`, dan parsing label di MP13–MP16 disesuaikan.
4. **Model terpilih = logistic regression**:
   - Kode buku MP15/MP16 mengasumsikan `svm.coef_`. Karena itu kodenya dibuat **family-aware**, dengan jalur linear `coef_`/`intercept_` atau jalur tree INT8.
   - Nama model `classical_logreg_int8`. Gate MP16 menerima model klasik INT8 apa pun dari manifest.
   - Jalur SVM INT8 buku tetap didemonstrasikan di fitur tambahan MP15 (arsip linear SVM MP11, hash diverifikasi).
5. **MP14**: kunci `sampling_rate_hz` di buku menjadi `sample_rate_hz` di `project_config.json` (dibaca dengan fallback). WAV sumber stream diambil dari split train.
6. **MP13**: buku menaruh skrip di `scripts/`, sedangkan di sini skrip konsisten ada di root folder MP. Bila MP13 dijalankan ulang, mode *reproduction run* memverifikasi hash dan metrik identik, dan test tetap dianggap dibuka sekali.
7. **PyTorch 2.6**: `torch.load(..., weights_only=False)` ditulis eksplisit untuk checkpoint berisi metadata.

## 6. Catatan kejujuran ilmiah (penting dibaca)

- **Dataset = rekaman asli 4 anggota kelompok**, memakai mikrofon laptop/HP masing-masing lewat browser (16 kHz).
  - Level suara dan derau latar berbeda antar penutur: peak median 0,12–0,40, derau berbeda hingga ±12 dB.
  - Ini realistis, tetapi juga membuat performa berbeda antar penutur.
- **Test berisi penutur yang sudah dikenal**, yaitu block 6 dari 4 penutur yang sama.
  - Akurasi 15/16 adalah estimasi untuk *sesi baru dari penutur yang dikenal*, bukan untuk penutur baru.
  - Test hanya 16 sampel, sehingga interval kepercayaan lebar (Wilson 95% akurasi [0,72; 0,99]).
- **Tiny CNN resmi underfitting** (F1 test 0,560), karena data dan langkah optimizer sedikit.
  - Varian DS-CNN lebih baik di validation, tetapi dipilih sesudah kontrak MP12, jadi hanya eksploratif dan tidak dipakai.
  - McNemar logreg vs CNN: p = 0,031.
- **Sanity window bawaan buku di MP14 memprediksi kelas yang salah.**
  - Uji ini memakai satu WAV nyala yang diulang. Konteks 975 ms tanpa endpoint tidak sama dengan klip training 1,2 s.
  - Karena itu streaming memakai **VAD-endpoint trigger** (fitur tambahan).
- **Ucapan non-keyword untuk uji penolakan MP16 adalah TTS** (Windows SAPI), karena kelompok hanya merekam 4 keyword.
  - Laju penolakan non-keyword masih rendah (4/8) dan belum mewakili suara manusia.
  - Bila sistem dipakai sungguhan, perlu data "unknown" asli.
- **Semua latency diukur di host PC (Windows, Python).**
  - Angka ESP32 di MP14 adalah *proyeksi berbasis MAC*, bukan pengukuran.
  - Sketch ESP32 disediakan sebagai baseline porting. Front-end 39-D masih perlu diport ke C (lihat `firmware/esp32_kws_mp16/`).
- **Uji stream MP14/MP16 memakai data yang sudah dilihat model** (train/val, dan test yang sudah dibuka MP13). Jadi ini **uji integrasi sistem**, bukan estimasi akurasi baru.
