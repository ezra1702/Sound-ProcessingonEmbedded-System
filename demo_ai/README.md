# demo_ai — Siklus 2: KWS deep learning di ESP32

Lanjutan MP16 untuk demo: semua pemrosesan suara dan AI berjalan **di ESP32 / ESP32-S3**. Web hanya menampilkan hasil.
Folder `Mini Project 07–16` tidak diubah oleh siklus 2.

| Fitur | Isi |
|---|---|
| KWS deep learning | DS-CNN INT8, 6 kelas `nyala, mati, naik, turun, diam, lain`, augmentasi kanal berat |
| Kenali penutur | kepala embedding 32-D + prototipe per penutur (cosine), "kunci suara" → `REJECT_SPEAKER` |
| Visualisasi AI | log-Mel input model + Class Activation Map, bar probabilitas, banding dengan logreg MP15 |
| Pengguna baru | few-shot: 3× tiap kata, prototipe disimpan di NVS, **tanpa melatih ulang** |

## File

| File | Isi |
|---|---|
| `common.py` | konstanta + front-end log-Mel 98×40 (identik dengan C) + loader 96 klip kelompok |
| `build_dataset.py` | dataset augmentasi → `artifacts/dataset_{train,val,test}.npz` |
| `model.py` | DS-CNN multi-tugas (kata + penutur + embedding) |
| `train.py` | training + pilih lebar 32/48/64 dari **validation saja** |
| `fewshot.py` | eksperimen enrol leave-one-speaker-out → ambang penutur |
| `int8_ref.py` | model INT8 referensi numpy (bit-exact dengan firmware) |
| `export_c.py` | kuantisasi, header C, `web_page.h`, verifikasi gcc C == numpy |
| `evaluate.py` | evaluasi akhir test (**sekali**) + banding pipeline klasik MP16 |
| `firmware_src/` | engine C (front-end FFT/mel/CMN, INT8, CAM, penutur) + `test_engine.c` |
| `esp32_kws_ai/` | sketch Arduino (I2S INMP441, WiFi/AP, HTTP + WebSocket, NVS, OLED) |

## Menjalankan pipeline

Python 3.11 + PyTorch 2.6 dalam env terpisah (jangan upgrade PyTorch global; MP12/MP13 bergantung padanya):
```bash
conda create -n kws-demo python=3.11 && conda activate kws-demo
pip install "torch==2.6.*" numpy scipy matplotlib scikit-learn websockets sounddevice

cd demo_ai
python build_dataset.py      # ±1 menit (M1). TTS 'lain' diambil dari artifacts/tts (di-commit)
python train.py              # CUDA / MPS (Mac Apple Silicon) / CPU otomatis
python fewshot.py            # LOSO 4 fold -> artifacts/fewshot_report.json (FEWSHOT_EPOCHS, default 20)
python export_c.py           # INT8 + header C + cek gcc/clang
python evaluate.py           # SEKALI SAJA -> artifacts/eval_report.json + figures/
python ../demo_web/mock_board.py --mic   # uji di laptop: http://localhost:8080
```
Urutan wajib: `export_c.py` membaca ambang dari `fewshot_report.json`; `evaluate.py` dan `mock_board.py` membaca
`export_report.json` + `int8_params.npz`.

## Hasil

Run 17 Nov 2026 di Mac M1 (MPS): **DS-CNN lebar 32, 20 epoch** (versi cepat untuk demo; few-shot LOSO dilewati,
ambang penutur memakai default 0,6). Model dilatih dari 96 rekaman asli kelompok + augmentasi.

| Metrik (test block 6, 1212 jendela) | DS-CNN INT8 (di ESP32) |
|---|---|
| Akurasi kata, 16 klip bersih | **16/16 = 1,00** (logreg MP16: 0,938) |
| Macro-F1 6 kelas | 0,785 |
| Recall kata perintah | 0,887 |
| Salah-terima (diam/lain dianggap perintah) | 0,171 |
| Penolakan kata asing / diam | 0,776 / 0,93 |
| Akurasi kata pada SNR 0–4 dB / 10–19 dB | 0,689 / 0,979 |
| Kenali penutur (klip bersih) | 0,812 |
| INT8 vs float agreement | 0,975 |
| C (firmware) == Python, end-to-end | **1,00** (clang) |
| Ukuran model | 15 KB, 3,5 juta MAC/inferensi, 8.486 parameter |
| Build ESP32-S3 / ESP32 | flash 51% / 52%, RAM global 121 KB / 123 KB |

Latency di board belum diukur (tidak ada board saat build); baca di Serial Monitor / dashboard setelah flash.

### Flash cepat
```bash
./flash_esp32.sh                          # compile + tampilkan port
./flash_esp32.sh /dev/cu.usbmodem1101     # ESP32-S3: compile + upload + serial monitor
BOARD=esp32 ./flash_esp32.sh /dev/cu.usbserial-0001   # ESP32 klasik
```
Dashboard: sambung ke WiFi `KWS-Lampu` (sandi `kelompok16`) lalu buka `http://192.168.4.1/`.

## Firmware ESP32

1. Arduino IDE 2 / `arduino-cli`, core **esp32 3.x**, library *Adafruit SSD1306* + *Adafruit GFX* (bila `USE_OLED 1`).
2. Jalankan `export_c.py` dulu: ia menulis `esp32_kws_ai/kws_model_int8.h`, `kws_tables.h`, dan `web_page.h`.
3. Atur `config.h`: WiFi (kosong = board membuat AP `KWS-Lampu`), pin INMP441 (SCK/WS/SD, **L/R ke GND**), LED lampu.
4. Board **ESP32S3 Dev Module**, Partition Scheme **No OTA (2MB APP/2MB SPIFFS)** (sketch ±1 MB; skema default juga cukup).
5. Upload, buka Serial Monitor 115200. Board mencetak alamat web. `MIC_SHIFT` dan `GATE_DB` disetel di `config.h`.

```bash
arduino-cli compile --fqbn esp32:esp32:esp32s3:PartitionScheme=no_ota demo_ai/esp32_kws_ai
arduino-cli upload  --fqbn esp32:esp32:esp32s3:PartitionScheme=no_ota -p /dev/cu.usbmodemXXXX demo_ai/esp32_kws_ai
```

## Catatan kejujuran (wajib dibaca sebelum mengutip angka)
- **Kelas `lain`** sebagian besar berasal dari **TTS en-US Windows SAPI** (Zira/David) yang membaca kata bahasa Indonesia,
  ditambah turunan rekaman kelompok (dibalik, disambung, terpotong) dan bunyi non-ucapan. Ini **bukan** ucapan manusia
  non-keyword, jadi tingkat penolakan kata asing dari orang sungguhan bisa lebih rendah.
- **Block 6 (test)** sudah pernah dibuka di siklus 1 (MP13) untuk model klasik. Di siklus 2 block 6 tidak pernah dipakai
  untuk training atau pemilihan model, tetapi ini bukan test set yang benar-benar "segar".
- **Hanya 4 penutur.** Pengenal penutur terbuka (open-set) lemah: ambang dari 4 fold LOSO, dan orang yang suaranya mirip
  anggota bisa lolos kunci suara. Kunci suara adalah fitur demo, **bukan** fitur keamanan.
- **Latency di `mock_board.py` = PC**, bukan ESP32. Footer web menampilkan `device` dari board; angka ESP32 hanya valid
  bila datang dari firmware di board.
- Pipeline siklus 2 dijalankan di **Mac M1 (MPS)**. Hasil training di GPU tidak bit-identik dengan CPU/CUDA, tetapi
  INT8 C == numpy tetap diverifikasi pada model yang diekspor.
