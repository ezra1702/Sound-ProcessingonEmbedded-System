# Web Demo — Lampu Kendali Suara

Halaman demo untuk menunjukkan sistem keyword spotting (nyala / mati / naik / turun) yang berjalan di board.

| File | Isi |
|---|---|
| `index.html` | Halaman demo, satu file tanpa library eksternal, sehingga bisa disajikan langsung dari board tanpa internet |
| `mock_board.py` | Board tiruan di PC / Raspberry Pi. Menjalankan **pipeline asli** proyek: front-end beku MP9, model INT8 MP15, gate OOD dan state machine MP16 |
| `PROTOKOL.md` | Format pesan WebSocket antara board dan halaman. Firmware ESP32 nanti wajib mengikuti format ini |

## Cara mencoba sekarang (tanpa board)

**A. Mode simulasi.** Buka `index.html` di browser, lalu klik **Mode simulasi**. Tombol ucapan tiruan akan muncul.
Banner ungu dan label `SIM` di log menandai bahwa hasilnya bukan dari board. Mode ini untuk latihan saja, jangan dipakai saat demo resmi.

**B. Board tiruan dengan pipeline asli.** Jalankan dari folder proyek:
```bash
python demo_web/mock_board.py            # kontrol manual + suara dari web
python demo_web/mock_board.py --replay   # putar 16 WAV test bergiliran tiap 3 s (uji tampilan)
python demo_web/mock_board.py --mic      # mic PC/Pi, deteksi kontinu (endpoint trigger)
```
Lalu buka `http://localhost:8080`. Halaman langsung tersambung otomatis.

Dari HP, buka `http://<IP-PC>:8080` dengan HP tersambung ke WiFi yang sama (IP PC bisa dilihat dengan perintah `ipconfig`).
Bila tidak bisa dibuka, izinkan Python di Windows Firewall.

Pipeline board tiruan sudah diverifikasi memberi prediksi yang **sama dengan MP13 pada 16/16 WAV test** (ambang OOD 11,81, payload 164 B).

## Isi halaman
- **Lampu**: status ON/OFF, level 0–10, dan state machine (IDLE / COOLDOWN / FAULT).
- **Deteksi terakhir**: kata, aksi, margin keyakinan, jarak OOD, latency di board, dan sumber suara.
- **Level suara mikrofon** board secara live, dengan indikator suara terdeteksi.
- **Kontrol manual** ON/OFF/terang/redup, sebagai cadangan.
- **Bicara lewat HP**: HP merekam 1,2 s, mengirim PCM 16 kHz ke board, dan board yang mengenali kata.
- **Statistik sesi** dan **event log**, bisa diunduh sebagai JSONL untuk lampiran laporan.

## Catatan mic HP (HTTPS)
Browser hanya mengizinkan mikrofon di halaman **HTTPS** atau `localhost`. Bila halaman dibuka dari `http://<ip-board>`, tombol mic akan menampilkan peringatan. Solusinya:
- **Chrome Android/desktop**: buka `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, isi `http://<ip-board>`, pilih Enabled, lalu restart Chrome.
- Atau buka halaman di laptop lewat `http://localhost:8080`, bila server berjalan di laptop itu.

Mode utama demo (mic di board) tidak terpengaruh masalah ini.

## Kejujuran angka
Field `device` di pesan `hello` tampil di footer halaman. `mock_board.py` mengisinya dengan **"PC (board tiruan, BUKAN ESP32)"**,
sehingga latency yang tampil adalah latency PC. Angka ESP32 baru valid bila dikirim dari firmware yang berjalan di board.

## Siklus 2: model DS-CNN dan pengenal penutur

Default `mock_board.py` sekarang `--model dl` (DS-CNN INT8 dari `demo_ai/`). Artefak dibuat dulu dengan pipeline di
`demo_ai/README.md` (`build_dataset.py` → `train.py` → `fewshot.py` → `export_c.py`). Mode lama tetap ada: `--model classical`.

Panel tambahan di halaman (terisi bila event dari board membawa field siklus 2):
- **AI**: spectrogram log-Mel yang dilihat model + peta CAM, bar probabilitas 6 kelas, latency front-end vs jaringan,
  dan perbandingan dengan model klasik MP15 (hanya di `mock_board.py`).
- **Penutur**: siapa yang bicara + skor cosine tiap pengguna, tombol **kunci suara**.
- **Daftarkan pengguna**: 3× tiap kata (12 ucapan). Disimpan di NVS (board) / `demo_ai/artifacts/mock_users.npz` (mock).

Detail pesan ada di `PROTOKOL.md`.

