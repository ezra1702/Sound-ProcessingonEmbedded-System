# Mini Project 09 — Akuisisi Dataset Keyword, Metadata, dan Split Grup

**Anggota Kelompok**
| Nama | NIM |
|---|---|
| Christama Ezra Yudianto | 245150307111009 |
| Abdillah Fatah Suryanto | 245150300111008 |
| Ivang Arfandia | 245150307111003 |
| Jahzeel Erason Nicodemus | 245150300111002 |

## Tujuan
Membangun dataset 4 kelas (**nyala, mati, naik, turun**) dari **rekaman suara asli 4 anggota kelompok**:
- 4 penutur × 6 block × 4 kata = **96 WAV** (16 kHz, mono, 16-bit, 1,2 s).
- Metadata per file.
- Ekstraksi fitur 39-D + log-Mel dengan front-end beku MP6–MP8.
- **Split berbasis grup (acquisition block)**:

| Split | Block | Jumlah | Per penutur | Per kelas |
|---|---|---|---|---|
| train | 1–4 | 64 | 16 | 16 |
| validation | 5 | 16 | 4 | 4 |
| test | 6 | 16 | 4 | 4 |

Tidak ada overlap grup antar split. Status → `ready_for_mp10`.

## Sumber audio & penyimpangan dari buku
Buku meminta **1 penutur × 24 WAV**. Kelompok merekam **4 penutur × 24 WAV**, agar model tidak hanya mengenal satu suara
dan hasil evaluasi lebih bermakna. Unit split tetap **block**, sesuai buku, sehingga setiap penutur ada di train, validation, dan test.
Penyimpangan ini dicatat di `config/dataset_config.json` (`deviation_from_book`).

Alur perekaman:
1. Setiap anggota merekam lewat web perekam `Perekam_Dataset_MP09.html` (di-host di Vercel), dari mikrofon laptop/HP masing-masing.
2. Web memandu Rekaman 1–6 berisi 4 kata, dengan waktu bicara 10 detik.
3. Kata diverifikasi otomatis dengan speech recognition (bahasa Indonesia). Kata lain ditolak.
4. Rekaman yang terlalu pelan, clipping, terpotong, atau terlalu panjang juga ditolak.
5. Rekaman dipotong menjadi jendela 1,2 s di sekitar ucapan, lalu disimpan sebagai WAV ke Google Drive (Apps Script).
6. File diunduh ke `../dataset_rekaman/<Nama>/block_0N/<kata>_b0N.wav`.
7. MP09 (mode `group`) memvalidasi format (16 kHz, int16, mono) dan menyalinnya **byte-exact** ke `data/raw/mp09_dataset/<penutur>_<kata>_b0N.wav`.

Metadata mencatat `subject_id` (ezra / abdillah / ivang / jahzeel), `session_id`, dan `device_id = browser_mic_<penutur>`.

Mode lain tetap tersedia lewat `MP09_MODE`: `mic` (rekam langsung, prosedur buku), `tts` (versi awal sebelum data kelompok ada), dan `demo` (sinyal sintetis untuk uji skrip).

## Cara menjalankan
```bash
python run_all_mini_projects.py --only 9       # baca ../dataset_rekaman (mode default: group)
python run_all_mini_projects.py --start 10     # propagasi ulang MP10..MP16 setelah dataset berubah
```
Lokasi dataset dapat diganti dengan environment `MP09_GROUP_DIR`.

## File penting
| File | Isi |
|---|---|
| `mini_project_09.py` | Kode wajib (+ mode akuisisi group/mic/tts/demo) |
| `features/kws_frontend.py` | **Front-end beku sebagai modul** (`KWSFrontend`), bit-exact dengan MP9, dipakai MP13–MP16 & dashboard |
| `features/mp09_extension_dataset_audit.py` | **FITUR TAMBAHAN** |
| `data/raw/mp09_dataset/*.wav` | 96 WAV |
| `data/metadata/mp09_metadata.csv` | Metadata (label, group/block, subject, session, device, split) |
| `features/dataset/mp09_features_39d.npz` | Matriks fitur 96×39 + mask split + array subject |
| `features/logmel/mp09/*.npz` | log-Mel per sampel (input CNN MP12) |
| `splits/mp09_split.json` | Split grup |

## Hasil (catat hasil)
| Besaran | Nilai |
|---|---|
| Sampel / dimensi fitur | 96 / 39 (semua finite) |
| Distribusi kelas | 24 per kelas (train 16, val 4, test 4 per kelas) |
| Distribusi penutur | 24 per penutur (train 16, val 4, test 4) |
| Overlap grup train–val / train–test | tidak ada / tidak ada |
| Durasi rekaman | 1,2 s |
| Durasi aktif setelah VAD | 0,185–0,795 s (rata-rata 0,47 s) |
| Panjang log-Mel | 17–78 frame |

## FITUR TAMBAHAN
1. **Audit kualitas per WAV** (`results/metrics/mp09_extension_quality_audit.csv`): memeriksa terlalu pelan, clipping, kata menyentuh tepi, SNR rendah, dan DC offset.
   - **1 sampel ditandai**: `jahzeel_turun_b06`, karena terlalu pelan. Sampel tetap dipakai dan tercatat.
   - Estimasi SNR keseluruhan 11,0–41,7 dB (rata-rata 22,6 dB).
   - **SNR per penutur** (median): Ezra 29,9 · Abdillah 21,0 · Ivang 20,6 · Jahzeel 17,3 dB. Perbedaan kualitas mikrofon/ruangan terlihat jelas.
2. **Manifest reproducibility SHA-256** (103 file + versi library), di `mp09_extension_reproducibility_manifest.json`.
3. **Leakage guard**: tidak ada duplikat persis atau hampir-persis lintas split (cosine maks lintas split 0,939, antara dua ucapan Ivang di block berbeda), dan scaler di-fit train-only.
4. **Self-test front-end** `kws_frontend.py`: bit-exact (selisih maks 0,0) terhadap fitur MP9 pada 96 sampel.
5. **Fisher score (train-only)** untuk melihat fitur paling diskriminatif. Teratas: `mfcc04_std`, `mfcc01_mean`, `mfcc10_std`, `centroid_hz_mean`.
6. **Galeri log-Mel** kelas × penutur, serta boxplot SNR per penutur.

## Adaptasi
- `active_duration_s`, `subject_id`, dan `session_id` ditambahkan ke metadata (buku hanya mencatat durasi rekaman 1,2 s).
- Nama file diberi prefiks penutur (`ezra_nyala_b01.wav`). Label tetap dibaca dari bagian sebelum `_b`.
- Akuisisi dapat dipilih (`MP09_MODE`) dan idempoten.
