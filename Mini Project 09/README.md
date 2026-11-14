# Mini Project 09 — Akuisisi Dataset Keyword, Metadata, dan Split Grup

**Anggota Kelompok**
| Nama | NIM |
|---|---|
| Christama Ezra Yudianto | 245150307111009 |
| Abdillah Fatah Suryanto | 245150300111008 |
| Ivang Arfandia | 245150307111003 |
| Jahzeel Erason Nicodemus | 245150300111002 |

## Tujuan
Membangun dataset 4 kelas (**nyala, mati, naik, turun**) × 6 block rekaman = 24 WAV (16 kHz, mono, 16-bit, 1,2 s),
metadata, ekstraksi fitur 39-D + log-Mel memakai front-end beku MP6–MP8, dan **split berbasis grup (block)**:
train block 1–4 (16), validation block 5 (4), test block 6 (4) — tanpa overlap grup. Status → `ready_for_mp10`.

## ⚠️ Catatan penting tentang sumber audio
Pengerjaan dilakukan otomatis tanpa kesempatan merekam suara anggota kelompok, sehingga mode default adalah **`tts`**:
ucapan dibangkitkan dengan **Windows SAPI** (suara Zira & David, ejaan fonetik) lalu disimulasikan kanal per block
(laju bicara, speed perturbation, jarak/gain, reverb RT60, noise floor SNR 29–34 dB, hum 50 Hz, onset acak) agar
setiap block benar-benar berbeda seperti instruksi buku "ubah posisi mikrofon antar block".
Metadata mencatatnya dengan jujur: `subject_id = tts_zira/tts_david`, `device_id = sapi_tts_simulated_mic`.

**Merekam ulang dengan mikrofon (disarankan untuk laporan akhir):**
```bash
# hapus WAV lama lalu rekam 24 ucapan (diminta satu per satu)
del "Mini Project 09\data\raw\mp09_dataset\*.wav"
set MP09_MODE=mic
python run_all_mini_projects.py --only 9
python run_all_mini_projects.py --start 10      # propagasi ulang MP10..MP16
```
Mode lain: `MP09_MODE=demo` (sinyal sintetis untuk uji skrip).

## Cara menjalankan
```bash
python run_all_mini_projects.py --only 9     # WAV yang sudah ada dipakai ulang (reproducible)
```

## File penting
| File | Isi |
|---|---|
| `mini_project_09.py` | Kode wajib (+ mode akuisisi mic/tts/demo) |
| `features/kws_frontend.py` | **Front-end beku sebagai modul** (`KWSFrontend`) — bit-exact dengan MP9, dipakai MP13–MP16 & dashboard |
| `features/mp09_extension_dataset_audit.py` | **FITUR TAMBAHAN** |
| `data/raw/mp09_dataset/*.wav` | 24 WAV |
| `data/metadata/mp09_metadata.csv` | Metadata (label, group, subject, device, split) |
| `features/dataset/mp09_features_39d.npz` | Matriks fitur 24×39 + mask split |
| `features/logmel/mp09/*.npz` | log-Mel per sampel (input CNN MP12) |
| `splits/mp09_split.json` | Split grup |

## Hasil (catat hasil)
| Besaran | Nilai |
|---|---|
| Sampel / dimensi fitur | 24 / 39 (semua finite) |
| Distribusi kelas | 6 per kelas (train 4, val 1, test 1 per kelas) |
| Overlap grup train–val / train–test | tidak ada / tidak ada |
| Durasi rekaman | 1,2 s |
| Durasi aktif setelah VAD | 0,275–0,725 s (rata-rata 0,48 s) |
| Panjang log-Mel | 26–71 frame |

## FITUR TAMBAHAN
1. **Audit kualitas per WAV** (`results/metrics/mp09_extension_quality_audit.csv`): terlalu pelan, clipping, kata menyentuh tepi, SNR rendah, DC offset → **0 sampel ditandai**; estimasi SNR 19,8–27,9 dB.
2. **Manifest reproducibility SHA-256** (31 file) — `mp09_extension_reproducibility_manifest.json`.
3. **Leakage guard**: tidak ada duplikat persis/hampir-persis lintas split (cosine maks lintas split 0,796), scaler fit train-only.
4. **Self-test front-end** `kws_frontend.py`: bit-exact (selisih maks 0,0) terhadap fitur MP9 pada 24 sampel.
5. **Fisher score (train-only)** untuk melihat fitur paling diskriminatif (teratas: `mfcc02_std`, `mfcc05_mean`, …) + **galeri log-Mel** per kelas/block.

## Adaptasi
- `active_duration_s` ditambahkan ke metadata (buku hanya mencatat durasi rekaman 1,2 s).
- Akuisisi dibuat dapat dipilih (`MP09_MODE`) dan idempoten: WAV yang sudah ada tidak direkam ulang.
