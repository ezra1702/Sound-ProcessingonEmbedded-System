# Mini Project 07 — Preprocessing Audio untuk Keyword Spotting

**Anggota Kelompok**
| Nama | NIM |
|---|---|
| Christama Ezra Yudianto | 245150307111009 |
| Abdillah Fatah Suryanto | 245150300111008 |
| Ivang Arfandia | 245150307111003 |
| Jahzeel Erason Nicodemus | 245150300111002 |

## Tujuan
Mengubah audio hasil filter final MP6 menjadi sinyal yang siap diekstraksi fiturnya: **DC removal → normalisasi peak →
framing 25 ms / hop 10 ms → VAD energi adaptif → spectral subtraction → pre-emphasis + Hamming window**.
Output dipakai MP8 (`features/intermediate/mp07_preprocessed_frames.npz`). Status: `ready_for_mp07` → `ready_for_mp08`.

## Cara menjalankan
```bash
# dari root repo (menyalin artefak MP06 -> MP07, membangun & mengeksekusi notebook)
python run_all_mini_projects.py --only 7
# atau langsung skripnya (artefak MP06 sudah ada di folder ini)
cd "Mini Project 07"
python mini_project_07.py
python features/mp07_extension_preprocessing_analysis.py   # FITUR TAMBAHAN
```
Notebook hasil eksekusi (berisi output & gambar): `Mini_Project_07.ipynb`.

## File penting
| File | Isi |
|---|---|
| `mini_project_07.py` | Kode wajib (sesuai buku, format sel `# %%`) |
| `features/mp07_extension_preprocessing_analysis.py` | **FITUR TAMBAHAN** |
| `config/preprocessing_config.json` | Parameter preprocessing (dipakai MP8–MP16) |
| `data/processed/mp07_preprocessing/nyala_001_preprocessed.wav` | Audio hasil preprocessing |
| `features/intermediate/mp07_preprocessed_frames.npz` | Frame siap fitur |
| `results/metrics/mp07_preprocessing_metrics.json` | Metrik wajib |
| `results/figures/mp07_preprocessing_overview.png`, `mp07_vad.png` | Gambar wajib |

## Hasil (catat hasil)
| Besaran | Nilai |
|---|---|
| Durasi input → output | 2,35 s → 0,615 s |
| Active ratio (VAD) | 0,258 |
| Jumlah frame input → output | 233 → 60 |
| RMS input → output | 0,1056 → 0,2061 |
| Ambang VAD (relatif 0,25 × RMS maks) | 0,0602 |
| Perubahan noise floor (proxy) | −5,99 dB |
| Sampel clipping output | 0 |

## Tabel validasi
| Pemeriksaan | Kriteria | Hasil |
|---|---|---|
| Status awal | `ready_for_mp07` | ✔ |
| Frame/hop | 400/160 sampel (25/10 ms @16 kHz) | ✔ |
| VAD membuang diam | active ratio < 1 | ✔ 0,258 |
| Tidak clipping | 0 sampel | ✔ |
| Status akhir | `ready_for_mp08` | ✔ |

## FITUR TAMBAHAN (di luar tugas wajib)
1. **Perbandingan normalisasi** none / peak 0,95 / RMS −20 dBFS (crest factor, headroom).
2. **VAD ganda energi + ZCR + hangover** dibanding VAD MP7: agreement 0,901; energi ucapan yang dipertahankan 99,4% (MP7) vs 100% (dual).
3. **Uji denoising terkontrol**: audio MP6 hampir tanpa derau aditif (≈ −80…−107 dB), sehingga perbandingan asli tidak informatif.
   Kami menambahkan white noise pada SNR terkontrol dan membandingkan **spectral subtraction** vs **Wiener decision-directed**:

   | SNR input | Spectral subtraction | Wiener DD |
   |---|---|---|
   | 0 dB | 4,92 dB | 11,83 dB |
   | 5 dB | 8,92 dB | 15,05 dB |
   | 10 dB | 13,51 dB | 18,66 dB |
   | 20 dB | 23,11 dB | 26,80 dB |
4. **Implementasi fixed-point Q15** DC-blocker + pre-emphasis untuk MCU: SQNR 60,7 dB vs float; header `firmware/mp07_preprocessing_q15.h`.
5. Metrik & gambar: `results/metrics/mp07_extension_preprocessing_metrics.json`, `results/figures/mp07_extension_preprocessing_analysis.png`.

## Adaptasi terhadap kode buku
- Path tidak lagi hard-coded; `ROOT` dihitung dari lokasi skrip sehingga bisa dijalankan dari notebook/terminal mana pun.
- Kunci status lama (`next_mini_project`, `next_input`) dibersihkan agar `project_status.json` konsisten.
