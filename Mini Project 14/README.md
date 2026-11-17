# Mini Project 14 — Menyusun dan Menguji Scheduler Streaming Real-Time

**Anggota Kelompok**
| Nama | NIM |
|---|---|
| Christama Ezra Yudianto | 245150307111009 |
| Abdillah Fatah Suryanto | 245150300111008 |
| Ivang Arfandia | 245150307111003 |
| Jahzeel Erason Nicodemus | 245150300111002 |

## Tujuan
Mengubah artefak MP13 menjadi **kontrak arsitektur streaming**:
- DMA double-buffer 160 sampel (10 ms), frame 25 ms, hop 10 ms.
- Context 96 frame (975 ms), inference stride 10 frame (100 ms), ring buffer 4 blok.

Simulasi berjalan blok demi blok dengan *virtual CPU*, memakai audio train `ezra_nyala_b01.wav` (diambil dari split train) yang diulang 4×.
Model, fitur, dan split tidak diubah. Status → `ready_for_mp15`.

## Cara menjalankan
```bash
python run_all_mini_projects.py --only 14
```

## Hasil (catat hasil)
| Besaran | Nilai |
|---|---|
| Blok DMA / inferensi | 480 / 39 (stream 4,8 s) |
| Compute per blok mean / p95 / max | 0,31 / **1,33** / 1,51 ms (periode 10 ms) |
| Antrean maks / overflow | 0 blok / **0** |
| Realtime condition | **True** |
| Inferensi classical (logreg) / Tiny CNN (mean host) | 0,105 ms / 0,296 ms |
| Komponen RAM proxy terbesar | aktivasi CNN float32 (upper proxy) **78,0 KiB** |

Memory budget proxy: DMA double buffer 640 B · ring PCM 800 B · konteks log-Mel 9.984 B · state frame klasik 7.680 B ·
bobot CNN 5.360 B · model klasik serialized 1.074 B.

Header `firmware/mp14_realtime_constants.h`:
`AUDIO_FS_HZ 16000, DMA_BLOCK_SAMPLES 160, DMA_BUFFER_COUNT 2, AUDIO_FRAME_SAMPLES 400, FEATURE_HOP_SAMPLES 160,
FEATURE_CONTEXT_FRAMES 96, INFERENCE_STRIDE_FRAMES 10, AUDIO_RING_CAPACITY_BLOCKS 4`.

| Pemeriksaan | Kriteria lulus | Hasil |
|---|---|---|
| Status awal | ready_for_mp14 | ✔ |
| DMA period | 10 ms | 10,0 ms |
| Context | 96 frame / 975 ms | 96 / 975,0 ms |
| Inference stride | 100 ms | 100,0 ms |
| Overflow count | 0 | 0 |
| Realtime condition | True | True |
| Model selection | False / deferred to MP15 | False / MP15 |
| Header C | tersedia | ✔ |
| Status akhir | ready_for_mp15 | ✔ |

**Catatan jujur tentang sanity-check:** pada simulasi buku, label *sanity-check* seluruh 39 jendela memprediksi *mati* untuk audio nyala.
- Penyebabnya, front-end streaming buku memakai jendela tetap 975 ms tanpa VAD-trim dan normalisasi per ucapan. Ini tidak sama dengan cara fitur training dihitung (klip 1,2 s, VAD, normalisasi peak).
- Prediksi ini memang hanya untuk mengecek aliran data, bukan akurasi.
- Streaming dengan front-end yang konsisten dibahas di fitur tambahan (5).

## FITUR TAMBAHAN (`features/mp14_extension_realtime_analysis.py`)
1. **Profil per-stage host** (median): IIR/blok 0,005 ms · fitur/frame 0,133 ms · agregasi 39-D 0,362 ms · logreg 0,050 ms · CNN 0,267 ms.
2. **Proyeksi MCU ESP32-S3 @240 MHz** (asumsi eksplisit: 4 siklus/MAC float, 1 siklus/MAC INT8; MAC dari MP8/MP10/MP11/MP12):

   | Pipeline | Blok terburuk | Utilisasi CPU rata-rata | Muat 10 ms? |
   |---|---|---|---|
   | A: front-end 39-D + Tiny CNN float | 16,2 ms | 27,3% | **tidak** (inferensi harus dipecah / INT8) |
   | B: front-end 39-D + classical (logreg) | 1,23 ms | 12,3% | ya |
   | C: front-end terpilih MP10 + classical | 0,17 ms | **1,7%** | ya |
   | D: log-Mel + Tiny CNN INT8 | 3,91 ms | 5,4% | ya |
3. **CPU-load stress sweep** (compute diperlambat 1×–256×): overflow pertama terjadi pada **×32** (utilisasi ≈ 84%), sedangkan ×16 masih aman.
4. **Inference-stride sweep** 50/100/200/500 ms: utilisasi pipeline A 42/27/20/15%, dengan latensi deteksi terburuk 65/115/215/515 ms.
5. **Streaming KWS kontinu**:
   - Input: seluruh **80 ucapan train+val dari 4 penutur** disambung dengan jeda derau. Derau jeda mengikuti derau latar rekaman tiap penutur.
   - Jendela 975 ms dihitung tiap 100 ms dengan front-end beku. Stream berdurasi 169 s.

   | Strategi | Hit | Substitusi | False alarm |
   |---|---|---|---|
   | raw tiap 100 ms | 33/80 | 34 | 325 (115,6/menit) |
   | energy gate 12 dB | 34/80 | 30 | 178 |
   | gate + debounce + refractory | 44/80 | 30 | 40 |
   | gate + smoothing + threshold + debounce | 48/80 | 22 | 33 |
   | **VAD-endpoint trigger** (1 inferensi per ucapan) | **57/80** | **4** | **0** |

   - Endpoint trigger hanya menjalankan **61 dari 392** inferensi (hemat 96%), sehingga cocok untuk MCU hemat daya.
   - **Sweep ambang endpoint** (gate 6/8/10/12 dB × end 3/6 dB, hanya train+val): gate 10/end 6 memberi 58 hit tetapi 1 false alarm. Konfigurasi 12/6 dB (0 false alarm) dipertahankan.
   - **Per penutur** [hit, substitusi, total]: Abdillah 18/0/20 · Jahzeel 18/1/20 · Ivang 11/3/20 · Ezra 10/0/20.
     Dugaan penyebab miss Ezra: rekamannya punya lantai derau paling rendah (≈ −68 dBFS), sehingga syarat endpoint absolut
     ("energi < noise floor + 6 dB") jarang tercapai sebelum jendela habis.
     Dugaan ini didukung MP16: setelah endpoint diberi syarat drop relatif 20 dB dan jendela 1,2 s, Ezra naik ke 23/24 dan total 92/96.
6. **Ring buffer C** drop-oldest (`firmware/mp14_ring_buffer.h`, critical-section macro), dikompilasi gcc `-Wall -Wextra -Werror`.
   Urutan blok & overflow (183) **identik** dengan model Python.
7. **Gantt chart** ISR / IIR / fitur / inferensi (durasi proyeksi ESP32), yang memperlihatkan CNN float melewati batas blok.
8. Output: `results/metrics/mp14_extension_realtime_metrics.json`, `results/figures/mp14_extension_realtime_analysis.png`.

## Adaptasi
- `project_cfg['sampling_rate_hz']` (buku) → kunci di proyek ini `sample_rate_hz`; dibaca dengan fallback.
- `completed_mini_projects` & `sanity_labels` ditambahkan ke status/metrik untuk audit.
