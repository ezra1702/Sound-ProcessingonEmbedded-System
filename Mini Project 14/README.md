# Mini Project 14 — Menyusun dan Menguji Scheduler Streaming Real-Time

**Anggota Kelompok**
| Nama | NIM |
|---|---|
| Christama Ezra Yudianto | 245150307111009 |
| Abdillah Fatah Suryanto | 245150300111008 |
| Ivang Arfandia | 245150307111003 |
| Jahzeel Erason Nicodemus | 245150300111002 |

## Tujuan
Mengubah artefak MP13 menjadi **kontrak arsitektur streaming**: DMA double-buffer 160 sampel (10 ms), frame 25 ms, hop 10 ms,
context 96 frame (975 ms), inference stride 10 frame (100 ms), ring buffer 4 blok. Simulasi blok-per-blok dengan *virtual CPU*
(audio train `nyala_b01.wav` diulang 4×). Model/fitur/split tidak diubah. Status → `ready_for_mp15`.

## Cara menjalankan
```bash
python run_all_mini_projects.py --only 14
```

## Hasil (catat hasil)
| Besaran | Nilai |
|---|---|
| Blok DMA / inferensi | 480 / 39 (stream 4,8 s) |
| Compute per blok mean / p95 / max | 0,30 / **1,31** / 1,58 ms (periode 10 ms) |
| Antrean maks / overflow | 0 blok / **0** |
| Realtime condition | **True** |
| Inferensi classical / Tiny CNN (mean host) | 0,116 ms / 0,292 ms |
| Komponen RAM proxy terbesar | aktivasi CNN float32 (upper proxy) **78,0 KiB** |

Memory budget proxy: DMA double buffer 640 B · ring PCM 800 B · konteks log-Mel 9.984 B · state frame klasik 7.680 B ·
bobot CNN 5.360 B · model klasik serialized 1.857 B. Header `firmware/mp14_realtime_constants.h`:
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

Catatan: label *sanity-check* pada simulasi buku (mis. classical "turun" untuk audio nyala) wajar, karena front-end streaming buku
tidak melakukan VAD-trim & normalisasi per ucapan seperti saat training — prediksi ini memang hanya untuk cek aliran data.
Streaming dengan front-end yang konsisten dibahas di fitur tambahan (4).

## FITUR TAMBAHAN (`features/mp14_extension_realtime_analysis.py`)
1. **Profil per-stage host** (median): IIR/blok 0,005 ms · fitur/frame 0,134 ms · agregasi 39-D 0,357 ms · tree 0,058 ms · CNN 0,23 ms.
2. **Proyeksi MCU ESP32-S3 @240 MHz** (asumsi eksplisit: 4 siklus/MAC float, 1 siklus/MAC INT8; MAC dari MP8/MP10/MP12):

   | Pipeline | Blok terburuk | Utilisasi CPU rata-rata | Muat 10 ms? |
   |---|---|---|---|
   | A: front-end 39-D + Tiny CNN float | 16,2 ms | 27,3% | **tidak** (inferensi harus dipecah / INT8) |
   | B: front-end 39-D + tree | 1,23 ms | 12,3% | ya |
   | C: front-end terpilih MP10 + tree | 0,18 ms | **1,8%** | ya |
   | D: log-Mel + Tiny CNN INT8 | 3,93 ms | 5,6% | ya |
3. **CPU-load stress sweep** (compute diperlambat 1×–256×): overflow pertama pada **×32** (utilisasi ≈97%); margin aman ±16× terhadap host.
4. **Inference-stride sweep** 50/100/200/500 ms: utilisasi pipeline A 42/27/20/15% vs latensi deteksi terburuk 65/115/215/515 ms.
5. **Streaming KWS kontinu** (20 ucapan train+val disambung + jeda derau, jendela 975 ms tiap 100 ms, front-end beku):

   | Strategi | Hit | Substitusi | False alarm |
   |---|---|---|---|
   | raw tiap 100 ms | 13/20 | 7 | 58 (79,6/menit) |
   | energy gate 12 dB | 13/20 | 5 | 32 |
   | gate + debounce + refractory | 10/20 | 9 | 2 |
   | gate + smoothing + threshold + debounce | 10/20 | 9 | 3 |
   | **VAD-endpoint trigger** (1 inferensi per ucapan) | **20/20** | **0** | **0** |

   Endpoint trigger hanya menjalankan **20 dari 428** inferensi (hemat 95%) — cocok untuk MCU hemat daya.
6. **Ring buffer C** drop-oldest (`firmware/mp14_ring_buffer.h`, critical-section macro) dikompilasi gcc `-Wall -Wextra -Werror`;
   urutan blok & overflow (183) **identik** dengan model Python.
7. **Gantt chart** ISR / IIR / fitur / inferensi (durasi proyeksi ESP32) — memperlihatkan CNN float melewati batas blok.
8. Output: `results/metrics/mp14_extension_realtime_metrics.json`, `results/figures/mp14_extension_realtime_analysis.png`.

## Adaptasi
- `project_cfg['sampling_rate_hz']` (buku) → kunci di proyek ini `sample_rate_hz`; dibaca dengan fallback.
- `completed_mini_projects` & `sanity_labels` ditambahkan ke status/metrik untuk audit.
