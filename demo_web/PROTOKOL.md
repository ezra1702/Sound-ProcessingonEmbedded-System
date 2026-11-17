# Protokol Board ↔ Web Demo

- **HTTP**
  - `GET /` mengembalikan `index.html` (di firmware: tersimpan gzip di `web_page.h`).
- **WebSocket**: `ws://<board>/ws`, dengan pesan teks JSON (UTF-8) dan pesan biner untuk audio.

Waktu `t_ms` = milidetik sejak board menyala. Nama aksi dan state sama dengan `firmware/mp16_command_fsm.h`.
Siklus 2 (DS-CNN) **menambah** field dan pesan, tetapi tidak mengubah arti field siklus 1. Web lama tetap bisa membaca event baru.

## Board → web

| `type` | Kapan | Field |
|---|---|---|
| `hello` | sekali, saat web tersambung | `device`, `model`, `payload_bytes`, `fs`, `power`, `level`, `state`. Siklus 1: `model_sha`, `ood_threshold`. Siklus 2: `macs`, `classes`, `features`, `lock`, `users`, `spk_threshold`, `kw_min_prob`, `free_heap`, `psram` |
| `level` | ±10×/detik | `rms_db` (dBFS blok terakhir), `vad` (bool: endpoint sedang terbuka) |
| `state` | saat state berubah tanpa event (mis. COOLDOWN → IDLE) | `power`, `level`, `state` |
| `event` | setiap keputusan | lihat di bawah |
| `settings` | balasan `set` | `lock` (bool) |
| `users` | setelah enrol selesai / pengguna dihapus | `list`: `[{"name":"Ezra","enrolled":false}, …]`. `enrolled:false` = penutur bawaan (data train), `true` = didaftarkan few-shot |
| `enroll` | selama pendaftaran | lihat tabel enrol |
| `error` | opsional | `message` |

`features` di `hello` menyatakan kemampuan board: `probs`, `speaker`, `cam`, `spec`, `enroll`, `lock`.
Panel AI di web selalu tampil, tetapi hanya terisi bila event membawa field-nya, sehingga event firmware siklus 1 tetap bisa dibaca.

### Field pesan `event`

| Field | Nilai |
|---|---|
| `kind` | `keyword` · `reject` · `manual` · `failsafe` · `recover` |
| `label` | `nyala` / `mati` / `naik` / `turun` / `null`. Untuk `REJECT_SPEAKER`, label berisi kata yang dikenali tetapi tidak dijalankan |
| `action` | `ON` · `OFF` · `LEVEL_UP` · `LEVEL_DOWN` · `IGNORED_BOUND` · `NONE` · `REJECT` · **`REJECT_SPEAKER`** · `FAILSAFE_OFF` · `RECOVER` |
| `power`, `level`, `state` | status setelah aksi (`state`: `IDLE` / `COOLDOWN` / `FAULT`) |
| `score` | siklus 1: margin logit top-1 − top-2. Siklus 2: probabilitas kelas terpilih (0..1) |
| `ood` | siklus 1 saja: jarak Mahalanobis (ditolak bila > `ood_threshold`) |
| `latency_ms` | waktu fitur + inferensi **di board** |
| `source` | `board_mic` · `web_mic` · `web` · `replay` |
| `t_ms` | waktu board |

Field tambahan siklus 2 (hanya pada `kind` `keyword` / `reject` dari model DS-CNN):

| Field | Nilai |
|---|---|
| `probs` | array 6 probabilitas, urut `classes`: nyala, mati, naik, turun, diam, lain |
| `speaker` | nama penutur dikenal, atau `null` bila cosine terbaik < `spk_threshold` |
| `speaker_cos` | cosine embedding ke prototipe terbaik |
| `speaker_scores` | objek `{nama: cosine}` untuk semua pengguna |
| `lat_fe_ms`, `lat_nn_ms` | rincian latency: front-end (FFT, mel, CMN) dan jaringan INT8 |
| `spec_t`, `spec_f`, `spec_b64` | log-Mel INT8 input model, `spec_t × spec_f` byte (uint8 = int8 + 128), base64, urutan baris = waktu |
| `cam_h`, `cam_w`, `cam_b64` | peta Class Activation Map (uint8 0..255) untuk kelas terpilih, atau keyword terkuat bila ditolak |
| `classical` | hanya `mock_board.py`: hasil model klasik MP15 pada audio yang sama, `{"label", "reject", "ood"}` |
| `pred_without_gate` | hanya mode klasik `mock_board.py --model classical`: label sebelum gate OOD |

Aturan keputusan siklus 2:
1. Bila kelas terpilih `diam`/`lain`, atau probabilitasnya < `kw_min_prob` → `kind:"reject"`, `action:"REJECT"`.
2. Bila kunci suara aktif (`lock:true`) dan `speaker` = `null` → `kind:"reject"`, `action:"REJECT_SPEAKER"`.
3. Selain itu → `kind:"keyword"` dan aksi dari state machine.

Contoh (angka ilustrasi, bukan hasil ukur):
```json
{"type":"event","kind":"keyword","label":"naik","action":"LEVEL_UP","power":true,"level":6,"state":"COOLDOWN",
 "t_ms":41520,"source":"board_mic","score":0.973,"latency_ms":38.1,"lat_fe_ms":9.4,"lat_nn_ms":28.7,
 "probs":[0.004,0.006,0.973,0.008,0.002,0.007],"speaker":"Ivang","speaker_cos":0.812,
 "speaker_scores":{"Ezra":0.31,"Abdillah":0.22,"Ivang":0.812,"Jahzeel":0.40},
 "cam_h":25,"cam_w":10,"cam_b64":"…","spec_t":98,"spec_f":40,"spec_b64":"…"}
```

### Pesan `enroll`

| `state` | Field | Arti |
|---|---|---|
| `prompt` | `name`, `index`, `total`, `word` | minta pengguna mengucapkan `word` (contoh ke-`index` dari `total`) |
| `retry` | `reason` | contoh terakhir ditolak (terlalu pelan / tak terdengar). `prompt` berikutnya menyusul |
| `done` | `name` | prototipe penutur disimpan (NVS di board, `demo_ai/artifacts/mock_users.npz` di mock). Lalu dikirim `users` |
| `cancelled` | — | pendaftaran dibatalkan |

Selama pendaftaran, audio board/web **tidak** menjalankan perintah lampu.

## Web → board

| Pesan | Arti |
|---|---|
| `{"type":"hello","client":"web_demo"}` | perkenalan. Board membalas `hello` |
| `{"type":"cmd","action":"ON"\|"OFF"\|"LEVEL_UP"\|"LEVEL_DOWN"}` | kontrol manual. Board menjalankan FSM tanpa cooldown, lalu membalas `event` dengan `kind:"manual"` |
| `{"type":"set","lock":true\|false}` | nyalakan/matikan kunci suara. Balasan `settings` |
| `{"type":"enroll_start","name":"Budi"}` | mulai pendaftaran few-shot (3× tiap kata = 12 contoh). Balasan `enroll{state:"prompt"}` |
| `{"type":"enroll_cancel"}` | batalkan pendaftaran. Balasan `enroll{state:"cancelled"}` |
| `{"type":"user_delete","name":"Budi"}` | hapus pengguna hasil enrol (penutur bawaan tidak bisa dihapus). Balasan `users` |
| **biner** | audio dari mic HP: **PCM16 little-endian mono 16 kHz, 19.200 sampel (1,2 s)**. Board menjalankan pipeline penuh lalu membalas `event` dengan `source:"web_mic"` (atau melanjutkan enrol) |

## Aturan state machine (sama dengan MP16)
- `nyala` → ON, `mati` → OFF.
- `naik` / `turun` hanya berlaku saat ON, dalam batas 0..10. Di luar batas, aksinya `IGNORED_BOUND`.
- Cooldown 0,8 s setelah aksi yang mengubah lampu. Selama cooldown, perintah suara berikutnya → `NONE`.
- Watchdog: bila tidak ada blok audio selama 500 ms → `FAILSAFE_OFF` (state `FAULT`, lampu OFF). Saat audio kembali → `RECOVER`.
- `REJECT_SPEAKER` tidak mengubah lampu dan tidak memulai cooldown.
