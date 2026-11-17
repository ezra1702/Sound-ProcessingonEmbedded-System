# Protokol Board ↔ Web Demo

- **HTTP**: `GET /` mengembalikan `index.html`.
- **WebSocket**: `ws://<board>/ws`, dengan pesan teks JSON (UTF-8) dan pesan biner untuk audio.

Waktu `t_ms` = milidetik sejak board menyala. Nama aksi dan state sama dengan `firmware/mp16_command_fsm.h`.

## Board → web

| `type` | Kapan | Field |
|---|---|---|
| `hello` | sekali, saat web tersambung | `device` (mis. `"ESP32-S3 DevKitC"`), `model`, `model_sha`, `payload_bytes`, `fs`, `ood_threshold`, `power`, `level`, `state` |
| `level` | ±10×/detik | `rms_db` (dBFS blok terakhir), `vad` (bool: endpoint sedang terbuka) |
| `state` | saat state berubah tanpa event (mis. COOLDOWN → IDLE) | `power`, `level`, `state` |
| `event` | setiap keputusan | lihat di bawah |
| `error` | opsional | `message` |

Field pesan `event`:

| Field | Nilai |
|---|---|
| `kind` | `keyword` · `reject` · `manual` · `failsafe` · `recover` |
| `label` | `nyala` / `mati` / `naik` / `turun` / `null` |
| `action` | `ON` · `OFF` · `LEVEL_UP` · `LEVEL_DOWN` · `IGNORED_BOUND` · `NONE` · `REJECT` · `FAILSAFE_OFF` · `RECOVER` |
| `power`, `level`, `state` | status setelah aksi (`state`: `IDLE` / `COOLDOWN` / `FAULT`) |
| `score` | margin logit top-1 − top-2 |
| `ood` | jarak Mahalanobis (ditolak bila > `ood_threshold`) |
| `latency_ms` | waktu fitur + inferensi **di board** |
| `source` | `board_mic` · `web_mic` · `web` · `replay` |
| `t_ms` | waktu board |

Contoh:
```json
{"type":"event","kind":"keyword","label":"naik","action":"LEVEL_UP","power":true,"level":6,
 "state":"COOLDOWN","score":4.47,"ood":6.39,"latency_ms":4.69,"source":"board_mic","t_ms":41520}
```

## Web → board

| Pesan | Arti |
|---|---|
| `{"type":"hello","client":"web_demo"}` | perkenalan (boleh diabaikan) |
| `{"type":"cmd","action":"ON"\|"OFF"\|"LEVEL_UP"\|"LEVEL_DOWN"}` | kontrol manual. Board menjalankan FSM tanpa cooldown, lalu membalas `event` dengan `kind:"manual"` |
| **biner** | audio dari mic HP: **PCM16 little-endian mono 16 kHz, 19.200 sampel (1,2 s)**. Board menjalankan pipeline penuh lalu membalas `event` dengan `source:"web_mic"` |

## Aturan state machine (sama dengan MP16)
- `nyala` → ON, `mati` → OFF.
- `naik` / `turun` hanya berlaku saat ON, dalam batas 0..10. Di luar batas, aksinya `IGNORED_BOUND`.
- Cooldown 0,8 s setelah aksi yang mengubah lampu. Selama cooldown, perintah suara berikutnya → `NONE`.
- Watchdog: bila tidak ada blok audio selama 500 ms → `FAILSAFE_OFF` (state `FAULT`, lampu OFF). Saat audio kembali → `RECOVER`.
