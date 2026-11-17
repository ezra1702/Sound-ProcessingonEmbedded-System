/* Engine Keyword Spotting INT8 (DS-CNN multi-tugas) — C99 murni, dipakai firmware ESP32 dan uji gcc di PC.
 * Kelompok: Christama Ezra Yudianto, Abdillah Fatah Suryanto, Ivang Arfandia, Jahzeel Erason Nicodemus.
 *
 * Alur: PCM int16 1,0 s @16 kHz -> kws_frontend() -> log-Mel 98x40 int8 -> kws_infer() -> logit kata (6),
 *       embedding kata f (KWS_WIDTH), embedding penutur e (KWS_EMB), peta fitur terakhir (untuk CAM).
 */
#ifndef KWS_ENGINE_H
#define KWS_ENGINE_H
#include <stdint.h>
#include "kws_model_int8.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KWS_FS        16000
#define KWS_WIN       16000
#define KWS_FRAMES    98
#define KWS_MELS      40
#define KWS_N_CLASSES 6
#define KWS_CAM_H     KWS_LAST_H
#define KWS_CAM_W     KWS_LAST_W
#define KWS_MAX_USERS 8            /* 4 anggota (dari training) + maks 4 pengguna yang didaftarkan */

typedef struct {
    float logits[KWS_N_CLASSES];
    float prob[KWS_N_CLASSES];
    float f[KWS_WIDTH];            /* fitur GAP (embedding kata) */
    float e[KWS_EMB];              /* embedding penutur, ternormalisasi L2 */
    const int8_t *last;            /* peta fitur terakhir HWC (KWS_LAST_H x KWS_LAST_W x KWS_WIDTH) */
    int   kw;                      /* kelas kata akhir (setelah fusi prototipe bila ada) */
    float kw_prob;
    int   speaker;                 /* indeks pengguna, -1 = tidak dikenal */
    float speaker_cos;             /* kemiripan cosine terbaik */
    float speaker_cos_all[KWS_MAX_USERS];
} kws_result_t;

typedef struct {
    char  name[24];
    float spk[KWS_EMB];            /* prototipe penutur (ternormalisasi) */
    float kw[4][KWS_WIDTH];        /* prototipe kata (rata-rata f ternormalisasi), hanya pengguna enrol */
    uint8_t has_kw;
} kws_user_t;

extern const char *const KWS_CLASS_NAMES[KWS_N_CLASSES];

/* Alokasi buffer kerja (±100 KB untuk lebar 48). Panggil sekali di setup(); return 1 bila sukses. */
int  kws_init(void);
/* Front-end: pcm 16000 sampel int16 -> xq 98*40 int8 (baris = frame). lm_out (opsional, boleh NULL) = log-Mel float. */
void kws_frontend(const int16_t *pcm, int8_t *xq, float *lm_out);
/* Inferensi jaringan + kepala. */
void kws_infer(const int8_t *xq, kws_result_t *r);
/* Keputusan akhir: identifikasi penutur (cosine ke prototipe) + fusi prototipe kata untuk pengguna enrol. */
void kws_decide(kws_result_t *r, const kws_user_t *users, int n_users, float spk_threshold);
/* Class Activation Map kelas k -> cam[KWS_CAM_H*KWS_CAM_W] 0..255. */
void kws_cam(const kws_result_t *r, int k, uint8_t *cam);
/* Muat 4 anggota kelompok (prototipe hasil training) ke array users; return jumlah. */
int  kws_load_builtin_users(kws_user_t *users);

#ifdef __cplusplus
}
#endif
#endif
