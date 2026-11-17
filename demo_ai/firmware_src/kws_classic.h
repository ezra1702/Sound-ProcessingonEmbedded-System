/* kws_classic — jalur klasik MP6..MP16 di ESP32 (C99, tanpa Arduino; diuji di PC dengan clang/gcc).
 *
 *  Bab 5-6  : filter IIR elliptic high-pass MP6 (lfilter b/a, double, DF2T)
 *  Bab 7    : DC removal, normalisasi peak, framing 25/10 ms, VAD RMS + margin, trim, pre-emphasis, Hamming
 *  Bab 3, 8 : FFT 400 titik (mixed radix 4x4x5x5), RMS, ZCR, centroid, bandwidth, roll-off, log-Mel 26, MFCC 13,
 *             pitch autokorelasi -> vektor 39-D (identik dengan features/kws_frontend.py MP9/MP16)
 *  Bab 10   : 8 fitur RFE-8
 *  Bab 11,15: logistic regression INT8 (MP15)
 *  Bab 16   : gate penolakan non-keyword (Mahalanobis Ledoit-Wolf)
 *  Analisis : profil akustik penutur (Gaussian diagonal, statistik train) untuk menjelaskan "mengapa Ezra/Ivang/..."
 *
 * Memori: sinyal tidak disalin ke float; filter dijalankan ulang per pass (DC -> peak -> VAD -> fitur). RAM kerja < 8 KB.
 */
#ifndef KWS_CLASSIC_H
#define KWS_CLASSIC_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define KC_DIM 39
#define KC_N_CLASSES 4
#define KC_N_SPK 4
#define KC_N_DISP 5

typedef struct {
    float v39[KC_DIM];            /* vektor fitur 39-D (urutan feature_config MP8) */
    float score[KC_N_CLASSES];    /* skor logreg INT8 (urutan KC_CLASS_NAMES) */
    int   cls;                    /* indeks kelas klasik */
    float margin;                 /* skor terbaik - kedua */
    float ood;                    /* jarak Mahalanobis */
    int   reject;                 /* 1 = ditolak gate OOD (bukan keyword) */
    float prof_prob[KC_N_SPK];    /* profil akustik penutur (softmax), urutan KC_SPK_NAMES */
    int   prof_best;
    int   n_frames;               /* jumlah frame aktif setelah VAD trim */
    int   s0, s1;                 /* batas trim VAD (sampel) */
} kc_result_t;

extern const char *const KC_CLASS_NAMES[KC_N_CLASSES];
extern const char *const KC_SPK_NAMES[KC_N_SPK];
extern const char *const KC_DISP_NAMES[KC_N_DISP];   /* fitur yang ditampilkan di web (F0, intonasi, ...) */
extern const int KC_DISP_IDX[KC_N_DISP];
extern const float KC_DISP_MU[KC_N_SPK][KC_N_DISP];  /* rata-rata per penutur (train) */

/* Ekstraksi 39-D saja. pcm int16, n sampel @16 kHz. Return 1 bila sukses. */
int  kc_features(const int16_t *pcm, int n, float *v39, int *n_frames, int *s0, int *s1);
/* Pipeline lengkap: fitur -> RFE-8 -> logreg INT8 -> gate OOD -> profil penutur. */
int  kc_run(const int16_t *pcm, int n, kc_result_t *r);

#ifdef __cplusplus
}
#endif
#endif
