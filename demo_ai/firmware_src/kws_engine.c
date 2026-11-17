/* Implementasi engine KWS INT8. Lihat kws_engine.h. Hasil integer identik dengan referensi numpy demo_ai/int8_ref.py. */
#include "kws_engine.h"
#include "kws_tables.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

const char *const KWS_CLASS_NAMES[KWS_N_CLASSES] = {"nyala", "mati", "naik", "turun", "diam", "lain"};

/* ------------------------------------------------------------------ front-end log-Mel */
static float s_re[512], s_im[512];

static void fft512(float *re, float *im) {
    for (int i = 0; i < 512; ++i) {               /* permutasi bit-reversal */
        int j = KWS_BITREV[i];
        if (j > i) { float t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (int len = 2; len <= 512; len <<= 1) {
        int half = len >> 1, step = 512 / len;
        for (int i = 0; i < 512; i += len) {
            for (int k = 0; k < half; ++k) {
                float wr = KWS_TW_COS[k * step], wi = KWS_TW_SIN[k * step];
                float xr = re[i + k + half], xi = im[i + k + half];
                float tr = xr * wr - xi * wi, ti = xr * wi + xi * wr;
                re[i + k + half] = re[i + k] - tr; im[i + k + half] = im[i + k] - ti;
                re[i + k] += tr; im[i + k] += ti;
            }
        }
    }
}

static float  *s_lm;                               /* log-Mel 98x40 float (heap) */
static int8_t *s_bufA, *s_bufB;                   /* buffer aktivasi ping-pong (heap) */

int kws_init(void) {
    if (s_lm) return 1;
    s_lm   = (float *)malloc(sizeof(float) * KWS_FRAMES * KWS_MELS);
    s_bufA = (int8_t *)malloc(KWS_C1_H * KWS_C1_W * KWS_WIDTH);
    s_bufB = (int8_t *)malloc(KWS_C1_H * KWS_C1_W * KWS_WIDTH);
    return s_lm && s_bufA && s_bufB;
}

static inline float pre_emph(const int16_t *pcm, int i, float mean) {
    /* DC removal + pre-emphasis 0,97 dihitung langsung dari PCM (tanpa buffer float 64 KB) */
    float cur = (float)pcm[i] / 32768.0f - mean;
    if (i == 0) return cur;
    float prev = (float)pcm[i - 1] / 32768.0f - mean;
    return cur - 0.97f * prev;
}

void kws_frontend(const int16_t *pcm, int8_t *xq, float *lm_out) {
    float *lm = s_lm;
    double acc = 0.0;
    for (int i = 0; i < KWS_WIN; ++i) acc += (float)pcm[i] / 32768.0f;
    float mean = (float)(acc / KWS_WIN);
    for (int t = 0; t < KWS_FRAMES; ++t) {
        for (int i = 0; i < 400; ++i) { s_re[i] = pre_emph(pcm, t * 160 + i, mean) * KWS_HAMMING[i]; s_im[i] = 0.0f; }
        for (int i = 400; i < 512; ++i) { s_re[i] = 0.0f; s_im[i] = 0.0f; }
        fft512(s_re, s_im);
        for (int m = 0; m < KWS_MELS; ++m) {
            float s = 0.0f; int b0 = KWS_MEL_START[m], n = KWS_MEL_LEN[m];
            const float *w = KWS_MEL_W + KWS_MEL_OFS[m];
            for (int k = 0; k < n; ++k) { int b = b0 + k; s += w[k] * (s_re[b] * s_re[b] + s_im[b] * s_im[b]); }
            lm[t * KWS_MELS + m] = logf(s + 1e-6f);
        }
    }
    for (int m = 0; m < KWS_MELS; ++m) {          /* CMN per band mel */
        double s = 0.0;
        for (int t = 0; t < KWS_FRAMES; ++t) s += lm[t * KWS_MELS + m];
        float mu = (float)(s / KWS_FRAMES);
        for (int t = 0; t < KWS_FRAMES; ++t) lm[t * KWS_MELS + m] -= mu;
    }
    for (int i = 0; i < KWS_FRAMES * KWS_MELS; ++i) {
        float q = nearbyintf(lm[i] / KWS_S_IN);
        xq[i] = (int8_t)(q > 127.f ? 127 : (q < -127.f ? -127 : q));
        if (lm_out) lm_out[i] = lm[i];
    }
}

/* ------------------------------------------------------------------ lapisan INT8 (HWC) */
static inline int8_t requant(int32_t acc, int32_t m0, int32_t rs) {
    int64_t v = ((int64_t)acc * (int64_t)m0 + ((int64_t)1 << (rs - 1))) >> rs;
    return (int8_t)(v < 0 ? 0 : (v > 127 ? 127 : v));
}

static void conv1(const int8_t *x, int8_t *out) {
    const int H = KWS_FRAMES, W = KWS_MELS;
    for (int ho = 0; ho < KWS_C1_H; ++ho)
        for (int wo = 0; wo < KWS_C1_W; ++wo)
            for (int co = 0; co < KWS_WIDTH; ++co) {
                int32_t acc = KWS_C1_B[co];
                const int8_t *w = KWS_C1_WT + co * KWS_C1_KH * KWS_C1_KW;
                for (int i = 0; i < KWS_C1_KH; ++i) {
                    int hi = ho * KWS_C1_SH - KWS_C1_PH + i; if (hi < 0 || hi >= H) continue;
                    for (int j = 0; j < KWS_C1_KW; ++j) {
                        int wi = wo * KWS_C1_SW - KWS_C1_PW + j; if (wi < 0 || wi >= W) continue;
                        acc += (int32_t)w[i * KWS_C1_KW + j] * (int32_t)x[hi * W + wi];
                    }
                }
                out[(ho * KWS_C1_W + wo) * KWS_WIDTH + co] = requant(acc, KWS_C1_M0[co], KWS_C1_RS[co]);
            }
}

static void dwconv(const int8_t *x, int H, int W, int s, int8_t *out, int Ho, int Wo,
                   const int8_t *wt, const int32_t *b, const int32_t *m0, const int32_t *rs) {
    const int C = KWS_WIDTH;
    for (int ho = 0; ho < Ho; ++ho)
        for (int wo = 0; wo < Wo; ++wo)
            for (int c = 0; c < C; ++c) {
                int32_t acc = b[c];
                for (int i = 0; i < 3; ++i) {
                    int hi = ho * s - 1 + i; if (hi < 0 || hi >= H) continue;
                    for (int j = 0; j < 3; ++j) {
                        int wi = wo * s - 1 + j; if (wi < 0 || wi >= W) continue;
                        acc += (int32_t)wt[c * 9 + i * 3 + j] * (int32_t)x[(hi * W + wi) * C + c];
                    }
                }
                out[(ho * Wo + wo) * C + c] = requant(acc, m0[c], rs[c]);
            }
}

static void pwconv(const int8_t *x, int HW, int8_t *out,
                   const int8_t *wt, const int32_t *b, const int32_t *m0, const int32_t *rs) {
    const int C = KWS_WIDTH;
    for (int p = 0; p < HW; ++p) {
        const int8_t *xi = x + p * C;
        for (int co = 0; co < C; ++co) {
            int32_t acc = b[co]; const int8_t *w = wt + co * C;
            for (int ci = 0; ci < C; ++ci) acc += (int32_t)w[ci] * (int32_t)xi[ci];
            out[p * C + co] = requant(acc, m0[co], rs[co]);
        }
    }
}

void kws_infer(const int8_t *xq, kws_result_t *r) {
    int8_t *a = s_bufA, *b = s_bufB;
    conv1(xq, a);
    int H = KWS_C1_H, W = KWS_C1_W;
    for (int k = 0; k < KWS_N_BLOCKS; ++k) {
        const kws_block_t *B = &KWS_BLOCKS[k];
        int Ho = (H - 1) / B->stride + 1, Wo = (W - 1) / B->stride + 1;
        dwconv(a, H, W, B->stride, b, Ho, Wo, B->dw_w, B->dw_b, B->dw_m0, B->dw_rs);
        pwconv(b, Ho * Wo, a, B->pw_w, B->pw_b, B->pw_m0, B->pw_rs);
        H = Ho; W = Wo;
    }
    r->last = a;
    for (int c = 0; c < KWS_WIDTH; ++c) {                      /* Global Average Pool */
        int32_t s = 0;
        for (int p = 0; p < H * W; ++p) s += a[p * KWS_WIDTH + c];
        r->f[c] = (float)s * KWS_GAP_SCALE;
    }
    float mx = -1e30f;
    for (int k = 0; k < KWS_N_CLASSES; ++k) {
        float z = KWS_KW_B[k];
        for (int c = 0; c < KWS_WIDTH; ++c) z += KWS_KW_W[k * KWS_WIDTH + c] * r->f[c];
        r->logits[k] = z; if (z > mx) mx = z;
    }
    float se = 0.f;
    for (int k = 0; k < KWS_N_CLASSES; ++k) { r->prob[k] = expf(r->logits[k] - mx); se += r->prob[k]; }
    for (int k = 0; k < KWS_N_CLASSES; ++k) r->prob[k] /= se;
    float n2 = 0.f;
    for (int j = 0; j < KWS_EMB; ++j) {
        float z = KWS_E_B[j];
        for (int c = 0; c < KWS_WIDTH; ++c) z += KWS_E_W[j * KWS_WIDTH + c] * r->f[c];
        r->e[j] = z; n2 += z * z;
    }
    n2 = sqrtf(n2) + 1e-9f;
    for (int j = 0; j < KWS_EMB; ++j) r->e[j] /= n2;
    int best = 0;
    for (int k = 1; k < KWS_N_CLASSES; ++k) if (r->prob[k] > r->prob[best]) best = k;
    r->kw = best; r->kw_prob = r->prob[best]; r->speaker = -1; r->speaker_cos = -1.f;
}

static float dotf(const float *a, const float *b, int n) { float s = 0.f; for (int i = 0; i < n; ++i) s += a[i] * b[i]; return s; }

void kws_decide(kws_result_t *r, const kws_user_t *users, int n_users, float spk_threshold) {
    int bs = -1; float bc = -2.f;
    for (int u = 0; u < n_users && u < KWS_MAX_USERS; ++u) {
        float c = dotf(r->e, users[u].spk, KWS_EMB); r->speaker_cos_all[u] = c;
        if (c > bc) { bc = c; bs = u; }
    }
    r->speaker_cos = bc; r->speaker = (bc >= spk_threshold) ? bs : -1;
    /* fusi prototipe kata (few-shot) untuk pengguna yang didaftarkan: p = 0,5*p_model + 0,5*p_proto pada 4 keyword */
    if (r->speaker >= 0 && users[r->speaker].has_kw && r->kw < 4) {
        float fn[KWS_WIDTH], n2 = 0.f;
        for (int c = 0; c < KWS_WIDTH; ++c) { fn[c] = r->f[c]; n2 += fn[c] * fn[c]; }
        n2 = sqrtf(n2) + 1e-9f;
        float pp[4], mx = -1e30f, se = 0.f, pk = 0.f;
        for (int k = 0; k < 4; ++k) { pp[k] = KWS_PROTO_TAU * dotf(fn, users[r->speaker].kw[k], KWS_WIDTH) / n2; if (pp[k] > mx) mx = pp[k]; }
        for (int k = 0; k < 4; ++k) { pp[k] = expf(pp[k] - mx); se += pp[k]; }
        for (int k = 0; k < 4; ++k) pk += r->prob[k];
        int best = 0; float bv = -1.f;
        for (int k = 0; k < 4; ++k) { float v = 0.5f * r->prob[k] + 0.5f * pk * pp[k] / se; if (v > bv) { bv = v; best = k; } }
        r->kw = best; r->kw_prob = bv;
    }
}

void kws_cam(const kws_result_t *r, int k, uint8_t *cam) {
    float c[KWS_CAM_H * KWS_CAM_W], mx = 1e-9f;
    for (int p = 0; p < KWS_CAM_H * KWS_CAM_W; ++p) {
        float s = 0.f;
        for (int ch = 0; ch < KWS_WIDTH; ++ch) s += KWS_KW_W[k * KWS_WIDTH + ch] * (float)r->last[p * KWS_WIDTH + ch];
        s *= KWS_S_LAST; c[p] = s > 0.f ? s : 0.f; if (c[p] > mx) mx = c[p];
    }
    for (int p = 0; p < KWS_CAM_H * KWS_CAM_W; ++p) cam[p] = (uint8_t)(255.f * c[p] / mx + 0.5f);
}

int kws_load_builtin_users(kws_user_t *users) {
    static const char *names[4] = {"Ezra", "Abdillah", "Ivang", "Jahzeel"};
    for (int u = 0; u < 4; ++u) {
        memset(&users[u], 0, sizeof(kws_user_t));
        strncpy(users[u].name, names[u], sizeof(users[u].name) - 1);
        memcpy(users[u].spk, KWS_SPK_PROTO + u * KWS_EMB, sizeof(float) * KWS_EMB);
    }
    return 4;
}
