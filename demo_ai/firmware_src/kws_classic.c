/* kws_classic.c — lihat kws_classic.h. Referensi Python: Mini Project 16/features/kws_frontend.py + mp15 INT8 + gate MP16. */
#include "kws_classic.h"
#include "kws_classic_tables.h"
#include <math.h>
#include <string.h>

const char *const KC_CLASS_NAMES[KC_N_CLASSES] = KC_CLASS_NAMES_INIT;
const char *const KC_SPK_NAMES[KC_N_SPK] = KC_SPK_NAMES_INIT;
const char *const KC_DISP_NAMES[KC_N_DISP] = KC_DISP_NAMES_INIT;

/* ------------------------------------------------------------------ sumber sinyal ter-filter (MP6) + normalisasi (MP7) */
typedef struct {
    const int16_t *pcm; int n;       /* sinyal asli */
    double z[KC_FORD];               /* state DF2T lfilter */
    int i;                           /* indeks sampel berikut */
    double mean, gain;               /* DC & gain normalisasi (pass 2+) */
} src_t;

static void src_reset(src_t *s) { memset(s->z, 0, sizeof s->z); s->i = 0; }

/* scipy.signal.lfilter (direct form II transposed), a[0] = 1 */
static double src_filt(src_t *s) {
    double x = s->i < s->n ? (double)s->pcm[s->i] / 32767.0 : 0.0;
    s->i++;
    double y = KC_FB[0] * x + s->z[0];
    for (int k = 1; k < KC_FORD; ++k) s->z[k - 1] = KC_FB[k] * x - KC_FA[k] * y + (k < KC_FORD - 1 ? s->z[k] : 0.0);
    return y;
}
/* sampel ternormalisasi (setelah DC removal + peak); di luar sinyal asli = 0 (padding Python) */
static float src_norm(src_t *s) {
    int idx = s->i; double y = src_filt(s);
    return idx < s->n ? (float)((y - s->mean) * s->gain) : 0.0f;
}

/* ------------------------------------------------------------------ FFT 400 mixed radix (Bab 3) */
typedef struct { float r, i; } cpx;
static const int FAC400[] = {4, 4, 5, 5};

static void fft_rec(cpx *out, const cpx *in, int n, int stride, const int *fac) {
    int p = fac[0], m = n / p;
    if (m == 1) { for (int q = 0; q < p; ++q) out[q] = in[q * stride]; }
    else for (int q = 0; q < p; ++q) fft_rec(out + q * m, in + q * stride, m, stride * p, fac + 1);
    int tws = KC_N / n;                                    /* W_n^j = W_400^(j*tws) */
    cpx t[5];
    for (int k = 0; k < m; ++k) {
        for (int q = 0; q < p; ++q) {
            cpx a = out[q * m + k]; int j = (q * k * tws) % KC_N;
            float c = KC_COS400[j], sn = KC_SIN400[j];
            t[q].r = a.r * c - a.i * sn; t[q].i = a.r * sn + a.i * c;
        }
        for (int u = 0; u < p; ++u) {
            float sr = 0.f, si = 0.f;
            for (int q = 0; q < p; ++q) {
                int j = ((u * q) % p) * (KC_N / p);
                float c = KC_COS400[j], sn = KC_SIN400[j];
                sr += t[q].r * c - t[q].i * sn; si += t[q].r * sn + t[q].i * c;
            }
            out[u * m + k].r = sr; out[u * m + k].i = si;
        }
    }
}

/* ------------------------------------------------------------------ fitur per frame (Bab 8) */
typedef struct { double s, s2; int n; } acc_t;
static void acc_add(acc_t *a, double v) { a->s += v; a->s2 += v * v; a->n++; }
static float acc_mean(const acc_t *a) { return a->n ? (float)(a->s / a->n) : 0.f; }
static float acc_std(const acc_t *a) {
    if (!a->n) return 0.f; double m = a->s / a->n, v = a->s2 / a->n - m * m; return (float)sqrt(v > 0 ? v : 0);
}

static cpx s_in[KC_N], s_out[KC_N];
static float s_x[KC_N], s_pw[KC_NB];

static float pitch_hz(const float *x) {
    double mu = 0; for (int i = 0; i < KC_N; ++i) mu += x[i]; mu /= KC_N;
    double ac0 = 0; for (int i = 0; i < KC_N; ++i) { double d = x[i] - mu; ac0 += d * d; }
    if (ac0 <= 1e-12) return 0.f;
    int best = KC_PITCH_LO; double bv = -1e300;
    for (int lag = KC_PITCH_LO; lag <= KC_PITCH_HI; ++lag) {
        double s = 0; for (int i = 0; i + lag < KC_N; ++i) s += (x[i] - mu) * (x[i + lag] - mu);
        s /= (ac0 + 1e-12);
        if (s > bv) { bv = s; best = lag; }
    }
    return bv < KC_PITCH_CONF ? 0.f : (float)KC_FS / (float)best;
}

int kc_features(const int16_t *pcm, int n, float *v39, int *n_frames, int *s0_out, int *s1_out) {
    if (!pcm || n <= 0) return 0;
    src_t s = {pcm, n, {0}, 0, 0.0, 1.0};
    /* pass 1: DC (rata-rata sinyal ter-filter) */
    double sum = 0; src_reset(&s); for (int i = 0; i < n; ++i) sum += src_filt(&s);
    s.mean = sum / n;
    /* pass 2: peak */
    double mx = 0; src_reset(&s); for (int i = 0; i < n; ++i) { double d = fabs(src_filt(&s) - s.mean); if (d > mx) mx = d; }
    s.gain = KC_TARGET_PEAK / (mx + 1e-12);
    /* pass 3: VAD RMS frame (ambang relatif terhadap RMS maksimum) + margin */
    int L = n < KC_N ? KC_N : n;
    int nfr = (L - KC_N) / KC_H + 1;
    float rmsmax = 0.f; int first = -1, last = -1;
    static float rmsv[KC_MAX_FRAMES];
    if (nfr > KC_MAX_FRAMES) return 0;
    src_reset(&s);
    for (int i = 0; i < KC_N; ++i) s_x[i] = src_norm(&s);
    for (int f = 0; f < nfr; ++f) {
        if (f) { memmove(s_x, s_x + KC_H, sizeof(float) * (KC_N - KC_H)); for (int i = KC_N - KC_H; i < KC_N; ++i) s_x[i] = src_norm(&s); }
        double e = 0; for (int i = 0; i < KC_N; ++i) e += (double)s_x[i] * s_x[i];
        rmsv[f] = (float)sqrt(e / KC_N + 1e-12); if (rmsv[f] > rmsmax) rmsmax = rmsv[f];
    }
    for (int f = 0; f < nfr; ++f) if (rmsv[f] >= KC_VAD_FACTOR * rmsmax) { if (first < 0) first = f; last = f; }
    int s0 = 0, s1 = L;
    if (first >= 0) {
        int i0 = first - KC_MARGIN; if (i0 < 0) i0 = 0;
        int i1 = last + KC_MARGIN; if (i1 > nfr - 1) i1 = nfr - 1;
        s0 = i0 * KC_H; s1 = i1 * KC_H + KC_N; if (s1 > L) s1 = L;
    }
    int Lt = s1 - s0; int Lp = Lt < KC_N ? KC_N : Lt;
    int nfr2 = (Lp - KC_N) / KC_H + 1;
    /* pass 4: fitur pada segmen ter-trim */
    acc_t mf[KC_NMFCC], a_rms = {0}, a_zcr = {0}, a_cen = {0}, a_bw = {0}, a_rol = {0}, a_f0 = {0};
    memset(mf, 0, sizeof mf);
    int voiced = 0;
    src_reset(&s);
    for (int i = 0; i < s0; ++i) src_filt(&s);
    static float fr[KC_N];
    for (int i = 0; i < KC_N; ++i) { int idx = s0 + i; float v = src_norm(&s); fr[i] = idx < s1 ? v : 0.f; }
    for (int f = 0; f < nfr2; ++f) {
        if (f) {
            memmove(fr, fr + KC_H, sizeof(float) * (KC_N - KC_H));
            for (int i = KC_N - KC_H; i < KC_N; ++i) { int idx = s0 + f * KC_H + i; float v = src_norm(&s); fr[i] = idx < s1 ? v : 0.f; }
        }
        /* pre-emphasis per frame + Hamming */
        s_x[0] = fr[0] * KC_HAMM[0];
        for (int i = 1; i < KC_N; ++i) s_x[i] = (fr[i] - KC_ALPHA * fr[i - 1]) * KC_HAMM[i];
        double e = 0; int zc = 0;
        for (int i = 0; i < KC_N; ++i) { e += (double)s_x[i] * s_x[i]; if (i && (signbit(s_x[i]) != signbit(s_x[i - 1]))) zc++; }
        acc_add(&a_rms, sqrt(e / KC_N + 1e-12));
        acc_add(&a_zcr, (double)zc / (KC_N - 1));
        for (int i = 0; i < KC_N; ++i) { s_in[i].r = s_x[i]; s_in[i].i = 0.f; }
        fft_rec(s_out, s_in, KC_N, 1, FAC400);
        double den = 0, cen = 0;
        for (int k = 0; k < KC_NB; ++k) {
            float a = sqrtf(s_out[k].r * s_out[k].r + s_out[k].i * s_out[k].i) + 1e-12f;
            s_pw[k] = a * a; den += s_pw[k]; cen += (double)s_pw[k] * (k * KC_DF);
        }
        double tot = den; den += 1e-12; cen /= den;
        double bw = 0; for (int k = 0; k < KC_NB; ++k) { double d = k * KC_DF - cen; bw += s_pw[k] * d * d; }
        bw = sqrt(bw / den);
        double cum = 0; int ri = KC_NB - 1;
        for (int k = 0; k < KC_NB; ++k) { cum += s_pw[k]; if (cum >= 0.85 * tot) { ri = k; break; } }
        acc_add(&a_cen, cen); acc_add(&a_bw, bw); acc_add(&a_rol, ri * KC_DF);
        float lm[KC_NMEL];
        for (int m = 0; m < KC_NMEL; ++m) {
            double sm = 0; const float *w = KC_MEL_W + KC_MEL_OFS[m];
            for (int k = 0; k < KC_MEL_LEN[m]; ++k) sm += (double)w[k] * s_pw[KC_MEL_START[m] + k];
            lm[m] = (float)log(sm > 1e-12 ? sm : 1e-12);
        }
        for (int c = 0; c < KC_NMFCC; ++c) {
            double sm = 0; for (int m = 0; m < KC_NMEL; ++m) sm += (double)KC_DCT[c * KC_NMEL + m] * lm[m];
            acc_add(&mf[c], sm);
        }
        float f0 = pitch_hz(s_x);
        if (f0 > 0) { acc_add(&a_f0, f0); voiced++; }
    }
    int o = 0;
    for (int c = 0; c < KC_NMFCC; ++c) { v39[o++] = acc_mean(&mf[c]); v39[o++] = acc_std(&mf[c]); }
    const acc_t *sc[5] = {&a_rms, &a_zcr, &a_cen, &a_bw, &a_rol};
    for (int j = 0; j < 5; ++j) { v39[o++] = acc_mean(sc[j]); v39[o++] = acc_std(sc[j]); }
    v39[o++] = acc_mean(&a_f0); v39[o++] = acc_std(&a_f0); v39[o++] = (float)voiced / nfr2;
    for (int i = 0; i < KC_DIM; ++i) if (!isfinite(v39[i])) return 0;
    if (n_frames) *n_frames = nfr2;
    if (s0_out) *s0_out = s0;
    if (s1_out) *s1_out = s1;
    return 1;
}

/* ------------------------------------------------------------------ model (Bab 10, 11, 15, 16) */
int kc_run(const int16_t *pcm, int n, kc_result_t *r) {
    memset(r, 0, sizeof *r);
    if (!kc_features(pcm, n, r->v39, &r->n_frames, &r->s0, &r->s1)) return 0;
    /* RFE-8 -> standardisasi -> kuantisasi input INT8 (round half to even = numpy) -> logreg INT8 */
    int8_t zq[KC_SEL_N];
    for (int i = 0; i < KC_SEL_N; ++i) {
        float z = (r->v39[KC_SEL_IDX[i]] - KC_SEL_MEAN[i]) / KC_SEL_SCALE[i];
        float q = nearbyintf(z / KC_IN_SCALE);
        zq[i] = (int8_t)(q > 127.f ? 127 : (q < -127.f ? -127 : q));
    }
    int best = 0;
    for (int c = 0; c < KC_N_CLASSES; ++c) {
        int32_t acc = 0; for (int i = 0; i < KC_SEL_N; ++i) acc += (int32_t)KC_WQ[c * KC_SEL_N + i] * zq[i];
        r->score[c] = (float)acc * KC_WSCALE[c] * KC_IN_SCALE + KC_BIAS[c];
        if (r->score[c] > r->score[best]) best = c;
    }
    float second = -1e30f; for (int c = 0; c < KC_N_CLASSES; ++c) if (c != best && r->score[c] > second) second = r->score[c];
    r->cls = best; r->margin = r->score[best] - second;
    /* gate OOD: Mahalanobis Ledoit-Wolf pada z-score 39-D */
    float z[KC_DIM];
    for (int i = 0; i < KC_DIM; ++i) z[i] = (r->v39[i] - KC_OOD_MU[i]) / KC_OOD_SD[i] - KC_OOD_LOC[i];
    double d2 = 0;
    for (int i = 0; i < KC_DIM; ++i) { double t = 0; for (int j = 0; j < KC_DIM; ++j) t += (double)KC_OOD_PREC[i * KC_DIM + j] * z[j]; d2 += t * z[i]; }
    r->ood = (float)sqrt(d2 > 0 ? d2 : 0); r->reject = r->ood > KC_OOD_THR;
    /* profil akustik penutur: Gaussian diagonal (varian gabungan), softmax log-likelihood */
    double ll[KC_N_SPK], mxl = -1e300;
    for (int s = 0; s < KC_N_SPK; ++s) {
        double a = 0; for (int k = 0; k < KC_PROF_K; ++k) { double d = (r->v39[KC_PROF_IDX[k]] - KC_PROF_MU[s][k]) / KC_PROF_SD[k]; a -= 0.5 * d * d; }
        ll[s] = a; if (a > mxl) { mxl = a; r->prof_best = s; }
    }
    double zs = 0; for (int s = 0; s < KC_N_SPK; ++s) zs += exp(ll[s] - mxl);
    for (int s = 0; s < KC_N_SPK; ++s) r->prof_prob[s] = (float)(exp(ll[s] - mxl) / zs);
    return 1;
}
