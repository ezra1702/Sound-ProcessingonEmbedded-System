/* Harness uji gcc di PC untuk kws_engine.c.
 * mode "wave": in = [int32 n][n x 16000 int16]  -> out per sampel: xq[3920] int8, logits[6] f32, f[W] f32, e[E] f32, last[HxWxC] int8
 * mode "xq"  : in = [int32 n][n x 3920 int8]    -> out sama (front-end dilewati; menguji jaringan INT8 saja)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kws_engine.h"

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "pakai: %s wave|xq in.bin out.bin\n", argv[0]); return 1; }
    FILE *fi = fopen(argv[2], "rb"), *fo = fopen(argv[3], "wb");
    if (!fi || !fo) { fprintf(stderr, "gagal buka file\n"); return 1; }
    if (!kws_init()) { fprintf(stderr, "kws_init gagal\n"); return 1; }
    int32_t n = 0; if (fread(&n, 4, 1, fi) != 1) return 1;
    static int16_t pcm[KWS_WIN]; static int8_t xq[KWS_FRAMES * KWS_MELS];
    kws_result_t r;
    int wave = strcmp(argv[1], "wave") == 0;
    for (int i = 0; i < n; ++i) {
        if (wave) { if (fread(pcm, 2, KWS_WIN, fi) != KWS_WIN) return 2; kws_frontend(pcm, xq, NULL); }
        else if (fread(xq, 1, KWS_FRAMES * KWS_MELS, fi) != KWS_FRAMES * KWS_MELS) return 2;
        kws_infer(xq, &r);
        fwrite(xq, 1, KWS_FRAMES * KWS_MELS, fo);
        fwrite(r.logits, 4, KWS_N_CLASSES, fo);
        fwrite(r.f, 4, KWS_WIDTH, fo);
        fwrite(r.e, 4, KWS_EMB, fo);
        fwrite(r.last, 1, KWS_LAST_H * KWS_LAST_W * KWS_WIDTH, fo);
    }
    fclose(fi); fclose(fo);
    return 0;
}
