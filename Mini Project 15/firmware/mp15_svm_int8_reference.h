// MP15 FITUR TAMBAHAN: referensi jalur INT8 buku (linear SVM arsip MP11) - BUKAN model terpilih
#pragma once
#include <stdint.h>
#include <math.h>
static const float SVM_IN_SCALE = 0.0291919783f;
static const int32_t SVM_IDX[8] = {0, 4, 6, 7, 11, 12, 21, 26};
static const float SVM_MEAN[8] = {-3.71215049f, -3.82435837f, 1.19398804f, 3.01063671f, 1.45001238f, -0.0335881028f, 0.955705956f, 0.0395954548f};
static const float SVM_SCALE[8] = {4.18022037f, 3.68923748f, 2.69562483f, 1.15169474f, 0.494389732f, 1.44786538f, 0.304245014f, 0.0159747109f};
static const int8_t SVM_WQ[32] = {-87, -22, -29, 127, -32, 107, -10, -83, 67, 7, 108, 24, 93, -127, -48, 41, 75, -68, -127, 68, -39, -22, 71, 9, 35, 44, 21, -127, -30, 14, 6, -20};
static const float SVM_WS[4] = {0.0103177286f, 0.018562377f, 0.0248148791f, 0.0193759304f};
static const float SVM_B[4] = {-1.38364012f, -1.56277062f, -3.17419818f, -1.99162642f};
static const uint8_t SVM_CLASS[4] = {0, 1, 2, 3};
static inline int svm_int8_predict(const float *x39){ int8_t zq[8]; int best=0; float bs=-1e30f;
  for(int i=0;i<8;++i){ float q=roundf(((x39[SVM_IDX[i]]-SVM_MEAN[i])/SVM_SCALE[i])/SVM_IN_SCALE); zq[i]=(int8_t)(q>127.f?127:(q<-127.f?-127:q)); }
  for(int c=0;c<4;++c){ int32_t acc=0; for(int i=0;i<8;++i) acc+=(int32_t)SVM_WQ[c*8+i]*zq[i];
    float s=(float)acc*SVM_WS[c]*SVM_IN_SCALE+SVM_B[c]; if(s>bs){bs=s;best=c;} }
  return SVM_CLASS[best]; }
