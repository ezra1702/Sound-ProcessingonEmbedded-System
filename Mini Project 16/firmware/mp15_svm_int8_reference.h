// MP15 FITUR TAMBAHAN: referensi jalur INT8 buku (linear SVM arsip MP11) - BUKAN model terpilih
#pragma once
#include <stdint.h>
#include <math.h>
static const float SVM_IN_SCALE = 0.0151333837f;
static const int32_t SVM_IDX[8] = {2, 3, 7, 8, 28, 29, 30, 31};
static const float SVM_MEAN[8] = {0.526658719f, 4.87777164f, 2.87890223f, -2.58045432f, 0.27257485f, 0.0974067861f, 1965.99218f, 871.798551f};
static const float SVM_SCALE[8] = {2.36671478f, 2.65812235f, 0.827549787f, 1.93496642f, 0.0601364703f, 0.0419211745f, 500.994766f, 379.860914f};
static const int8_t SVM_WQ[32] = {-40, 2, 127, -97, 84, 87, 100, 91, -21, -102, 127, 119, 30, -113, 22, -89, 34, -37, -66, -127, -40, 6, -36, -11, 23, 116, -109, 127, -34, 26, -42, 26};
static const float SVM_WS[4] = {0.00186276528f, 0.00235459819f, 0.00306426709f, 0.00254802055f};
static const float SVM_B[4] = {-0.458516217f, -0.417422295f, -0.397752653f, -0.384422482f};
static const uint8_t SVM_CLASS[4] = {0, 1, 2, 3};
static inline int svm_int8_predict(const float *x39){ int8_t zq[8]; int best=0; float bs=-1e30f;
  for(int i=0;i<8;++i){ float q=roundf(((x39[SVM_IDX[i]]-SVM_MEAN[i])/SVM_SCALE[i])/SVM_IN_SCALE); zq[i]=(int8_t)(q>127.f?127:(q<-127.f?-127:q)); }
  for(int c=0;c<4;++c){ int32_t acc=0; for(int i=0;i<8;++i) acc+=(int32_t)SVM_WQ[c*8+i]*zq[i];
    float s=(float)acc*SVM_WS[c]*SVM_IN_SCALE+SVM_B[c]; if(s>bs){bs=s;best=c;} }
  return SVM_CLASS[best]; }
