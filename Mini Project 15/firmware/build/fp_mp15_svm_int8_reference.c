#include "mp15_svm_int8_reference.h"
int entry(const float *x){ return svm_int8_predict(x); }
