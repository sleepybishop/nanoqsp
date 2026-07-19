#include "nanoqsp_blas.h"
#include <math.h>

void v_copy(double *restrict dest, const double *restrict src, int n) {
  for (int i = 0; i < n; i++) {
    dest[i] = src[i];
  }
}

void v_zero(double *restrict dest, int n) {
  for (int i = 0; i < n; i++) {
    dest[i] = 0.0;
  }
}

double v_dot(const double *restrict x, const double *restrict y, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += x[i] * y[i];
  }
  return sum;
}

double v_norm1(const double *restrict x, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += fabs(x[i]);
  }
  return sum;
}

void v_axpy(double *restrict y, const double *restrict x, double alpha, int n) {
  if (alpha == 0.0)
    return;
  for (int i = 0; i < n; i++) {
    y[i] += alpha * x[i];
  }
}
