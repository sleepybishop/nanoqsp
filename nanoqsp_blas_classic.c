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
  double sum0 = 0.0;
  double sum1 = 0.0;
  double sum2 = 0.0;
  double sum3 = 0.0;
  int i = 0;
  for (; i <= n - 4; i += 4) {
    sum0 += x[i] * y[i];
    sum1 += x[i + 1] * y[i + 1];
    sum2 += x[i + 2] * y[i + 2];
    sum3 += x[i + 3] * y[i + 3];
  }
  double sum = (sum0 + sum1) + (sum2 + sum3);
  for (; i < n; i++)
    sum += x[i] * y[i];
  return sum;
}

double v_norm1(const double *restrict x, int n) {
  double sum0 = 0.0;
  double sum1 = 0.0;
  double sum2 = 0.0;
  double sum3 = 0.0;
  int i = 0;
  for (; i <= n - 4; i += 4) {
    sum0 += fabs(x[i]);
    sum1 += fabs(x[i + 1]);
    sum2 += fabs(x[i + 2]);
    sum3 += fabs(x[i + 3]);
  }
  double sum = (sum0 + sum1) + (sum2 + sum3);
  for (; i < n; i++)
    sum += fabs(x[i]);
  return sum;
}

void v_axpy(double *restrict y, const double *restrict x, double alpha, int n) {
  if (alpha == 0.0)
    return;
  int i = 0;
  for (; i <= n - 4; i += 4) {
    y[i] += alpha * x[i];
    y[i + 1] += alpha * x[i + 1];
    y[i + 2] += alpha * x[i + 2];
    y[i + 3] += alpha * x[i + 3];
  }
  for (; i < n; i++) {
    y[i] += alpha * x[i];
  }
}
