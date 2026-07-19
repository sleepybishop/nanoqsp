#include "nanoqsp_blas.h"
#include <immintrin.h> /* AVX */
#include <math.h>

void v_copy(double *restrict dest, const double *restrict src, int n) {
  int i = 0;
  for (; i <= n - 4; i += 4) {
    _mm256_storeu_pd(&dest[i], _mm256_loadu_pd(&src[i]));
  }
  for (; i < n; i++) {
    dest[i] = src[i];
  }
}

void v_zero(double *restrict dest, int n) {
  __m256d z = _mm256_setzero_pd();
  int i = 0;
  for (; i <= n - 4; i += 4) {
    _mm256_storeu_pd(&dest[i], z);
  }
  for (; i < n; i++) {
    dest[i] = 0.0;
  }
}

double v_dot(const double *restrict x, const double *restrict y, int n) {
  __m256d acc = _mm256_setzero_pd();
  int i = 0;
  for (; i <= n - 4; i += 4) {
    __m256d xi = _mm256_loadu_pd(&x[i]);
    __m256d yi = _mm256_loadu_pd(&y[i]);
    acc = _mm256_add_pd(acc, _mm256_mul_pd(xi, yi));
  }
  double temp[4];
  _mm256_storeu_pd(temp, acc);
  double sum = temp[0] + temp[1] + temp[2] + temp[3];
  for (; i < n; i++) {
    sum += x[i] * y[i];
  }
  return sum;
}

double v_norm1(const double *restrict x, int n) {
  __m256d acc = _mm256_setzero_pd();
  __m256d sign_bit = _mm256_set1_pd(-0.0);
  int i = 0;
  for (; i <= n - 4; i += 4) {
    __m256d xi = _mm256_loadu_pd(&x[i]);
    __m256d abs_xi = _mm256_andnot_pd(sign_bit, xi);
    acc = _mm256_add_pd(acc, abs_xi);
  }
  double temp[4];
  _mm256_storeu_pd(temp, acc);
  double sum = temp[0] + temp[1] + temp[2] + temp[3];
  for (; i < n; i++) {
    sum += fabs(x[i]);
  }
  return sum;
}

void v_axpy(double *restrict y, const double *restrict x, double alpha, int n) {
  if (alpha == 0.0)
    return;
  __m256d a = _mm256_set1_pd(alpha);
  int i = 0;
  for (; i <= n - 4; i += 4) {
    __m256d xi = _mm256_loadu_pd(&x[i]);
    __m256d yi = _mm256_loadu_pd(&y[i]);
    _mm256_storeu_pd(&y[i], _mm256_add_pd(yi, _mm256_mul_pd(a, xi)));
  }
  for (; i < n; i++) {
    y[i] += alpha * x[i];
  }
}
