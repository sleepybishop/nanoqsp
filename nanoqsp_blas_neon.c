#include "nanoqsp_blas.h"
#include <arm_neon.h>
#include <math.h>

void v_copy(double *restrict dest, const double *restrict src, int n) {
  int i = 0;
  for (; i <= n - 2; i += 2) {
    vst1q_f64(&dest[i], vld1q_f64(&src[i]));
  }
  for (; i < n; i++) {
    dest[i] = src[i];
  }
}

void v_zero(double *restrict dest, int n) {
  float64x2_t zero = vdupq_n_f64(0.0);
  int i = 0;
  for (; i <= n - 2; i += 2) {
    vst1q_f64(&dest[i], zero);
  }
  for (; i < n; i++) {
    dest[i] = 0.0;
  }
}

double v_dot(const double *restrict x, const double *restrict y, int n) {
  float64x2_t sum_vec = vdupq_n_f64(0.0);
  int i = 0;
  for (; i <= n - 2; i += 2) {
    float64x2_t xi = vld1q_f64(&x[i]);
    float64x2_t yi = vld1q_f64(&y[i]);
    sum_vec = vfmaq_f64(sum_vec, xi, yi);
  }
  double sum = vgetq_lane_f64(sum_vec, 0) + vgetq_lane_f64(sum_vec, 1);
  for (; i < n; i++) {
    sum += x[i] * y[i];
  }
  return sum;
}

double v_norm1(const double *restrict x, int n) {
  float64x2_t sum_vec = vdupq_n_f64(0.0);
  int i = 0;
  for (; i <= n - 2; i += 2) {
    float64x2_t xi = vld1q_f64(&x[i]);
    sum_vec = vaddq_f64(sum_vec, vabsq_f64(xi));
  }
  double sum = vgetq_lane_f64(sum_vec, 0) + vgetq_lane_f64(sum_vec, 1);
  for (; i < n; i++) {
    sum += fabs(x[i]);
  }
  return sum;
}

void v_axpy(double *restrict y, const double *restrict x, double alpha, int n) {
  if (alpha == 0.0) return;
  float64x2_t a = vdupq_n_f64(alpha);
  int i = 0;
  for (; i <= n - 2; i += 2) {
    float64x2_t xi = vld1q_f64(&x[i]);
    float64x2_t yi = vld1q_f64(&y[i]);
    vst1q_f64(&y[i], vfmaq_f64(yi, a, xi));
  }
  for (; i < n; i++) {
    y[i] += alpha * x[i];
  }
}
