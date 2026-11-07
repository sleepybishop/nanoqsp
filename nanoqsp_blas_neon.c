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
  float64x2_t sum0 = vdupq_n_f64(0.0);
  float64x2_t sum1 = vdupq_n_f64(0.0);
  float64x2_t sum2 = vdupq_n_f64(0.0);
  float64x2_t sum3 = vdupq_n_f64(0.0);
  int i = 0;
  for (; i <= n - 8; i += 8) {
    sum0 = vfmaq_f64(sum0, vld1q_f64(&x[i]), vld1q_f64(&y[i]));
    sum1 = vfmaq_f64(sum1, vld1q_f64(&x[i + 2]), vld1q_f64(&y[i + 2]));
    sum2 = vfmaq_f64(sum2, vld1q_f64(&x[i + 4]), vld1q_f64(&y[i + 4]));
    sum3 = vfmaq_f64(sum3, vld1q_f64(&x[i + 6]), vld1q_f64(&y[i + 6]));
  }
  float64x2_t sum_vec = vaddq_f64(vaddq_f64(sum0, sum1), vaddq_f64(sum2, sum3));
  for (; i <= n - 2; i += 2) {
    sum_vec = vfmaq_f64(sum_vec, vld1q_f64(&x[i]), vld1q_f64(&y[i]));
  }
  double sum = vgetq_lane_f64(sum_vec, 0) + vgetq_lane_f64(sum_vec, 1);
  for (; i < n; i++) {
    sum += x[i] * y[i];
  }
  return sum;
}

double v_norm1(const double *restrict x, int n) {
  float64x2_t sum0 = vdupq_n_f64(0.0);
  float64x2_t sum1 = vdupq_n_f64(0.0);
  float64x2_t sum2 = vdupq_n_f64(0.0);
  float64x2_t sum3 = vdupq_n_f64(0.0);
  int i = 0;
  for (; i <= n - 8; i += 8) {
    sum0 = vaddq_f64(sum0, vabsq_f64(vld1q_f64(&x[i])));
    sum1 = vaddq_f64(sum1, vabsq_f64(vld1q_f64(&x[i + 2])));
    sum2 = vaddq_f64(sum2, vabsq_f64(vld1q_f64(&x[i + 4])));
    sum3 = vaddq_f64(sum3, vabsq_f64(vld1q_f64(&x[i + 6])));
  }
  float64x2_t sum_vec = vaddq_f64(vaddq_f64(sum0, sum1), vaddq_f64(sum2, sum3));
  for (; i <= n - 2; i += 2) {
    sum_vec = vaddq_f64(sum_vec, vabsq_f64(vld1q_f64(&x[i])));
  }
  double sum = vgetq_lane_f64(sum_vec, 0) + vgetq_lane_f64(sum_vec, 1);
  for (; i < n; i++) {
    sum += fabs(x[i]);
  }
  return sum;
}

void v_axpy(double *restrict y, const double *restrict x, double alpha, int n) {
  if (alpha == 0.0)
    return;
  float64x2_t a = vdupq_n_f64(alpha);
  int i = 0;
  for (; i <= n - 8; i += 8) {
    vst1q_f64(&y[i], vfmaq_f64(vld1q_f64(&y[i]), a, vld1q_f64(&x[i])));
    vst1q_f64(&y[i + 2],
              vfmaq_f64(vld1q_f64(&y[i + 2]), a, vld1q_f64(&x[i + 2])));
    vst1q_f64(&y[i + 4],
              vfmaq_f64(vld1q_f64(&y[i + 4]), a, vld1q_f64(&x[i + 4])));
    vst1q_f64(&y[i + 6],
              vfmaq_f64(vld1q_f64(&y[i + 6]), a, vld1q_f64(&x[i + 6])));
  }
  for (; i <= n - 2; i += 2) {
    vst1q_f64(&y[i], vfmaq_f64(vld1q_f64(&y[i]), a, vld1q_f64(&x[i])));
  }
  for (; i < n; i++) {
    y[i] += alpha * x[i];
  }
}
