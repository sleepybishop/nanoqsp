#include "nanoqsp_blas.h"
#include <math.h>

#define NANOQSP_SIMD_IMPL(VEC_T, LANES, ZERO, SET1, LOAD, STORE, ADD, FMA,     \
                          ABS, REDUCE)                                         \
  void v_copy(double *restrict dest, const double *restrict src, int n) {      \
    int i = 0;                                                                 \
    for (; i <= n - (LANES); i += (LANES)) {                                   \
      STORE(&dest[i], LOAD(&src[i]));                                          \
    }                                                                          \
    for (; i < n; i++) {                                                       \
      dest[i] = src[i];                                                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  void v_zero(double *restrict dest, int n) {                                  \
    VEC_T z = ZERO();                                                          \
    int i = 0;                                                                 \
    for (; i <= n - (LANES); i += (LANES)) {                                   \
      STORE(&dest[i], z);                                                      \
    }                                                                          \
    for (; i < n; i++) {                                                       \
      dest[i] = 0.0;                                                           \
    }                                                                          \
  }                                                                            \
                                                                               \
  double v_dot(const double *restrict x, const double *restrict y, int n) {    \
    VEC_T acc0 = ZERO();                                                       \
    VEC_T acc1 = ZERO();                                                       \
    VEC_T acc2 = ZERO();                                                       \
    VEC_T acc3 = ZERO();                                                       \
    int i = 0;                                                                 \
    for (; i <= n - 4 * (LANES); i += 4 * (LANES)) {                           \
      acc0 = FMA(acc0, LOAD(&x[i]), LOAD(&y[i]));                              \
      acc1 = FMA(acc1, LOAD(&x[i + (LANES)]), LOAD(&y[i + (LANES)]));          \
      acc2 = FMA(acc2, LOAD(&x[i + 2 * (LANES)]), LOAD(&y[i + 2 * (LANES)]));  \
      acc3 = FMA(acc3, LOAD(&x[i + 3 * (LANES)]), LOAD(&y[i + 3 * (LANES)]));  \
    }                                                                          \
    VEC_T acc = ADD(ADD(acc0, acc1), ADD(acc2, acc3));                         \
    for (; i <= n - (LANES); i += (LANES)) {                                   \
      acc = FMA(acc, LOAD(&x[i]), LOAD(&y[i]));                                \
    }                                                                          \
    double sum = REDUCE(acc);                                                  \
    for (; i < n; i++) {                                                       \
      sum += x[i] * y[i];                                                      \
    }                                                                          \
    return sum;                                                                \
  }                                                                            \
                                                                               \
  double v_norm1(const double *restrict x, int n) {                            \
    VEC_T acc0 = ZERO();                                                       \
    VEC_T acc1 = ZERO();                                                       \
    VEC_T acc2 = ZERO();                                                       \
    VEC_T acc3 = ZERO();                                                       \
    int i = 0;                                                                 \
    for (; i <= n - 4 * (LANES); i += 4 * (LANES)) {                           \
      acc0 = ADD(acc0, ABS(LOAD(&x[i])));                                      \
      acc1 = ADD(acc1, ABS(LOAD(&x[i + (LANES)])));                            \
      acc2 = ADD(acc2, ABS(LOAD(&x[i + 2 * (LANES)])));                        \
      acc3 = ADD(acc3, ABS(LOAD(&x[i + 3 * (LANES)])));                        \
    }                                                                          \
    VEC_T acc = ADD(ADD(acc0, acc1), ADD(acc2, acc3));                         \
    for (; i <= n - (LANES); i += (LANES)) {                                   \
      acc = ADD(acc, ABS(LOAD(&x[i])));                                        \
    }                                                                          \
    double sum = REDUCE(acc);                                                  \
    for (; i < n; i++) {                                                       \
      sum += fabs(x[i]);                                                       \
    }                                                                          \
    return sum;                                                                \
  }                                                                            \
                                                                               \
  void v_axpy(double *restrict y, const double *restrict x, double alpha,      \
              int n) {                                                         \
    if (alpha == 0.0)                                                          \
      return;                                                                  \
    VEC_T a = SET1(alpha);                                                     \
    int i = 0;                                                                 \
    for (; i <= n - 4 * (LANES); i += 4 * (LANES)) {                           \
      STORE(&y[i], FMA(LOAD(&y[i]), a, LOAD(&x[i])));                          \
      STORE(&y[i + (LANES)],                                                   \
            FMA(LOAD(&y[i + (LANES)]), a, LOAD(&x[i + (LANES)])));             \
      STORE(&y[i + 2 * (LANES)],                                               \
            FMA(LOAD(&y[i + 2 * (LANES)]), a, LOAD(&x[i + 2 * (LANES)])));     \
      STORE(&y[i + 3 * (LANES)],                                               \
            FMA(LOAD(&y[i + 3 * (LANES)]), a, LOAD(&x[i + 3 * (LANES)])));     \
    }                                                                          \
    for (; i <= n - (LANES); i += (LANES)) {                                   \
      STORE(&y[i], FMA(LOAD(&y[i]), a, LOAD(&x[i])));                          \
    }                                                                          \
    for (; i < n; i++) {                                                       \
      y[i] += alpha * x[i];                                                    \
    }                                                                          \
  }

#if defined(NANOQSP_AVX)
#include <immintrin.h>

static inline __m256d avx_fma(__m256d acc, __m256d a, __m256d b) {
  return _mm256_add_pd(acc, _mm256_mul_pd(a, b));
}

static inline __m256d avx_abs(__m256d v) {
  return _mm256_andnot_pd(_mm256_set1_pd(-0.0), v);
}

static inline double avx_reduce(__m256d v) {
  double t[4];
  _mm256_storeu_pd(t, v);
  return (t[0] + t[1]) + (t[2] + t[3]);
}

NANOQSP_SIMD_IMPL(__m256d, 4, _mm256_setzero_pd, _mm256_set1_pd,
                  _mm256_loadu_pd, _mm256_storeu_pd, _mm256_add_pd, avx_fma,
                  avx_abs, avx_reduce)

#elif defined(NANOQSP_SSE)
#include <emmintrin.h>

static inline __m128d sse_fma(__m128d acc, __m128d a, __m128d b) {
  return _mm_add_pd(acc, _mm_mul_pd(a, b));
}

static inline __m128d sse_abs(__m128d v) {
  return _mm_andnot_pd(_mm_set1_pd(-0.0), v);
}

static inline double sse_reduce(__m128d v) {
  double t[2];
  _mm_storeu_pd(t, v);
  return t[0] + t[1];
}

NANOQSP_SIMD_IMPL(__m128d, 2, _mm_setzero_pd, _mm_set1_pd, _mm_loadu_pd,
                  _mm_storeu_pd, _mm_add_pd, sse_fma, sse_abs, sse_reduce)

#elif defined(NANOQSP_NEON)
#include <arm_neon.h>

#define NEON_ZERO() vdupq_n_f64(0.0)
#define NEON_REDUCE(v) (vgetq_lane_f64((v), 0) + vgetq_lane_f64((v), 1))

NANOQSP_SIMD_IMPL(float64x2_t, 2, NEON_ZERO, vdupq_n_f64, vld1q_f64, vst1q_f64,
                  vaddq_f64, vfmaq_f64, vabsq_f64, NEON_REDUCE)

#elif defined(NANOQSP_RVV)
#include <riscv_vector.h>

void v_copy(double *restrict dest, const double *restrict src, int n) {
  size_t vl;
  for (int i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e64m4(n - i);
    vfloat64m4_t s = __riscv_vle64_v_f64m4(&src[i], vl);
    __riscv_vse64_v_f64m4(&dest[i], s, vl);
  }
}

void v_zero(double *restrict dest, int n) {
  size_t vl;
  for (int i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e64m4(n - i);
    vfloat64m4_t zero = __riscv_vfmv_v_f_f64m4(0.0, vl);
    __riscv_vse64_v_f64m4(&dest[i], zero, vl);
  }
}

double v_dot(const double *restrict x, const double *restrict y, int n) {
  size_t vlmax = __riscv_vsetvlmax_e64m4();
  vfloat64m4_t sum_vec = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
  size_t vl;
  for (int i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e64m4(n - i);
    vfloat64m4_t xi = __riscv_vle64_v_f64m4(&x[i], vl);
    vfloat64m4_t yi = __riscv_vle64_v_f64m4(&y[i], vl);
    /* preserve partial sums in tail lanes for full-width reduction */
    sum_vec = __riscv_vfmacc_vv_f64m4_tu(sum_vec, xi, yi, vl);
  }

  vfloat64m1_t red = __riscv_vfmv_v_f_f64m1(0.0, 1);
  red = __riscv_vfredusum_vs_f64m4_f64m1(sum_vec, red, vlmax);
  return __riscv_vfmv_f_s_f64m1_f64(red);
}

double v_norm1(const double *restrict x, int n) {
  size_t vlmax = __riscv_vsetvlmax_e64m4();
  vfloat64m4_t sum_vec = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
  size_t vl;
  for (int i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e64m4(n - i);
    vfloat64m4_t xi = __riscv_vle64_v_f64m4(&x[i], vl);
    xi = __riscv_vfabs_v_f64m4(xi, vl);
    /* preserve partial sums in tail lanes for full-width reduction */
    sum_vec = __riscv_vfadd_vv_f64m4_tu(sum_vec, sum_vec, xi, vl);
  }

  vfloat64m1_t red = __riscv_vfmv_v_f_f64m1(0.0, 1);
  red = __riscv_vfredusum_vs_f64m4_f64m1(sum_vec, red, vlmax);
  return __riscv_vfmv_f_s_f64m1_f64(red);
}

void v_axpy(double *restrict y, const double *restrict x, double alpha, int n) {
  if (alpha == 0.0)
    return;
  size_t vl;
  for (int i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e64m4(n - i);
    vfloat64m4_t xi = __riscv_vle64_v_f64m4(&x[i], vl);
    vfloat64m4_t yi = __riscv_vle64_v_f64m4(&y[i], vl);
    yi = __riscv_vfmacc_vf_f64m4(yi, alpha, xi, vl);
    __riscv_vse64_v_f64m4(&y[i], yi, vl);
  }
}

#else

#define CLASSIC_ZERO() 0.0
#define CLASSIC_SET1(v) (v)
#define CLASSIC_LOAD(p) (*(p))
#define CLASSIC_STORE(p, v) (*(p) = (v))
#define CLASSIC_ADD(a, b) ((a) + (b))
#define CLASSIC_FMA(acc, a, b) ((acc) + (a) * (b))
#define CLASSIC_ABS(v) fabs(v)
#define CLASSIC_REDUCE(v) (v)

NANOQSP_SIMD_IMPL(double, 1, CLASSIC_ZERO, CLASSIC_SET1, CLASSIC_LOAD,
                  CLASSIC_STORE, CLASSIC_ADD, CLASSIC_FMA, CLASSIC_ABS,
                  CLASSIC_REDUCE)

#endif
