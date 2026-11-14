#include "qsp_blas.h"
#include <math.h>
#include <stddef.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
    defined(_M_IX86)
#if defined(__GNUC__) || defined(__clang__)
#define QSP_ARCH_X86 1
#include <immintrin.h>
#endif
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)
#define QSP_ARCH_NEON 1
#include <arm_neon.h>
#elif defined(__riscv) && defined(__riscv_vector)
#define QSP_ARCH_RVV 1
#include <riscv_vector.h>
#endif

/* parameterized SIMD kernel generator for fixed-width vector ISAs */
#define QSP_DEFINE_KERNELS(SUFFIX, ATTR, VEC_T, LANES, ZERO, SET1, LOAD,       \
                           STORE, ADD, FMA, ABS, REDUCE)                       \
  ATTR static void qsp_copy_##SUFFIX(double *restrict dst,                     \
                                     const double *restrict src, int n) {      \
    int i = 0;                                                                 \
    for (; i <= n - (LANES); i += (LANES))                                     \
      STORE(&dst[i], LOAD(&src[i]));                                           \
    for (; i < n; i++)                                                         \
      dst[i] = src[i];                                                         \
  }                                                                            \
                                                                               \
  ATTR static void qsp_zero_##SUFFIX(double *restrict dst, int n) {            \
    VEC_T z = ZERO();                                                          \
    int i = 0;                                                                 \
    for (; i <= n - (LANES); i += (LANES))                                     \
      STORE(&dst[i], z);                                                       \
    for (; i < n; i++)                                                         \
      dst[i] = 0.0;                                                            \
  }                                                                            \
                                                                               \
  ATTR static double qsp_dot_##SUFFIX(const double *restrict x,                \
                                      const double *restrict y, int n) {       \
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
    for (; i <= n - (LANES); i += (LANES))                                     \
      acc = FMA(acc, LOAD(&x[i]), LOAD(&y[i]));                                \
    double sum = REDUCE(acc);                                                  \
    for (; i < n; i++)                                                         \
      sum += x[i] * y[i];                                                      \
    return sum;                                                                \
  }                                                                            \
                                                                               \
  ATTR static double qsp_norm1_##SUFFIX(const double *restrict x, int n) {     \
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
    for (; i <= n - (LANES); i += (LANES))                                     \
      acc = ADD(acc, ABS(LOAD(&x[i])));                                        \
    double sum = REDUCE(acc);                                                  \
    for (; i < n; i++)                                                         \
      sum += fabs(x[i]);                                                       \
    return sum;                                                                \
  }                                                                            \
                                                                               \
  ATTR static void qsp_axpy_##SUFFIX(                                          \
      double *restrict y, const double *restrict x, double alpha, int n) {     \
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
    for (; i <= n - (LANES); i += (LANES))                                     \
      STORE(&y[i], FMA(LOAD(&y[i]), a, LOAD(&x[i])));                          \
    for (; i < n; i++)                                                         \
      y[i] += alpha * x[i];                                                    \
  }                                                                            \
                                                                               \
  /* 4-row register-blocked dense matrix-vector product */                     \
  ATTR static void qsp_gemv_##SUFFIX(int m, int n, const double *restrict A,   \
                                     const double *restrict x,                 \
                                     double *restrict y) {                     \
    int i = 0;                                                                 \
    size_t stride = (size_t)n;                                                 \
    for (; i <= m - 4; i += 4) {                                               \
      const double *restrict r0 = &A[(size_t)i * stride];                      \
      const double *restrict r1 = r0 + stride;                                 \
      const double *restrict r2 = r1 + stride;                                 \
      const double *restrict r3 = r2 + stride;                                 \
      VEC_T acc0 = ZERO();                                                     \
      VEC_T acc1 = ZERO();                                                     \
      VEC_T acc2 = ZERO();                                                     \
      VEC_T acc3 = ZERO();                                                     \
      int j = 0;                                                               \
      for (; j <= n - (LANES); j += (LANES)) {                                 \
        VEC_T xv = LOAD(&x[j]);                                                \
        acc0 = FMA(acc0, LOAD(&r0[j]), xv);                                    \
        acc1 = FMA(acc1, LOAD(&r1[j]), xv);                                    \
        acc2 = FMA(acc2, LOAD(&r2[j]), xv);                                    \
        acc3 = FMA(acc3, LOAD(&r3[j]), xv);                                    \
      }                                                                        \
      double s0 = REDUCE(acc0);                                                \
      double s1 = REDUCE(acc1);                                                \
      double s2 = REDUCE(acc2);                                                \
      double s3 = REDUCE(acc3);                                                \
      for (; j < n; j++) {                                                     \
        double xj = x[j];                                                      \
        s0 += r0[j] * xj;                                                      \
        s1 += r1[j] * xj;                                                      \
        s2 += r2[j] * xj;                                                      \
        s3 += r3[j] * xj;                                                      \
      }                                                                        \
      y[i + 0] = s0;                                                           \
      y[i + 1] = s1;                                                           \
      y[i + 2] = s2;                                                           \
      y[i + 3] = s3;                                                           \
    }                                                                          \
    for (; i < m; i++)                                                         \
      y[i] = qsp_dot_##SUFFIX(&A[(size_t)i * stride], x, n);                   \
  }

/* portable scalar kernels */
#define SCALAR_ZERO() 0.0
#define SCALAR_SET1(v) (v)
#define SCALAR_LOAD(p) (*(p))
#define SCALAR_STORE(p, v) (*(p) = (v))
#define SCALAR_ADD(a, b) ((a) + (b))
#define SCALAR_FMA(acc, a, b) ((acc) + (a) * (b))
#define SCALAR_ABS(v) fabs(v)
#define SCALAR_REDUCE(v) (v)

QSP_DEFINE_KERNELS(scalar, , double, 1, SCALAR_ZERO, SCALAR_SET1, SCALAR_LOAD,
                   SCALAR_STORE, SCALAR_ADD, SCALAR_FMA, SCALAR_ABS,
                   SCALAR_REDUCE)

#if defined(QSP_ARCH_X86)

/* SSE2 kernels */
#define QSP_ATTR_SSE2 __attribute__((target("sse2")))
#define SSE2_FMA(acc, a, b) _mm_add_pd((acc), _mm_mul_pd((a), (b)))
#define SSE2_ABS(v) _mm_andnot_pd(_mm_set1_pd(-0.0), (v))

QSP_ATTR_SSE2 static inline double qsp_reduce_sse2(__m128d v) {
  return _mm_cvtsd_f64(_mm_add_sd(v, _mm_unpackhi_pd(v, v)));
}

QSP_DEFINE_KERNELS(sse2, QSP_ATTR_SSE2, __m128d, 2, _mm_setzero_pd, _mm_set1_pd,
                   _mm_loadu_pd, _mm_storeu_pd, _mm_add_pd, SSE2_FMA, SSE2_ABS,
                   qsp_reduce_sse2)

/* AVX2 + FMA kernels */
#define QSP_ATTR_AVX2 __attribute__((target("avx2,fma")))
#define AVX2_FMA(acc, a, b) _mm256_fmadd_pd((a), (b), (acc))
#define AVX2_ABS(v) _mm256_andnot_pd(_mm256_set1_pd(-0.0), (v))

QSP_ATTR_AVX2 static inline double qsp_reduce_avx2(__m256d v) {
  __m128d lo = _mm256_castpd256_pd128(v);
  __m128d hi = _mm256_extractf128_pd(v, 1);
  __m128d sum = _mm_add_pd(lo, hi);
  return _mm_cvtsd_f64(_mm_add_sd(sum, _mm_unpackhi_pd(sum, sum)));
}

QSP_DEFINE_KERNELS(avx2, QSP_ATTR_AVX2, __m256d, 4, _mm256_setzero_pd,
                   _mm256_set1_pd, _mm256_loadu_pd, _mm256_storeu_pd,
                   _mm256_add_pd, AVX2_FMA, AVX2_ABS, qsp_reduce_avx2)

/* AVX-512 kernels */
#define QSP_ATTR_AVX512 __attribute__((target("avx512f,avx512dq,avx512vl")))
#define AVX512_FMA(acc, a, b) _mm512_fmadd_pd((a), (b), (acc))
#define AVX512_ABS(v) _mm512_andnot_pd(_mm512_set1_pd(-0.0), (v))

QSP_ATTR_AVX512 static inline double qsp_reduce_avx512(__m512d v) {
  __m256d lo = _mm512_castpd512_pd256(v);
  __m256d hi = _mm512_extractf64x4_pd(v, 1);
  __m256d s256 = _mm256_add_pd(lo, hi);
  __m128d s_lo = _mm256_castpd256_pd128(s256);
  __m128d s_hi = _mm256_extractf128_pd(s256, 1);
  __m128d s128 = _mm_add_pd(s_lo, s_hi);
  return _mm_cvtsd_f64(_mm_add_sd(s128, _mm_unpackhi_pd(s128, s128)));
}

QSP_DEFINE_KERNELS(avx512, QSP_ATTR_AVX512, __m512d, 8, _mm512_setzero_pd,
                   _mm512_set1_pd, _mm512_loadu_pd, _mm512_storeu_pd,
                   _mm512_add_pd, AVX512_FMA, AVX512_ABS, qsp_reduce_avx512)

#elif defined(QSP_ARCH_NEON)

#define NEON_ZERO() vdupq_n_f64(0.0)
#define NEON_FMA(acc, a, b) vfmaq_f64((acc), (a), (b))
#define NEON_REDUCE(v) (vgetq_lane_f64((v), 0) + vgetq_lane_f64((v), 1))

QSP_DEFINE_KERNELS(neon, , float64x2_t, 2, NEON_ZERO, vdupq_n_f64, vld1q_f64,
                   vst1q_f64, vaddq_f64, NEON_FMA, vabsq_f64, NEON_REDUCE)

#elif defined(QSP_ARCH_RVV)

static void qsp_copy_rvv(double *restrict dst, const double *restrict src,
                         int n) {
  size_t vl;
  for (int i = 0; i < n; i += (int)vl) {
    vl = __riscv_vsetvl_e64m4((size_t)(n - i));
    vfloat64m4_t s = __riscv_vle64_v_f64m4(&src[i], vl);
    __riscv_vse64_v_f64m4(&dst[i], s, vl);
  }
}

static void qsp_zero_rvv(double *restrict dst, int n) {
  size_t vl;
  for (int i = 0; i < n; i += (int)vl) {
    vl = __riscv_vsetvl_e64m4((size_t)(n - i));
    vfloat64m4_t zero = __riscv_vfmv_v_f_f64m4(0.0, vl);
    __riscv_vse64_v_f64m4(&dst[i], zero, vl);
  }
}

static double qsp_dot_rvv(const double *restrict x, const double *restrict y,
                          int n) {
  size_t vlmax = __riscv_vsetvlmax_e64m4();
  vfloat64m4_t sum_vec = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
  size_t vl;
  for (int i = 0; i < n; i += (int)vl) {
    vl = __riscv_vsetvl_e64m4((size_t)(n - i));
    vfloat64m4_t xi = __riscv_vle64_v_f64m4(&x[i], vl);
    vfloat64m4_t yi = __riscv_vle64_v_f64m4(&y[i], vl);
    /* preserve partial sums in tail lanes for full-width reduction */
    sum_vec = __riscv_vfmacc_vv_f64m4_tu(sum_vec, xi, yi, vl);
  }
  vfloat64m1_t red = __riscv_vfmv_v_f_f64m1(0.0, 1);
  red = __riscv_vfredusum_vs_f64m4_f64m1(sum_vec, red, vlmax);
  return __riscv_vfmv_f_s_f64m1_f64(red);
}

static double qsp_norm1_rvv(const double *restrict x, int n) {
  size_t vlmax = __riscv_vsetvlmax_e64m4();
  vfloat64m4_t sum_vec = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
  size_t vl;
  for (int i = 0; i < n; i += (int)vl) {
    vl = __riscv_vsetvl_e64m4((size_t)(n - i));
    vfloat64m4_t xi = __riscv_vle64_v_f64m4(&x[i], vl);
    xi = __riscv_vfabs_v_f64m4(xi, vl);
    sum_vec = __riscv_vfadd_vv_f64m4_tu(sum_vec, sum_vec, xi, vl);
  }
  vfloat64m1_t red = __riscv_vfmv_v_f_f64m1(0.0, 1);
  red = __riscv_vfredusum_vs_f64m4_f64m1(sum_vec, red, vlmax);
  return __riscv_vfmv_f_s_f64m1_f64(red);
}

static void qsp_axpy_rvv(double *restrict y, const double *restrict x,
                         double alpha, int n) {
  if (alpha == 0.0)
    return;
  size_t vl;
  for (int i = 0; i < n; i += (int)vl) {
    vl = __riscv_vsetvl_e64m4((size_t)(n - i));
    vfloat64m4_t xi = __riscv_vle64_v_f64m4(&x[i], vl);
    vfloat64m4_t yi = __riscv_vle64_v_f64m4(&y[i], vl);
    yi = __riscv_vfmacc_vf_f64m4(yi, alpha, xi, vl);
    __riscv_vse64_v_f64m4(&y[i], yi, vl);
  }
}

static void qsp_gemv_rvv(int m, int n, const double *restrict A,
                         const double *restrict x, double *restrict y) {
  size_t vlmax = __riscv_vsetvlmax_e64m4();
  size_t stride = (size_t)n;
  int i = 0;
  for (; i <= m - 4; i += 4) {
    const double *restrict r0 = &A[(size_t)i * stride];
    const double *restrict r1 = r0 + stride;
    const double *restrict r2 = r1 + stride;
    const double *restrict r3 = r2 + stride;
    vfloat64m4_t a0 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    vfloat64m4_t a1 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    vfloat64m4_t a2 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    vfloat64m4_t a3 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    size_t vl;
    for (int j = 0; j < n; j += (int)vl) {
      vl = __riscv_vsetvl_e64m4((size_t)(n - j));
      vfloat64m4_t xv = __riscv_vle64_v_f64m4(&x[j], vl);
      a0 = __riscv_vfmacc_vv_f64m4_tu(a0, __riscv_vle64_v_f64m4(&r0[j], vl), xv,
                                      vl);
      a1 = __riscv_vfmacc_vv_f64m4_tu(a1, __riscv_vle64_v_f64m4(&r1[j], vl), xv,
                                      vl);
      a2 = __riscv_vfmacc_vv_f64m4_tu(a2, __riscv_vle64_v_f64m4(&r2[j], vl), xv,
                                      vl);
      a3 = __riscv_vfmacc_vv_f64m4_tu(a3, __riscv_vle64_v_f64m4(&r3[j], vl), xv,
                                      vl);
    }
    vfloat64m1_t zero1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    y[i + 0] = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredusum_vs_f64m4_f64m1(a0, zero1, vlmax));
    y[i + 1] = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredusum_vs_f64m4_f64m1(a1, zero1, vlmax));
    y[i + 2] = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredusum_vs_f64m4_f64m1(a2, zero1, vlmax));
    y[i + 3] = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredusum_vs_f64m4_f64m1(a3, zero1, vlmax));
  }
  for (; i < m; i++)
    y[i] = qsp_dot_rvv(&A[(size_t)i * stride], x, n);
}

#endif

#define QSP_MAKE_OPS(SUFFIX, NAME, KIND)                                       \
  ((qsp_blas_ops){.copy = qsp_copy_##SUFFIX,                                   \
                  .zero = qsp_zero_##SUFFIX,                                   \
                  .dot = qsp_dot_##SUFFIX,                                     \
                  .norm1 = qsp_norm1_##SUFFIX,                                 \
                  .axpy = qsp_axpy_##SUFFIX,                                   \
                  .gemv = qsp_gemv_##SUFFIX,                                   \
                  .name = (NAME),                                              \
                  .kind = (KIND)})

qsp_blas_ops nanoqsp_blas = {
    .copy = qsp_copy_scalar,
    .zero = qsp_zero_scalar,
    .dot = qsp_dot_scalar,
    .norm1 = qsp_norm1_scalar,
    .axpy = qsp_axpy_scalar,
    .gemv = qsp_gemv_scalar,
    .name = "scalar",
    .kind = QSP_SIMD_SCALAR,
};

static unsigned qsp_blas_state = 0;

static int qsp_get_ops(qsp_simd_kind kind, qsp_blas_ops *out) {
#if defined(QSP_ARCH_X86)
  __builtin_cpu_init();
#endif
  switch (kind) {
  case QSP_SIMD_SCALAR:
    *out = QSP_MAKE_OPS(scalar, "scalar", QSP_SIMD_SCALAR);
    return 1;
#if defined(QSP_ARCH_X86)
  case QSP_SIMD_SSE2:
    if (__builtin_cpu_supports("sse2")) {
      *out = QSP_MAKE_OPS(sse2, "sse2", QSP_SIMD_SSE2);
      return 1;
    }
    return 0;
  case QSP_SIMD_AVX2:
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
      *out = QSP_MAKE_OPS(avx2, "avx2+fma", QSP_SIMD_AVX2);
      return 1;
    }
    return 0;
  case QSP_SIMD_AVX512:
    if (__builtin_cpu_supports("avx512f") &&
        __builtin_cpu_supports("avx512dq") &&
        __builtin_cpu_supports("avx512vl")) {
      *out = QSP_MAKE_OPS(avx512, "avx512", QSP_SIMD_AVX512);
      return 1;
    }
    return 0;
#elif defined(QSP_ARCH_NEON)
  case QSP_SIMD_NEON:
    *out = QSP_MAKE_OPS(neon, "neon", QSP_SIMD_NEON);
    return 1;
#elif defined(QSP_ARCH_RVV)
  case QSP_SIMD_RVV:
    *out = QSP_MAKE_OPS(rvv, "rvv", QSP_SIMD_RVV);
    return 1;
#endif
  default:
    return 0;
  }
}

static qsp_blas_ops qsp_detect_best(void) {
  qsp_blas_ops ops = QSP_MAKE_OPS(scalar, "scalar", QSP_SIMD_SCALAR);
  qsp_simd_kind order[] = {QSP_SIMD_AVX512, QSP_SIMD_AVX2, QSP_SIMD_SSE2,
                           QSP_SIMD_NEON, QSP_SIMD_RVV};
  for (size_t i = 0; i < sizeof(order) / sizeof(order[0]); i++) {
    if (qsp_get_ops(order[i], &ops))
      return ops;
  }
  return ops;
}

void nanoqsp_blas_init(void) {
#if defined(__GNUC__) || defined(__clang__)
  if (__atomic_load_n(&qsp_blas_state, __ATOMIC_ACQUIRE) == 2)
    return;
  unsigned expected = 0;
  if (__atomic_compare_exchange_n(&qsp_blas_state, &expected, 1, 0,
                                  __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
    nanoqsp_blas = qsp_detect_best();
    __atomic_store_n(&qsp_blas_state, 2, __ATOMIC_RELEASE);
    return;
  }
  while (__atomic_load_n(&qsp_blas_state, __ATOMIC_ACQUIRE) != 2) {
  }
#else
  if (qsp_blas_state == 0) {
    nanoqsp_blas = qsp_detect_best();
    qsp_blas_state = 2;
  }
#endif
}

int nanoqsp_blas_select(qsp_simd_kind kind) {
  qsp_blas_ops ops;
  if (!qsp_get_ops(kind, &ops))
    return 0;
  nanoqsp_blas = ops;
#if defined(__GNUC__) || defined(__clang__)
  __atomic_store_n(&qsp_blas_state, 2, __ATOMIC_RELEASE);
#else
  qsp_blas_state = 2;
#endif
  return 1;
}
