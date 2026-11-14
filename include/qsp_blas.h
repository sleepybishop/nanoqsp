#ifndef QSP_BLAS_H
#define QSP_BLAS_H

typedef enum {
  QSP_SIMD_SCALAR = 0,
  QSP_SIMD_SSE2,
  QSP_SIMD_AVX2,
  QSP_SIMD_AVX512,
  QSP_SIMD_NEON,
  QSP_SIMD_RVV,
  QSP_SIMD_COUNT
} qsp_simd_kind;

typedef struct {
  void (*copy)(double *restrict dst, const double *restrict src, int n);
  void (*zero)(double *restrict dst, int n);
  double (*dot)(const double *restrict x, const double *restrict y, int n);
  double (*norm1)(const double *restrict x, int n);
  void (*axpy)(double *restrict y, const double *restrict x, double alpha,
               int n);
  void (*gemv)(int m, int n, const double *restrict A, const double *restrict x,
               double *restrict y);
  const char *name;
  qsp_simd_kind kind;
} qsp_blas_ops;

extern qsp_blas_ops nanoqsp_blas;

/* thread-safe one-time runtime SIMD selection */
void nanoqsp_blas_init(void);

/* select a specific SIMD backend if supported on host; returns 1 on success */
int nanoqsp_blas_select(qsp_simd_kind kind);

#endif /* QSP_BLAS_H */
