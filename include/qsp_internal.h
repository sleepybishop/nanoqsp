#ifndef QSP_INTERNAL_H
#define QSP_INTERNAL_H

#include "nanoqsp.h"
#include "qsp_arena.h"
#include "qsp_blas.h"
#include <math.h>
#include <stddef.h>

#define NANOQSP_MAX_DOUBLES 268435455LL

typedef struct {
  int rows;
  int cols;
  const double *dense;
  const NanoqspCSR *csr;
} qsp_mat;

static inline qsp_mat qsp_mat_dense(int rows, int cols, const double *dense) {
  qsp_mat m = {rows, cols, dense, NULL};
  return m;
}

static inline qsp_mat qsp_mat_csr(int rows, int cols, const NanoqspCSR *csr) {
  qsp_mat m = {rows, cols, NULL, csr};
  return m;
}

static inline double qsp_clamp(double val, double lb, double ub) {
  if (val < lb)
    return lb;
  if (val > ub)
    return ub;
  return val;
}

static inline double qsp_vec_norm_inf(int n, const double *restrict v) {
  double max_val = 0.0;
  for (int i = 0; i < n; i++) {
    double val = fabs(v[i]);
    if (val > max_val)
      max_val = val;
  }
  return max_val;
}

static inline void qsp_mat_vec(const qsp_mat *M, const double *restrict x,
                               double *restrict y) {
  if (M->dense != NULL) {
    nanoqsp_blas.gemv(M->rows, M->cols, M->dense, x, y);
    return;
  }
  const NanoqspCSR *S = M->csr;
  for (int i = 0; i < S->n; i++) {
    double sum = 0.0;
    int start = S->row_ptr[i];
    int end = S->row_ptr[i + 1];
    for (int idx = start; idx < end; idx++)
      sum += S->values[idx] * x[S->col_indices[idx]];
    y[i] = sum;
  }
}

static inline void qsp_mat_tvec_add(const qsp_mat *M, const double *restrict v,
                                    double alpha, double *restrict y) {
  if (alpha == 0.0)
    return;
  if (M->dense != NULL) {
    for (int i = 0; i < M->rows; i++) {
      double a = alpha * v[i];
      if (a != 0.0)
        nanoqsp_blas.axpy(y, &M->dense[(size_t)i * (size_t)M->cols], a,
                          M->cols);
    }
    return;
  }
  const NanoqspCSR *S = M->csr;
  for (int i = 0; i < M->rows; i++) {
    double a = alpha * v[i];
    if (a == 0.0)
      continue;
    int start = S->row_ptr[i];
    int end = S->row_ptr[i + 1];
    for (int idx = start; idx < end; idx++)
      y[S->col_indices[idx]] += S->values[idx] * a;
  }
}

static inline void qsp_mat_tvec(const qsp_mat *M, const double *restrict v,
                                double *restrict y) {
  nanoqsp_blas.zero(y, M->cols);
  qsp_mat_tvec_add(M, v, 1.0, y);
}

static inline void qsp_mat_extract_diag(const qsp_mat *D,
                                        double *restrict diag) {
  int n = D->rows;
  if (D->dense != NULL) {
    size_t stride = (size_t)n;
    for (int i = 0; i < n; i++)
      diag[i] = D->dense[(size_t)i * stride + (size_t)i];
    return;
  }
  const NanoqspCSR *S = D->csr;
  for (int i = 0; i < n; i++) {
    double d_ii = 0.0;
    int start = S->row_ptr[i];
    int end = S->row_ptr[i + 1];
    for (int idx = start; idx < end; idx++) {
      if (S->col_indices[idx] == i)
        d_ii += S->values[idx];
    }
    diag[i] = d_ii;
  }
}

static inline double qsp_mat_norm_inf(const qsp_mat *M) {
  double max_row = 0.0;
  if (M->dense != NULL) {
    for (int i = 0; i < M->rows; i++) {
      double s =
          nanoqsp_blas.norm1(&M->dense[(size_t)i * (size_t)M->cols], M->cols);
      if (s > max_row)
        max_row = s;
    }
    return max_row;
  }
  const NanoqspCSR *S = M->csr;
  for (int i = 0; i < S->n; i++) {
    int start = S->row_ptr[i];
    int end = S->row_ptr[i + 1];
    double s = nanoqsp_blas.norm1(&S->values[start], end - start);
    if (s > max_row)
      max_row = s;
  }
  return max_row;
}

/* overflow/underflow-safe frobenius norm: returns scale, writes sum_sq */
static inline double qsp_mat_scaled_fro(const qsp_mat *M, double *out_sum_sq) {
  const double *vals = M->dense ? M->dense : M->csr->values;
  size_t count =
      M->dense ? (size_t)M->rows * (size_t)M->cols : (size_t)M->csr->nnz;
  double scale = 0.0;
  double sum_sq = 1.0;
  for (size_t i = 0; i < count; i++) {
    double a = fabs(vals[i]);
    if (!isfinite(a)) {
      *out_sum_sq = a;
      return a;
    }
    if (a == 0.0)
      continue;
    if (a > scale) {
      double r = scale / a;
      sum_sq = (scale == 0.0) ? 1.0 : (1.0 + sum_sq * r * r);
      scale = a;
    } else {
      double r = a / scale;
      sum_sq += r * r;
    }
  }
  if (scale == 0.0)
    sum_sq = 0.0;
  *out_sum_sq = sum_sq;
  return scale;
}

int nanoqsp_check_kkt(int n, const double *restrict dx,
                      const double *restrict d, const double *restrict lb,
                      const double *restrict ub, const double *restrict x,
                      double d_max, double tol);

int nanoqsp_solve_cd(const qsp_mat *D, const double *restrict d,
                     const double *restrict lb, const double *restrict ub,
                     double *restrict x, int max_iter, double tol,
                     qsp_arena *ws);

int nanoqsp_solve_pg(const qsp_mat *D, const double *restrict d,
                     const double *restrict lb, const double *restrict ub,
                     double *restrict x, int max_iter, double tol,
                     qsp_arena *ws);

int nanoqsp_solve_fista(const qsp_mat *D, const double *restrict d,
                        const double *restrict lb, const double *restrict ub,
                        double *restrict x, int max_iter, double tol,
                        qsp_arena *ws);

int nanoqsp_solve_spg(const qsp_mat *D, const double *restrict d,
                      const double *restrict lb, const double *restrict ub,
                      double *restrict x, int max_iter, double tol,
                      qsp_arena *ws);

int nanoqsp_solve_admm(const qsp_mat *D, const double *restrict d,
                       const qsp_mat *A, const double *restrict lb_x,
                       const double *restrict ub_x, const double *restrict lb_A,
                       const double *restrict ub_A, double *restrict x,
                       int max_iter, double tol, qsp_arena *ws);

int nanoqsp_solve_alm_spg(const qsp_mat *D, const double *restrict d,
                          const qsp_mat *A, const double *restrict lb_x,
                          const double *restrict ub_x,
                          const double *restrict lb_A,
                          const double *restrict ub_A, double *restrict x,
                          int max_iter, double tol, qsp_arena *ws);

#endif /* QSP_INTERNAL_H */
