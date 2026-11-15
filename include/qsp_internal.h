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

/* accumulate one CSR row into zeroed scratch; callers clear touched columns */
static inline void qsp_csr_sum_row(const NanoqspCSR *S, int row,
                                   double *restrict scratch) {
  for (int idx = S->row_ptr[row]; idx < S->row_ptr[row + 1]; idx++)
    scratch[S->col_indices[idx]] += S->values[idx];
}

static inline double qsp_csr_take_sum(double *restrict scratch, int col) {
  double value = scratch[col];
  scratch[col] = 0.0;
  return value;
}

static inline double qsp_mat_norm_inf(const qsp_mat *M,
                                      double *restrict scratch) {
  double max_row = 0.0;
  if (M->dense != NULL) {
    for (int i = 0; i < M->rows; i++) {
      double s =
          nanoqsp_blas.norm1(&M->dense[(size_t)i * (size_t)M->cols], M->cols);
      if (isnan(s))
        return s;
      if (s > max_row)
        max_row = s;
    }
    return max_row;
  }
  const NanoqspCSR *S = M->csr;
  nanoqsp_blas.zero(scratch, M->cols);
  for (int row = 0; row < S->n; row++) {
    qsp_csr_sum_row(S, row, scratch);
    double s = 0.0;
    for (int idx = S->row_ptr[row]; idx < S->row_ptr[row + 1]; idx++)
      s += fabs(qsp_csr_take_sum(scratch, S->col_indices[idx]));
    if (isnan(s))
      return s;
    if (s > max_row)
      max_row = s;
  }
  return max_row;
}

static inline void qsp_scaled_squares_add(double a, double *scale,
                                          double *sum_sq) {
  if (a == 0.0)
    return;
  if (a > *scale) {
    double r = *scale / a;
    *sum_sq = (*scale == 0.0) ? 1.0 : (1.0 + *sum_sq * r * r);
    *scale = a;
  } else {
    double r = a / *scale;
    *sum_sq += r * r;
  }
}

/* overflow/underflow-safe frobenius norm: returns scale, writes sum_sq */
static inline double qsp_mat_scaled_fro(const qsp_mat *M,
                                        const double *restrict col_scale,
                                        double *out_sum_sq,
                                        double *restrict scratch) {
  double scale = 0.0;
  double sum_sq = 0.0;
  if (!M->dense)
    nanoqsp_blas.zero(scratch, M->cols);
  for (int row = 0; row < M->rows; row++) {
    int start = M->dense ? 0 : M->csr->row_ptr[row];
    int end = M->dense ? M->cols : M->csr->row_ptr[row + 1];
    if (!M->dense)
      qsp_csr_sum_row(M->csr, row, scratch);
    for (int idx = start; idx < end; idx++) {
      int col = M->dense ? idx : M->csr->col_indices[idx];
      double value = M->dense ? M->dense[(size_t)row * (size_t)M->cols + col]
                              : qsp_csr_take_sum(scratch, col);
      double a = fabs(value) * (col_scale ? col_scale[col] : 1.0);
      if (!isfinite(a)) {
        *out_sum_sq = a;
        return a;
      }
      qsp_scaled_squares_add(a, &scale, &sum_sq);
    }
  }
  *out_sum_sq = sum_sq;
  return scale;
}

/* add diag((factor * A)^T (factor * A)), summing CSR duplicates first */
static inline void qsp_mat_add_gram_diag(const qsp_mat *A, double factor,
                                         double *restrict diag,
                                         double *restrict scratch) {
  if (!A->dense)
    nanoqsp_blas.zero(scratch, A->cols);
  for (int row = 0; row < A->rows; row++) {
    int start = A->dense ? 0 : A->csr->row_ptr[row];
    int end = A->dense ? A->cols : A->csr->row_ptr[row + 1];
    if (!A->dense)
      qsp_csr_sum_row(A->csr, row, scratch);
    for (int idx = start; idx < end; idx++) {
      int col = A->dense ? idx : A->csr->col_indices[idx];
      double value = A->dense ? A->dense[(size_t)row * (size_t)A->cols + col]
                              : qsp_csr_take_sum(scratch, col);
      double t = factor * value;
      diag[col] += t * t;
    }
  }
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
