#include "nanoqsp.h"
#include "nanoqsp_blas.h"
#include <math.h>
#include <stddef.h>
#ifndef NANOQSP_NO_LIBC
#include <stdlib.h>
#endif

#define NANOQSP_MAX_DOUBLES 268435455LL

static inline int within_bounds(long long rows, long long cols) {
  return rows > 0 && cols >= 0 && rows * cols <= NANOQSP_MAX_DOUBLES;
}

static inline double *ws_take(double *restrict *cursor, int count) {
  double *p = *cursor;
  *cursor += count;
  return p;
}

static inline double clamp(double val, double lb, double ub) {
  if (val < lb)
    return lb;
  if (val > ub)
    return ub;
  return val;
}

static int valid_strategy(NanoqspStrategy strategy) {
  return strategy >= NANOQSP_STRATEGY_COORDINATE_DESCENT &&
         strategy <= NANOQSP_STRATEGY_ADMM;
}

static int validate_bounds(int n, const double *restrict lb,
                           const double *restrict ub) {
  for (int i = 0; i < n; i++) {
    double lower = lb ? lb[i] : -INFINITY;
    double upper = ub ? ub[i] : INFINITY;
    if (isnan(lower) || isnan(upper) || lower > upper)
      return 0;
  }
  return 1;
}

static void project_bounds(int n, const double *restrict lb,
                           const double *restrict ub, double *restrict x) {
  for (int i = 0; i < n; i++) {
    double lower = lb ? lb[i] : -INFINITY;
    double upper = ub ? ub[i] : INFINITY;
    x[i] = clamp(x[i], lower, upper);
  }
}

static int validate_vector(int n, const double *restrict v) {
  for (int i = 0; i < n; i++) {
    if (isnan(v[i]))
      return 0;
  }
  return 1;
}

static int validate_config(const NanoqspConfig *config) {
  if (config == NULL)
    return 1;
  if (isnan(config->tolerance) || isinf(config->tolerance))
    return 0;
  return 1;
}

static int validate_csr(const NanoqspCSR *matrix, int rows, int cols) {
  if (matrix == NULL || rows < 0 || cols <= 0 || matrix->n != rows ||
      matrix->nnz < 0 || matrix->row_ptr == NULL)
    return 0;
  if (matrix->row_ptr[0] != 0 || matrix->row_ptr[rows] != matrix->nnz)
    return 0;
  if (matrix->nnz > 0 &&
      (matrix->values == NULL || matrix->col_indices == NULL))
    return 0;
  for (int i = 0; i < rows; i++) {
    int start = matrix->row_ptr[i];
    int end = matrix->row_ptr[i + 1];
    if (start < 0 || end < start || end > matrix->nnz)
      return 0;
  }
  for (int idx = 0; idx < matrix->nnz; idx++) {
    if (matrix->col_indices[idx] < 0 || matrix->col_indices[idx] >= cols ||
        isnan(matrix->values[idx]))
      return 0;
  }
  return 1;
}

int nanoqsp_workspace_size_box(int n, NanoqspStrategy strategy) {
  if (!valid_strategy(strategy) || !within_bounds(n, 6))
    return NANOQSP_ERR_INVALID_ARG;
  if (strategy == NANOQSP_STRATEGY_COORDINATE_DESCENT)
    return 0;
  if (strategy == NANOQSP_STRATEGY_PROJECTED_GRADIENT)
    return n;
  if (strategy == NANOQSP_STRATEGY_ACCELERATED_GRADIENT)
    return 2 * n;
  if (strategy == NANOQSP_STRATEGY_SPECTRAL_GRADIENT)
    return 3 * n;
  return 6 * n;
}

int nanoqsp_workspace_size_box_sparse(int n, NanoqspStrategy strategy) {
  if (!valid_strategy(strategy) || !within_bounds(n, 6))
    return NANOQSP_ERR_INVALID_ARG;
  if (strategy == NANOQSP_STRATEGY_PROJECTED_GRADIENT)
    return n;
  if (strategy == NANOQSP_STRATEGY_COORDINATE_DESCENT ||
      strategy == NANOQSP_STRATEGY_ACCELERATED_GRADIENT)
    return 2 * n;
  if (strategy == NANOQSP_STRATEGY_SPECTRAL_GRADIENT)
    return 3 * n;
  return 6 * n;
}

int nanoqsp_workspace_size_least_squares(int n, NanoqspStrategy strategy) {
  int box_size = nanoqsp_workspace_size_box(n, strategy);
  if (box_size < 0 || !within_bounds(n, n))
    return NANOQSP_ERR_INVALID_ARG;
  long long total = (long long)n * n + n + box_size;
  if (total > NANOQSP_MAX_DOUBLES)
    return NANOQSP_ERR_INVALID_ARG;
  return (int)total;
}

int nanoqsp_workspace_size_affine(int n, int m) {
  if (n <= 0 || m < 0)
    return NANOQSP_ERR_INVALID_ARG;
  long long total = 7LL * n + 3LL * m;
  if (total > NANOQSP_MAX_DOUBLES)
    return NANOQSP_ERR_INVALID_ARG;
  return (int)total;
}

/* matrix-vector and norm helpers */

static void dense_mat_vec(int m, int n, const double *restrict A,
                          const double *restrict x, double *restrict y) {
  for (int i = 0; i < m; i++)
    y[i] = v_dot(&A[i * n], x, n);
}

static void dense_mat_trans_vec(int m, int n, const double *restrict A,
                                const double *restrict v, double *restrict y) {
  v_zero(y, n);
  for (int j = 0; j < m; j++)
    v_axpy(y, &A[j * n], v[j], n);
}

static void sparse_mat_vec(const NanoqspCSR *A, const double *restrict x,
                           double *restrict y) {
  for (int i = 0; i < A->n; i++) {
    double sum = 0.0;
    int start = A->row_ptr[i];
    int end = A->row_ptr[i + 1];
    for (int idx = start; idx < end; idx++) {
      sum += A->values[idx] * x[A->col_indices[idx]];
    }
    y[i] = sum;
  }
}

static void sparse_mat_trans_vec(int m, int n, const NanoqspCSR *A,
                                 const double *restrict v, double *restrict y) {
  v_zero(y, n);
  for (int j = 0; j < m; j++) {
    double val_v = v[j];
    if (val_v == 0.0)
      continue;
    int start = A->row_ptr[j];
    int end = A->row_ptr[j + 1];
    for (int idx = start; idx < end; idx++) {
      y[A->col_indices[idx]] += A->values[idx] * val_v;
    }
  }
}

static double get_matrix_norm_inf(int n, const double *restrict D) {
  double max_row_sum = 0.0;
  for (int i = 0; i < n; i++) {
    double row_sum = v_norm1(&D[i * n], n);
    if (row_sum > max_row_sum) {
      max_row_sum = row_sum;
    }
  }
  return max_row_sum;
}

static double get_sparse_matrix_norm_inf(const NanoqspCSR *D) {
  double max_row_sum = 0.0;
  for (int i = 0; i < D->n; i++) {
    int start = D->row_ptr[i];
    int end = D->row_ptr[i + 1];
    double row_sum = v_norm1(&D->values[start], end - start);
    if (row_sum > max_row_sum) {
      max_row_sum = row_sum;
    }
  }
  return max_row_sum;
}

static double get_vec_norm_inf(int n, const double *restrict v) {
  double max_val = 0.0;
  for (int i = 0; i < n; i++) {
    double val = fabs(v[i]);
    if (val > max_val)
      max_val = val;
  }
  return max_val;
}

static inline void eval_D_vec(int n, const double *restrict D_dense,
                              const NanoqspCSR *D_sparse,
                              const double *restrict x, double *restrict y) {
  if (D_dense != NULL)
    dense_mat_vec(n, n, D_dense, x, y);
  else
    sparse_mat_vec(D_sparse, x, y);
}

static inline double eval_D_norm_inf(int n, const double *restrict D_dense,
                                     const NanoqspCSR *D_sparse) {
  if (D_dense != NULL)
    return get_matrix_norm_inf(n, D_dense);
  return get_sparse_matrix_norm_inf(D_sparse);
}

static inline void eval_A_vec(int m, int n, const double *restrict A_dense,
                              const NanoqspCSR *A_sparse,
                              const double *restrict x, double *restrict y) {
  if (A_dense != NULL)
    dense_mat_vec(m, n, A_dense, x, y);
  else
    sparse_mat_vec(A_sparse, x, y);
}

static inline void eval_AT_vec(int m, int n, const double *restrict A_dense,
                               const NanoqspCSR *A_sparse,
                               const double *restrict v, double *restrict y) {
  if (A_dense != NULL)
    dense_mat_trans_vec(m, n, A_dense, v, y);
  else
    sparse_mat_trans_vec(m, n, A_sparse, v, y);
}

/* shared KKT optimality check given Dx = D * x */
static int
check_kkt_from_Dx(int n, const double *restrict Dx, const double *restrict d,
                  const double *restrict lb, const double *restrict ub,
                  const double *restrict x, double d_max, double tol) {
  double max_kkt = 0.0;
  double dx_max = 0.0;
  for (int i = 0; i < n; i++) {
    double dx_i = Dx[i];
    double grad_i = dx_i - d[i];
    if (isnan(grad_i))
      return 0;
    double lower = lb ? lb[i] : -INFINITY;
    double upper = ub ? ub[i] : INFINITY;
    double viol = 0.0;
    if (lower == upper) {
      /* fixed coordinate has no feasible descent direction */
      viol = 0.0;
    } else if (x[i] <= lower) {
      if (grad_i < 0.0)
        viol = -grad_i;
    } else if (x[i] >= upper) {
      if (grad_i > 0.0)
        viol = grad_i;
    } else {
      viol = fabs(grad_i);
    }
    if (viol > max_kkt)
      max_kkt = viol;
    double val_dx = fabs(dx_i);
    if (val_dx > dx_max)
      dx_max = val_dx;
  }
  double scale = 1.0 + (d_max > dx_max ? d_max : dx_max);
  return max_kkt < tol * scale;
}

/* strategy 1a: dense coordinate descent */
static int solve_coordinate_descent_dense(int n, const double *restrict D,
                                          const double *restrict d,
                                          const double *restrict lb,
                                          const double *restrict ub,
                                          double *restrict x, int max_iter,
                                          double tol) {
  double d_max = get_vec_norm_inf(n, d);
  for (int iter = 0; iter < max_iter; iter++) {
    double max_diff = 0.0;
    double x_max = 0.0;
    for (int i = 0; i < n; i++) {
      double D_ii = D[i * n + i];
      double sum = v_dot(&D[i * n], x, n) - D_ii * x[i];
      double old_val = x[i];
      if (D_ii > 0.0) {
        double lower = lb ? lb[i] : -INFINITY;
        double upper = ub ? ub[i] : INFINITY;
        x[i] = clamp((d[i] - sum) / D_ii, lower, upper);
      }

      double diff = fabs(x[i] - old_val);
      if (diff > max_diff)
        max_diff = diff;
      double val_x = fabs(x[i]);
      if (val_x > x_max)
        x_max = val_x;
    }
    if (max_diff < tol * (1.0 + x_max)) {
      double max_kkt = 0.0;
      double dx_max = 0.0;
      int nan_found = 0;
      for (int i = 0; i < n; i++) {
        double dx_i = v_dot(&D[i * n], x, n);
        double grad_i = dx_i - d[i];
        if (isnan(grad_i)) {
          nan_found = 1;
          break;
        }
        double lower = lb ? lb[i] : -INFINITY;
        double upper = ub ? ub[i] : INFINITY;
        double viol = 0.0;
        if (lower == upper) {
          viol = 0.0;
        } else if (x[i] <= lower) {
          if (grad_i < 0.0)
            viol = -grad_i;
        } else if (x[i] >= upper) {
          if (grad_i > 0.0)
            viol = grad_i;
        } else {
          viol = fabs(grad_i);
        }
        if (viol > max_kkt)
          max_kkt = viol;
        double val_dx = fabs(dx_i);
        if (val_dx > dx_max)
          dx_max = val_dx;
      }
      double scale = 1.0 + (d_max > dx_max ? d_max : dx_max);
      if (!nan_found && max_kkt < tol * scale) {
        return iter + 1;
      }
    }
  }
  return NANOQSP_ERR_MAX_ITERATIONS;
}

/* strategy 1b: sparse coordinate descent */
static int solve_coordinate_descent_sparse(int n, const NanoqspCSR *D,
                                           const double *restrict d,
                                           const double *restrict lb,
                                           const double *restrict ub,
                                           double *restrict x, int max_iter,
                                           double tol, double *restrict ws) {
  double *restrict diag = ws_take(&ws, n);
  double *restrict g_temp = ws_take(&ws, n);
  double d_max = get_vec_norm_inf(n, d);

  for (int i = 0; i < n; i++) {
    double d_ii = 0.0;
    int start = D->row_ptr[i];
    int end = D->row_ptr[i + 1];
    for (int idx = start; idx < end; idx++) {
      if (D->col_indices[idx] == i) {
        d_ii += D->values[idx];
      }
    }
    diag[i] = d_ii;
  }

  for (int iter = 0; iter < max_iter; iter++) {
    double max_diff = 0.0;
    double x_max = 0.0;
    for (int i = 0; i < n; i++) {
      double sum = 0.0;
      int start = D->row_ptr[i];
      int end = D->row_ptr[i + 1];
      for (int idx = start; idx < end; idx++) {
        int j = D->col_indices[idx];
        if (i != j) {
          sum += D->values[idx] * x[j];
        }
      }
      double old_val = x[i];
      double D_ii = diag[i];
      if (D_ii > 0.0) {
        double lower = lb ? lb[i] : -INFINITY;
        double upper = ub ? ub[i] : INFINITY;
        x[i] = clamp((d[i] - sum) / D_ii, lower, upper);
      }

      double diff = fabs(x[i] - old_val);
      if (diff > max_diff)
        max_diff = diff;
      double val_x = fabs(x[i]);
      if (val_x > x_max)
        x_max = val_x;
    }
    if (max_diff < tol * (1.0 + x_max)) {
      sparse_mat_vec(D, x, g_temp);
      if (check_kkt_from_Dx(n, g_temp, d, lb, ub, x, d_max, tol)) {
        return iter + 1;
      }
    }
  }
  return NANOQSP_ERR_MAX_ITERATIONS;
}

/* strategy 2: projected gradient descent */
static int solve_projected_gradient(int n, const double *restrict D_dense,
                                    const NanoqspCSR *D_sparse,
                                    const double *restrict d,
                                    const double *restrict lb,
                                    const double *restrict ub,
                                    double *restrict x, int max_iter,
                                    double tol, double *restrict ws) {
  double norm_inf = eval_D_norm_inf(n, D_dense, D_sparse);
  double matrix_scale = norm_inf > 0.0 ? norm_inf : 10.0;
  double d_max = get_vec_norm_inf(n, d);
  double *restrict dx = ws;

  for (int iter = 0; iter < max_iter; iter++) {
    eval_D_vec(n, D_dense, D_sparse, x, dx);
    if (check_kkt_from_Dx(n, dx, d, lb, ub, x, d_max, tol))
      return iter;

    for (int i = 0; i < n; i++) {
      double lower = lb ? lb[i] : -INFINITY;
      double upper = ub ? ub[i] : INFINITY;
      x[i] = clamp(x[i] - (dx[i] - d[i]) / matrix_scale, lower, upper);
    }
  }

  eval_D_vec(n, D_dense, D_sparse, x, dx);
  if (check_kkt_from_Dx(n, dx, d, lb, ub, x, d_max, tol))
    return max_iter;
  return NANOQSP_ERR_MAX_ITERATIONS;
}

/* strategy 3: nesterov accelerated projected gradient with adaptive restarts */
static int solve_accelerated_gradient(int n, const double *restrict D_dense,
                                      const NanoqspCSR *D_sparse,
                                      const double *restrict d,
                                      const double *restrict lb,
                                      const double *restrict ub,
                                      double *restrict x, int max_iter,
                                      double tol, double *restrict ws) {
  double norm_inf = eval_D_norm_inf(n, D_dense, D_sparse);
  double matrix_scale = norm_inf > 0.0 ? norm_inf : 10.0;
  double d_max = get_vec_norm_inf(n, d);

  double *restrict y = ws_take(&ws, n);
  double *restrict next_x = ws_take(&ws, n);

  v_copy(y, x, n);

  double t = 1.0;
  for (int iter = 0; iter < max_iter; iter++) {
    double max_diff = 0.0;
    double x_max = 0.0;

    eval_D_vec(n, D_dense, D_sparse, y, next_x);

    for (int i = 0; i < n; i++) {
      double grad_i = next_x[i] - d[i];
      double lower = lb ? lb[i] : -INFINITY;
      double upper = ub ? ub[i] : INFINITY;
      next_x[i] = clamp(y[i] - grad_i / matrix_scale, lower, upper);

      double diff = fabs(next_x[i] - x[i]);
      if (diff > max_diff)
        max_diff = diff;
      double val_x = fabs(next_x[i]);
      if (val_x > x_max)
        x_max = val_x;
    }

    /* compute restart condition before KKT check reuses y */
    double restart_check = 0.0;
    for (int i = 0; i < n; i++) {
      restart_check += (y[i] - next_x[i]) * (next_x[i] - x[i]);
    }

    if (max_diff < tol * (1.0 + x_max)) {
      eval_D_vec(n, D_dense, D_sparse, next_x, y);
      if (check_kkt_from_Dx(n, y, d, lb, ub, next_x, d_max, tol)) {
        v_copy(x, next_x, n);
        return iter + 1;
      }
    }

    if (restart_check > 0.0) {
      t = 1.0;
      v_copy(y, next_x, n);
      v_copy(x, next_x, n);
    } else {
      double t_next = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t * t));
      double beta = (t - 1.0) / t_next;
      for (int i = 0; i < n; i++) {
        y[i] = next_x[i] + beta * (next_x[i] - x[i]);
        x[i] = next_x[i];
      }
      t = t_next;
    }
  }
  return NANOQSP_ERR_MAX_ITERATIONS;
}

/* strategy 4: spectral projected gradient (SPG) */
static int solve_spectral_gradient(int n, const double *restrict D_dense,
                                   const NanoqspCSR *D_sparse,
                                   const double *restrict d,
                                   const double *restrict lb,
                                   const double *restrict ub,
                                   double *restrict x, int max_iter, double tol,
                                   double *restrict ws) {
  double norm_inf = eval_D_norm_inf(n, D_dense, D_sparse);
  double matrix_scale = norm_inf > 0.0 ? norm_inf : 1.0;
  /* store step relative to matrix_scale to avoid reciprocal overflow */
  double alpha = 1.0;
  double d_max = get_vec_norm_inf(n, d);
  double *restrict next_x = ws_take(&ws, n);
  double *restrict dx = ws_take(&ws, n);
  double *restrict next_dx = ws_take(&ws, n);

  eval_D_vec(n, D_dense, D_sparse, x, dx);

  for (int iter = 0; iter < max_iter; iter++) {
    if (check_kkt_from_Dx(n, dx, d, lb, ub, x, d_max, tol))
      return iter;

    double direction_max = 0.0;
    for (int i = 0; i < n; i++) {
      double lower = lb ? lb[i] : -INFINITY;
      double upper = ub ? ub[i] : INFINITY;
      double gradient = (dx[i] - d[i]) / matrix_scale;
      next_x[i] = clamp(x[i] - alpha * gradient, lower, upper) - x[i];
      double magnitude = fabs(next_x[i]);
      if (magnitude > direction_max)
        direction_max = magnitude;
    }

    if (!(direction_max > 0.0) || !isfinite(direction_max))
      return NANOQSP_ERR_MAX_ITERATIONS;

    /* normalize direction q before forming curvature to avoid BB underflow */
    for (int i = 0; i < n; i++)
      next_x[i] /= direction_max;
    eval_D_vec(n, D_dense, D_sparse, next_x, next_dx);

    double q_dot_q = 0.0;
    double curvature = 0.0;
    double slope = 0.0;
    for (int i = 0; i < n; i++) {
      double q_i = next_x[i];
      q_dot_q += q_i * q_i;
      curvature += q_i * (next_dx[i] / matrix_scale);
      slope += q_i * ((dx[i] - d[i]) / matrix_scale);
    }

    /* minimize quadratic along feasible segment to prevent BB cycling */
    double length = direction_max;
    if (curvature > 0.0) {
      double minimizer = -slope / curvature;
      if (minimizer < length)
        length = minimizer;
    }
    if (!(length > 0.0))
      return NANOQSP_ERR_MAX_ITERATIONS;

    for (int i = 0; i < n; i++) {
      double lower = lb ? lb[i] : -INFINITY;
      double upper = ub ? ub[i] : INFINITY;
      /* recreate full projected step exactly, including active bounds */
      double candidate = length == direction_max
                             ? x[i] - alpha * ((dx[i] - d[i]) / matrix_scale)
                             : x[i] + length * next_x[i];
      x[i] = clamp(candidate, lower, upper);
    }

    /* BB1 ratio q^T q / q^T D q relative to matrix_scale */
    double spectral_alpha = q_dot_q / curvature;
    if (spectral_alpha > 0.0 && isfinite(spectral_alpha)) {
      alpha = spectral_alpha < 1e12 ? spectral_alpha : 1e12;
    } else {
      alpha = 1.0;
    }
    eval_D_vec(n, D_dense, D_sparse, x, dx);
  }

  if (check_kkt_from_Dx(n, dx, d, lb, ub, x, d_max, tol))
    return max_iter;
  return NANOQSP_ERR_MAX_ITERATIONS;
}

/* strategy 5: unified ADMM for box and general affine constraints */

static void cg_solve_unified(
    int n, int m, const double *restrict D_dense, const NanoqspCSR *D_sparse,
    const double *restrict A_dense, const NanoqspCSR *A_sparse, double rho,
    double affine_rho, double r_tol_sq, const double *restrict d,
    const double *restrict z, const double *restrict y, double *restrict x,
    double *restrict r, double *restrict p, double *restrict Ap,
    double *restrict tmp_n, double *restrict tmp_m) {
  int has_A = (m > 0 && (A_dense != NULL || A_sparse != NULL));

  /* form residual from constraint differences before multiplying by A^T */
  eval_D_vec(n, D_dense, D_sparse, x, r);
  for (int i = 0; i < n; i++)
    r[i] = d[i] - r[i] + rho * ((z[i] - x[i]) - y[i]);
  if (has_A) {
    eval_A_vec(m, n, A_dense, A_sparse, x, tmp_m);
    for (int i = 0; i < m; i++)
      tmp_m[i] = (z[n + i] - tmp_m[i]) - y[n + i];
    eval_AT_vec(m, n, A_dense, A_sparse, tmp_m, tmp_n);
    v_axpy(r, tmp_n, affine_rho, n);
  }

  v_copy(p, r, n);
  double r_sq = v_dot(r, r, n);

  for (int iter = 0; iter < n + 5; iter++) {
    if (r_sq <= r_tol_sq)
      break;

    eval_D_vec(n, D_dense, D_sparse, p, Ap);
    /* separate positive quadratic forms avoid cancellation in p^T(A^T A)p */
    double pAp = v_dot(p, Ap, n) + rho * v_dot(p, p, n);

    if (has_A) {
      eval_A_vec(m, n, A_dense, A_sparse, p, tmp_m);
      pAp += affine_rho * v_dot(tmp_m, tmp_m, m);
      eval_AT_vec(m, n, A_dense, A_sparse, tmp_m, tmp_n);
      for (int i = 0; i < n; i++) {
        Ap[i] += rho * p[i] + affine_rho * tmp_n[i];
      }
    } else {
      for (int i = 0; i < n; i++) {
        Ap[i] += rho * p[i];
      }
    }

    if (!(pAp > 0.0) || !isfinite(pAp) || !isfinite(r_sq))
      break;

    double alpha = r_sq / pAp;
    if (!isfinite(alpha))
      break;
    v_axpy(x, p, alpha, n);
    v_axpy(r, Ap, -alpha, n);

    double next_r_sq = v_dot(r, r, n);
    double beta = next_r_sq / r_sq;
    for (int i = 0; i < n; i++) {
      p[i] = r[i] + beta * p[i];
    }
    r_sq = next_r_sq;
  }
}

static int
solve_admm_unified(int n, int m, const double *restrict D_dense,
                   const NanoqspCSR *D_sparse, const double *restrict d,
                   const double *restrict A_dense, const NanoqspCSR *A_sparse,
                   const double *restrict lb_x, const double *restrict ub_x,
                   const double *restrict lb_A, const double *restrict ub_A,
                   double *restrict x, int max_iter, double tol,
                   double *restrict ws) {
  double rho = 1.0;
  double affine_weight = 1.0;
  int has_A = (m > 0 && (A_dense != NULL || A_sparse != NULL));
  if (has_A) {
    /* weight affine penalty independently of box penalty */
    const double *values = A_dense ? A_dense : A_sparse->values;
    int count = A_dense ? m * n : A_sparse->nnz;
    double scale = 1.0;
    double sum_squares = 1.0;
    for (int i = 0; i < count; i++) {
      double value = fabs(values[i]);
      if (value > scale) {
        double ratio = scale / value;
        sum_squares = 1.0 + sum_squares * ratio * ratio;
        scale = value;
      } else {
        double ratio = value / scale;
        sum_squares += ratio * ratio;
      }
    }
    double weight = ((1.0 / scale) / scale) / sum_squares;
    if (weight > 0.0 && isfinite(weight))
      affine_weight = weight;
  }
  double d_max = get_vec_norm_inf(n, d);
  double cg_tol = 1e-2 * tol * (1.0 + d_max);
  double cg_tol_sq = cg_tol * cg_tol;
  if (cg_tol_sq > 1e-16)
    cg_tol_sq = 1e-16;

  double *restrict z = ws_take(&ws, n + m);
  double *restrict y = ws_take(&ws, n + m);
  double *restrict c = ws_take(&ws, n);
  double *restrict r = ws_take(&ws, n);
  double *restrict p = ws_take(&ws, n);
  double *restrict Ap = ws_take(&ws, n);
  double *restrict Ax = ws_take(&ws, m);
  double *restrict tmp_n = has_A ? ws_take(&ws, n) : NULL;

  v_copy(z, x, n);
  if (has_A) {
    eval_A_vec(m, n, A_dense, A_sparse, x, &z[n]);
  }
  v_zero(y, n + m);

  for (int iter = 0; iter < max_iter; iter++) {
    cg_solve_unified(n, m, D_dense, D_sparse, A_dense, A_sparse, rho,
                     rho * affine_weight, cg_tol_sq, d, z, y, x, r, p, Ap,
                     tmp_n, Ax);

    double prim_res = 0.0;
    double x_max = 0.0;
    double z_max = 0.0;
    int nonfinite_found = 0;

    for (int i = 0; i < n; i++) {
      double old_z = z[i];
      double lower = lb_x ? lb_x[i] : -INFINITY;
      double upper = ub_x ? ub_x[i] : INFINITY;
      double z_i = clamp(x[i] + y[i], lower, upper);
      double y_i = y[i] + (x[i] - z_i);
      z[i] = z_i;
      y[i] = y_i;
      c[i] = rho * (z_i - old_z);

      if (!isfinite(x[i]) || !isfinite(z_i) || !isfinite(y_i))
        nonfinite_found = 1;

      double res = fabs(x[i] - z_i);
      if (res > prim_res)
        prim_res = res;

      double val_x = fabs(x[i]);
      if (val_x > x_max)
        x_max = val_x;
      double val_z = fabs(z_i);
      if (val_z > z_max)
        z_max = val_z;
    }

    if (has_A) {
      eval_A_vec(m, n, A_dense, A_sparse, x, Ax);
      for (int i = 0; i < m; i++) {
        double old_z = z[n + i];
        double lower = lb_A ? lb_A[i] : -INFINITY;
        double upper = ub_A ? ub_A[i] : INFINITY;
        double z_i = clamp(Ax[i] + y[n + i], lower, upper);
        double y_i = y[n + i] + (Ax[i] - z_i);
        z[n + i] = z_i;
        y[n + i] = y_i;

        if (!isfinite(Ax[i]) || !isfinite(z_i) || !isfinite(y_i))
          nonfinite_found = 1;

        double res = fabs(Ax[i] - z_i);
        if (res > prim_res)
          prim_res = res;

        double val_z = fabs(z_i);
        if (val_z > z_max)
          z_max = val_z;
        Ax[i] = rho * affine_weight * (z_i - old_z);
      }
      eval_AT_vec(m, n, A_dense, A_sparse, Ax, r);
      v_axpy(c, r, 1.0, n);

      /* check affine feasibility at the returned point z_x */
      eval_A_vec(m, n, A_dense, A_sparse, z, Ax);
      for (int i = 0; i < m; i++) {
        if (!isfinite(Ax[i]))
          nonfinite_found = 1;
        double res = fabs(Ax[i] - z[n + i]);
        if (res > prim_res)
          prim_res = res;
        double val_x = fabs(Ax[i]);
        if (val_x > x_max)
          x_max = val_x;
      }
    }

    /* use transformed z change to balance penalties */
    double admm_dual_res = get_vec_norm_inf(n, c);

    /* test actual stationarity at the returned point z_x */
    eval_D_vec(n, D_dense, D_sparse, z, c);
    if (has_A)
      eval_AT_vec(m, n, A_dense, A_sparse, &y[n], r);
    double dx_max = 0.0;
    double y_max = 0.0;
    double dual_res = 0.0;
    for (int i = 0; i < n; i++) {
      double dual_i = rho * y[i] + (has_A ? rho * affine_weight * r[i] : 0.0);
      double residual = c[i] - d[i] + dual_i;
      if (!isfinite(c[i]) || !isfinite(dual_i) || !isfinite(residual))
        nonfinite_found = 1;
      double val_dx = fabs(c[i]);
      if (val_dx > dx_max)
        dx_max = val_dx;
      double val_y = fabs(dual_i);
      if (val_y > y_max)
        y_max = val_y;
      double res = fabs(residual);
      if (res > dual_res)
        dual_res = res;
    }

    double prim_scale = 1.0 + (x_max > z_max ? x_max : z_max);
    double dual_scale =
        1.0 + (y_max > d_max ? (y_max > dx_max ? y_max : dx_max)
                             : (d_max > dx_max ? d_max : dx_max));

    if (!nonfinite_found && dual_res < tol * dual_scale &&
        prim_res < tol * prim_scale) {
      v_copy(x, z, n);
      return iter + 1;
    }

    /* balance relative residuals periodically */
    if (!nonfinite_found && (iter + 1) % 10 == 0) {
      double relative_primal = prim_res / prim_scale;
      double relative_dual = admm_dual_res / dual_scale;
      double new_rho = rho;
      if (relative_primal > 10.0 * relative_dual)
        new_rho = 2.0 * rho;
      else if (relative_dual > 10.0 * relative_primal)
        new_rho = 0.5 * rho;
      if (new_rho > 0.0 && isfinite(new_rho) && new_rho != rho) {
        double multiplier_scale = rho / new_rho;
        int can_rescale = 1;
        for (int i = 0; i < n + m; i++) {
          if (!isfinite(y[i] * multiplier_scale))
            can_rescale = 0;
        }
        if (can_rescale) {
          for (int i = 0; i < n + m; i++)
            y[i] *= multiplier_scale;
          rho = new_rho;
        }
      }
    }
  }
  v_copy(x, z, n);
  return NANOQSP_ERR_MAX_ITERATIONS;
}

static int prepare_workspace(const NanoqspConfig *config, int required_ws,
                             int *max_iterations, double *tolerance,
                             double *restrict *ws_out, int *needs_free) {
  *max_iterations = 1000;
  *tolerance = 1e-6;
  double *restrict ws = NULL;
  int ws_size = 0;

  if (config != NULL) {
    if (config->max_iterations > 0)
      *max_iterations = config->max_iterations;
    if (config->tolerance > 0.0)
      *tolerance = config->tolerance;
    ws = config->workspace;
    ws_size = config->workspace_size;
  }

  *needs_free = 0;
  if (required_ws > 0 && (ws == NULL || ws_size < required_ws)) {
#ifdef NANOQSP_NO_LIBC
    return NANOQSP_ERR_OUT_OF_MEMORY;
#else
    ws = (double *restrict)malloc((size_t)required_ws * sizeof(double));
    if (!ws)
      return NANOQSP_ERR_OUT_OF_MEMORY;
    *needs_free = 1;
#endif
  }
  *ws_out = ws;
  return NANOQSP_SUCCESS;
}

static void release_workspace(double *ws, int needs_free) {
#ifdef NANOQSP_NO_LIBC
  (void)ws;
  (void)needs_free;
#else
  if (needs_free)
    free(ws);
#endif
}

/* public API implementations */

int nanoqsp_solve_box(int n, const double *restrict D, const double *restrict d,
                      const double *restrict lb, const double *restrict ub,
                      double *restrict x, const NanoqspConfig *config) {
  if (!within_bounds(n, n) || D == NULL || d == NULL || x == NULL ||
      !validate_config(config) || !validate_bounds(n, lb, ub) ||
      !validate_vector(n, x) || !validate_vector(n, d)) {
    return NANOQSP_ERR_INVALID_ARG;
  }

  NanoqspStrategy strategy =
      config ? config->strategy : NANOQSP_STRATEGY_COORDINATE_DESCENT;
  int required_ws = nanoqsp_workspace_size_box(n, strategy);
  if (required_ws < 0)
    return valid_strategy(strategy) ? required_ws
                                    : NANOQSP_ERR_INVALID_STRATEGY;

  int max_iterations, needs_free;
  double tolerance;
  double *restrict ws = NULL;
  int status = prepare_workspace(config, required_ws, &max_iterations,
                                 &tolerance, &ws, &needs_free);
  if (status < 0)
    return status;

  project_bounds(n, lb, ub, x);

  int iters = NANOQSP_ERR_INVALID_STRATEGY;
  switch (strategy) {
  case NANOQSP_STRATEGY_COORDINATE_DESCENT:
    iters = solve_coordinate_descent_dense(n, D, d, lb, ub, x, max_iterations,
                                           tolerance);
    break;
  case NANOQSP_STRATEGY_PROJECTED_GRADIENT:
    iters = solve_projected_gradient(n, D, NULL, d, lb, ub, x, max_iterations,
                                     tolerance, ws);
    break;
  case NANOQSP_STRATEGY_ACCELERATED_GRADIENT:
    iters = solve_accelerated_gradient(n, D, NULL, d, lb, ub, x, max_iterations,
                                       tolerance, ws);
    break;
  case NANOQSP_STRATEGY_SPECTRAL_GRADIENT:
    iters = solve_spectral_gradient(n, D, NULL, d, lb, ub, x, max_iterations,
                                    tolerance, ws);
    break;
  case NANOQSP_STRATEGY_ADMM:
    iters = solve_admm_unified(n, 0, D, NULL, d, NULL, NULL, lb, ub, NULL, NULL,
                               x, max_iterations, tolerance, ws);
    break;
  }

  release_workspace(ws, needs_free);
  return iters;
}

int nanoqsp_solve_box_sparse(int n, const NanoqspCSR *D,
                             const double *restrict d,
                             const double *restrict lb,
                             const double *restrict ub, double *restrict x,
                             const NanoqspConfig *config) {
  if (!within_bounds(n, 6) || d == NULL || x == NULL ||
      !validate_config(config) || !validate_csr(D, n, n) ||
      !validate_bounds(n, lb, ub) || !validate_vector(n, x) ||
      !validate_vector(n, d)) {
    return NANOQSP_ERR_INVALID_ARG;
  }

  NanoqspStrategy strategy =
      config ? config->strategy : NANOQSP_STRATEGY_COORDINATE_DESCENT;
  int required_ws = nanoqsp_workspace_size_box_sparse(n, strategy);
  if (required_ws < 0)
    return valid_strategy(strategy) ? required_ws
                                    : NANOQSP_ERR_INVALID_STRATEGY;

  int max_iterations, needs_free;
  double tolerance;
  double *restrict ws = NULL;
  int status = prepare_workspace(config, required_ws, &max_iterations,
                                 &tolerance, &ws, &needs_free);
  if (status < 0)
    return status;

  project_bounds(n, lb, ub, x);

  int iters = NANOQSP_ERR_INVALID_STRATEGY;
  switch (strategy) {
  case NANOQSP_STRATEGY_COORDINATE_DESCENT:
    iters = solve_coordinate_descent_sparse(n, D, d, lb, ub, x, max_iterations,
                                            tolerance, ws);
    break;
  case NANOQSP_STRATEGY_PROJECTED_GRADIENT:
    iters = solve_projected_gradient(n, NULL, D, d, lb, ub, x, max_iterations,
                                     tolerance, ws);
    break;
  case NANOQSP_STRATEGY_ACCELERATED_GRADIENT:
    iters = solve_accelerated_gradient(n, NULL, D, d, lb, ub, x, max_iterations,
                                       tolerance, ws);
    break;
  case NANOQSP_STRATEGY_SPECTRAL_GRADIENT:
    iters = solve_spectral_gradient(n, NULL, D, d, lb, ub, x, max_iterations,
                                    tolerance, ws);
    break;
  case NANOQSP_STRATEGY_ADMM:
    iters = solve_admm_unified(n, 0, NULL, D, d, NULL, NULL, lb, ub, NULL, NULL,
                               x, max_iterations, tolerance, ws);
    break;
  }

  release_workspace(ws, needs_free);
  return iters;
}

double nanoqsp_predict(int n, const double *restrict x,
                       const double *restrict feature_vector) {
  if (n <= 0 || x == NULL || feature_vector == NULL)
    return 0.0;
  return v_dot(x, feature_vector, n);
}

int nanoqsp_solve_least_squares(int m, int n, const double *restrict A,
                                const double *restrict b,
                                const double *restrict lb,
                                const double *restrict ub, double *restrict x,
                                const NanoqspConfig *config) {
  if (!within_bounds(m, n) || !within_bounds(n, n) || m <= 0 || A == NULL ||
      b == NULL || x == NULL || !validate_config(config) ||
      !validate_bounds(n, lb, ub) || !validate_vector(m * n, A) ||
      !validate_vector(n, x) || !validate_vector(m, b))
    return NANOQSP_ERR_INVALID_ARG;

  NanoqspStrategy strategy =
      config ? config->strategy : NANOQSP_STRATEGY_COORDINATE_DESCENT;
  int required_total = nanoqsp_workspace_size_least_squares(n, strategy);
  if (required_total < 0)
    return valid_strategy(strategy) ? required_total
                                    : NANOQSP_ERR_INVALID_STRATEGY;

  int max_iterations, needs_free;
  double tolerance;
  double *restrict ws = NULL;
  int status = prepare_workspace(config, required_total, &max_iterations,
                                 &tolerance, &ws, &needs_free);
  if (status < 0)
    return status;

  double *restrict cursor = ws;
  double *restrict D = ws_take(&cursor, n * n);
  double *restrict d = ws_take(&cursor, n);

  v_zero(D, n * n);
  v_zero(d, n);

  for (int k = 0; k < m; k++) {
    const double *restrict A_row = &A[k * n];
    v_axpy(d, A_row, b[k], n);
    for (int i = 0; i < n; i++) {
      v_axpy(&D[i * n + i], &A_row[i], A_row[i], n - i);
    }
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++) {
      D[i * n + j] = D[j * n + i];
    }
  }

  double norm_inf = get_matrix_norm_inf(n, D);
  double epsilon = (norm_inf > 0.0) ? (norm_inf * 1e-8) : 1e-8;
  for (int i = 0; i < n; i++)
    D[i * n + i] += epsilon;

  NanoqspConfig sub_config = {.strategy = strategy,
                              .max_iterations = max_iterations,
                              .tolerance = tolerance,
                              .workspace = cursor,
                              .workspace_size = required_total - (n * n + n)};

  int iters = nanoqsp_solve_box(n, D, d, lb, ub, x, &sub_config);

  release_workspace(ws, needs_free);
  return iters;
}

int nanoqsp_solve_affine(int n, int m, const double *restrict D,
                         const double *restrict d, const double *restrict A,
                         const double *restrict lb_x,
                         const double *restrict ub_x,
                         const double *restrict lb_A,
                         const double *restrict ub_A, double *restrict x,
                         const NanoqspConfig *config) {
  if (!within_bounds(n, n) || !within_bounds(n, m) || D == NULL || d == NULL ||
      x == NULL || (m > 0 && A == NULL) || !validate_config(config) ||
      !validate_bounds(n, lb_x, ub_x) || !validate_bounds(m, lb_A, ub_A) ||
      !validate_vector(n, x) || !validate_vector(n, d)) {
    return NANOQSP_ERR_INVALID_ARG;
  }
  if (config != NULL && config->strategy != NANOQSP_STRATEGY_ADMM)
    return NANOQSP_ERR_INVALID_STRATEGY;

  int required_ws = nanoqsp_workspace_size_affine(n, m);
  if (required_ws < 0)
    return required_ws;

  int max_iterations, needs_free;
  double tolerance;
  double *restrict ws = NULL;
  int status = prepare_workspace(config, required_ws, &max_iterations,
                                 &tolerance, &ws, &needs_free);
  if (status < 0)
    return status;

  project_bounds(n, lb_x, ub_x, x);

  int iters = solve_admm_unified(n, m, D, NULL, d, A, NULL, lb_x, ub_x, lb_A,
                                 ub_A, x, max_iterations, tolerance, ws);

  release_workspace(ws, needs_free);
  return iters;
}

int nanoqsp_solve_affine_sparse(int n, int m, const NanoqspCSR *D,
                                const double *restrict d, const NanoqspCSR *A,
                                const double *restrict lb_x,
                                const double *restrict ub_x,
                                const double *restrict lb_A,
                                const double *restrict ub_A, double *restrict x,
                                const NanoqspConfig *config) {
  int required_ws = nanoqsp_workspace_size_affine(n, m);
  if (required_ws < 0 || d == NULL || x == NULL || !validate_config(config) ||
      !validate_csr(D, n, n) || (m > 0 && !validate_csr(A, m, n)) ||
      !validate_bounds(n, lb_x, ub_x) || !validate_bounds(m, lb_A, ub_A) ||
      !validate_vector(n, x) || !validate_vector(n, d)) {
    return NANOQSP_ERR_INVALID_ARG;
  }
  if (config != NULL && config->strategy != NANOQSP_STRATEGY_ADMM)
    return NANOQSP_ERR_INVALID_STRATEGY;

  int max_iterations, needs_free;
  double tolerance;
  double *restrict ws = NULL;
  int status = prepare_workspace(config, required_ws, &max_iterations,
                                 &tolerance, &ws, &needs_free);
  if (status < 0)
    return status;

  project_bounds(n, lb_x, ub_x, x);

  int iters = solve_admm_unified(n, m, NULL, D, d, NULL, A, lb_x, ub_x, lb_A,
                                 ub_A, x, max_iterations, tolerance, ws);

  release_workspace(ws, needs_free);
  return iters;
}
