#include "nanoqsp.h"
#include "nanoqsp_blas.h"
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#define NANOQSP_MAX_DOUBLES 268435455LL

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
  if (n <= 0 || !valid_strategy(strategy) || 6LL * n > NANOQSP_MAX_DOUBLES)
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
  if (n <= 0 || !valid_strategy(strategy) || 6LL * n > NANOQSP_MAX_DOUBLES)
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
  long long total;
  if (box_size < 0 || (long long)n * n > NANOQSP_MAX_DOUBLES)
    return NANOQSP_ERR_INVALID_ARG;
  total = (long long)n * n + n + box_size;
  if (total > NANOQSP_MAX_DOUBLES)
    return NANOQSP_ERR_INVALID_ARG;
  return (int)total;
}

int nanoqsp_workspace_size_affine(int n, int m) {
  long long total;
  if (n <= 0 || m < 0)
    return NANOQSP_ERR_INVALID_ARG;
  total = 7LL * n + 3LL * m;
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
    if (x[i] <= lower) {
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
      if (fabs(D_ii) > 1e-12) {
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
        if (x[i] <= lower) {
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
  double *restrict diag = ws;
  double *restrict g_temp = ws + n;
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
      if (fabs(D_ii) > 1e-12) {
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
  double alpha = (norm_inf > 1e-12) ? (1.0 / norm_inf) : 0.1;
  double d_max = get_vec_norm_inf(n, d);
  double *restrict dx = ws;

  for (int iter = 0; iter < max_iter; iter++) {
    eval_D_vec(n, D_dense, D_sparse, x, dx);
    if (check_kkt_from_Dx(n, dx, d, lb, ub, x, d_max, tol))
      return iter;

    for (int i = 0; i < n; i++) {
      double lower = lb ? lb[i] : -INFINITY;
      double upper = ub ? ub[i] : INFINITY;
      x[i] = clamp(x[i] - alpha * (dx[i] - d[i]), lower, upper);
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
  double alpha = (norm_inf > 1e-12) ? (1.0 / norm_inf) : 0.1;
  double d_max = get_vec_norm_inf(n, d);

  double *restrict y = ws;
  double *restrict next_x = ws + n;

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
      next_x[i] = clamp(y[i] - alpha * grad_i, lower, upper);

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
  double alpha = (norm_inf > 1e-12) ? (1.0 / norm_inf) : 0.01;
  double d_max = get_vec_norm_inf(n, d);
  double *restrict next_x = ws;
  double *restrict dx = ws + n;
  double *restrict next_dx = ws + 2 * n;

  eval_D_vec(n, D_dense, D_sparse, x, dx);

  for (int iter = 0; iter < max_iter; iter++) {
    if (check_kkt_from_Dx(n, dx, d, lb, ub, x, d_max, tol))
      return iter;

    for (int i = 0; i < n; i++) {
      double lower = lb ? lb[i] : -INFINITY;
      double upper = ub ? ub[i] : INFINITY;
      next_x[i] = clamp(x[i] - alpha * (dx[i] - d[i]), lower, upper);
    }

    eval_D_vec(n, D_dense, D_sparse, next_x, next_dx);

    double s_dot_s = 0.0;
    double s_dot_y = 0.0;
    for (int i = 0; i < n; i++) {
      double s_i = next_x[i] - x[i];
      double y_i = next_dx[i] - dx[i];

      s_dot_s += s_i * s_i;
      s_dot_y += s_i * y_i;
    }

    if (s_dot_y > 1e-12) {
      double val = s_dot_s / s_dot_y;
      if (val < 1e-10)
        val = 1e-10;
      if (val > 1e10)
        val = 1e10;
      alpha = val;
    }

    v_copy(x, next_x, n);
    double *restrict tmp_dx = dx;
    dx = next_dx;
    next_dx = tmp_dx;
  }

  if (check_kkt_from_Dx(n, dx, d, lb, ub, x, d_max, tol))
    return max_iter;
  return NANOQSP_ERR_MAX_ITERATIONS;
}

/* strategy 5: unified ADMM for box and general affine constraints */

static void cg_solve_unified(int n, int m, const double *restrict D_dense,
                             const NanoqspCSR *D_sparse,
                             const double *restrict A_dense,
                             const NanoqspCSR *A_sparse, double rho,
                             double r_tol_sq, const double *restrict c,
                             double *restrict x, double *restrict r,
                             double *restrict p, double *restrict Ap,
                             double *restrict tmp_n, double *restrict tmp_m) {
  int has_A = (m > 0 && (A_dense != NULL || A_sparse != NULL));

  eval_D_vec(n, D_dense, D_sparse, x, r);

  if (has_A) {
    eval_A_vec(m, n, A_dense, A_sparse, x, tmp_m);
    eval_AT_vec(m, n, A_dense, A_sparse, tmp_m, tmp_n);
    for (int i = 0; i < n; i++) {
      r[i] = c[i] - (r[i] + rho * (x[i] + tmp_n[i]));
    }
  } else {
    for (int i = 0; i < n; i++) {
      r[i] = c[i] - (r[i] + rho * x[i]);
    }
  }

  v_copy(p, r, n);
  double r_sq = v_dot(r, r, n);

  for (int iter = 0; iter < n + 5; iter++) {
    if (r_sq <= r_tol_sq)
      break;

    double pAp = 0.0;
    eval_D_vec(n, D_dense, D_sparse, p, Ap);

    if (has_A) {
      eval_A_vec(m, n, A_dense, A_sparse, p, tmp_m);
      eval_AT_vec(m, n, A_dense, A_sparse, tmp_m, tmp_n);
      for (int i = 0; i < n; i++) {
        Ap[i] += rho * (p[i] + tmp_n[i]);
        pAp += p[i] * Ap[i];
      }
    } else {
      for (int i = 0; i < n; i++) {
        Ap[i] += rho * p[i];
        pAp += p[i] * Ap[i];
      }
    }

    if (pAp < 1e-20)
      break;

    double alpha = r_sq / pAp;
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
  int has_A = (m > 0 && (A_dense != NULL || A_sparse != NULL));
  double d_max = get_vec_norm_inf(n, d);
  double cg_tol = 1e-2 * tol * (1.0 + d_max);
  double cg_tol_sq = cg_tol * cg_tol;
  if (cg_tol_sq > 1e-16)
    cg_tol_sq = 1e-16;
  if (cg_tol_sq < 1e-24)
    cg_tol_sq = 1e-24;

  double *restrict z = ws;
  double *restrict y = ws + n + m;
  double *restrict c = ws + 2 * (n + m);
  double *restrict r = ws + 2 * (n + m) + n;
  double *restrict p = ws + 2 * (n + m) + 2 * n;
  double *restrict Ap = ws + 2 * (n + m) + 3 * n;
  double *restrict Ax = ws + 2 * (n + m) + 4 * n;
  double *restrict tmp_n = ws + 2 * (n + m) + 4 * n + m;

  v_copy(z, x, n);
  if (has_A) {
    eval_A_vec(m, n, A_dense, A_sparse, x, &z[n]);
  }
  v_zero(y, n + m);

  for (int iter = 0; iter < max_iter; iter++) {
    for (int i = 0; i < n; i++) {
      c[i] = d[i] + rho * (z[i] - y[i]);
    }
    if (has_A) {
      double *restrict diff_A = Ax;
      for (int i = 0; i < m; i++) {
        diff_A[i] = z[n + i] - y[n + i];
      }
      eval_AT_vec(m, n, A_dense, A_sparse, diff_A, r);
      v_axpy(c, r, rho, n);
    }

    /* solve system (D + rho * (I + A^T * A)) x = c */
    cg_solve_unified(n, m, D_dense, D_sparse, A_dense, A_sparse, rho, cg_tol_sq,
                     c, x, r, p, Ap, tmp_n, Ax);

    double max_diff = 0.0;
    double prim_res = 0.0;
    double x_max = 0.0;
    double z_max = 0.0;
    double y_max = 0.0;
    int nan_found = 0;

    for (int i = 0; i < n; i++) {
      double old_z = z[i];
      double lower = lb_x ? lb_x[i] : -INFINITY;
      double upper = ub_x ? ub_x[i] : INFINITY;
      double z_i = clamp(x[i] + y[i], lower, upper);
      double y_i = y[i] + (x[i] - z_i);
      z[i] = z_i;
      y[i] = y_i;

      if (isnan(x[i]) || isnan(z_i))
        nan_found = 1;

      double diff = rho * fabs(z_i - old_z);
      if (diff > max_diff)
        max_diff = diff;

      double res = fabs(x[i] - z_i);
      if (res > prim_res)
        prim_res = res;

      double val_x = fabs(x[i]);
      if (val_x > x_max)
        x_max = val_x;
      double val_z = fabs(z_i);
      if (val_z > z_max)
        z_max = val_z;
      double val_y = rho * fabs(y_i);
      if (val_y > y_max)
        y_max = val_y;
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

        if (isnan(Ax[i]) || isnan(z_i))
          nan_found = 1;

        double diff = rho * fabs(z_i - old_z);
        if (diff > max_diff)
          max_diff = diff;

        double res = fabs(Ax[i] - z_i);
        if (res > prim_res)
          prim_res = res;

        double val_z = fabs(z_i);
        if (val_z > z_max)
          z_max = val_z;
        double val_y = rho * fabs(y_i);
        if (val_y > y_max)
          y_max = val_y;
      }
    }

    double dx_max = 0.0;
    if (!has_A) {
      for (int i = 0; i < n; i++) {
        double val_dx = fabs(d[i] - rho * (y[i] - (z[i] - x[i])));
        if (val_dx > dx_max)
          dx_max = val_dx;
      }
    } else {
      /* estimate dual gradient scale from x-update optimality */
      double *restrict temp_n = c;
      for (int i = 0; i < n; i++) {
        temp_n[i] = rho * (y[i] - (z[i] - x[i]));
      }
      double *restrict temp_m = Ax;
      for (int i = 0; i < m; i++) {
        temp_m[i] = rho * (y[n + i] - (z[n + i] - Ax[i]));
      }
      eval_AT_vec(m, n, A_dense, A_sparse, temp_m, r);
      for (int i = 0; i < n; i++) {
        double val_dx = fabs(d[i] - (temp_n[i] + r[i]));
        if (val_dx > dx_max)
          dx_max = val_dx;
      }
    }

    double prim_scale = 1.0 + (x_max > z_max ? x_max : z_max);
    double dual_scale =
        1.0 + (y_max > d_max ? (y_max > dx_max ? y_max : dx_max)
                             : (d_max > dx_max ? d_max : dx_max));

    if (!nan_found && max_diff < tol * dual_scale &&
        prim_res < tol * prim_scale) {
      v_copy(x, z, n);
      return iter + 1;
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
    ws = (double *restrict)malloc((size_t)required_ws * sizeof(double));
    if (!ws)
      return NANOQSP_ERR_OUT_OF_MEMORY;
    *needs_free = 1;
  }
  *ws_out = ws;
  return NANOQSP_SUCCESS;
}

/* public API implementations */

int nanoqsp_solve_box(int n, const double *restrict D, const double *restrict d,
                      const double *restrict lb, const double *restrict ub,
                      double *restrict x, const NanoqspConfig *config) {
  if ((long long)n * n > NANOQSP_MAX_DOUBLES)
    return NANOQSP_ERR_INVALID_ARG;
  if (n <= 0 || D == NULL || d == NULL || x == NULL ||
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

  if (needs_free)
    free(ws);
  return iters;
}

int nanoqsp_solve_box_sparse(int n, const NanoqspCSR *D,
                             const double *restrict d,
                             const double *restrict lb,
                             const double *restrict ub, double *restrict x,
                             const NanoqspConfig *config) {
  if (6LL * n > NANOQSP_MAX_DOUBLES)
    return NANOQSP_ERR_INVALID_ARG;
  if (n <= 0 || d == NULL || x == NULL || !validate_config(config) ||
      !validate_csr(D, n, n) || !validate_bounds(n, lb, ub) ||
      !validate_vector(n, x) || !validate_vector(n, d)) {
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

  if (needs_free)
    free(ws);
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
  if ((long long)m * n > NANOQSP_MAX_DOUBLES ||
      (long long)n * n > NANOQSP_MAX_DOUBLES)
    return NANOQSP_ERR_INVALID_ARG;
  if (m <= 0 || n <= 0 || A == NULL || b == NULL || x == NULL ||
      !validate_config(config) || !validate_bounds(n, lb, ub) ||
      !validate_vector(m * n, A) || !validate_vector(n, x) ||
      !validate_vector(m, b))
    return NANOQSP_ERR_INVALID_ARG;

  NanoqspStrategy strategy = NANOQSP_STRATEGY_COORDINATE_DESCENT;
  double *restrict ws = NULL;
  int ws_size = 0;

  if (config != NULL) {
    strategy = config->strategy;
    ws = config->workspace;
    ws_size = config->workspace_size;
  }

  int required_total = nanoqsp_workspace_size_least_squares(n, strategy);
  if (required_total < 0)
    return valid_strategy(strategy) ? required_total
                                    : NANOQSP_ERR_INVALID_STRATEGY;
  double *restrict D = NULL;
  double *restrict d = NULL;
  int needs_free = 0;

  if (ws != NULL && ws_size >= required_total) {
    D = ws;
    d = ws + (n * n);
  } else {
    D = (double *restrict)malloc((size_t)n * (size_t)n * sizeof(double));
    d = (double *restrict)malloc((size_t)n * sizeof(double));
    if (!D || !d) {
      free(D);
      free(d);
      return NANOQSP_ERR_OUT_OF_MEMORY;
    }
    needs_free = 1;
  }

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
  double epsilon = (norm_inf > 1e-12) ? (norm_inf * 1e-8) : 1e-8;
  for (int i = 0; i < n; i++)
    D[i * n + i] += epsilon;

  NanoqspConfig safe_config;
  const NanoqspConfig *pass_config = config;

  if (!needs_free && ws != NULL) {
    if (config != NULL) {
      safe_config = *config;
    } else {
      safe_config.strategy = strategy;
      safe_config.max_iterations = 1000;
      safe_config.tolerance = 1e-6;
    }
    safe_config.workspace = ws + (n * n) + n;
    safe_config.workspace_size = ws_size - ((n * n) + n);
    pass_config = &safe_config;
  }

  int iters = nanoqsp_solve_box(n, D, d, lb, ub, x, pass_config);

  if (needs_free) {
    free(D);
    free(d);
  }
  return iters;
}

int nanoqsp_solve_affine(int n, int m, const double *restrict D,
                         const double *restrict d, const double *restrict A,
                         const double *restrict lb_x,
                         const double *restrict ub_x,
                         const double *restrict lb_A,
                         const double *restrict ub_A, double *restrict x,
                         const NanoqspConfig *config) {
  if (n <= 0 || m < 0 || D == NULL || d == NULL || x == NULL) {
    return NANOQSP_ERR_INVALID_ARG;
  }
  if ((long long)n * n > NANOQSP_MAX_DOUBLES ||
      (long long)m * n > NANOQSP_MAX_DOUBLES || (m > 0 && A == NULL) ||
      !validate_config(config) || !validate_bounds(n, lb_x, ub_x) ||
      !validate_bounds(m, lb_A, ub_A) || !validate_vector(n, x) ||
      !validate_vector(n, d)) {
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

  if (needs_free)
    free(ws);
  return iters;
}

int nanoqsp_solve_affine_sparse(int n, int m, const NanoqspCSR *D,
                                const double *restrict d, const NanoqspCSR *A,
                                const double *restrict lb_x,
                                const double *restrict ub_x,
                                const double *restrict lb_A,
                                const double *restrict ub_A, double *restrict x,
                                const NanoqspConfig *config) {
  if (n <= 0 || m < 0 || d == NULL || x == NULL) {
    return NANOQSP_ERR_INVALID_ARG;
  }
  if (7LL * n + 3LL * m > NANOQSP_MAX_DOUBLES || !validate_config(config) ||
      !validate_csr(D, n, n) || (m > 0 && !validate_csr(A, m, n)) ||
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

  int iters = solve_admm_unified(n, m, NULL, D, d, NULL, A, lb_x, ub_x, lb_A,
                                 ub_A, x, max_iterations, tolerance, ws);

  if (needs_free)
    free(ws);
  return iters;
}
