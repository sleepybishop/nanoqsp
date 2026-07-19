#include "nanoqsp.h"
#include "nanoqsp_blas.h"
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

static inline double clamp(double val, double lb, double ub) {
  if (val < lb)
    return lb;
  if (val > ub)
    return ub;
  return val;
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

/* Strategy 1: Coordinate Descent */
static int solve_coordinate_descent(int n, const double *restrict D,
                                    const double *restrict d,
                                    const double *restrict lb,
                                    const double *restrict ub,
                                    double *restrict x, int max_iter,
                                    double tol) {
  int iter;
  for (iter = 0; iter < max_iter; iter++) {
    double max_diff = 0.0;
    for (int i = 0; i < n; i++) {
      double sum = v_dot(&D[i * n], x, n) - D[i * n + i] * x[i];
      double old_val = x[i];
      double D_ii = D[i * n + i];
      if (fabs(D_ii) > 1e-12) {
        double lower = lb ? lb[i] : -INFINITY;
        double upper = ub ? ub[i] : INFINITY;
        x[i] = clamp((d[i] - sum) / D_ii, lower, upper);
      }

      double diff = fabs(x[i] - old_val);
      if (diff > max_diff) {
        max_diff = diff;
      }
    }
    if (max_diff < tol) {
      double max_kkt = 0.0;
      double d_max = 0.0;
      double dx_max = 0.0;
      for (int i = 0; i < n; i++) {
        double grad_i = v_dot(&D[i * n], x, n) - d[i];
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

        double val_d = fabs(d[i]);
        if (val_d > d_max)
          d_max = val_d;
        double val_dx = fabs(grad_i + d[i]);
        if (val_dx > dx_max)
          dx_max = val_dx;
      }
      double scale = 1.0 + (d_max > dx_max ? d_max : dx_max);
      if (max_kkt < tol * scale) {
        return iter + 1;
      }
    }
  }
  return max_iter;
}

/* Strategy 2: Projected Gradient Descent */
static int solve_projected_gradient(int n, const double *restrict D,
                                    const double *restrict d,
                                    const double *restrict lb,
                                    const double *restrict ub,
                                    double *restrict x, int max_iter,
                                    double tol, double *restrict ws) {
  double norm_inf = get_matrix_norm_inf(n, D);
  double alpha = (norm_inf > 1e-12) ? (1.0 / norm_inf) : 0.1;

  double *restrict next_x = ws;

  int iter;
  for (iter = 0; iter < max_iter; iter++) {
    v_zero(next_x, n);
    for (int j = 0; j < n; j++) {
      v_axpy(next_x, &D[j * n], x[j], n);
    }

    double max_kkt = 0.0;
    double d_max = 0.0;
    double dx_max = 0.0;
    for (int i = 0; i < n; i++) {
      double grad_i = next_x[i] - d[i];

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

      double val_d = fabs(d[i]);
      if (val_d > d_max)
        d_max = val_d;
      double val_dx = fabs(grad_i + d[i]);
      if (val_dx > dx_max)
        dx_max = val_dx;

      next_x[i] = clamp(x[i] - alpha * grad_i, lower, upper);
    }

    v_copy(x, next_x, n);

    double scale = 1.0 + (d_max > dx_max ? d_max : dx_max);
    if (max_kkt < tol * scale)
      return iter + 1;
  }
  return max_iter;
}

/* Strategy 3: Nesterov Accelerated Projected Gradient Descent with Adaptive
 * Restarts */
static int solve_accelerated_gradient(int n, const double *restrict D,
                                      const double *restrict d,
                                      const double *restrict lb,
                                      const double *restrict ub,
                                      double *restrict x, int max_iter,
                                      double tol, double *restrict ws) {
  double norm_inf = get_matrix_norm_inf(n, D);
  double alpha = (norm_inf > 1e-12) ? (1.0 / norm_inf) : 0.1;

  double *restrict y = ws;
  double *restrict next_x = ws + n;

  v_copy(y, x, n);

  double t = 1.0;
  int iter;
  for (iter = 0; iter < max_iter; iter++) {
    double max_diff = 0.0;

    v_zero(next_x, n);
    for (int j = 0; j < n; j++) {
      v_axpy(next_x, &D[j * n], y[j], n);
    }

    for (int i = 0; i < n; i++) {
      double grad_i = next_x[i] - d[i];

      double lower = lb ? lb[i] : -INFINITY;
      double upper = ub ? ub[i] : INFINITY;
      next_x[i] = clamp(y[i] - alpha * grad_i, lower, upper);

      double diff = fabs(next_x[i] - x[i]);
      if (diff > max_diff)
        max_diff = diff;
    }

    if (max_diff < tol) {
      v_zero(y, n);
      for (int j = 0; j < n; j++) {
        v_axpy(y, &D[j * n], next_x[j], n);
      }
      double max_kkt = 0.0;
      double d_max = 0.0;
      double dx_max = 0.0;
      for (int i = 0; i < n; i++) {
        double grad_i = y[i] - d[i];
        double lower = lb ? lb[i] : -INFINITY;
        double upper = ub ? ub[i] : INFINITY;
        double viol = 0.0;
        if (next_x[i] <= lower) {
          if (grad_i < 0.0)
            viol = -grad_i;
        } else if (next_x[i] >= upper) {
          if (grad_i > 0.0)
            viol = grad_i;
        } else {
          viol = fabs(grad_i);
        }
        if (viol > max_kkt)
          max_kkt = viol;

        double val_d = fabs(d[i]);
        if (val_d > d_max)
          d_max = val_d;
        double val_dx = fabs(grad_i + d[i]);
        if (val_dx > dx_max)
          dx_max = val_dx;
      }
      double scale = 1.0 + (d_max > dx_max ? d_max : dx_max);
      if (max_kkt < tol * scale) {
        v_copy(x, next_x, n);
        return iter + 1;
      }
    }

    /* Adaptive Restart Condition */
    double restart_check = 0.0;
    for (int i = 0; i < n; i++) {
      restart_check += (y[i] - next_x[i]) * (next_x[i] - x[i]);
    }

    if (restart_check > 0.0) {
      t = 1.0;
      for (int i = 0; i < n; i++) {
        y[i] = next_x[i];
        x[i] = next_x[i];
      }
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
  return max_iter;
}

/* Strategy 4: Spectral Projected Gradient (SPG) */
static int solve_spectral_gradient(int n, const double *restrict D,
                                   const double *restrict d,
                                   const double *restrict lb,
                                   const double *restrict ub,
                                   double *restrict x, int max_iter, double tol,
                                   double *restrict ws) {
  double alpha = 0.01;
  double *restrict next_x = ws;
  double *restrict grad = ws + n;
  double *restrict next_grad = ws + 2 * n;

  v_zero(grad, n);
  for (int j = 0; j < n; j++) {
    v_axpy(grad, &D[j * n], x[j], n);
  }
  for (int i = 0; i < n; i++) {
    grad[i] -= d[i];
  }

  int iter;
  for (iter = 0; iter < max_iter; iter++) {
    double max_kkt = 0.0;
    double d_max = 0.0;
    double dx_max = 0.0;
    for (int i = 0; i < n; i++) {
      double lower = lb ? lb[i] : -INFINITY;
      double upper = ub ? ub[i] : INFINITY;
      double viol = 0.0;
      if (x[i] <= lower) {
        if (grad[i] < 0.0)
          viol = -grad[i];
      } else if (x[i] >= upper) {
        if (grad[i] > 0.0)
          viol = grad[i];
      } else {
        viol = fabs(grad[i]);
      }
      if (viol > max_kkt)
        max_kkt = viol;

      double val_d = fabs(d[i]);
      if (val_d > d_max)
        d_max = val_d;
      double val_dx = fabs(grad[i] + d[i]);
      if (val_dx > dx_max)
        dx_max = val_dx;
    }

    double scale = 1.0 + (d_max > dx_max ? d_max : dx_max);
    if (max_kkt < tol * scale) {
      return iter;
    }

    for (int i = 0; i < n; i++) {
      double lower = lb ? lb[i] : -INFINITY;
      double upper = ub ? ub[i] : INFINITY;
      next_x[i] = clamp(x[i] - alpha * grad[i], lower, upper);
    }

    v_zero(next_grad, n);
    for (int j = 0; j < n; j++) {
      v_axpy(next_grad, &D[j * n], next_x[j], n);
    }
    for (int i = 0; i < n; i++) {
      next_grad[i] -= d[i];
    }

    double s_dot_s = 0.0;
    double s_dot_y = 0.0;
    for (int i = 0; i < n; i++) {
      double s_i = next_x[i] - x[i];
      double y_i = next_grad[i] - grad[i];

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
    v_copy(grad, next_grad, n);
  }
  return max_iter;
}

/* Strategy 5: Nano ADMM */
static void cg_solve(int n, const double *restrict D, double rho,
                     const double *restrict c, double *restrict x,
                     double *restrict r, double *restrict p,
                     double *restrict Ap) {
  v_zero(r, n);
  for (int j = 0; j < n; j++) {
    v_axpy(r, &D[j * n], x[j], n);
  }
  for (int i = 0; i < n; i++) {
    r[i] = c[i] - (r[i] + rho * x[i]);
    p[i] = r[i];
  }
  double r_sq = v_dot(r, r, n);

  for (int iter = 0; iter < n + 5; iter++) {
    if (r_sq < 1e-10)
      break;

    double pAp = 0.0;
    v_zero(Ap, n);
    for (int j = 0; j < n; j++) {
      v_axpy(Ap, &D[j * n], p[j], n);
    }
    for (int i = 0; i < n; i++) {
      Ap[i] += rho * p[i];
      pAp += p[i] * Ap[i];
    }
    if (pAp < 1e-12)
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

static int solve_admm(int n, const double *restrict D, const double *restrict d,
                      const double *restrict lb, const double *restrict ub,
                      double *restrict x, int max_iter, double tol,
                      double *restrict ws) {
  double rho = 1.0;
  double *restrict z = ws;
  double *restrict y = ws + n;
  double *restrict c = ws + 2 * n;
  double *restrict r = ws + 3 * n;
  double *restrict p = ws + 4 * n;
  double *restrict Ap = ws + 5 * n;

  v_copy(z, x, n);
  v_zero(y, n);

  for (int iter = 0; iter < max_iter; iter++) {
    for (int i = 0; i < n; i++) {
      c[i] = d[i] - y[i] + rho * z[i];
    }

    cg_solve(n, D, rho, c, x, r, p, Ap);

    double max_diff = 0.0;
    for (int i = 0; i < n; i++) {
      double old_z = z[i];
      double lower = lb ? lb[i] : -INFINITY;
      double upper = ub ? ub[i] : INFINITY;
      z[i] = clamp(x[i] + y[i] / rho, lower, upper);

      double diff = fabs(z[i] - old_z);
      if (diff > max_diff)
        max_diff = diff;
    }

    for (int i = 0; i < n; i++) {
      y[i] += rho * (x[i] - z[i]);
    }

    double prim_res = 0.0;
    double x_max = 0.0;
    double z_max = 0.0;
    for (int i = 0; i < n; i++) {
      double res = fabs(x[i] - z[i]);
      if (res > prim_res)
        prim_res = res;
      double val_x = fabs(x[i]);
      if (val_x > x_max)
        x_max = val_x;
      double val_z = fabs(z[i]);
      if (val_z > z_max)
        z_max = val_z;
    }

    double y_max = 0.0;
    double d_max = 0.0;
    double dx_max = 0.0;
    for (int i = 0; i < n; i++) {
      double val_y = fabs(y[i]);
      if (val_y > y_max)
        y_max = val_y;
      double val_d = fabs(d[i]);
      if (val_d > d_max)
        d_max = val_d;
      double val_dx = fabs(d[i] - y[i] + rho * (z[i] - x[i]));
      if (val_dx > dx_max)
        dx_max = val_dx;
    }

    double prim_scale = 1.0 + (x_max > z_max ? x_max : z_max);
    double dual_scale =
        1.0 + (y_max > d_max ? (y_max > dx_max ? y_max : dx_max)
                             : (d_max > dx_max ? d_max : dx_max));

    if (max_diff < tol * dual_scale && prim_res < tol * prim_scale) {
      v_copy(x, z, n);
      return iter + 1;
    }
  }
  v_copy(x, z, n);
  return max_iter;
}

int nanoqsp_solve_box(int n, const double *restrict D, const double *restrict d,
                      const double *restrict lb, const double *restrict ub,
                      double *restrict x, const NanoqspConfig *config) {
  if (n <= 0 || D == NULL || d == NULL || x == NULL) {
    return NANOQSP_ERR_INVALID_ARG;
  }

  NanoqspStrategy strategy = NANOQSP_STRATEGY_COORDINATE_DESCENT;
  int max_iterations = 1000;
  double tolerance = 1e-6;
  double *restrict ws = NULL;
  int ws_size = 0;

  if (config != NULL) {
    strategy = config->strategy;
    if (config->max_iterations > 0)
      max_iterations = config->max_iterations;
    if (config->tolerance > 0.0)
      tolerance = config->tolerance;
    ws = config->workspace;
    ws_size = config->workspace_size;
  }

  int required_ws = 0;
  if (strategy == NANOQSP_STRATEGY_PROJECTED_GRADIENT)
    required_ws = n;
  else if (strategy == NANOQSP_STRATEGY_ACCELERATED_GRADIENT)
    required_ws = 2 * n;
  else if (strategy == NANOQSP_STRATEGY_SPECTRAL_GRADIENT)
    required_ws = 3 * n;
  else if (strategy == NANOQSP_STRATEGY_ADMM)
    required_ws = 6 * n;

  int needs_free = 0;
  if (required_ws > 0) {
    if (ws == NULL || ws_size < required_ws) {
      ws = (double *restrict)malloc(required_ws * sizeof(double));
      if (!ws)
        return NANOQSP_ERR_OUT_OF_MEMORY;
      needs_free = 1;
    }
  }

  int iters = NANOQSP_ERR_INVALID_STRATEGY;
  switch (strategy) {
  case NANOQSP_STRATEGY_COORDINATE_DESCENT:
    iters =
        solve_coordinate_descent(n, D, d, lb, ub, x, max_iterations, tolerance);
    break;
  case NANOQSP_STRATEGY_PROJECTED_GRADIENT:
    iters = solve_projected_gradient(n, D, d, lb, ub, x, max_iterations,
                                     tolerance, ws);
    break;
  case NANOQSP_STRATEGY_ACCELERATED_GRADIENT:
    iters = solve_accelerated_gradient(n, D, d, lb, ub, x, max_iterations,
                                       tolerance, ws);
    break;
  case NANOQSP_STRATEGY_SPECTRAL_GRADIENT:
    iters = solve_spectral_gradient(n, D, d, lb, ub, x, max_iterations,
                                    tolerance, ws);
    break;
  case NANOQSP_STRATEGY_ADMM:
    iters = solve_admm(n, D, d, lb, ub, x, max_iterations, tolerance, ws);
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
  if (m <= 0 || n <= 0 || A == NULL || b == NULL || x == NULL)
    return NANOQSP_ERR_INVALID_ARG;

  NanoqspStrategy strategy = NANOQSP_STRATEGY_COORDINATE_DESCENT;
  double *restrict ws = NULL;
  int ws_size = 0;

  if (config != NULL) {
    strategy = config->strategy;
    ws = config->workspace;
    ws_size = config->workspace_size;
  }

  int required_box = 0;
  if (strategy == NANOQSP_STRATEGY_PROJECTED_GRADIENT)
    required_box = n;
  else if (strategy == NANOQSP_STRATEGY_ACCELERATED_GRADIENT)
    required_box = 2 * n;
  else if (strategy == NANOQSP_STRATEGY_SPECTRAL_GRADIENT)
    required_box = 3 * n;
  else if (strategy == NANOQSP_STRATEGY_ADMM)
    required_box = 6 * n;

  int required_total = (n * n) + n + required_box;
  double *restrict D = NULL;
  double *restrict d = NULL;
  int needs_free = 0;

  if (ws != NULL && ws_size >= required_total) {
    D = ws;
    d = ws + (n * n);
  } else {
    D = (double *restrict)malloc(n * n * sizeof(double));
    d = (double *restrict)malloc(n * sizeof(double));
    if (!D || !d) {
      if (D)
        free(D);
      if (d)
        free(d);
      return NANOQSP_ERR_OUT_OF_MEMORY;
    }
    needs_free = 1;
  }

  v_zero(D, n * n);
  v_zero(d, n);

  for (int k = 0; k < m; k++) {
    for (int i = 0; i < n; i++) {
      double A_ki = A[k * n + i];
      d[i] += A_ki * b[k];
      for (int j = i; j < n; j++) {
        D[i * n + j] += A_ki * A[k * n + j];
      }
    }
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++) {
      D[i * n + j] = D[j * n + i];
    }
  }

  double norm_inf = get_matrix_norm_inf(n, D);
  double epsilon = 1e-8;
  if (norm_inf > 1e-12) {
    epsilon = norm_inf * 1e-8;
  }
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

static void sparse_mat_vec(const NanoqspCSR *D, const double *restrict x,
                           double *restrict y) {
  for (int i = 0; i < D->n; i++) {
    double sum = 0.0;
    int start = D->row_ptr[i];
    int end = D->row_ptr[i + 1];
    for (int idx = start; idx < end; idx++) {
      sum += D->values[idx] * x[D->col_indices[idx]];
    }
    y[i] = sum;
  }
}

static double get_sparse_matrix_norm_inf(const NanoqspCSR *D) {
  double max_row_sum = 0.0;
  for (int i = 0; i < D->n; i++) {
    double row_sum = 0.0;
    int start = D->row_ptr[i];
    int end = D->row_ptr[i + 1];
    for (int idx = start; idx < end; idx++) {
      row_sum += fabs(D->values[idx]);
    }
    if (row_sum > max_row_sum) {
      max_row_sum = row_sum;
    }
  }
  return max_row_sum;
}

static int solve_coordinate_descent_sparse(int n, const NanoqspCSR *D,
                                           const double *restrict d,
                                           const double *restrict lb,
                                           const double *restrict ub,
                                           double *restrict x, int max_iter,
                                           double tol, double *restrict ws) {
  double *restrict diag = ws;
  for (int i = 0; i < n; i++) {
    diag[i] = 0.0;
    int start = D->row_ptr[i];
    int end = D->row_ptr[i + 1];
    for (int idx = start; idx < end; idx++) {
      if (D->col_indices[idx] == i) {
        diag[i] = D->values[idx];
        break;
      }
    }
  }

  int iter;
  for (iter = 0; iter < max_iter; iter++) {
    double max_diff = 0.0;
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
      if (diff > max_diff) {
        max_diff = diff;
      }
    }
    if (max_diff < tol) {
      double max_kkt = 0.0;
      double d_max = 0.0;
      double dx_max = 0.0;
      double *restrict g_temp = ws;
      sparse_mat_vec(D, x, g_temp);
      for (int i = 0; i < n; i++) {
        double grad_i = g_temp[i] - d[i];
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

        double val_d = fabs(d[i]);
        if (val_d > d_max)
          d_max = val_d;
        double val_dx = fabs(grad_i + d[i]);
        if (val_dx > dx_max)
          dx_max = val_dx;
      }
      double scale = 1.0 + (d_max > dx_max ? d_max : dx_max);
      if (max_kkt < tol * scale) {
        return iter + 1;
      }
    }
  }
  return max_iter;
}

static int solve_projected_gradient_sparse(int n, const NanoqspCSR *D,
                                           const double *restrict d,
                                           const double *restrict lb,
                                           const double *restrict ub,
                                           double *restrict x, int max_iter,
                                           double tol, double *restrict ws) {
  double norm_inf = get_sparse_matrix_norm_inf(D);
  double alpha = (norm_inf > 1e-12) ? (1.0 / norm_inf) : 0.1;

  double *restrict next_x = ws;
  double *restrict g = ws + n;

  int iter;
  for (iter = 0; iter < max_iter; iter++) {
    sparse_mat_vec(D, x, g);

    double max_kkt = 0.0;
    double d_max = 0.0;
    double dx_max = 0.0;
    for (int i = 0; i < n; i++) {
      double grad_i = g[i] - d[i];
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

      double val_d = fabs(d[i]);
      if (val_d > d_max)
        d_max = val_d;
      double val_dx = fabs(grad_i + d[i]);
      if (val_dx > dx_max)
        dx_max = val_dx;

      next_x[i] = clamp(x[i] - alpha * grad_i, lower, upper);
    }

    for (int i = 0; i < n; i++)
      x[i] = next_x[i];

    double scale = 1.0 + (d_max > dx_max ? d_max : dx_max);
    if (max_kkt < tol * scale) {
      return iter + 1;
    }
  }
  return max_iter;
}

static int solve_accelerated_gradient_sparse(int n, const NanoqspCSR *D,
                                             const double *restrict d,
                                             const double *restrict lb,
                                             const double *restrict ub,
                                             double *restrict x, int max_iter,
                                             double tol, double *restrict ws) {
  double norm_inf = get_sparse_matrix_norm_inf(D);
  double alpha = (norm_inf > 1e-12) ? (1.0 / norm_inf) : 0.1;

  double *restrict y = ws;
  double *restrict next_x = ws + n;
  double *restrict g = ws + 2 * n;

  for (int i = 0; i < n; i++) {
    y[i] = x[i];
  }

  double t = 1.0;
  int iter;
  for (iter = 0; iter < max_iter; iter++) {
    double max_diff = 0.0;
    sparse_mat_vec(D, y, g);

    for (int i = 0; i < n; i++) {
      double grad_i = g[i] - d[i];
      double lower = lb ? lb[i] : -INFINITY;
      double upper = ub ? ub[i] : INFINITY;
      next_x[i] = clamp(y[i] - alpha * grad_i, lower, upper);

      double diff = fabs(next_x[i] - x[i]);
      if (diff > max_diff)
        max_diff = diff;
    }

    if (max_diff < tol) {
      sparse_mat_vec(D, next_x, g);
      double max_kkt = 0.0;
      double d_max = 0.0;
      double dx_max = 0.0;
      for (int i = 0; i < n; i++) {
        double grad_i = g[i] - d[i];
        double lower = lb ? lb[i] : -INFINITY;
        double upper = ub ? ub[i] : INFINITY;
        double viol = 0.0;
        if (next_x[i] <= lower) {
          if (grad_i < 0.0)
            viol = -grad_i;
        } else if (next_x[i] >= upper) {
          if (grad_i > 0.0)
            viol = grad_i;
        } else {
          viol = fabs(grad_i);
        }
        if (viol > max_kkt)
          max_kkt = viol;

        double val_d = fabs(d[i]);
        if (val_d > d_max)
          d_max = val_d;
        double val_dx = fabs(grad_i + d[i]);
        if (val_dx > dx_max)
          dx_max = val_dx;
      }
      double scale = 1.0 + (d_max > dx_max ? d_max : dx_max);
      if (max_kkt < tol * scale) {
        for (int i = 0; i < n; i++)
          x[i] = next_x[i];
        return iter + 1;
      }
    }

    double restart_check = 0.0;
    for (int i = 0; i < n; i++) {
      restart_check += (y[i] - next_x[i]) * (next_x[i] - x[i]);
    }

    if (restart_check > 0.0) {
      t = 1.0;
      for (int i = 0; i < n; i++) {
        y[i] = next_x[i];
        x[i] = next_x[i];
      }
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
  return max_iter;
}

static int solve_spectral_gradient_sparse(int n, const NanoqspCSR *D,
                                          const double *restrict d,
                                          const double *restrict lb,
                                          const double *restrict ub,
                                          double *restrict x, int max_iter,
                                          double tol, double *restrict ws) {
  double alpha = 0.01;
  double *restrict next_x = ws;
  double *restrict grad = ws + n;
  double *restrict next_grad = ws + 2 * n;

  sparse_mat_vec(D, x, grad);
  for (int i = 0; i < n; i++) {
    grad[i] -= d[i];
  }

  int iter;
  for (iter = 0; iter < max_iter; iter++) {
    double max_kkt = 0.0;
    double d_max = 0.0;
    double dx_max = 0.0;
    for (int i = 0; i < n; i++) {
      double lower = lb ? lb[i] : -INFINITY;
      double upper = ub ? ub[i] : INFINITY;
      double viol = 0.0;
      if (x[i] <= lower) {
        if (grad[i] < 0.0)
          viol = -grad[i];
      } else if (x[i] >= upper) {
        if (grad[i] > 0.0)
          viol = grad[i];
      } else {
        viol = fabs(grad[i]);
      }
      if (viol > max_kkt)
        max_kkt = viol;

      double val_d = fabs(d[i]);
      if (val_d > d_max)
        d_max = val_d;
      double val_dx = fabs(grad[i] + d[i]);
      if (val_dx > dx_max)
        dx_max = val_dx;
    }

    double scale = 1.0 + (d_max > dx_max ? d_max : dx_max);
    if (max_kkt < tol * scale) {
      return iter;
    }

    for (int i = 0; i < n; i++) {
      double lower = lb ? lb[i] : -INFINITY;
      double upper = ub ? ub[i] : INFINITY;
      next_x[i] = clamp(x[i] - alpha * grad[i], lower, upper);
    }

    sparse_mat_vec(D, next_x, next_grad);
    for (int i = 0; i < n; i++) {
      next_grad[i] -= d[i];
    }

    double s_dot_s = 0.0;
    double s_dot_y = 0.0;
    for (int i = 0; i < n; i++) {
      double s_i = next_x[i] - x[i];
      double y_i = next_grad[i] - grad[i];

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

    for (int i = 0; i < n; i++) {
      x[i] = next_x[i];
      grad[i] = next_grad[i];
    }
  }
  return max_iter;
}

static void cg_solve_sparse(int n, const NanoqspCSR *D, double rho,
                            const double *restrict c, double *restrict x,
                            double *restrict r, double *restrict p,
                            double *restrict Ap) {
  sparse_mat_vec(D, x, r);
  for (int i = 0; i < n; i++) {
    r[i] = c[i] - (r[i] + rho * x[i]);
    p[i] = r[i];
  }
  double r_sq = 0.0;
  for (int i = 0; i < n; i++)
    r_sq += r[i] * r[i];

  for (int iter = 0; iter < n + 5; iter++) {
    if (r_sq < 1e-10)
      break;

    sparse_mat_vec(D, p, Ap);
    double pAp = 0.0;
    for (int i = 0; i < n; i++) {
      Ap[i] += rho * p[i];
      pAp += p[i] * Ap[i];
    }
    if (pAp < 1e-12)
      break;

    double alpha = r_sq / pAp;
    double next_r_sq = 0.0;
    for (int i = 0; i < n; i++) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
      next_r_sq += r[i] * r[i];
    }

    double beta = next_r_sq / r_sq;
    for (int i = 0; i < n; i++) {
      p[i] = r[i] + beta * p[i];
    }
    r_sq = next_r_sq;
  }
}

static int solve_admm_sparse(int n, const NanoqspCSR *D,
                             const double *restrict d,
                             const double *restrict lb,
                             const double *restrict ub, double *restrict x,
                             int max_iter, double tol, double *restrict ws) {
  double rho = 1.0;
  double *restrict z = ws;
  double *restrict y = ws + n;
  double *restrict c = ws + 2 * n;
  double *restrict r = ws + 3 * n;
  double *restrict p = ws + 4 * n;
  double *restrict Ap = ws + 5 * n;

  for (int i = 0; i < n; i++) {
    z[i] = x[i];
    y[i] = 0.0;
  }

  for (int iter = 0; iter < max_iter; iter++) {
    for (int i = 0; i < n; i++) {
      c[i] = d[i] - y[i] + rho * z[i];
    }

    cg_solve_sparse(n, D, rho, c, x, r, p, Ap);

    double max_diff = 0.0;
    for (int i = 0; i < n; i++) {
      double old_z = z[i];
      double lower = lb ? lb[i] : -INFINITY;
      double upper = ub ? ub[i] : INFINITY;
      z[i] = clamp(x[i] + y[i] / rho, lower, upper);

      double diff = fabs(z[i] - old_z);
      if (diff > max_diff)
        max_diff = diff;
    }

    for (int i = 0; i < n; i++) {
      y[i] += rho * (x[i] - z[i]);
    }

    double prim_res = 0.0;
    double x_max = 0.0;
    double z_max = 0.0;
    for (int i = 0; i < n; i++) {
      double res = fabs(x[i] - z[i]);
      if (res > prim_res)
        prim_res = res;
      double val_x = fabs(x[i]);
      if (val_x > x_max)
        x_max = val_x;
      double val_z = fabs(z[i]);
      if (val_z > z_max)
        z_max = val_z;
    }

    double y_max = 0.0;
    double d_max = 0.0;
    double dx_max = 0.0;
    for (int i = 0; i < n; i++) {
      double val_y = fabs(y[i]);
      if (val_y > y_max)
        y_max = val_y;
      double val_d = fabs(d[i]);
      if (val_d > d_max)
        d_max = val_d;
      double val_dx = fabs(d[i] - y[i] + rho * (z[i] - x[i]));
      if (val_dx > dx_max)
        dx_max = val_dx;
    }

    double prim_scale = 1.0 + (x_max > z_max ? x_max : z_max);
    double dual_scale =
        1.0 + (y_max > d_max ? (y_max > dx_max ? y_max : dx_max)
                             : (d_max > dx_max ? d_max : dx_max));

    if (max_diff < tol * dual_scale && prim_res < tol * prim_scale) {
      for (int i = 0; i < n; i++)
        x[i] = z[i];
      return iter + 1;
    }
  }
  for (int i = 0; i < n; i++)
    x[i] = z[i];
  return max_iter;
}

int nanoqsp_solve_box_sparse(int n, const NanoqspCSR *D,
                             const double *restrict d,
                             const double *restrict lb,
                             const double *restrict ub, double *restrict x,
                             const NanoqspConfig *config) {
  if (n <= 0 || D == NULL || d == NULL || x == NULL) {
    return NANOQSP_ERR_INVALID_ARG;
  }

  NanoqspStrategy strategy = NANOQSP_STRATEGY_COORDINATE_DESCENT;
  int max_iterations = 1000;
  double tolerance = 1e-6;
  double *restrict ws = NULL;
  int ws_size = 0;

  if (config != NULL) {
    strategy = config->strategy;
    if (config->max_iterations > 0)
      max_iterations = config->max_iterations;
    if (config->tolerance > 0.0)
      tolerance = config->tolerance;
    ws = config->workspace;
    ws_size = config->workspace_size;
  }

  int required_ws = 0;
  if (strategy == NANOQSP_STRATEGY_COORDINATE_DESCENT)
    required_ws = n;
  else if (strategy == NANOQSP_STRATEGY_PROJECTED_GRADIENT)
    required_ws = 2 * n;
  else if (strategy == NANOQSP_STRATEGY_ACCELERATED_GRADIENT)
    required_ws = 3 * n;
  else if (strategy == NANOQSP_STRATEGY_SPECTRAL_GRADIENT)
    required_ws = 3 * n;
  else if (strategy == NANOQSP_STRATEGY_ADMM)
    required_ws = 6 * n;

  int needs_free = 0;
  if (required_ws > 0) {
    if (ws == NULL || ws_size < required_ws) {
      ws = (double *restrict)malloc(required_ws * sizeof(double));
      if (!ws)
        return NANOQSP_ERR_OUT_OF_MEMORY;
      needs_free = 1;
    }
  }

  int iters = NANOQSP_ERR_INVALID_STRATEGY;
  switch (strategy) {
  case NANOQSP_STRATEGY_COORDINATE_DESCENT:
    iters = solve_coordinate_descent_sparse(n, D, d, lb, ub, x, max_iterations,
                                            tolerance, ws);
    break;
  case NANOQSP_STRATEGY_PROJECTED_GRADIENT:
    iters = solve_projected_gradient_sparse(n, D, d, lb, ub, x, max_iterations,
                                            tolerance, ws);
    break;
  case NANOQSP_STRATEGY_ACCELERATED_GRADIENT:
    iters = solve_accelerated_gradient_sparse(n, D, d, lb, ub, x,
                                              max_iterations, tolerance, ws);
    break;
  case NANOQSP_STRATEGY_SPECTRAL_GRADIENT:
    iters = solve_spectral_gradient_sparse(n, D, d, lb, ub, x, max_iterations,
                                           tolerance, ws);
    break;
  case NANOQSP_STRATEGY_ADMM:
    iters =
        solve_admm_sparse(n, D, d, lb, ub, x, max_iterations, tolerance, ws);
    break;
  }

  if (needs_free)
    free(ws);
  return iters;
}

/* General Affine Constraints Helpers & Solvers */

static void mat_vec_dense(int m, int n, const double *restrict A,
                          const double *restrict x, double *restrict y) {
  for (int i = 0; i < m; i++) {
    double sum = 0.0;
    for (int j = 0; j < n; j++) {
      sum += A[i * n + j] * x[j];
    }
    y[i] = sum;
  }
}

static void mat_trans_vec_dense(int m, int n, const double *restrict A,
                                const double *restrict v, double *restrict y) {
  v_zero(y, n);
  for (int j = 0; j < m; j++) {
    v_axpy(y, &A[j * n], v[j], n);
  }
}

static void cg_solve_affine(int n, int m, const double *restrict D,
                            const double *restrict A, double rho,
                            const double *restrict c, double *restrict x,
                            double *restrict r, double *restrict p,
                            double *restrict Ap, double *restrict tmp_n,
                            double *restrict tmp_m) {
  /* Compute initial residual r = c - (D*x + rho*(x + A^T*(A*x))) */
  v_zero(r, n);
  for (int j = 0; j < n; j++) {
    v_axpy(r, &D[j * n], x[j], n);
  }

  if (m > 0 && A != NULL) {
    mat_vec_dense(m, n, A, x, tmp_m);
    mat_trans_vec_dense(m, n, A, tmp_m, tmp_n);
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
    if (r_sq < 1e-10)
      break;

    double pAp = 0.0;
    v_zero(Ap, n);
    for (int j = 0; j < n; j++) {
      v_axpy(Ap, &D[j * n], p[j], n);
    }

    if (m > 0 && A != NULL) {
      mat_vec_dense(m, n, A, p, tmp_m);
      mat_trans_vec_dense(m, n, A, tmp_m, tmp_n);
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

    if (pAp < 1e-12)
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

static int solve_admm_affine(int n, int m, const double *restrict D,
                             const double *restrict d, const double *restrict A,
                             const double *restrict lb_x,
                             const double *restrict ub_x,
                             const double *restrict lb_A,
                             const double *restrict ub_A, double *restrict x,
                             int max_iter, double tol, double *restrict ws) {
  double rho = 1.0;

  double *restrict z = ws;
  double *restrict y = ws + n + m;
  double *restrict c = ws + 2 * (n + m);
  double *restrict r = ws + 2 * (n + m) + n;
  double *restrict p = ws + 2 * (n + m) + 2 * n;
  double *restrict Ap = ws + 2 * (n + m) + 3 * n;
  double *restrict Ax = ws + 2 * (n + m) + 4 * n;

  for (int i = 0; i < n; i++) {
    z[i] = x[i];
  }
  if (m > 0 && A != NULL) {
    mat_vec_dense(m, n, A, x, &z[n]);
  }
  v_zero(y, n + m);

  for (int iter = 0; iter < max_iter; iter++) {
    for (int i = 0; i < n; i++) {
      c[i] = d[i] + rho * (z[i] - y[i]);
    }
    if (m > 0 && A != NULL) {
      double *restrict diff_A = Ax;
      for (int i = 0; i < m; i++) {
        diff_A[i] = z[n + i] - y[n + i];
      }
      mat_trans_vec_dense(m, n, A, diff_A, r);
      for (int i = 0; i < n; i++) {
        c[i] += rho * r[i];
      }
    }

    /* Solve system (D + rho * (I + A^T * A)) x = c */
    double *restrict tmp_n = ws + 2 * (n + m) + 4 * n + m;
    cg_solve_affine(n, m, D, A, rho, c, x, r, p, Ap, tmp_n, Ax);

    double max_diff = 0.0;
    for (int i = 0; i < n; i++) {
      double old_z = z[i];
      double lower = lb_x ? lb_x[i] : -INFINITY;
      double upper = ub_x ? ub_x[i] : INFINITY;
      z[i] = clamp(x[i] + y[i], lower, upper);
      y[i] += x[i] - z[i];

      double diff = fabs(z[i] - old_z);
      if (diff > max_diff)
        max_diff = diff;
    }
    if (m > 0 && A != NULL) {
      mat_vec_dense(m, n, A, x, Ax);
      for (int i = 0; i < m; i++) {
        double old_z = z[n + i];
        double lower = lb_A ? lb_A[i] : -INFINITY;
        double upper = ub_A ? ub_A[i] : INFINITY;
        z[n + i] = clamp(Ax[i] + y[n + i], lower, upper);
        y[n + i] += Ax[i] - z[n + i];

        double diff = fabs(z[n + i] - old_z);
        if (diff > max_diff)
          max_diff = diff;
      }
    }

    double prim_res = 0.0;
    double x_max = 0.0;
    double z_max = 0.0;
    for (int i = 0; i < n; i++) {
      double res = fabs(x[i] - z[i]);
      if (res > prim_res)
        prim_res = res;
      double val_x = fabs(x[i]);
      if (val_x > x_max)
        x_max = val_x;
      double val_z = fabs(z[i]);
      if (val_z > z_max)
        z_max = val_z;
    }
    if (m > 0 && A != NULL) {
      for (int i = 0; i < m; i++) {
        double res = fabs(Ax[i] - z[n + i]);
        if (res > prim_res)
          prim_res = res;
        double val_z = fabs(z[n + i]);
        if (val_z > z_max)
          z_max = val_z;
      }
    }

    double y_max = 0.0;
    double d_max = 0.0;
    double dx_max = 0.0;
    for (int i = 0; i < n; i++) {
      double val_y = fabs(y[i]);
      if (val_y > y_max)
        y_max = val_y;
      double val_d = fabs(d[i]);
      if (val_d > d_max)
        d_max = val_d;
    }
    if (m > 0 && A != NULL) {
      for (int i = 0; i < m; i++) {
        double val_y = fabs(y[n + i]);
        if (val_y > y_max)
          y_max = val_y;
      }
    }

    double *restrict temp_n = c;
    for (int i = 0; i < n; i++) {
      temp_n[i] = y[i] - rho * (z[i] - x[i]);
    }
    if (m > 0 && A != NULL) {
      double *restrict temp_m = Ax;
      for (int i = 0; i < m; i++) {
        temp_m[i] = y[n + i] - rho * (z[n + i] - Ax[i]);
      }
      mat_trans_vec_dense(m, n, A, temp_m, r);
      for (int i = 0; i < n; i++) {
        temp_n[i] += r[i];
      }
    }
    for (int i = 0; i < n; i++) {
      double val_dx = fabs(d[i] - temp_n[i]);
      if (val_dx > dx_max)
        dx_max = val_dx;
    }

    double prim_scale = 1.0 + (x_max > z_max ? x_max : z_max);
    double dual_scale =
        1.0 + (y_max > d_max ? (y_max > dx_max ? y_max : dx_max)
                             : (d_max > dx_max ? d_max : dx_max));

    if (max_diff < tol * dual_scale && prim_res < tol * prim_scale) {
      return iter + 1;
    }
  }
  return max_iter;
}

int nanoqsp_solve_affine(int n, int m, const double *restrict D,
                         const double *restrict d, const double *restrict A,
                         const double *restrict lb_x,
                         const double *restrict ub_x,
                         const double *restrict lb_A,
                         const double *restrict ub_A, double *restrict x,
                         const NanoqspConfig *config) {
  if (n <= 0 || D == NULL || d == NULL || x == NULL) {
    return NANOQSP_ERR_INVALID_ARG;
  }
  if (m > 0 && A == NULL) {
    return NANOQSP_ERR_INVALID_ARG;
  }

  int max_iterations = 1000;
  double tolerance = 1e-6;
  double *restrict ws = NULL;
  int ws_size = 0;

  if (config != NULL) {
    if (config->max_iterations > 0)
      max_iterations = config->max_iterations;
    if (config->tolerance > 0.0)
      tolerance = config->tolerance;
    ws = config->workspace;
    ws_size = config->workspace_size;
  }

  int required_ws = 7 * n + 3 * m;
  int needs_free = 0;
  if (ws == NULL || ws_size < required_ws) {
    ws = (double *restrict)malloc(required_ws * sizeof(double));
    if (!ws)
      return NANOQSP_ERR_OUT_OF_MEMORY;
    needs_free = 1;
  }

  int iters = solve_admm_affine(n, m, D, d, A, lb_x, ub_x, lb_A, ub_A, x,
                                max_iterations, tolerance, ws);

  if (needs_free)
    free(ws);
  return iters;
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

static void cg_solve_affine_sparse(int n, int m, const NanoqspCSR *D,
                                   const NanoqspCSR *A, double rho,
                                   const double *restrict c, double *restrict x,
                                   double *restrict r, double *restrict p,
                                   double *restrict Ap, double *restrict tmp_n,
                                   double *restrict tmp_m) {
  v_zero(r, n);
  sparse_mat_vec(D, x, r);

  if (m > 0 && A != NULL) {
    sparse_mat_vec(A, x, tmp_m);
    sparse_mat_trans_vec(m, n, A, tmp_m, tmp_n);
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
    if (r_sq < 1e-10)
      break;

    double pAp = 0.0;
    v_zero(Ap, n);
    sparse_mat_vec(D, p, Ap);

    if (m > 0 && A != NULL) {
      sparse_mat_vec(A, p, tmp_m);
      sparse_mat_trans_vec(m, n, A, tmp_m, tmp_n);
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

    if (pAp < 1e-12)
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

static int solve_admm_affine_sparse(
    int n, int m, const NanoqspCSR *D, const double *restrict d,
    const NanoqspCSR *A, const double *restrict lb_x,
    const double *restrict ub_x, const double *restrict lb_A,
    const double *restrict ub_A, double *restrict x, int max_iter, double tol,
    double *restrict ws) {
  double rho = 1.0;

  double *restrict z = ws;
  double *restrict y = ws + n + m;
  double *restrict c = ws + 2 * (n + m);
  double *restrict r = ws + 2 * (n + m) + n;
  double *restrict p = ws + 2 * (n + m) + 2 * n;
  double *restrict Ap = ws + 2 * (n + m) + 3 * n;
  double *restrict Ax = ws + 2 * (n + m) + 4 * n;

  for (int i = 0; i < n; i++) {
    z[i] = x[i];
  }
  if (m > 0 && A != NULL) {
    sparse_mat_vec(A, x, &z[n]);
  }
  v_zero(y, n + m);

  for (int iter = 0; iter < max_iter; iter++) {
    for (int i = 0; i < n; i++) {
      c[i] = d[i] + rho * (z[i] - y[i]);
    }
    if (m > 0 && A != NULL) {
      double *restrict diff_A = Ax;
      for (int i = 0; i < m; i++) {
        diff_A[i] = z[n + i] - y[n + i];
      }
      sparse_mat_trans_vec(m, n, A, diff_A, r);
      for (int i = 0; i < n; i++) {
        c[i] += rho * r[i];
      }
    }

    /* Solve system (D + rho * (I + A^T * A)) x = c */
    double *restrict tmp_n = ws + 2 * (n + m) + 4 * n + m;
    cg_solve_affine_sparse(n, m, D, A, rho, c, x, r, p, Ap, tmp_n, Ax);

    double max_diff = 0.0;
    for (int i = 0; i < n; i++) {
      double old_z = z[i];
      double lower = lb_x ? lb_x[i] : -INFINITY;
      double upper = ub_x ? ub_x[i] : INFINITY;
      z[i] = clamp(x[i] + y[i], lower, upper);
      y[i] += x[i] - z[i];

      double diff = fabs(z[i] - old_z);
      if (diff > max_diff)
        max_diff = diff;
    }
    if (m > 0 && A != NULL) {
      sparse_mat_vec(A, x, Ax);
      for (int i = 0; i < m; i++) {
        double old_z = z[n + i];
        double lower = lb_A ? lb_A[i] : -INFINITY;
        double upper = ub_A ? ub_A[i] : INFINITY;
        z[n + i] = clamp(Ax[i] + y[n + i], lower, upper);
        y[n + i] += Ax[i] - z[n + i];

        double diff = fabs(z[n + i] - old_z);
        if (diff > max_diff)
          max_diff = diff;
      }
    }

    double prim_res = 0.0;
    double x_max = 0.0;
    double z_max = 0.0;
    for (int i = 0; i < n; i++) {
      double res = fabs(x[i] - z[i]);
      if (res > prim_res)
        prim_res = res;
      double val_x = fabs(x[i]);
      if (val_x > x_max)
        x_max = val_x;
      double val_z = fabs(z[i]);
      if (val_z > z_max)
        z_max = val_z;
    }
    if (m > 0 && A != NULL) {
      for (int i = 0; i < m; i++) {
        double res = fabs(Ax[i] - z[n + i]);
        if (res > prim_res)
          prim_res = res;
        double val_z = fabs(z[n + i]);
        if (val_z > z_max)
          z_max = val_z;
      }
    }

    double y_max = 0.0;
    double d_max = 0.0;
    double dx_max = 0.0;
    for (int i = 0; i < n; i++) {
      double val_y = fabs(y[i]);
      if (val_y > y_max)
        y_max = val_y;
      double val_d = fabs(d[i]);
      if (val_d > d_max)
        d_max = val_d;
    }
    if (m > 0 && A != NULL) {
      for (int i = 0; i < m; i++) {
        double val_y = fabs(y[n + i]);
        if (val_y > y_max)
          y_max = val_y;
      }
    }

    double *restrict temp_n = c;
    for (int i = 0; i < n; i++) {
      temp_n[i] = y[i] - rho * (z[i] - x[i]);
    }
    if (m > 0 && A != NULL) {
      double *restrict temp_m = Ax;
      for (int i = 0; i < m; i++) {
        temp_m[i] = y[n + i] - rho * (z[n + i] - Ax[i]);
      }
      sparse_mat_trans_vec(m, n, A, temp_m, r);
      for (int i = 0; i < n; i++) {
        temp_n[i] += r[i];
      }
    }
    for (int i = 0; i < n; i++) {
      double val_dx = fabs(d[i] - temp_n[i]);
      if (val_dx > dx_max)
        dx_max = val_dx;
    }

    double prim_scale = 1.0 + (x_max > z_max ? x_max : z_max);
    double dual_scale =
        1.0 + (y_max > d_max ? (y_max > dx_max ? y_max : dx_max)
                             : (d_max > dx_max ? d_max : dx_max));

    if (max_diff < tol * dual_scale && prim_res < tol * prim_scale) {
      for (int i = 0; i < n; i++)
        x[i] = z[i];
      return iter + 1;
    }
  }
  for (int i = 0; i < n; i++)
    x[i] = z[i];
  return max_iter;
}

int nanoqsp_solve_affine_sparse(int n, int m, const NanoqspCSR *D,
                                const double *restrict d, const NanoqspCSR *A,
                                const double *restrict lb_x,
                                const double *restrict ub_x,
                                const double *restrict lb_A,
                                const double *restrict ub_A, double *restrict x,
                                const NanoqspConfig *config) {
  if (n <= 0 || D == NULL || d == NULL || x == NULL) {
    return NANOQSP_ERR_INVALID_ARG;
  }
  if (m > 0 && A == NULL) {
    return NANOQSP_ERR_INVALID_ARG;
  }

  int max_iterations = 1000;
  double tolerance = 1e-6;
  double *restrict ws = NULL;
  int ws_size = 0;

  if (config != NULL) {
    if (config->max_iterations > 0)
      max_iterations = config->max_iterations;
    if (config->tolerance > 0.0)
      tolerance = config->tolerance;
    ws = config->workspace;
    ws_size = config->workspace_size;
  }

  int required_ws = 7 * n + 3 * m;
  int needs_free = 0;
  if (ws == NULL || ws_size < required_ws) {
    ws = (double *restrict)malloc(required_ws * sizeof(double));
    if (!ws)
      return NANOQSP_ERR_OUT_OF_MEMORY;
    needs_free = 1;
  }

  int iters = solve_admm_affine_sparse(n, m, D, d, A, lb_x, ub_x, lb_A, ub_A, x,
                                       max_iterations, tolerance, ws);

  if (needs_free)
    free(ws);
  return iters;
}
