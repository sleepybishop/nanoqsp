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

static double get_matrix_norm_inf(int n, const double *D) {
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
static int solve_coordinate_descent(int n, const double *D, const double *d,
                                    const double *lb, const double *ub,
                                    double *x, int max_iter, double tol) {
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
      return iter + 1;
    }
  }
  return max_iter;
}

/* Strategy 2: Projected Gradient Descent */
static int solve_projected_gradient(int n, const double *D, const double *d,
                                    const double *lb, const double *ub,
                                    double *x, int max_iter, double tol,
                                    double *ws) {
  double norm_inf = get_matrix_norm_inf(n, D);
  double alpha = (norm_inf > 1e-12) ? (1.0 / norm_inf) : 0.1;

  double *next_x = ws;

  int iter;
  for (iter = 0; iter < max_iter; iter++) {
    double max_diff = 0.0;

    for (int i = 0; i < n; i++) {
      double grad_i = v_dot(&D[i * n], x, n) - d[i];

      double lower = lb ? lb[i] : -INFINITY;
      double upper = ub ? ub[i] : INFINITY;
      next_x[i] = clamp(x[i] - alpha * grad_i, lower, upper);

      double diff = fabs(next_x[i] - x[i]);
      if (diff > max_diff)
        max_diff = diff;
    }

    v_copy(x, next_x, n);

    if (max_diff < tol)
      return iter + 1;
  }
  return max_iter;
}

/* Strategy 3: Nesterov Accelerated Projected Gradient Descent with Adaptive
 * Restarts */
static int solve_accelerated_gradient(int n, const double *D, const double *d,
                                      const double *lb, const double *ub,
                                      double *x, int max_iter, double tol,
                                      double *ws) {
  double norm_inf = get_matrix_norm_inf(n, D);
  double alpha = (norm_inf > 1e-12) ? (1.0 / norm_inf) : 0.1;

  double *y = ws;
  double *next_x = ws + n;

  v_copy(y, x, n);

  double t = 1.0;
  int iter;
  for (iter = 0; iter < max_iter; iter++) {
    double max_diff = 0.0;

    for (int i = 0; i < n; i++) {
      double grad_i = v_dot(&D[i * n], y, n) - d[i];

      double lower = lb ? lb[i] : -INFINITY;
      double upper = ub ? ub[i] : INFINITY;
      next_x[i] = clamp(y[i] - alpha * grad_i, lower, upper);

      double diff = fabs(next_x[i] - x[i]);
      if (diff > max_diff)
        max_diff = diff;
    }

    if (max_diff < tol) {
      v_copy(x, next_x, n);
      return iter + 1;
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
static int solve_spectral_gradient(int n, const double *D, const double *d,
                                   const double *lb, const double *ub,
                                   double *x, int max_iter, double tol,
                                   double *ws) {
  double alpha = 0.01;
  double *next_x = ws;
  double *grad = ws + n;
  double *next_grad = ws + 2 * n;

  for (int i = 0; i < n; i++) {
    grad[i] = v_dot(&D[i * n], x, n) - d[i];
  }

  int iter;
  for (iter = 0; iter < max_iter; iter++) {
    double max_diff = 0.0;

    for (int i = 0; i < n; i++) {
      double lower = lb ? lb[i] : -INFINITY;
      double upper = ub ? ub[i] : INFINITY;
      next_x[i] = clamp(x[i] - alpha * grad[i], lower, upper);

      double diff = fabs(next_x[i] - x[i]);
      if (diff > max_diff)
        max_diff = diff;
    }

    if (max_diff < tol) {
      v_copy(x, next_x, n);
      return iter + 1;
    }

    for (int i = 0; i < n; i++) {
      next_grad[i] = v_dot(&D[i * n], next_x, n) - d[i];
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
static void cg_solve(int n, const double *D, double rho, const double *c,
                     double *x, double *r, double *p, double *Ap) {
  for (int i = 0; i < n; i++) {
    double Dx_i = v_dot(&D[i * n], x, n);
    r[i] = c[i] - (Dx_i + rho * x[i]);
    p[i] = r[i];
  }
  double r_sq = v_dot(r, r, n);

  for (int iter = 0; iter < n + 5; iter++) {
    if (r_sq < 1e-10)
      break;

    double pAp = 0.0;
    for (int i = 0; i < n; i++) {
      double Dp_i = v_dot(&D[i * n], p, n);
      Ap[i] = Dp_i + rho * p[i];
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

static int solve_admm(int n, const double *D, const double *d, const double *lb,
                      const double *ub, double *x, int max_iter, double tol,
                      double *ws) {
  double rho = 1.0;
  double *z = ws;
  double *y = ws + n;
  double *c = ws + 2 * n;
  double *r = ws + 3 * n;
  double *p = ws + 4 * n;
  double *Ap = ws + 5 * n;

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
    for (int i = 0; i < n; i++) {
      double res = fabs(x[i] - z[i]);
      if (res > prim_res)
        prim_res = res;
    }

    if (max_diff < tol && prim_res < tol) {
      v_copy(x, z, n);
      return iter + 1;
    }
  }
  v_copy(x, z, n);
  return max_iter;
}

int nanoqsp_solve_box(int n, const double *D, const double *d, const double *lb,
                      const double *ub, double *x,
                      const NanoqspConfig *config) {
  if (n <= 0 || D == NULL || d == NULL || x == NULL) {
    return NANOQSP_ERR_INVALID_ARG;
  }

  NanoqspStrategy strategy = NANOQSP_STRATEGY_COORDINATE_DESCENT;
  int max_iterations = 1000;
  double tolerance = 1e-6;
  double *ws = NULL;
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
      ws = (double *)malloc(required_ws * sizeof(double));
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

double nanoqsp_predict(int n, const double *x, const double *feature_vector) {
  if (n <= 0 || x == NULL || feature_vector == NULL)
    return 0.0;
  return v_dot(x, feature_vector, n);
}

int nanoqsp_solve_least_squares(int m, int n, const double *A, const double *b,
                                const double *lb, const double *ub, double *x,
                                const NanoqspConfig *config) {
  if (m <= 0 || n <= 0 || A == NULL || b == NULL || x == NULL)
    return NANOQSP_ERR_INVALID_ARG;

  NanoqspStrategy strategy = NANOQSP_STRATEGY_COORDINATE_DESCENT;
  double *ws = NULL;
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
  double *D = NULL;
  double *d = NULL;
  int needs_free = 0;

  if (ws != NULL && ws_size >= required_total) {
    D = ws;
    d = ws + (n * n);
  } else {
    D = (double *)malloc(n * n * sizeof(double));
    d = (double *)malloc(n * sizeof(double));
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
