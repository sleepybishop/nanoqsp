#include "qsp_internal.h"
#include <float.h>

int nanoqsp_check_kkt(int n, const double *restrict dx,
                      const double *restrict d, const double *restrict lb,
                      const double *restrict ub, const double *restrict x,
                      double d_max, double tol) {
  double max_kkt = 0.0;
  double dx_max = 0.0;
  for (int i = 0; i < n; i++) {
    double dx_i = dx[i];
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

/* residual-maintaining projected coordinate descent for dense and CSR D */
int nanoqsp_solve_cd(const qsp_mat *D, const double *restrict d,
                     const double *restrict lb, const double *restrict ub,
                     double *restrict x, int max_iter, double tol,
                     qsp_arena *ws) {
  int n = D->rows;
  int is_sparse = (D->dense == NULL);
  double *restrict dx = qsp_arena_take(ws, n);
  double *restrict diag = is_sparse ? qsp_arena_take(ws, n) : NULL;
  if (!dx || (is_sparse && !diag))
    return NANOQSP_ERR_OUT_OF_MEMORY;

  if (is_sparse)
    qsp_mat_extract_diag(D, diag);

  double d_max = qsp_vec_norm_inf(n, d);
  qsp_mat_vec(D, x, dx);

  for (int iter = 0; iter < max_iter; iter++) {
    double max_diff = 0.0;
    double x_max = 0.0;

    if (!is_sparse) {
      const double *restrict dm = D->dense;
      size_t stride = (size_t)n;
      for (int i = 0; i < n; i++) {
        const double *restrict row = &dm[(size_t)i * stride];
        double D_ii = row[i];
        double old_val = x[i];
        if (D_ii > 0.0) {
          double lower = lb ? lb[i] : -INFINITY;
          double upper = ub ? ub[i] : INFINITY;
          double new_val =
              qsp_clamp(old_val - (dx[i] - d[i]) / D_ii, lower, upper);
          double delta = new_val - old_val;
          if (delta != 0.0) {
            x[i] = new_val;
            nanoqsp_blas.axpy(dx, row, delta, n);
          }
        }
        double diff = fabs(x[i] - old_val);
        if (diff > max_diff)
          max_diff = diff;
        double val_x = fabs(x[i]);
        if (val_x > x_max)
          x_max = val_x;
      }
    } else {
      const NanoqspCSR *S = D->csr;
      for (int i = 0; i < n; i++) {
        double D_ii = diag[i];
        double old_val = x[i];
        if (D_ii > 0.0) {
          double lower = lb ? lb[i] : -INFINITY;
          double upper = ub ? ub[i] : INFINITY;
          double new_val =
              qsp_clamp(old_val - (dx[i] - d[i]) / D_ii, lower, upper);
          double delta = new_val - old_val;
          if (delta != 0.0) {
            x[i] = new_val;
            int start = S->row_ptr[i];
            int end = S->row_ptr[i + 1];
            for (int idx = start; idx < end; idx++)
              dx[S->col_indices[idx]] += S->values[idx] * delta;
          }
        }
        double diff = fabs(x[i] - old_val);
        if (diff > max_diff)
          max_diff = diff;
        double val_x = fabs(x[i]);
        if (val_x > x_max)
          x_max = val_x;
      }
    }

    int small_step = (max_diff < tol * (1.0 + x_max));
    if (small_step || (iter + 1) % 64 == 0) {
      qsp_mat_vec(D, x, dx);
      if (small_step && nanoqsp_check_kkt(n, dx, d, lb, ub, x, d_max, tol))
        return iter + 1;
    }
  }
  return NANOQSP_ERR_MAX_ITERATIONS;
}

/* projected gradient descent with gershgorin step size */
int nanoqsp_solve_pg(const qsp_mat *D, const double *restrict d,
                     const double *restrict lb, const double *restrict ub,
                     double *restrict x, int max_iter, double tol,
                     qsp_arena *ws) {
  int n = D->rows;
  double *restrict dx = qsp_arena_take(ws, n);
  if (!dx)
    return NANOQSP_ERR_OUT_OF_MEMORY;

  double norm_inf = qsp_mat_norm_inf(D);
  double d_max = qsp_vec_norm_inf(n, d);
  double matrix_scale = norm_inf > 0.0 ? norm_inf : (d_max > 0.0 ? d_max : 1.0);

  for (int iter = 0; iter < max_iter; iter++) {
    qsp_mat_vec(D, x, dx);
    if (nanoqsp_check_kkt(n, dx, d, lb, ub, x, d_max, tol))
      return iter;

    for (int i = 0; i < n; i++) {
      double lower = lb ? lb[i] : -INFINITY;
      double upper = ub ? ub[i] : INFINITY;
      x[i] = qsp_clamp(x[i] - (dx[i] - d[i]) / matrix_scale, lower, upper);
    }
  }

  qsp_mat_vec(D, x, dx);
  if (nanoqsp_check_kkt(n, dx, d, lb, ub, x, d_max, tol))
    return max_iter;
  return NANOQSP_ERR_MAX_ITERATIONS;
}

/* nesterov accelerated projected gradient (FISTA) with adaptive restarts */
int nanoqsp_solve_fista(const qsp_mat *D, const double *restrict d,
                        const double *restrict lb, const double *restrict ub,
                        double *restrict x, int max_iter, double tol,
                        qsp_arena *ws) {
  int n = D->rows;
  double *restrict y = qsp_arena_take(ws, n);
  double *restrict next_x = qsp_arena_take(ws, n);
  if (!y || !next_x)
    return NANOQSP_ERR_OUT_OF_MEMORY;

  double norm_inf = qsp_mat_norm_inf(D);
  double d_max = qsp_vec_norm_inf(n, d);
  double matrix_scale = norm_inf > 0.0 ? norm_inf : (d_max > 0.0 ? d_max : 1.0);

  nanoqsp_blas.copy(y, x, n);
  int y_feasible = 1;
  double t = 1.0;

  for (int iter = 0; iter < max_iter; iter++) {
    qsp_mat_vec(D, y, next_x);

    /* free KKT check at feasible y avoids extra matvec on optimal iterates */
    if (y_feasible && nanoqsp_check_kkt(n, next_x, d, lb, ub, y, d_max, tol)) {
      nanoqsp_blas.copy(x, y, n);
      return iter;
    }

    double max_diff = 0.0;
    double step_y_max = 0.0;
    double x_max = 0.0;
    double dy_max = 0.0;

    for (int i = 0; i < n; i++) {
      double dy_i = next_x[i];
      double val_dy = fabs(dy_i);
      if (val_dy > dy_max)
        dy_max = val_dy;

      double lower = lb ? lb[i] : -INFINITY;
      double upper = ub ? ub[i] : INFINITY;
      double nx_i =
          qsp_clamp(y[i] - (dy_i - d[i]) / matrix_scale, lower, upper);
      next_x[i] = nx_i;

      double diff = fabs(nx_i - x[i]);
      if (diff > max_diff)
        max_diff = diff;
      double step_y = fabs(nx_i - y[i]);
      if (step_y > step_y_max)
        step_y_max = step_y;
      double val_x = fabs(nx_i);
      if (val_x > x_max)
        x_max = val_x;
    }

    double restart_check = 0.0;
    for (int i = 0; i < n; i++)
      restart_check += (y[i] - next_x[i]) * (next_x[i] - x[i]);

    double kkt_scale = 1.0 + (d_max > dy_max ? d_max : dy_max);
    if ((max_diff < tol * (1.0 + x_max) &&
         matrix_scale * step_y_max < 10.0 * tol * kkt_scale) ||
        iter == max_iter - 1) {
      qsp_mat_vec(D, next_x, y);
      if (nanoqsp_check_kkt(n, y, d, lb, ub, next_x, d_max, tol)) {
        nanoqsp_blas.copy(x, next_x, n);
        return iter + 1;
      }
    }

    if (restart_check > 0.0) {
      t = 1.0;
      nanoqsp_blas.copy(y, next_x, n);
      nanoqsp_blas.copy(x, next_x, n);
      y_feasible = 1;
    } else {
      double t_next = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t * t));
      double beta = (t - 1.0) / t_next;
      y_feasible = 1;
      for (int i = 0; i < n; i++) {
        double yi = next_x[i] + beta * (next_x[i] - x[i]);
        double lower = lb ? lb[i] : -INFINITY;
        double upper = ub ? ub[i] : INFINITY;
        if (yi < lower || yi > upper)
          y_feasible = 0;
        y[i] = yi;
        x[i] = next_x[i];
      }
      t = t_next;
    }
  }
  return NANOQSP_ERR_MAX_ITERATIONS;
}

static double qsp_scaled_quad_obj(int n, const double *restrict x,
                                  const double *restrict dx,
                                  const double *restrict d,
                                  double matrix_scale) {
  double obj = 0.0;
  for (int i = 0; i < n; i++)
    obj += x[i] * ((0.5 * dx[i] - d[i]) / matrix_scale);
  return obj;
}

/* spectral projected gradient with GLL non-monotone line search */
int nanoqsp_solve_spg(const qsp_mat *D, const double *restrict d,
                      const double *restrict lb, const double *restrict ub,
                      double *restrict x, int max_iter, double tol,
                      qsp_arena *ws) {
  enum { GLL_WINDOW = 8 };
  int n = D->rows;
  double *restrict q = qsp_arena_take(ws, n);
  double *restrict dx = qsp_arena_take(ws, n);
  double *restrict dq = qsp_arena_take(ws, n);
  if (!q || !dx || !dq)
    return NANOQSP_ERR_OUT_OF_MEMORY;

  double norm_inf = qsp_mat_norm_inf(D);
  double matrix_scale = norm_inf > 0.0 ? norm_inf : 1.0;
  double alpha = 1.0;
  double d_max = qsp_vec_norm_inf(n, d);

  qsp_mat_vec(D, x, dx);
  double f_curr = qsp_scaled_quad_obj(n, x, dx, d, matrix_scale);
  double f_hist[GLL_WINDOW];
  f_hist[0] = f_curr;
  int dx_exact = 1;

  for (int iter = 0; iter < max_iter; iter++) {
    if (nanoqsp_check_kkt(n, dx, d, lb, ub, x, d_max, tol)) {
      if (dx_exact)
        return iter;
      qsp_mat_vec(D, x, dx);
      dx_exact = 1;
      f_curr = qsp_scaled_quad_obj(n, x, dx, d, matrix_scale);
      f_hist[iter % GLL_WINDOW] = f_curr;
      if (nanoqsp_check_kkt(n, dx, d, lb, ub, x, d_max, tol))
        return iter;
    }

    double dir_max = 0.0;
    for (int i = 0; i < n; i++) {
      double lower = lb ? lb[i] : -INFINITY;
      double upper = ub ? ub[i] : INFINITY;
      double grad = (dx[i] - d[i]) / matrix_scale;
      double step_i = qsp_clamp(x[i] - alpha * grad, lower, upper) - x[i];
      q[i] = step_i;
      double mag = fabs(step_i);
      if (mag > dir_max)
        dir_max = mag;
    }

    if (!(dir_max > 0.0) || !isfinite(dir_max)) {
      if (!dx_exact) {
        qsp_mat_vec(D, x, dx);
        dx_exact = 1;
        if (nanoqsp_check_kkt(n, dx, d, lb, ub, x, d_max, tol))
          return iter;
      }
      return NANOQSP_ERR_MAX_ITERATIONS;
    }

    /* power-of-two scaling preserves exact mantissa bits when rescaling q */
    double scale_q = dir_max;
    if (dir_max >= DBL_MIN) {
      int exp_q = 0;
      (void)frexp(dir_max, &exp_q);
      scale_q = ldexp(1.0, exp_q - 1);
    }
    double inv_scale_q = 1.0 / scale_q;
    for (int i = 0; i < n; i++)
      q[i] *= inv_scale_q;

    qsp_mat_vec(D, q, dq);

    double q_dot_q = 0.0;
    double curvature = 0.0;
    double slope = 0.0;
    for (int i = 0; i < n; i++) {
      double qi = q[i];
      q_dot_q += qi * qi;
      curvature += qi * (dq[i] / matrix_scale);
      slope += qi * ((dx[i] - d[i]) / matrix_scale);
    }

    /* accept full BB step under GLL window; shrink to exact minimizer if not */
    double length = scale_q;
    int hist_len = iter + 1 < GLL_WINDOW ? iter + 1 : GLL_WINDOW;
    double f_ref = f_hist[0];
    for (int k = 1; k < hist_len; k++) {
      if (f_hist[k] > f_ref)
        f_ref = f_hist[k];
    }

    double delta_f = length * slope + 0.5 * length * length * curvature;
    if (!isfinite(f_curr) || !isfinite(delta_f) ||
        !(f_curr + delta_f <= f_ref + 1e-4 * length * slope)) {
      if (curvature > 0.0) {
        double minimizer = -slope / curvature;
        if (minimizer < length)
          length = minimizer;
      }
      delta_f = length * slope + 0.5 * length * length * curvature;
    }

    if (!(length > 0.0) || !isfinite(length))
      return NANOQSP_ERR_MAX_ITERATIONS;

    for (int i = 0; i < n; i++) {
      double lower = lb ? lb[i] : -INFINITY;
      double upper = ub ? ub[i] : INFINITY;
      double candidate = (length == scale_q)
                             ? x[i] - alpha * ((dx[i] - d[i]) / matrix_scale)
                             : x[i] + length * q[i];
      x[i] = qsp_clamp(candidate, lower, upper);
    }

    if ((iter + 1) % 32 == 0 || !isfinite(f_curr + delta_f)) {
      qsp_mat_vec(D, x, dx);
      f_curr = qsp_scaled_quad_obj(n, x, dx, d, matrix_scale);
      dx_exact = 1;
    } else {
      nanoqsp_blas.axpy(dx, dq, length, n);
      f_curr += delta_f;
      dx_exact = 0;
    }
    f_hist[(iter + 1) % GLL_WINDOW] = f_curr;

    double abs_curv = fabs(curvature);
    if (abs_curv > 0.0 && isfinite(abs_curv)) {
      double spectral_alpha = qsp_clamp(q_dot_q / abs_curv, 1e-12, 1e12);
      alpha = isfinite(spectral_alpha) ? spectral_alpha : 1.0;
    } else if (curvature == 0.0 && slope < 0.0) {
      alpha = alpha * 2.0 < 1e12 ? alpha * 2.0 : 1e12;
    } else {
      alpha = 1.0;
    }
  }

  if (!dx_exact)
    qsp_mat_vec(D, x, dx);
  if (nanoqsp_check_kkt(n, dx, d, lb, ub, x, d_max, tol))
    return max_iter;
  return NANOQSP_ERR_MAX_ITERATIONS;
}
