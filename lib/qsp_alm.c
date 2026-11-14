#include "qsp_internal.h"
#include <float.h>

/* evaluate affine ALM state, exact global KKT residuals, and inner gradient */
static double qsp_alm_eval_state(
    int n, int m, const double *restrict x, const double *restrict x_prox,
    const double *restrict dx, const double *restrict d,
    const double *restrict w_diag, const qsp_mat *A, const double *restrict ax,
    const double *restrict bar_y, const double *restrict lb_x,
    const double *restrict ub_x, const double *restrict lb_A,
    const double *restrict ub_A, double mu, double sigma, double sigma_A,
    double eff_scale, double d_max, double *restrict aq_scratch,
    double *restrict g_out, double *out_inner_kkt, double *out_inner_step,
    double *out_x_scale, double *out_global_dual, double *out_dual_scale,
    double *out_aff_prim, double *out_aff_scale, int *out_nonfinite) {
  int nonfinite = 0;
  double aff_obj = 0.0;
  double aff_prim = 0.0;
  double ax_max = 0.0;
  double z_max = 0.0;
  double inv_eff = 1.0 / eff_scale;
  double mu_scaled = mu * inv_eff;

  for (int j = 0; j < m; j++) {
    double lo = lb_A ? lb_A[j] : -INFINITY;
    double hi = ub_A ? ub_A[j] : INFINITY;
    double ax_j = ax[j];
    double sj = ax_j + sigma_A * (bar_y[j] / mu);
    double zj = qsp_clamp(sj, lo, hi);
    double bar_ej = (sj - zj) / sigma_A;

    if (!isfinite(ax_j) || !isfinite(sj) || !isfinite(bar_ej))
      nonfinite = 1;

    aq_scratch[j] = (mu * bar_ej) / sigma_A;
    aff_obj += 0.5 * mu_scaled * bar_ej * bar_ej;

    double err = fabs(ax_j - zj);
    if (err > aff_prim)
      aff_prim = err;
    double v_ax = fabs(ax_j);
    if (v_ax > ax_max)
      ax_max = v_ax;
    double v_z = fabs(zj);
    if (v_z > z_max)
      z_max = v_z;
  }

  qsp_mat_tvec(A, aq_scratch, g_out);

  double quad_obj = 0.0;
  double x_max = 0.0;
  double dx_max = 0.0;
  double aty_max = 0.0;
  double global_dual = 0.0;
  double inner_kkt = 0.0;
  double inner_step = 0.0;

  for (int i = 0; i < n; i++) {
    double xi = x[i];
    double dxi = dx[i];
    double atyi = g_out[i];
    double diff_prox = xi - x_prox[i];
    double raw_grad = dxi - d[i] + atyi;
    double tot_grad = raw_grad + sigma * diff_prox;

    if (!isfinite(xi) || !isfinite(raw_grad) || !isfinite(tot_grad))
      nonfinite = 1;

    g_out[i] = tot_grad;
    quad_obj += xi * ((0.5 * dxi - d[i]) * inv_eff) +
                0.5 * (sigma * inv_eff) * diff_prox * diff_prox;

    double v_x = fabs(xi);
    if (v_x > x_max)
      x_max = v_x;
    double v_dx = fabs(dxi);
    if (v_dx > dx_max)
      dx_max = v_dx;
    double v_aty = fabs(atyi);
    if (v_aty > aty_max)
      aty_max = v_aty;

    double lo = lb_x ? lb_x[i] : -INFINITY;
    double hi = ub_x ? ub_x[i] : INFINITY;
    double g_viol = 0.0;
    double in_viol = 0.0;
    if (lo < hi) {
      if (xi <= lo) {
        if (raw_grad < 0.0)
          g_viol = -raw_grad;
        if (tot_grad < 0.0)
          in_viol = -tot_grad;
      } else if (xi >= hi) {
        if (raw_grad > 0.0)
          g_viol = raw_grad;
        if (tot_grad > 0.0)
          in_viol = tot_grad;
      } else {
        g_viol = fabs(raw_grad);
        in_viol = fabs(tot_grad);
      }
    }
    if (g_viol > global_dual)
      global_dual = g_viol;
    if (in_viol > inner_kkt)
      inner_kkt = in_viol;
    double step_viol = in_viol / w_diag[i];
    if (step_viol > inner_step)
      inner_step = step_viol;
  }

  *out_inner_kkt = inner_kkt;
  *out_inner_step = inner_step;
  *out_x_scale = 1.0 + x_max;
  *out_global_dual = global_dual;
  *out_dual_scale =
      1.0 + (aty_max > d_max ? (aty_max > dx_max ? aty_max : dx_max)
                             : (d_max > dx_max ? d_max : dx_max));
  *out_aff_prim = aff_prim;
  *out_aff_scale = 1.0 + (ax_max > z_max ? ax_max : z_max);
  *out_nonfinite = nonfinite;
  return quad_obj + aff_obj;
}

/* evaluate piecewise-quadratic 1D objective difference, slope, and curvature */
static void
qsp_alm_line_eval(int m, const double *restrict ax, const double *restrict aq,
                  const double *restrict bar_y, const double *restrict lb_A,
                  const double *restrict ub_A, double mu, double sigma_A,
                  double eff_scale, double slope_quad, double curv_quad,
                  double t, double *out_delta, double *out_slope,
                  double *out_curv) {
  double mu_scaled = mu / eff_scale;
  double delta_aff = 0.0;
  double slope_aff = 0.0;
  double curv_aff = 0.0;

  for (int j = 0; j < m; j++) {
    double lo = lb_A ? lb_A[j] : -INFINITY;
    double hi = ub_A ? ub_A[j] : INFINITY;
    double s0 = ax[j] + sigma_A * (bar_y[j] / mu);
    double z0 = qsp_clamp(s0, lo, hi);
    double e0 = (s0 - z0) / sigma_A;

    double st = s0 + t * aq[j];
    double zt = qsp_clamp(st, lo, hi);
    double et = (st - zt) / sigma_A;
    double aq_bar = aq[j] / sigma_A;

    delta_aff += 0.5 * mu_scaled * (et - e0) * (et + e0);
    slope_aff += mu_scaled * et * aq_bar;
    if (st <= lo || st >= hi)
      curv_aff += mu_scaled * aq_bar * aq_bar;
  }

  *out_delta = t * slope_quad + 0.5 * t * t * curv_quad + delta_aff;
  *out_slope = slope_quad + t * curv_quad + slope_aff;
  *out_curv = curv_quad + curv_aff;
}

/* build diagonal curvature preconditioner for inner box SPG */
static void qsp_alm_update_diag(const qsp_mat *D, const qsp_mat *A, double mu,
                                double sigma, double sigma_A, double fallback,
                                double *restrict w_diag) {
  int n = D->rows;
  int m = A->rows;
  qsp_mat_extract_diag(D, w_diag);
  for (int i = 0; i < n; i++)
    w_diag[i] = fabs(w_diag[i]) + sigma;

  double factor = sqrt(mu) / sigma_A;
  if (factor > 0.0 && isfinite(factor)) {
    if (A->dense != NULL) {
      size_t stride = (size_t)n;
      for (int j = 0; j < m; j++) {
        const double *restrict row = &A->dense[(size_t)j * stride];
        for (int i = 0; i < n; i++) {
          double t = factor * row[i];
          w_diag[i] += t * t;
        }
      }
    } else {
      const NanoqspCSR *S = A->csr;
      for (int idx = 0; idx < S->nnz; idx++) {
        double t = factor * S->values[idx];
        w_diag[S->col_indices[idx]] += t * t;
      }
    }
  }

  for (int i = 0; i < n; i++) {
    if (!(w_diag[i] > 0.0) || !isfinite(w_diag[i]))
      w_diag[i] = fallback;
  }
}

/* proximal augmented lagrangian outer loop with SPG inner solver */
int nanoqsp_solve_alm_spg(const qsp_mat *D, const double *restrict d,
                          const qsp_mat *A, const double *restrict lb_x,
                          const double *restrict ub_x,
                          const double *restrict lb_A,
                          const double *restrict ub_A, double *restrict x,
                          int max_iter, double tol, qsp_arena *ws) {
  enum { GLL_WINDOW = 8 };
  int n = D->rows;
  int m = A ? A->rows : 0;
  if (A == NULL || m <= 0)
    return nanoqsp_solve_spg(D, d, lb_x, ub_x, x, max_iter, tol, ws);

  double d_norm_inf = qsp_mat_norm_inf(D);
  double a_sum_sq = 0.0;
  double a_scale_raw = qsp_mat_scaled_fro(A, &a_sum_sq);
  if (!isfinite(d_norm_inf) || !isfinite(a_scale_raw) || !isfinite(a_sum_sq))
    return NANOQSP_ERR_MAX_ITERATIONS;

  double *restrict dx = qsp_arena_take(ws, n);
  double *restrict g = qsp_arena_take(ws, n);
  double *restrict q = qsp_arena_take(ws, n);
  double *restrict dq = qsp_arena_take(ws, n);
  double *restrict x_prox = qsp_arena_take(ws, n);
  double *restrict w_diag = qsp_arena_take(ws, n);
  double *restrict bar_y = qsp_arena_take(ws, m);
  double *restrict ax = qsp_arena_take(ws, m);
  double *restrict aq = qsp_arena_take(ws, m);
  if (!dx || !g || !q || !dq || !x_prox || !w_diag || !bar_y || !ax || !aq)
    return NANOQSP_ERR_OUT_OF_MEMORY;

  double sigma_A = 1.0;
  if (a_scale_raw > 0.0 && a_sum_sq > 0.0) {
    double s = a_scale_raw * sqrt(a_sum_sq / (double)m);
    if (s > 0.0 && isfinite(s))
      sigma_A = s;
  }

  qsp_mat_extract_diag(D, g);
  double diag_min = INFINITY;
  double diag_max = 0.0;
  for (int i = 0; i < n; i++) {
    double di = fabs(g[i]);
    if (di > 0.0) {
      if (di < diag_min)
        diag_min = di;
      if (di > diag_max)
        diag_max = di;
    }
  }

  double d_max = qsp_vec_norm_inf(n, d);
  double mu = 1.0;
  if (diag_max > 0.0) {
    mu = sqrt(diag_min) * sqrt(diag_max);
    if (mu > 10.0 * diag_min)
      mu = 10.0 * diag_min;
  } else if (d_norm_inf > 0.0) {
    mu = d_norm_inf / sqrt((double)n);
  } else if (d_max > 0.0) {
    mu = d_max;
  }
  if (!(mu > 0.0) || !isfinite(mu))
    mu = 1.0;

  double mu_0 = mu;
  double mu_ceil_base = diag_max > mu_0 ? diag_max : mu_0;
  double mu_max = (mu_ceil_base <= 1e180) ? (mu_ceil_base * 1e8) : 1e188;
  double sigma = 1e-6 * mu_0;
  double init_in = 100.0 * tol;
  if (init_in > 1e-3)
    init_in = 1e-3;
  double inner_tol = tol > init_in ? tol : init_in;
  double prev_rel_aff = INFINITY;
  int inner_cap = n < 128 ? 512 : (4 * n < 2048 ? 4 * n : 2048);

  nanoqsp_blas.zero(bar_y, m);
  nanoqsp_blas.copy(x_prox, x, n);
  qsp_mat_vec(D, x, dx);
  qsp_mat_vec(A, x, ax);
  int exact = 1;

  for (int outer = 0; outer < max_iter; outer++) {
    double eff_scale = (d_norm_inf + mu > 0.0 && isfinite(d_norm_inf + mu))
                           ? (d_norm_inf + mu)
                           : 1.0;
    qsp_alm_update_diag(D, A, mu, sigma, sigma_A, eff_scale, w_diag);
    double alpha = 1.0;
    double f_hist[GLL_WINDOW];
    int hist_count = 0;

    int inner_converged = 0;
    for (int inner = 0; inner < inner_cap; inner++) {
      double inner_kkt = 0.0;
      double inner_step = 0.0;
      double x_scale = 1.0;
      double global_dual = 0.0;
      double dual_scale = 1.0;
      double aff_prim = 0.0;
      double aff_scale = 1.0;
      int nonfinite = 0;

      double f_curr = qsp_alm_eval_state(
          n, m, x, x_prox, dx, d, w_diag, A, ax, bar_y, lb_x, ub_x, lb_A, ub_A,
          mu, sigma, sigma_A, eff_scale, d_max, aq, g, &inner_kkt, &inner_step,
          &x_scale, &global_dual, &dual_scale, &aff_prim, &aff_scale,
          &nonfinite);

      if (nonfinite)
        return NANOQSP_ERR_MAX_ITERATIONS;

      if (global_dual < tol * dual_scale && aff_prim < tol * aff_scale) {
        if (exact)
          return outer + 1;
        qsp_mat_vec(D, x, dx);
        qsp_mat_vec(A, x, ax);
        exact = 1;
        f_curr = qsp_alm_eval_state(
            n, m, x, x_prox, dx, d, w_diag, A, ax, bar_y, lb_x, ub_x, lb_A,
            ub_A, mu, sigma, sigma_A, eff_scale, d_max, aq, g, &inner_kkt,
            &inner_step, &x_scale, &global_dual, &dual_scale, &aff_prim,
            &aff_scale, &nonfinite);
        if (!nonfinite && global_dual < tol * dual_scale &&
            aff_prim < tol * aff_scale)
          return outer + 1;
      }

      if (inner_kkt < inner_tol * dual_scale &&
          inner_step < inner_tol * x_scale) {
        if (exact) {
          inner_converged = 1;
          break;
        }
        qsp_mat_vec(D, x, dx);
        qsp_mat_vec(A, x, ax);
        exact = 1;
        f_curr = qsp_alm_eval_state(
            n, m, x, x_prox, dx, d, w_diag, A, ax, bar_y, lb_x, ub_x, lb_A,
            ub_A, mu, sigma, sigma_A, eff_scale, d_max, aq, g, &inner_kkt,
            &inner_step, &x_scale, &global_dual, &dual_scale, &aff_prim,
            &aff_scale, &nonfinite);
        if (nonfinite)
          return NANOQSP_ERR_MAX_ITERATIONS;
        if (inner_kkt < inner_tol * dual_scale &&
            inner_step < inner_tol * x_scale) {
          inner_converged = 1;
          break;
        }
      }

      f_hist[hist_count % GLL_WINDOW] = f_curr;
      hist_count++;

      double dir_max = 0.0;
      double inv_eff = 1.0 / eff_scale;
      for (int i = 0; i < n; i++) {
        double lo = lb_x ? lb_x[i] : -INFINITY;
        double hi = ub_x ? ub_x[i] : INFINITY;
        double step_i =
            qsp_clamp(x[i] - alpha * (g[i] / w_diag[i]), lo, hi) - x[i];
        q[i] = step_i;
        double mag = fabs(step_i);
        if (mag > dir_max)
          dir_max = mag;
      }

      if (!(dir_max > 0.0) || !isfinite(dir_max)) {
        if (!exact) {
          qsp_mat_vec(D, x, dx);
          qsp_mat_vec(A, x, ax);
          exact = 1;
          continue;
        }
        break;
      }

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
      qsp_mat_vec(A, q, aq);

      double q_w_q = 0.0;
      double curv_quad = 0.0;
      double slope_quad = 0.0;
      double dphi_0 = 0.0;
      for (int i = 0; i < n; i++) {
        double qi = q[i];
        q_w_q += qi * qi * (w_diag[i] * inv_eff);
        curv_quad += qi * ((dq[i] + sigma * qi) * inv_eff);
        slope_quad +=
            qi * ((dx[i] - d[i] + sigma * (x[i] - x_prox[i])) * inv_eff);
        dphi_0 += qi * (g[i] * inv_eff);
      }

      if (!(dphi_0 < 0.0) || !isfinite(dphi_0)) {
        if (!exact) {
          qsp_mat_vec(D, x, dx);
          qsp_mat_vec(A, x, ax);
          exact = 1;
          continue;
        }
        break;
      }

      int win = hist_count < GLL_WINDOW ? hist_count : GLL_WINDOW;
      double f_ref = f_hist[0];
      for (int k = 1; k < win; k++) {
        if (f_hist[k] > f_ref)
          f_ref = f_hist[k];
      }

      double length = scale_q;
      double delta_f = 0.0;
      double slope_end = 0.0;
      double curv_end = 0.0;
      qsp_alm_line_eval(m, ax, aq, bar_y, lb_A, ub_A, mu, sigma_A, eff_scale,
                        slope_quad, curv_quad, length, &delta_f, &slope_end,
                        &curv_end);

      if (!isfinite(f_curr) || !isfinite(delta_f) ||
          !(f_curr + delta_f <= f_ref + 1e-4 * length * dphi_0)) {
        if (slope_end < 0.0) {
          for (int ls = 0; ls < 40 && !(delta_f <= 1e-4 * length * dphi_0);
               ls++) {
            length *= 0.5;
            qsp_alm_line_eval(m, ax, aq, bar_y, lb_A, ub_A, mu, sigma_A,
                              eff_scale, slope_quad, curv_quad, length,
                              &delta_f, &slope_end, &curv_end);
          }
        } else {
          double lo_t = 0.0;
          double hi_t = scale_q;
          double t = (curv_end > 0.0) ? (scale_q - slope_end / curv_end)
                                      : (0.5 * scale_q);
          if (!(t > 0.0 && t < scale_q))
            t = 0.5 * scale_q;

          for (int ns = 0; ns < 12; ns++) {
            qsp_alm_line_eval(m, ax, aq, bar_y, lb_A, ub_A, mu, sigma_A,
                              eff_scale, slope_quad, curv_quad, t, &delta_f,
                              &slope_end, &curv_end);
            if (fabs(slope_end) <= 1e-12 * fabs(dphi_0))
              break;
            if (slope_end > 0.0)
              hi_t = t;
            else
              lo_t = t;
            double next_t = (curv_end > 0.0) ? (t - slope_end / curv_end)
                                             : 0.5 * (lo_t + hi_t);
            if (!(next_t > lo_t && next_t < hi_t))
              next_t = 0.5 * (lo_t + hi_t);
            if (fabs(next_t - t) <= 1e-14 * (lo_t + hi_t)) {
              t = next_t;
              break;
            }
            t = next_t;
          }
          length = t;
          qsp_alm_line_eval(m, ax, aq, bar_y, lb_A, ub_A, mu, sigma_A,
                            eff_scale, slope_quad, curv_quad, length, &delta_f,
                            &slope_end, &curv_end);
          for (int ls = 0; ls < 40 && !(delta_f <= 1e-4 * length * dphi_0);
               ls++) {
            length *= 0.5;
            qsp_alm_line_eval(m, ax, aq, bar_y, lb_A, ub_A, mu, sigma_A,
                              eff_scale, slope_quad, curv_quad, length,
                              &delta_f, &slope_end, &curv_end);
          }
        }
      }

      if (!(length > 0.0) || !isfinite(length))
        break;

      for (int i = 0; i < n; i++) {
        double lo = lb_x ? lb_x[i] : -INFINITY;
        double hi = ub_x ? ub_x[i] : INFINITY;
        double cand = (length == scale_q) ? x[i] - alpha * (g[i] / w_diag[i])
                                          : x[i] + length * q[i];
        x[i] = qsp_clamp(cand, lo, hi);
      }

      if ((inner + 1) % 32 == 0 || !isfinite(delta_f)) {
        qsp_mat_vec(D, x, dx);
        qsp_mat_vec(A, x, ax);
        exact = 1;
      } else {
        nanoqsp_blas.axpy(dx, dq, length, n);
        nanoqsp_blas.axpy(ax, aq, length, m);
        exact = 0;
      }

      double secant_curv = fabs(slope_end - dphi_0) / length;
      if (secant_curv > 0.0 && isfinite(secant_curv)) {
        double sp_alpha = qsp_clamp(q_w_q / secant_curv, 1e-12, 1e12);
        alpha = isfinite(sp_alpha) ? sp_alpha : 1.0;
      } else if (slope_end == dphi_0 && dphi_0 < 0.0) {
        alpha = alpha * 2.0 < 1e12 ? alpha * 2.0 : 1e12;
      } else {
        alpha = 1.0;
      }
    }

    if (!exact) {
      qsp_mat_vec(D, x, dx);
      qsp_mat_vec(A, x, ax);
      exact = 1;
    }

    double inner_kkt = 0.0;
    double inner_step = 0.0;
    double x_scale = 1.0;
    double global_dual = 0.0;
    double dual_scale = 1.0;
    double aff_prim = 0.0;
    double aff_scale = 1.0;
    int nonfinite = 0;
    (void)qsp_alm_eval_state(
        n, m, x, x_prox, dx, d, w_diag, A, ax, bar_y, lb_x, ub_x, lb_A, ub_A,
        mu, sigma, sigma_A, eff_scale, d_max, aq, g, &inner_kkt, &inner_step,
        &x_scale, &global_dual, &dual_scale, &aff_prim, &aff_scale, &nonfinite);
    if (nonfinite)
      return NANOQSP_ERR_MAX_ITERATIONS;
    if (global_dual < tol * dual_scale && aff_prim < tol * aff_scale)
      return outer + 1;

    /* update normalized multipliers bar_y and proximal center x_prox */
    for (int j = 0; j < m; j++) {
      double lo = lb_A ? lb_A[j] : -INFINITY;
      double hi = ub_A ? ub_A[j] : INFINITY;
      double sj = ax[j] + sigma_A * (bar_y[j] / mu);
      double zj = qsp_clamp(sj, lo, hi);
      double next_by = mu * ((sj - zj) / sigma_A);
      if (isfinite(next_by))
        bar_y[j] = next_by;
    }
    nanoqsp_blas.copy(x_prox, x, n);

    double rel_aff = aff_prim / aff_scale;
    if (inner_converged) {
      if (outer > 0 && rel_aff > 0.25 * prev_rel_aff) {
        if (mu < mu_max) {
          double next_mu = mu * 5.0;
          if (next_mu > mu_max)
            next_mu = mu_max;
          if (isfinite(next_mu))
            mu = next_mu;
        }
        inner_tol *= 0.5;
      } else {
        inner_tol *= 0.2;
      }
      if (inner_tol < 0.25 * tol)
        inner_tol = 0.25 * tol;
      prev_rel_aff = rel_aff;
    }

    double min_sigma = 1e-14 * mu_0;
    sigma *= 0.2;
    if (sigma < min_sigma)
      sigma = min_sigma;
  }

  return NANOQSP_ERR_MAX_ITERATIONS;
}
