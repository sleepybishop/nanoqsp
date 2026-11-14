#include "qsp_internal.h"

/* precompute inverse diagonal of D + rho * W_x + (rho / sigma_A^2) A^T A */
static void qsp_admm_update_precond(const qsp_mat *D, const qsp_mat *A,
                                    double rho, double sigma_A,
                                    const double *restrict w_x,
                                    double *restrict inv_diag) {
  int n = D->rows;
  int m = A ? A->rows : 0;

  qsp_mat_extract_diag(D, inv_diag);
  for (int i = 0; i < n; i++)
    inv_diag[i] += rho * w_x[i];

  if (A != NULL && m > 0 && rho > 0.0 && sigma_A > 0.0) {
    double factor = sqrt(rho) / sigma_A;
    if (factor > 0.0 && isfinite(factor)) {
      if (A->dense != NULL) {
        size_t stride = (size_t)n;
        for (int j = 0; j < m; j++) {
          const double *restrict row = &A->dense[(size_t)j * stride];
          for (int i = 0; i < n; i++) {
            double t = factor * row[i];
            inv_diag[i] += t * t;
          }
        }
      } else {
        const NanoqspCSR *S = A->csr;
        for (int idx = 0; idx < S->nnz; idx++) {
          double t = factor * S->values[idx];
          inv_diag[S->col_indices[idx]] += t * t;
        }
      }
    }
  }

  for (int i = 0; i < n; i++) {
    double d_ii = inv_diag[i];
    double inv = (d_ii > 0.0 && isfinite(d_ii)) ? (1.0 / d_ii) : 1.0;
    inv_diag[i] = (inv > 0.0 && isfinite(inv)) ? inv : 1.0;
  }
}

/* jacobi-preconditioned CG with dual and preconditioned stopping norms */
static void qsp_cg_solve(const qsp_mat *D, const qsp_mat *A, double rho,
                         double sigma_A, double r_tol, double x_tol,
                         int is_first_outer, const double *restrict w_x,
                         const double *restrict inv_diag,
                         const double *restrict d, const double *restrict z,
                         const double *restrict y, double *restrict x,
                         double *restrict r, double *restrict p,
                         double *restrict Ap, double *restrict tmp_m) {
  int n = D->rows;
  int m = A ? A->rows : 0;
  int has_A = (A != NULL && m > 0);
  double rho_over_sig = has_A ? (rho / sigma_A) : 0.0;

  /* form RHS residual from constraint differences before multiplying by A^T */
  qsp_mat_vec(D, x, r);
  for (int i = 0; i < n; i++)
    r[i] = d[i] - r[i] + rho * w_x[i] * ((z[i] - x[i]) - y[i]);
  if (has_A) {
    qsp_mat_vec(A, x, tmp_m);
    for (int j = 0; j < m; j++)
      tmp_m[j] = (((z[n + j] - tmp_m[j]) - y[n + j]) / sigma_A) * rho_over_sig;
    qsp_mat_tvec_add(A, tmp_m, 1.0, r);
  }

  double r_inf = 0.0;
  double z_inf = 0.0;
  double rz = 0.0;
  for (int i = 0; i < n; i++) {
    double ri = r[i];
    double wi = inv_diag[i] * ri;
    p[i] = wi;
    double abs_ri = fabs(ri);
    if (abs_ri > r_inf)
      r_inf = abs_ri;
    double abs_wi = fabs(wi);
    if (abs_wi > z_inf)
      z_inf = abs_wi;
    rz += ri * wi;
  }

  if (is_first_outer) {
    double init_rtol = 0.15 * r_inf;
    double init_xtol = 0.15 * z_inf;
    if (init_rtol > r_tol)
      r_tol = init_rtol;
    if (init_xtol > x_tol)
      x_tol = init_xtol;
  }

  for (int iter = 0; iter < n + 5; iter++) {
    if (r_inf <= r_tol && z_inf <= x_tol)
      break;

    qsp_mat_vec(D, p, Ap);
    double pAp = nanoqsp_blas.dot(p, Ap, n);
    double pw_p = 0.0;
    for (int i = 0; i < n; i++) {
      double pi = p[i];
      double rwi = rho * w_x[i] * pi;
      Ap[i] += rwi;
      pw_p += pi * rwi;
    }
    pAp += pw_p;

    if (has_A) {
      qsp_mat_vec(A, p, tmp_m);
      double ap_sq = 0.0;
      for (int j = 0; j < m; j++) {
        double s = tmp_m[j] / sigma_A;
        ap_sq += s * s;
        tmp_m[j] = s * rho_over_sig;
      }
      pAp += rho * ap_sq;
      qsp_mat_tvec_add(A, tmp_m, 1.0, Ap);
    }

    if (!(pAp > 0.0) || !(rz > 0.0) || !isfinite(pAp) || !isfinite(rz) ||
        !isfinite(r_inf) || !isfinite(z_inf))
      break;

    double alpha = rz / pAp;
    if (!isfinite(alpha))
      break;
    nanoqsp_blas.axpy(x, p, alpha, n);
    nanoqsp_blas.axpy(r, Ap, -alpha, n);

    double next_r_inf = 0.0;
    double next_z_inf = 0.0;
    double next_rz = 0.0;
    for (int i = 0; i < n; i++) {
      double ri = r[i];
      double wi = inv_diag[i] * ri;
      double abs_ri = fabs(ri);
      if (abs_ri > next_r_inf)
        next_r_inf = abs_ri;
      double abs_wi = fabs(wi);
      if (abs_wi > next_z_inf)
        next_z_inf = abs_wi;
      next_rz += ri * wi;
    }
    r_inf = next_r_inf;
    z_inf = next_z_inf;
    if ((r_inf <= r_tol && z_inf <= x_tol) || !(next_rz > 0.0) ||
        !isfinite(next_rz))
      break;

    double beta = next_rz / rz;
    for (int i = 0; i < n; i++)
      p[i] = inv_diag[i] * r[i] + beta * p[i];
    rz = next_rz;
  }
}

/* direct jacobi-preconditioned CG fast-path for unconstrained Dx = d */
static int qsp_solve_unconstrained_pcg(
    const qsp_mat *D, const double *restrict d, double *restrict x,
    double d_max, int max_iter, double tol, double *restrict inv_diag,
    double *restrict r, double *restrict p, double *restrict Ap) {
  int n = D->rows;
  if (max_iter <= 0)
    return NANOQSP_ERR_MAX_ITERATIONS;

  qsp_mat_extract_diag(D, inv_diag);
  for (int i = 0; i < n; i++) {
    double d_ii = inv_diag[i];
    if (d_ii < 0.0 || !isfinite(d_ii))
      return NANOQSP_ERR_MAX_ITERATIONS;
    double inv = (d_ii > 0.0) ? (1.0 / d_ii) : 1.0;
    inv_diag[i] = (inv > 0.0 && isfinite(inv)) ? inv : 1.0;
  }

  qsp_mat_vec(D, x, Ap);
  double r_inf = 0.0;
  double dx_max = 0.0;
  double rz = 0.0;
  for (int i = 0; i < n; i++) {
    double dxi = Ap[i];
    double ri = d[i] - dxi;
    if (!isfinite(x[i]) || !isfinite(dxi) || !isfinite(ri))
      return NANOQSP_ERR_MAX_ITERATIONS;
    r[i] = ri;
    double wi = inv_diag[i] * ri;
    p[i] = wi;
    double abs_dx = fabs(dxi);
    if (abs_dx > dx_max)
      dx_max = abs_dx;
    double abs_r = fabs(ri);
    if (abs_r > r_inf)
      r_inf = abs_r;
    rz += ri * wi;
  }

  double dual_scale = 1.0 + (d_max > dx_max ? d_max : dx_max);
  if (r_inf < tol * dual_scale)
    return 1;

  for (int iter = 0; iter < max_iter; iter++) {
    qsp_mat_vec(D, p, Ap);
    double pAp = nanoqsp_blas.dot(p, Ap, n);
    if (!(pAp > 0.0) || !(rz > 0.0) || !isfinite(pAp) || !isfinite(rz))
      return NANOQSP_ERR_MAX_ITERATIONS;

    double alpha = rz / pAp;
    if (!isfinite(alpha))
      return NANOQSP_ERR_MAX_ITERATIONS;
    nanoqsp_blas.axpy(x, p, alpha, n);
    nanoqsp_blas.axpy(r, Ap, -alpha, n);

    double next_r_inf = 0.0;
    double next_rz = 0.0;
    for (int i = 0; i < n; i++) {
      double ri = r[i];
      if (!isfinite(x[i]) || !isfinite(ri))
        return NANOQSP_ERR_MAX_ITERATIONS;
      double wi = inv_diag[i] * ri;
      double abs_r = fabs(ri);
      if (abs_r > next_r_inf)
        next_r_inf = abs_r;
      next_rz += ri * wi;
    }

    if (next_r_inf < tol * (1.0 + d_max + next_r_inf) || (iter + 1) % n == 0) {
      qsp_mat_vec(D, x, Ap);
      double true_r_inf = 0.0;
      double true_dx_max = 0.0;
      double true_rz = 0.0;
      for (int i = 0; i < n; i++) {
        double dxi = Ap[i];
        double ri = d[i] - dxi;
        if (!isfinite(dxi) || !isfinite(ri))
          return NANOQSP_ERR_MAX_ITERATIONS;
        r[i] = ri;
        double wi = inv_diag[i] * ri;
        double abs_dx = fabs(dxi);
        if (abs_dx > true_dx_max)
          true_dx_max = abs_dx;
        double abs_r = fabs(ri);
        if (abs_r > true_r_inf)
          true_r_inf = abs_r;
        true_rz += ri * wi;
      }
      dual_scale = 1.0 + (d_max > true_dx_max ? d_max : true_dx_max);
      if (true_r_inf < tol * dual_scale)
        return iter + 1;
      for (int i = 0; i < n; i++)
        p[i] = inv_diag[i] * r[i];
      rz = true_rz;
      continue;
    }

    if (!(next_rz > 0.0) || !isfinite(next_rz))
      return NANOQSP_ERR_MAX_ITERATIONS;

    double beta = next_rz / rz;
    for (int i = 0; i < n; i++)
      p[i] = inv_diag[i] * r[i] + beta * p[i];
    rz = next_rz;
  }

  return NANOQSP_ERR_MAX_ITERATIONS;
}

/* compute RMS row norm of A in equilibrated w_x coordinates without overflow */
static int qsp_equilibrated_a_scale(const qsp_mat *A,
                                    const double *restrict inv_sqrt_wx,
                                    double *out_sigma_A) {
  int n = A->cols;
  int m = A->rows;
  double scale = 0.0;
  double sum_sq = 1.0;

  if (A->dense != NULL) {
    size_t stride = (size_t)n;
    for (int j = 0; j < m; j++) {
      const double *restrict row = &A->dense[(size_t)j * stride];
      for (int i = 0; i < n; i++) {
        double a = fabs(row[i]) * inv_sqrt_wx[i];
        if (!isfinite(a))
          return 0;
        if (a == 0.0)
          continue;
        if (a > scale) {
          double ratio = scale / a;
          sum_sq = (scale == 0.0) ? 1.0 : (1.0 + sum_sq * ratio * ratio);
          scale = a;
        } else {
          double ratio = a / scale;
          sum_sq += ratio * ratio;
        }
      }
    }
  } else {
    const NanoqspCSR *S = A->csr;
    for (int idx = 0; idx < S->nnz; idx++) {
      double a = fabs(S->values[idx]) * inv_sqrt_wx[S->col_indices[idx]];
      if (!isfinite(a))
        return 0;
      if (a == 0.0)
        continue;
      if (a > scale) {
        double ratio = scale / a;
        sum_sq = (scale == 0.0) ? 1.0 : (1.0 + sum_sq * ratio * ratio);
        scale = a;
      } else {
        double ratio = a / scale;
        sum_sq += ratio * ratio;
      }
    }
  }

  double sigma = 1.0;
  if (scale > 0.0 && sum_sq > 0.0) {
    double s = scale * sqrt(sum_sq / (double)m);
    if (s > 0.0 && isfinite(s))
      sigma = s;
  }
  *out_sigma_A = sigma;
  return 1;
}

int nanoqsp_solve_admm(const qsp_mat *D, const double *restrict d,
                       const qsp_mat *A, const double *restrict lb_x,
                       const double *restrict ub_x, const double *restrict lb_A,
                       const double *restrict ub_A, double *restrict x,
                       int max_iter, double tol, qsp_arena *ws) {
  int n = D->rows;
  int m = A ? A->rows : 0;
  int has_A = (A != NULL && m > 0);

  double d_norm_inf = qsp_mat_norm_inf(D);
  if (!isfinite(d_norm_inf))
    return NANOQSP_ERR_MAX_ITERATIONS;

  double d_max = qsp_vec_norm_inf(n, d);
  int unconstrained = !has_A;
  if (unconstrained && (lb_x != NULL || ub_x != NULL)) {
    for (int i = 0; i < n; i++) {
      if ((lb_x && isfinite(lb_x[i])) || (ub_x && isfinite(ub_x[i]))) {
        unconstrained = 0;
        break;
      }
    }
  }
  if (unconstrained) {
    double *restrict inv_diag = qsp_arena_take(ws, n);
    double *restrict r = qsp_arena_take(ws, n);
    double *restrict p = qsp_arena_take(ws, n);
    double *restrict Ap = qsp_arena_take(ws, n);
    if (!inv_diag || !r || !p || !Ap)
      return NANOQSP_ERR_OUT_OF_MEMORY;
    return qsp_solve_unconstrained_pcg(D, d, x, d_max, max_iter, tol, inv_diag,
                                       r, p, Ap);
  }

  double *restrict z = qsp_arena_take(ws, n + m);
  double *restrict y = qsp_arena_take(ws, n + m);
  double *restrict w_x = qsp_arena_take(ws, n);
  double *restrict inv_diag = qsp_arena_take(ws, n);
  double *restrict r = qsp_arena_take(ws, n);
  double *restrict p = qsp_arena_take(ws, n);
  double *restrict Ap = qsp_arena_take(ws, n);
  double *restrict Ax = m > 0 ? qsp_arena_take(ws, m) : NULL;
  if (!z || !y || !w_x || !inv_diag || !r || !p || !Ap || (m > 0 && !Ax))
    return NANOQSP_ERR_OUT_OF_MEMORY;

  /* seed rho and per-coordinate diagonal weights from D */
  qsp_mat_extract_diag(D, r);
  double diag_min = INFINITY;
  double diag_max = 0.0;
  for (int i = 0; i < n; i++) {
    if (r[i] > 0.0) {
      if (r[i] < diag_min)
        diag_min = r[i];
      if (r[i] > diag_max)
        diag_max = r[i];
    }
  }

  double rho = 1.0;
  if (diag_max > 0.0) {
    rho = sqrt(diag_min) * sqrt(diag_max);
  } else if (d_norm_inf > 0.0) {
    rho = d_norm_inf / sqrt((double)n);
  } else if (d_max > 0.0) {
    rho = d_max;
  }
  if (!(rho > 0.0) || !isfinite(rho))
    rho = 1.0;

  double eq_d_max = 0.0;
  for (int i = 0; i < n; i++) {
    w_x[i] =
        (diag_max > 0.0 && r[i] > 0.0) ? qsp_clamp(r[i] / rho, 1e-6, 1e6) : 1.0;
    double inv_sx = 1.0 / sqrt(w_x[i]);
    p[i] = inv_sx;
    double vd = fabs(d[i]) * inv_sx;
    if (vd > eq_d_max)
      eq_d_max = vd;
  }

  double sigma_A = 1.0;
  if (has_A && !qsp_equilibrated_a_scale(A, p, &sigma_A))
    return NANOQSP_ERR_MAX_ITERATIONS;

  qsp_admm_update_precond(D, A, rho, sigma_A, w_x, inv_diag);

  nanoqsp_blas.copy(z, x, n);
  if (has_A)
    qsp_mat_vec(A, x, &z[n]);
  nanoqsp_blas.zero(y, n + m);

  double init_scale = 1.0 + (d_max > rho ? d_max : rho);
  if (!isfinite(init_scale))
    init_scale = 1.0;
  double rho_hi_base = rho > d_max ? rho : d_max;
  double rho_min = (rho * 1e-6 > 0.0) ? (rho * 1e-6) : 1e-12;
  double rho_max =
      isfinite(rho_hi_base * 1e6) ? (rho_hi_base * 1e6) : rho_hi_base;
  double cg_tol = 0.05 * tol * init_scale;
  double cg_xtol = 0.05 * tol;

  for (int iter = 0; iter < max_iter; iter++) {
    qsp_cg_solve(D, A, rho, sigma_A, cg_tol, cg_xtol, iter == 0, w_x, inv_diag,
                 d, z, y, x, r, p, Ap, Ax);

    double box_prim_res = 0.0;
    double aff_prim_res = 0.0;
    double x_max = 0.0;
    double zx_max = 0.0;
    double Az_max = 0.0;
    double zA_max = 0.0;
    int nonfinite_found = 0;

    /* p is free after PCG and holds the ADMM dual step and D * z_x */
    for (int i = 0; i < n; i++) {
      double old_z = z[i];
      double lower = lb_x ? lb_x[i] : -INFINITY;
      double upper = ub_x ? ub_x[i] : INFINITY;
      double z_i = qsp_clamp(x[i] + y[i], lower, upper);
      double y_i = y[i] + (x[i] - z_i);
      z[i] = z_i;
      y[i] = y_i;
      p[i] = rho * w_x[i] * (z_i - old_z);

      if (!isfinite(x[i]) || !isfinite(z_i) || !isfinite(y_i))
        nonfinite_found = 1;

      double res = fabs(x[i] - z_i);
      if (res > box_prim_res)
        box_prim_res = res;
      double val_x = fabs(x[i]);
      if (val_x > x_max)
        x_max = val_x;
      double val_z = fabs(z_i);
      if (val_z > zx_max)
        zx_max = val_z;
    }

    if (has_A) {
      double rho_over_sig = rho / sigma_A;
      qsp_mat_vec(A, x, Ax);
      for (int j = 0; j < m; j++) {
        double old_z = z[n + j];
        double lower = lb_A ? lb_A[j] : -INFINITY;
        double upper = ub_A ? ub_A[j] : INFINITY;
        double z_j = qsp_clamp(Ax[j] + y[n + j], lower, upper);
        double y_j = y[n + j] + (Ax[j] - z_j);
        z[n + j] = z_j;
        y[n + j] = y_j;

        if (!isfinite(Ax[j]) || !isfinite(z_j) || !isfinite(y_j))
          nonfinite_found = 1;

        double res = fabs(Ax[j] - z_j);
        if (res > aff_prim_res)
          aff_prim_res = res;
        double val_z = fabs(z_j);
        if (val_z > zA_max)
          zA_max = val_z;
        Ax[j] = ((z_j - old_z) / sigma_A) * rho_over_sig;
      }
      qsp_mat_tvec_add(A, Ax, 1.0, p);

      /* evaluate affine feasibility at the projected output point z_x */
      qsp_mat_vec(A, z, Ax);
      for (int j = 0; j < m; j++) {
        if (!isfinite(Ax[j]))
          nonfinite_found = 1;
        double res = fabs(Ax[j] - z[n + j]);
        if (res > aff_prim_res)
          aff_prim_res = res;
        double val_Az = fabs(Ax[j]);
        if (val_Az > Az_max)
          Az_max = val_Az;
        Ax[j] = (y[n + j] / sigma_A) * rho_over_sig;
      }
      qsp_mat_tvec(A, Ax, r);
    }

    int adapt_step = ((iter + 1) % 10 == 0);
    double eq_dual_step = 0.0;
    if (adapt_step) {
      for (int i = 0; i < n; i++) {
        double v = fabs(p[i]) / sqrt(w_x[i]);
        if (v > eq_dual_step)
          eq_dual_step = v;
      }
    }

    /* evaluate exact KKT stationarity at z_x */
    qsp_mat_vec(D, z, p);
    double dx_max = 0.0;
    double y_max = 0.0;
    double dual_res = 0.0;
    for (int i = 0; i < n; i++) {
      double dual_i = rho * w_x[i] * y[i] + (has_A ? r[i] : 0.0);
      double residual = p[i] - d[i] + dual_i;
      if (!isfinite(p[i]) || !isfinite(dual_i) || !isfinite(residual))
        nonfinite_found = 1;
      double val_dx = fabs(p[i]);
      if (val_dx > dx_max)
        dx_max = val_dx;
      double val_y = fabs(dual_i);
      if (val_y > y_max)
        y_max = val_y;
      double res = fabs(residual);
      if (res > dual_res)
        dual_res = res;
    }

    double box_scale = 1.0 + (x_max > zx_max ? x_max : zx_max);
    double aff_scale = 1.0 + (Az_max > zA_max ? Az_max : zA_max);
    double dual_scale =
        1.0 + (y_max > d_max ? (y_max > dx_max ? y_max : dx_max)
                             : (d_max > dx_max ? d_max : dx_max));

    if (!nonfinite_found && dual_res < tol * dual_scale &&
        box_prim_res < tol * box_scale &&
        (!has_A || aff_prim_res < tol * aff_scale)) {
      nanoqsp_blas.copy(x, z, n);
      return iter + 1;
    }

    double rel_box = box_prim_res / box_scale;
    double rel_aff = has_A ? (aff_prim_res / aff_scale) : 0.0;
    double rel_prim = rel_box > rel_aff ? rel_box : rel_aff;
    double rel_dual_kkt = dual_res / dual_scale;

    /* tie inner PCG accuracy to outer ADMM progress */
    if (!nonfinite_found) {
      double r_p = rel_prim > 0.0 ? rel_prim : rel_dual_kkt;
      double r_d = rel_dual_kkt > 0.0 ? rel_dual_kkt : rel_prim;
      double rel_min = r_p < r_d ? r_p : r_d;
      double rel_target = 0.15 * rel_min;
      double rel_floor = 0.05 * tol;
      if (rel_target < rel_floor)
        rel_target = rel_floor;
      cg_tol = rel_target * dual_scale;
      cg_xtol = rel_target * box_scale;
    }

    /* OSQP-style proportional penalty adaptation in equilibrated norms */
    if (!nonfinite_found && adapt_step) {
      double eq_box_prim = 0.0;
      double eq_box_norm = 0.0;
      double eq_dual_norm = eq_d_max;
      for (int i = 0; i < n; i++) {
        double sx = sqrt(w_x[i]);
        double inv_sx = 1.0 / sx;
        double bp = sx * fabs(x[i] - z[i]);
        if (bp > eq_box_prim)
          eq_box_prim = bp;
        double xm = fabs(x[i]) > fabs(z[i]) ? fabs(x[i]) : fabs(z[i]);
        if (sx * xm > eq_box_norm)
          eq_box_norm = sx * xm;
        double dual_i = rho * w_x[i] * y[i] + (has_A ? r[i] : 0.0);
        double dm = fabs(p[i]) > fabs(dual_i) ? fabs(p[i]) : fabs(dual_i);
        if (inv_sx * dm > eq_dual_norm)
          eq_dual_norm = inv_sx * dm;
      }
      double eq_rel_box = eq_box_prim / (1.0 + eq_box_norm);
      double eq_rel_aff = 0.0;
      if (has_A) {
        double am = Az_max > zA_max ? Az_max : zA_max;
        eq_rel_aff = (aff_prim_res / sigma_A) / (1.0 + am / sigma_A);
      }
      double eq_prim = eq_rel_box > eq_rel_aff ? eq_rel_box : eq_rel_aff;
      double eq_dual = eq_dual_step / (1.0 + eq_dual_norm);

      double new_rho = rho;
      if (eq_prim == 0.0 && eq_dual > 0.0 && rel_dual_kkt > tol) {
        new_rho = qsp_clamp(0.1 * rho, rho_min, rho_max);
      } else if (eq_prim > 0.0 && eq_dual == 0.0 && rel_prim > tol) {
        new_rho = qsp_clamp(10.0 * rho, rho_min, rho_max);
      } else if (eq_prim > 0.0 && eq_dual > 0.0 &&
                 (eq_prim > 3.0 * eq_dual || eq_dual > 3.0 * eq_prim)) {
        double factor = qsp_clamp(sqrt(eq_prim / eq_dual), 0.1, 10.0);
        new_rho = qsp_clamp(rho * factor, rho_min, rho_max);
      }
      if (new_rho > 0.0 && isfinite(new_rho) && new_rho != rho) {
        double mult_scale = rho / new_rho;
        int can_rescale = 1;
        for (int i = 0; i < n + m; i++) {
          if (!isfinite(y[i] * mult_scale))
            can_rescale = 0;
        }
        if (can_rescale) {
          for (int i = 0; i < n + m; i++)
            y[i] *= mult_scale;
          rho = new_rho;
          qsp_admm_update_precond(D, A, rho, sigma_A, w_x, inv_diag);
        }
      }
    }
  }

  nanoqsp_blas.copy(x, z, n);
  return NANOQSP_ERR_MAX_ITERATIONS;
}
