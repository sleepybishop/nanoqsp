#include "qsp_internal.h"

static void qsp_cg_solve(const qsp_mat *D, const qsp_mat *A, double rho,
                         double affine_rho, double r_tol_sq,
                         const double *restrict d, const double *restrict z,
                         const double *restrict y, double *restrict x,
                         double *restrict r, double *restrict p,
                         double *restrict Ap, double *restrict tmp_n,
                         double *restrict tmp_m) {
  int n = D->rows;
  int m = A ? A->rows : 0;
  int has_A = (A != NULL && m > 0);

  /* form RHS residual from constraint differences before multiplying by A^T */
  qsp_mat_vec(D, x, r);
  for (int i = 0; i < n; i++)
    r[i] = d[i] - r[i] + rho * ((z[i] - x[i]) - y[i]);
  if (has_A) {
    qsp_mat_vec(A, x, tmp_m);
    for (int i = 0; i < m; i++)
      tmp_m[i] = (z[n + i] - tmp_m[i]) - y[n + i];
    qsp_mat_tvec(A, tmp_m, tmp_n);
    nanoqsp_blas.axpy(r, tmp_n, affine_rho, n);
  }

  nanoqsp_blas.copy(p, r, n);
  double r_sq = nanoqsp_blas.dot(r, r, n);

  for (int iter = 0; iter < n + 5; iter++) {
    if (r_sq <= r_tol_sq)
      break;

    qsp_mat_vec(D, p, Ap);
    double pAp = nanoqsp_blas.dot(p, Ap, n) + rho * nanoqsp_blas.dot(p, p, n);

    if (has_A) {
      qsp_mat_vec(A, p, tmp_m);
      pAp += affine_rho * nanoqsp_blas.dot(tmp_m, tmp_m, m);
      qsp_mat_tvec(A, tmp_m, tmp_n);
      for (int i = 0; i < n; i++)
        Ap[i] += rho * p[i] + affine_rho * tmp_n[i];
    } else {
      nanoqsp_blas.axpy(Ap, p, rho, n);
    }

    if (!(pAp > 0.0) || !isfinite(pAp) || !isfinite(r_sq))
      break;

    double alpha = r_sq / pAp;
    if (!isfinite(alpha))
      break;
    nanoqsp_blas.axpy(x, p, alpha, n);
    nanoqsp_blas.axpy(r, Ap, -alpha, n);

    double next_r_sq = nanoqsp_blas.dot(r, r, n);
    double beta = next_r_sq / r_sq;
    for (int i = 0; i < n; i++)
      p[i] = r[i] + beta * p[i];
    r_sq = next_r_sq;
  }
}

int nanoqsp_solve_admm(const qsp_mat *D, const double *restrict d,
                       const qsp_mat *A, const double *restrict lb_x,
                       const double *restrict ub_x, const double *restrict lb_A,
                       const double *restrict ub_A, double *restrict x,
                       int max_iter, double tol, qsp_arena *ws) {
  int n = D->rows;
  int m = A ? A->rows : 0;
  int has_A = (A != NULL && m > 0);

  /* scale initial penalty with D so ADMM is objective-scale invariant */
  double d_norm_inf = qsp_mat_norm_inf(D);
  double rho = 1.0;
  if (d_norm_inf > 0.0 && isfinite(d_norm_inf))
    rho = d_norm_inf / sqrt((double)n);

  double affine_weight = 1.0;
  if (has_A) {
    double sum_sq = 0.0;
    double scale = qsp_mat_scaled_fro(A, &sum_sq);
    if (!isfinite(scale) || !isfinite(sum_sq))
      return NANOQSP_ERR_MAX_ITERATIONS;
    if (scale > 0.0 && sum_sq > 0.0) {
      double w = (((double)m / scale) / scale) / sum_sq;
      if (w > 0.0 && isfinite(w))
        affine_weight = w;
    }
  }

  double *restrict z = qsp_arena_take(ws, n + m);
  double *restrict y = qsp_arena_take(ws, n + m);
  double *restrict c = qsp_arena_take(ws, n);
  double *restrict r = qsp_arena_take(ws, n);
  double *restrict p = qsp_arena_take(ws, n);
  double *restrict Ap = qsp_arena_take(ws, n);
  double *restrict Ax = m > 0 ? qsp_arena_take(ws, m) : NULL;
  if (!z || !y || !c || !r || !p || !Ap || (m > 0 && !Ax))
    return NANOQSP_ERR_OUT_OF_MEMORY;

  nanoqsp_blas.copy(z, x, n);
  if (has_A)
    qsp_mat_vec(A, x, &z[n]);
  nanoqsp_blas.zero(y, n + m);

  double d_max = qsp_vec_norm_inf(n, d);
  double base_scale = d_max > rho ? d_max : rho;
  if (!(base_scale > 0.0) || !isfinite(base_scale))
    base_scale = 1.0;
  double admm_err = 0.1 * base_scale;

  for (int iter = 0; iter < max_iter; iter++) {
    double cg_min = 1e-2 * tol * (d_max > 0.0 ? d_max : base_scale);
    double cg_tol = 0.05 * admm_err;
    if (cg_tol < cg_min)
      cg_tol = cg_min;
    double cg_tol_sq = cg_tol * cg_tol;

    /* c is dead during CG and serves as the length-n scratch vector */
    qsp_cg_solve(D, A, rho, rho * affine_weight, cg_tol_sq, d, z, y, x, r, p,
                 Ap, c, Ax);

    double box_prim_res = 0.0;
    double aff_prim_res = 0.0;
    double x_max = 0.0;
    double zx_max = 0.0;
    double Az_max = 0.0;
    double zA_max = 0.0;
    int nonfinite_found = 0;

    for (int i = 0; i < n; i++) {
      double old_z = z[i];
      double lower = lb_x ? lb_x[i] : -INFINITY;
      double upper = ub_x ? ub_x[i] : INFINITY;
      double z_i = qsp_clamp(x[i] + y[i], lower, upper);
      double y_i = y[i] + (x[i] - z_i);
      z[i] = z_i;
      y[i] = y_i;
      c[i] = rho * (z_i - old_z);

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
      qsp_mat_vec(A, x, Ax);
      for (int i = 0; i < m; i++) {
        double old_z = z[n + i];
        double lower = lb_A ? lb_A[i] : -INFINITY;
        double upper = ub_A ? ub_A[i] : INFINITY;
        double z_i = qsp_clamp(Ax[i] + y[n + i], lower, upper);
        double y_i = y[n + i] + (Ax[i] - z_i);
        z[n + i] = z_i;
        y[n + i] = y_i;

        if (!isfinite(Ax[i]) || !isfinite(z_i) || !isfinite(y_i))
          nonfinite_found = 1;

        double res = fabs(Ax[i] - z_i);
        if (res > aff_prim_res)
          aff_prim_res = res;
        double val_z = fabs(z_i);
        if (val_z > zA_max)
          zA_max = val_z;
        Ax[i] = rho * affine_weight * (z_i - old_z);
      }
      qsp_mat_tvec(A, Ax, r);
      nanoqsp_blas.axpy(c, r, 1.0, n);

      /* evaluate affine feasibility at the projected output point z_x */
      qsp_mat_vec(A, z, Ax);
      for (int i = 0; i < m; i++) {
        if (!isfinite(Ax[i]))
          nonfinite_found = 1;
        double res = fabs(Ax[i] - z[n + i]);
        if (res > aff_prim_res)
          aff_prim_res = res;
        double val_Az = fabs(Ax[i]);
        if (val_Az > Az_max)
          Az_max = val_Az;
      }
    }

    double admm_dual_res = qsp_vec_norm_inf(n, c);

    /* evaluate exact KKT stationarity at z_x */
    qsp_mat_vec(D, z, c);
    if (has_A)
      qsp_mat_tvec(A, &y[n], r);
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

    if (!nonfinite_found) {
      double scaled_prim = rho * box_prim_res;
      if (has_A) {
        double aff_scaled = rho * sqrt(affine_weight) * aff_prim_res;
        if (aff_scaled > scaled_prim)
          scaled_prim = aff_scaled;
      }
      admm_err = dual_res > scaled_prim ? dual_res : scaled_prim;
      if (admm_dual_res > admm_err)
        admm_err = admm_dual_res;
    }

    /* balance relative primal and dual residuals periodically */
    if (!nonfinite_found && (iter + 1) % 10 == 0) {
      double rel_box = box_prim_res / box_scale;
      double rel_aff = has_A ? (aff_prim_res / aff_scale) : 0.0;
      double rel_prim = rel_box > rel_aff ? rel_box : rel_aff;
      double rel_dual = admm_dual_res / dual_scale;
      double new_rho = rho;
      if (rel_prim > 10.0 * rel_dual)
        new_rho = 2.0 * rho;
      else if (rel_dual > 10.0 * rel_prim)
        new_rho = 0.5 * rho;
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
        }
      }
    }
  }

  nanoqsp_blas.copy(x, z, n);
  return NANOQSP_ERR_MAX_ITERATIONS;
}
