#include "qsp_internal.h"
#ifndef NANOQSP_NO_LIBC
#include <stdlib.h>
#endif

static inline int within_bounds(long long rows, long long cols) {
  return rows > 0 && cols >= 0 && rows * cols <= NANOQSP_MAX_DOUBLES;
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
    if (isnan(lower) || isnan(upper) || lower > upper || lower == INFINITY ||
        upper == -INFINITY)
      return 0;
  }
  return 1;
}

static void project_bounds(int n, const double *restrict lb,
                           const double *restrict ub, double *restrict x) {
  for (int i = 0; i < n; i++) {
    double lower = lb ? lb[i] : -INFINITY;
    double upper = ub ? ub[i] : INFINITY;
    x[i] = qsp_clamp(x[i], lower, upper);
  }
}

static int validate_vector(size_t n, const double *restrict v) {
  for (size_t i = 0; i < n; i++) {
    if (!isfinite(v[i]))
      return 0;
  }
  return 1;
}

static int validate_config(const NanoqspConfig *config) {
  if (config == NULL)
    return 1;
  if (!isfinite(config->tolerance) || config->tolerance < 0.0 ||
      config->max_iterations < 0 || config->workspace_size < 0 ||
      (config->workspace == NULL && config->workspace_size > 0))
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
  if (!valid_strategy(strategy))
    return NANOQSP_ERR_INVALID_STRATEGY;
  if (!within_bounds(n, 7))
    return NANOQSP_ERR_INVALID_ARG;
  switch (strategy) {
  case NANOQSP_STRATEGY_COORDINATE_DESCENT:
  case NANOQSP_STRATEGY_PROJECTED_GRADIENT:
    return n;
  case NANOQSP_STRATEGY_ACCELERATED_GRADIENT:
    return 2 * n;
  case NANOQSP_STRATEGY_SPECTRAL_GRADIENT:
    return 3 * n;
  case NANOQSP_STRATEGY_ADMM:
    return 7 * n;
  }
  return NANOQSP_ERR_INVALID_STRATEGY;
}

int nanoqsp_workspace_size_box_sparse(int n, NanoqspStrategy strategy) {
  if (!valid_strategy(strategy))
    return NANOQSP_ERR_INVALID_STRATEGY;
  if (!within_bounds(n, 7))
    return NANOQSP_ERR_INVALID_ARG;
  switch (strategy) {
  case NANOQSP_STRATEGY_PROJECTED_GRADIENT:
    return n;
  case NANOQSP_STRATEGY_COORDINATE_DESCENT:
  case NANOQSP_STRATEGY_ACCELERATED_GRADIENT:
    return 2 * n;
  case NANOQSP_STRATEGY_SPECTRAL_GRADIENT:
    return 3 * n;
  case NANOQSP_STRATEGY_ADMM:
    return 7 * n;
  }
  return NANOQSP_ERR_INVALID_STRATEGY;
}

int nanoqsp_workspace_size_least_squares(int n, NanoqspStrategy strategy) {
  int box_size = nanoqsp_workspace_size_box(n, strategy);
  if (box_size < 0)
    return box_size;
  if (!within_bounds(n, n))
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

static int prepare_workspace(const NanoqspConfig *config, int required_ws,
                             int *max_iterations, double *tolerance,
                             qsp_arena *arena_out, double **owned_buf) {
  *max_iterations = 1000;
  *tolerance = 1e-6;
  *owned_buf = NULL;
  double *ws = NULL;
  int ws_size = 0;

  if (config != NULL) {
    if (config->max_iterations > 0)
      *max_iterations = config->max_iterations;
    if (config->tolerance > 0.0)
      *tolerance = config->tolerance;
    ws = config->workspace;
    ws_size = config->workspace_size;
  }

  if (required_ws > 0 && (ws == NULL || ws_size < required_ws)) {
#ifdef NANOQSP_NO_LIBC
    return NANOQSP_ERR_OUT_OF_MEMORY;
#else
    ws = (double *)malloc((size_t)required_ws * sizeof(double));
    if (!ws)
      return NANOQSP_ERR_OUT_OF_MEMORY;
    ws_size = required_ws;
    *owned_buf = ws;
#endif
  }

  *arena_out = qsp_arena_make(ws, ws_size);
  return NANOQSP_SUCCESS;
}

static void release_workspace(double *owned_buf) {
#ifdef NANOQSP_NO_LIBC
  (void)owned_buf;
#else
  free(owned_buf);
#endif
}

static int dispatch_box(const qsp_mat *D, const double *restrict d,
                        const double *restrict lb, const double *restrict ub,
                        double *restrict x, NanoqspStrategy strategy,
                        int max_iterations, double tolerance, qsp_arena *ws) {
  project_bounds(D->rows, lb, ub, x);
  switch (strategy) {
  case NANOQSP_STRATEGY_COORDINATE_DESCENT:
    return nanoqsp_solve_cd(D, d, lb, ub, x, max_iterations, tolerance, ws);
  case NANOQSP_STRATEGY_PROJECTED_GRADIENT:
    return nanoqsp_solve_pg(D, d, lb, ub, x, max_iterations, tolerance, ws);
  case NANOQSP_STRATEGY_ACCELERATED_GRADIENT:
    return nanoqsp_solve_fista(D, d, lb, ub, x, max_iterations, tolerance, ws);
  case NANOQSP_STRATEGY_SPECTRAL_GRADIENT:
    return nanoqsp_solve_spg(D, d, lb, ub, x, max_iterations, tolerance, ws);
  case NANOQSP_STRATEGY_ADMM:
    return nanoqsp_solve_admm(D, d, NULL, lb, ub, NULL, NULL, x, max_iterations,
                              tolerance, ws);
  }
  return NANOQSP_ERR_INVALID_STRATEGY;
}

int nanoqsp_solve_box(int n, const double *restrict D, const double *restrict d,
                      const double *restrict lb, const double *restrict ub,
                      double *restrict x, const NanoqspConfig *config) {
  if (!within_bounds(n, n) || D == NULL || d == NULL || x == NULL ||
      !validate_config(config) || !validate_bounds(n, lb, ub) ||
      !validate_vector((size_t)n, x) || !validate_vector((size_t)n, d))
    return NANOQSP_ERR_INVALID_ARG;

  NanoqspStrategy strategy =
      config ? config->strategy : NANOQSP_STRATEGY_COORDINATE_DESCENT;
  int required_ws = nanoqsp_workspace_size_box(n, strategy);
  if (required_ws < 0)
    return required_ws;

  nanoqsp_blas_init();

  int max_iterations;
  double tolerance;
  qsp_arena ws;
  double *owned = NULL;
  int status = prepare_workspace(config, required_ws, &max_iterations,
                                 &tolerance, &ws, &owned);
  if (status < 0)
    return status;

  qsp_mat mat_D = qsp_mat_dense(n, n, D);
  int iters = dispatch_box(&mat_D, d, lb, ub, x, strategy, max_iterations,
                           tolerance, &ws);
  release_workspace(owned);
  return iters;
}

int nanoqsp_solve_box_sparse(int n, const NanoqspCSR *D,
                             const double *restrict d,
                             const double *restrict lb,
                             const double *restrict ub, double *restrict x,
                             const NanoqspConfig *config) {
  if (!within_bounds(n, 7) || d == NULL || x == NULL ||
      !validate_config(config) || !validate_csr(D, n, n) ||
      !validate_bounds(n, lb, ub) || !validate_vector((size_t)n, x) ||
      !validate_vector((size_t)n, d))
    return NANOQSP_ERR_INVALID_ARG;

  NanoqspStrategy strategy =
      config ? config->strategy : NANOQSP_STRATEGY_COORDINATE_DESCENT;
  int required_ws = nanoqsp_workspace_size_box_sparse(n, strategy);
  if (required_ws < 0)
    return required_ws;

  nanoqsp_blas_init();

  int max_iterations;
  double tolerance;
  qsp_arena ws;
  double *owned = NULL;
  int status = prepare_workspace(config, required_ws, &max_iterations,
                                 &tolerance, &ws, &owned);
  if (status < 0)
    return status;

  qsp_mat mat_D = qsp_mat_csr(n, n, D);
  int iters = dispatch_box(&mat_D, d, lb, ub, x, strategy, max_iterations,
                           tolerance, &ws);
  release_workspace(owned);
  return iters;
}

int nanoqsp_solve_least_squares(int m, int n, const double *restrict A,
                                const double *restrict b,
                                const double *restrict lb,
                                const double *restrict ub, double *restrict x,
                                const NanoqspConfig *config) {
  if (!within_bounds(m, n) || !within_bounds(n, n) || m <= 0 || A == NULL ||
      b == NULL || x == NULL || !validate_config(config) ||
      !validate_bounds(n, lb, ub) ||
      !validate_vector((size_t)m * (size_t)n, A) ||
      !validate_vector((size_t)n, x) || !validate_vector((size_t)m, b))
    return NANOQSP_ERR_INVALID_ARG;

  NanoqspStrategy strategy =
      config ? config->strategy : NANOQSP_STRATEGY_COORDINATE_DESCENT;
  int required_total = nanoqsp_workspace_size_least_squares(n, strategy);
  if (required_total < 0)
    return required_total;

  nanoqsp_blas_init();

  int max_iterations;
  double tolerance;
  qsp_arena ws;
  double *owned = NULL;
  int status = prepare_workspace(config, required_total, &max_iterations,
                                 &tolerance, &ws, &owned);
  if (status < 0)
    return status;

  double *restrict D = qsp_arena_take(&ws, n * n);
  double *restrict d = qsp_arena_take(&ws, n);
  if (!D || !d) {
    release_workspace(owned);
    return NANOQSP_ERR_OUT_OF_MEMORY;
  }

  nanoqsp_blas.zero(D, n * n);
  nanoqsp_blas.zero(d, n);

  size_t stride = (size_t)n;
  for (int k = 0; k < m; k++) {
    const double *restrict A_row = &A[(size_t)k * stride];
    nanoqsp_blas.axpy(d, A_row, b[k], n);
    for (int i = 0; i < n; i++)
      nanoqsp_blas.axpy(&D[(size_t)i * stride + (size_t)i], &A_row[i], A_row[i],
                        n - i);
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++)
      D[(size_t)i * stride + (size_t)j] = D[(size_t)j * stride + (size_t)i];
  }

  qsp_mat mat_D = qsp_mat_dense(n, n, D);
  double norm_inf = qsp_mat_norm_inf(&mat_D);
  double epsilon = (norm_inf > 0.0) ? (norm_inf * 1e-8) : 1e-8;
  for (int i = 0; i < n; i++)
    D[(size_t)i * stride + (size_t)i] += epsilon;

  int iters = dispatch_box(&mat_D, d, lb, ub, x, strategy, max_iterations,
                           tolerance, &ws);
  release_workspace(owned);
  return iters;
}

static int valid_affine_strategy(NanoqspStrategy strategy) {
  return strategy == NANOQSP_STRATEGY_ADMM ||
         strategy == NANOQSP_STRATEGY_SPECTRAL_GRADIENT;
}

static int dispatch_affine(const qsp_mat *D, const double *restrict d,
                           const qsp_mat *A, const double *restrict lb_x,
                           const double *restrict ub_x,
                           const double *restrict lb_A,
                           const double *restrict ub_A, double *restrict x,
                           NanoqspStrategy strategy, int max_iterations,
                           double tolerance, qsp_arena *ws) {
  project_bounds(D->rows, lb_x, ub_x, x);
  if (strategy == NANOQSP_STRATEGY_SPECTRAL_GRADIENT) {
    return nanoqsp_solve_alm_spg(D, d, A, lb_x, ub_x, lb_A, ub_A, x,
                                 max_iterations, tolerance, ws);
  }
  return nanoqsp_solve_admm(D, d, A, lb_x, ub_x, lb_A, ub_A, x, max_iterations,
                            tolerance, ws);
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
      !validate_vector((size_t)n, x) || !validate_vector((size_t)n, d))
    return NANOQSP_ERR_INVALID_ARG;

  NanoqspStrategy strategy = config ? config->strategy : NANOQSP_STRATEGY_ADMM;
  if (!valid_affine_strategy(strategy))
    return NANOQSP_ERR_INVALID_STRATEGY;

  int required_ws = nanoqsp_workspace_size_affine(n, m);
  if (required_ws < 0)
    return required_ws;

  nanoqsp_blas_init();

  int max_iterations;
  double tolerance;
  qsp_arena ws;
  double *owned = NULL;
  int status = prepare_workspace(config, required_ws, &max_iterations,
                                 &tolerance, &ws, &owned);
  if (status < 0)
    return status;

  qsp_mat mat_D = qsp_mat_dense(n, n, D);
  qsp_mat mat_A = qsp_mat_dense(m, n, A);
  int iters =
      dispatch_affine(&mat_D, d, m > 0 ? &mat_A : NULL, lb_x, ub_x, lb_A, ub_A,
                      x, strategy, max_iterations, tolerance, &ws);
  release_workspace(owned);
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
      !validate_vector((size_t)n, x) || !validate_vector((size_t)n, d))
    return NANOQSP_ERR_INVALID_ARG;

  NanoqspStrategy strategy = config ? config->strategy : NANOQSP_STRATEGY_ADMM;
  if (!valid_affine_strategy(strategy))
    return NANOQSP_ERR_INVALID_STRATEGY;

  nanoqsp_blas_init();

  int max_iterations;
  double tolerance;
  qsp_arena ws;
  double *owned = NULL;
  int status = prepare_workspace(config, required_ws, &max_iterations,
                                 &tolerance, &ws, &owned);
  if (status < 0)
    return status;

  qsp_mat mat_D = qsp_mat_csr(n, n, D);
  qsp_mat mat_A = qsp_mat_csr(m, n, A);
  int iters =
      dispatch_affine(&mat_D, d, m > 0 ? &mat_A : NULL, lb_x, ub_x, lb_A, ub_A,
                      x, strategy, max_iterations, tolerance, &ws);
  release_workspace(owned);
  return iters;
}
