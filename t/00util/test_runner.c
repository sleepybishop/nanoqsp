#include "nanoqsp.h"
#include "qsp_blas.h"
#include <math.h>
#if defined(__riscv) && defined(__riscv_vector)
#include <riscv_vector.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long int rng_state = 1;

static int next_rand(void) {
  rng_state = rng_state * 1103515245UL + 12345UL;
  return (int)((unsigned int)(rng_state / 65536UL) % 32768U);
}

static double rand_uniform(double min, double max) {
  double r = (double)next_rand() / 32768.0;
  return min + r * (max - min);
}

static int verify_kkt_dense(int n, const double *D, const double *d,
                            const double *lb, const double *ub, const double *x,
                            double tol) {
  for (int i = 0; i < n; i++) {
    double grad_i = -d[i];
    for (int j = 0; j < n; j++)
      grad_i += D[i * n + j] * x[j];
    double lower = lb ? lb[i] : -INFINITY;
    double upper = ub ? ub[i] : INFINITY;
    double viol = 0.0;
    if (lower == upper) {
      viol = 0.0;
    } else if (x[i] <= lower + 1e-9) {
      if (grad_i < 0.0)
        viol = -grad_i;
    } else if (x[i] >= upper - 1e-9) {
      if (grad_i > 0.0)
        viol = grad_i;
    } else {
      viol = fabs(grad_i);
    }
    if (viol > tol) {
      fprintf(stderr,
              "KKT violation at index %d: x=%f, grad=%f, viol=%e (tol=%e)\n", i,
              x[i], grad_i, viol, tol);
      return 0;
    }
  }
  return 1;
}

static int verify_kkt_sparse(int n, const NanoqspCSR *D, const double *d,
                             const double *lb, const double *ub,
                             const double *x, double tol) {
  for (int i = 0; i < n; i++) {
    double grad_i = -d[i];
    int start = D->row_ptr[i];
    int end = D->row_ptr[i + 1];
    for (int idx = start; idx < end; idx++)
      grad_i += D->values[idx] * x[D->col_indices[idx]];
    double lower = lb ? lb[i] : -INFINITY;
    double upper = ub ? ub[i] : INFINITY;
    double viol = 0.0;
    if (lower == upper) {
      viol = 0.0;
    } else if (x[i] <= lower + 1e-9) {
      if (grad_i < 0.0)
        viol = -grad_i;
    } else if (x[i] >= upper - 1e-9) {
      if (grad_i > 0.0)
        viol = grad_i;
    } else {
      viol = fabs(grad_i);
    }
    if (viol > tol) {
      fprintf(stderr,
              "KKT violation at index %d: x=%f, grad=%f, viol=%e (tol=%e)\n", i,
              x[i], grad_i, viol, tol);
      return 0;
    }
  }
  return 1;
}

static void generate_random_qp(int n, double *D, double *d, double *lb,
                               double *ub, int type) {
  rng_state = 42; /* fixed seed for deterministic tests */

  double *M = malloc((size_t)n * (size_t)n * sizeof(double));
  for (int i = 0; i < n * n; i++)
    M[i] = rand_uniform(-1.0, 1.0);

  /* form SPD matrix D = M^T M + shift * I */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++)
        sum += M[k * n + i] * M[k * n + j];
      D[i * n + j] = sum;
    }
  }
  free(M);

  double shift = (type == 3) ? 1e-4 : 0.5;
  for (int i = 0; i < n; i++)
    D[i * n + i] += shift;

  if (type == 3) {
    /* congruence scaling creates ill-conditioning while preserving SPD */
    for (int i = 0; i < n; i++) {
      double scale = pow(10.0, rand_uniform(-1.5, 1.5));
      for (int j = 0; j < n; j++) {
        D[i * n + j] *= scale;
        D[j * n + i] *= scale;
      }
    }
  }

  for (int i = 0; i < n; i++)
    d[i] = rand_uniform(-2.0, 2.0);

  for (int i = 0; i < n; i++) {
    if (type == 1) {
      lb[i] = -INFINITY;
      ub[i] = INFINITY;
    } else if (type == 2) {
      lb[i] = 0.0;
      ub[i] = INFINITY;
    } else {
      lb[i] = rand_uniform(-1.0, 0.0);
      ub[i] = rand_uniform(0.1, 2.0);
    }
  }
}

static int test_cross_verify(int test_id) {
  int n = 10;
  int type = 0;

  if (test_id == 5) {
    n = 10;
    type = 0;
  } else if (test_id == 6) {
    n = 50;
    type = 0;
  } else if (test_id == 7) {
    n = 10;
    type = 1;
  } else if (test_id == 8) {
    n = 10;
    type = 2;
  } else if (test_id == 9) {
    n = 10;
    type = 3;
  } else {
    return -99;
  }

  double *D = malloc((size_t)n * (size_t)n * sizeof(double));
  double *d = malloc((size_t)n * sizeof(double));
  double *lb = malloc((size_t)n * sizeof(double));
  double *ub = malloc((size_t)n * sizeof(double));
  generate_random_qp(n, D, d, lb, ub, type);

  double **solutions = malloc(5 * sizeof(double *));
  int *status = malloc(5 * sizeof(int));
  for (int s = 0; s < 5; s++) {
    solutions[s] = calloc((size_t)n, sizeof(double));
    int ws_size = nanoqsp_workspace_size_box(n, (NanoqspStrategy)s);
    double *ws = malloc((size_t)ws_size * sizeof(double));
    NanoqspConfig config = {
        .strategy = (NanoqspStrategy)s,
        .max_iterations = (type == 3) ? 100000 : 5000,
        .tolerance = 1e-6,
        .workspace = ws,
        .workspace_size = ws_size,
    };
    status[s] = nanoqsp_solve_box(n, D, d, lb, ub, solutions[s], &config);
    free(ws);
  }

  int match = 1;
  for (int s = 0; s < 5; s++) {
    if (status[s] < 0) {
      fprintf(stderr, "Strategy %d failed with status %d\n", s, status[s]);
      match = 0;
      break;
    }
    double kkt_tol = (type == 3) ? 1e-1 : 1e-2;
    if (!verify_kkt_dense(n, D, d, lb, ub, solutions[s], kkt_tol)) {
      fprintf(stderr, "Strategy %d failed KKT verification\n", s);
      match = 0;
      break;
    }
  }

  if (match) {
    int ref_strat = 3;
    for (int s = 0; s < 5; s++) {
      if (s == ref_strat)
        continue;
      double max_diff = 0.0;
      for (int i = 0; i < n; i++) {
        double diff = fabs(solutions[s][i] - solutions[ref_strat][i]);
        if (diff > max_diff)
          max_diff = diff;
      }
      double tol = (s == 0 || type == 3) ? 1e-2 : 1e-3;
      if (max_diff > tol) {
        fprintf(
            stderr,
            "Mismatch between strategy %d and %d: max_diff = %e (tol = %e)\n",
            s, ref_strat, max_diff, tol);
        match = 0;
        break;
      }
    }
  }

  for (int s = 0; s < 5; s++)
    free(solutions[s]);
  free(solutions);
  free(status);
  free(D);
  free(d);
  free(lb);
  free(ub);
  return match ? 0 : -2;
}

/* test BLAS kernels and solver parity across all supported SIMD backends */
static int test_simd_backends(void) {
  int max_n = 257;
#if defined(__riscv) && defined(__riscv_vector)
  int rvv_n = (int)(2 * __riscv_vsetvlmax_e64m4() + 1);
  if (rvv_n > max_n)
    max_n = rvv_n;
#endif

  double *a_buf = malloc((size_t)(max_n + 2) * sizeof(double));
  double *b_buf = malloc((size_t)(max_n + 2) * sizeof(double));
  double *y_buf = malloc((size_t)(max_n + 2) * sizeof(double));
  double *A_mat = malloc(11 * (size_t)max_n * sizeof(double));
  if (!a_buf || !b_buf || !y_buf || !A_mat) {
    free(a_buf);
    free(b_buf);
    free(y_buf);
    free(A_mat);
    return -10;
  }

  /* offset by 1 element to exercise unaligned vector loads and stores */
  double *a = a_buf + 1;
  double *b = b_buf + 1;
  double *y = y_buf + 1;
  for (int i = 0; i < 11 * max_n; i++)
    A_mat[i] = (double)((i % 13) - 6) * 0.5;

  int qp_n = 37;
  double *D = malloc((size_t)qp_n * (size_t)qp_n * sizeof(double));
  double *d = malloc((size_t)qp_n * sizeof(double));
  double *lb = malloc((size_t)qp_n * sizeof(double));
  double *ub = malloc((size_t)qp_n * sizeof(double));
  double *x_ref = calloc((size_t)qp_n, sizeof(double));
  double *x_cur = calloc((size_t)qp_n, sizeof(double));
  generate_random_qp(qp_n, D, d, lb, ub, 0);

  int have_ref = 0;
  int ret = 0;

  for (int k = 0; k < QSP_SIMD_COUNT; k++) {
    if (!nanoqsp_blas_select((qsp_simd_kind)k))
      continue;

    double dot = 0.0;
    double norm = 0.0;
    for (int n = 0; n <= max_n; n++) {
      if (n > 0) {
        a[n - 1] = (double)((n % 7) - 3);
        b[n - 1] = (double)((n % 5) - 2);
        dot += a[n - 1] * b[n - 1];
        norm += fabs(a[n - 1]);
      }
      if (nanoqsp_blas.dot(a, b, n) != dot ||
          nanoqsp_blas.norm1(a, n) != norm) {
        fprintf(stderr, "Reduction mismatch on %s at n=%d\n", nanoqsp_blas.name,
                n);
        ret = -10;
        break;
      }
      nanoqsp_blas.zero(y, n);
      nanoqsp_blas.copy(y, a, n);
      nanoqsp_blas.axpy(y, b, -2.5, n);
      for (int i = 0; i < n; i++) {
        if (y[i] != a[i] - 2.5 * b[i]) {
          fprintf(stderr, "axpy/copy mismatch on %s at n=%d\n",
                  nanoqsp_blas.name, n);
          ret = -10;
          break;
        }
      }
      if (ret != 0)
        break;
    }
    if (ret != 0)
      break;

    /* exercise 4-row blocked and tail rows in gemv */
    for (int m = 0; m <= 11 && ret == 0; m++) {
      for (int n = 0; n <= 35; n += 7) {
        double mv[11];
        nanoqsp_blas.gemv(m, n, A_mat, a, mv);
        for (int r = 0; r < m; r++) {
          double expected = 0.0;
          for (int c = 0; c < n; c++)
            expected += A_mat[r * n + c] * a[c];
          if (fabs(mv[r] - expected) > 1e-12) {
            fprintf(stderr, "gemv mismatch on %s at m=%d n=%d\n",
                    nanoqsp_blas.name, m, n);
            ret = -10;
            break;
          }
        }
        if (ret != 0)
          break;
      }
    }
    if (ret != 0)
      break;

    memset(x_cur, 0, (size_t)qp_n * sizeof(double));
    NanoqspConfig cfg = {.strategy = NANOQSP_STRATEGY_SPECTRAL_GRADIENT,
                         .max_iterations = 2000,
                         .tolerance = 1e-9};
    int st = nanoqsp_solve_box(qp_n, D, d, lb, ub, x_cur, &cfg);
    if (st < 0) {
      ret = -10;
      break;
    }
    if (!have_ref) {
      memcpy(x_ref, x_cur, (size_t)qp_n * sizeof(double));
      have_ref = 1;
    } else {
      for (int i = 0; i < qp_n; i++) {
        if (fabs(x_cur[i] - x_ref[i]) > 1e-8) {
          fprintf(stderr, "QP mismatch on %s at i=%d\n", nanoqsp_blas.name, i);
          ret = -10;
          break;
        }
      }
    }
    if (ret != 0)
      break;
  }

  /* restore fastest available backend */
  for (int k = QSP_SIMD_COUNT - 1; k >= 0; k--) {
    if (nanoqsp_blas_select((qsp_simd_kind)k))
      break;
  }

  free(a_buf);
  free(b_buf);
  free(y_buf);
  free(A_mat);
  free(D);
  free(d);
  free(lb);
  free(ub);
  free(x_ref);
  free(x_cur);
  return ret;
}

/* numerical regressions allocate the exact advertised workspace size */
static int numerical_box(int sparse, int n, const double *D, const double *d,
                         const double *lb, const double *ub, double *x,
                         NanoqspConfig config) {
  int cols[] = {0, 1, 0, 1};
  int rows[] = {0, n, n * n};
  NanoqspCSR csr = {n, n * n, D, cols, rows};
  int size = sparse ? nanoqsp_workspace_size_box_sparse(n, config.strategy)
                    : nanoqsp_workspace_size_box(n, config.strategy);
  if (size < 0)
    return -10;
  double *ws = size ? malloc((size_t)size * sizeof(double)) : NULL;
  if (size && !ws)
    return -10;
  config.workspace = ws;
  config.workspace_size = size;
  int ret = sparse ? nanoqsp_solve_box_sparse(n, &csr, d, lb, ub, x, &config)
                   : nanoqsp_solve_box(n, D, d, lb, ub, x, &config);
  free(ws);
  return ret;
}

static int numerical_affine(int sparse, int n, const double *D, const double *d,
                            const double *A, const double *lb, const double *ub,
                            const double *lb_A, const double *ub_A, double *x,
                            NanoqspConfig config) {
  int cols[] = {0, 1, 0, 1};
  int rows[] = {0, n, n * n};
  NanoqspCSR csr_D = {n, n * n, D, cols, rows};
  NanoqspCSR csr_A = {1, n, A, cols, rows};
  int size = nanoqsp_workspace_size_affine(n, 1);
  double *ws = malloc((size_t)size * sizeof(double));
  if (!ws)
    return -10;
  config.workspace = ws;
  config.workspace_size = size;
  int ret = sparse ? nanoqsp_solve_affine_sparse(n, 1, &csr_D, d, &csr_A, lb,
                                                 ub, lb_A, ub_A, x, &config)
                   : nanoqsp_solve_affine(n, 1, D, d, A, lb, ub, lb_A, ub_A, x,
                                          &config);
  free(ws);
  return ret;
}

static int numerical_solution(int ret, int n, const double *x,
                              const double *expected, double tol) {
  if (ret < 0) {
    fprintf(stderr, "Numerical regression failed with status %d\n", ret);
    return -10;
  }
  for (int i = 0; i < n; i++) {
    if (!(fabs(x[i] - expected[i]) <= tol)) {
      fprintf(stderr, "x[%d]=%.17g, expected %.17g within %.3g\n", i, x[i],
              expected[i], tol);
      return -10;
    }
  }
  return 0;
}

static int run_numerical_test(int test_id, int strategy_id, double *x) {
  NanoqspConfig config = {.strategy = (NanoqspStrategy)strategy_id,
                          .max_iterations = 1000,
                          .tolerance = 1e-9};
  int sparse = test_id % 2 == 0;
  if (test_id == 33 || test_id == 34) {
    /* negative gradient at a fixed coordinate is still optimal */
    double D[] = {2.0, 1.0, 1.0, 2.0}, d[] = {2.0, 2.0};
    double lb[] = {0.0, -10.0}, ub[] = {0.0, 10.0};
    double expected[] = {0.0, 1.0};
    int ret = numerical_box(sparse, 2, D, d, lb, ub, x, config);
    return numerical_solution(ret, 2, x, expected, 1e-7);
  } else if (test_id == 35 || test_id == 36) {
    double D[] = {1e-13}, d[] = {1.0}, lb[] = {0.0}, ub[] = {1.0};
    double expected[] = {1.0};
    int ret = numerical_box(sparse, 1, D, d, lb, ub, x, config);
    return numerical_solution(ret, 1, x, expected, 1e-7);
  } else if (test_id == 37 || test_id == 38) {
    /* unsafeguarded BB steps cycle on this strictly convex QP */
    double D[] = {7.5, 3.9, 3.9, 2.2}, d[] = {2.3, 0.7};
    double lb[] = {-1.0, -1.0}, ub[] = {1.0, 1.0};
    double expected[] = {6.2 / 7.5, -1.0};
    int ret = numerical_box(sparse, 2, D, d, lb, ub, x, config);
    return numerical_solution(ret, 2, x, expected, 1e-7);
  } else if (test_id == 39 || test_id == 40) {
    double scales[] = {1e-14, 1e-12, 1.0, 1e12};
    for (int k = 0; k < 4; k++) {
      double scale = scales[k];
      double D[] = {scale, 0.0, 0.0, 2.0 * scale};
      double d[] = {scale, 2.0 * scale}, lb[] = {0.0, 0.0}, ub[] = {2.0, 2.0};
      double expected[] = {1.0, 1.0};
      config.tolerance = scale < 1.0 ? scale * 1e-8 : 1e-8;
      x[0] = x[1] = 0.0;
      int ret = numerical_box(sparse, 2, D, d, lb, ub, x, config);
      if (numerical_solution(ret, 2, x, expected, 1e-6))
        return -10;
    }
    return 0;
  } else if (test_id >= 41 && test_id <= 44) {
    double scales[] = {1e-6, 1.0, 1e3, 1e6};
    config.tolerance = 1e-8;
    for (int k = 0; k < 4; k++) {
      double scale = scales[k];
      x[0] = x[1] = 0.0;
      if (test_id <= 42) {
        double D[] = {1.0}, d[] = {1.0}, A[] = {scale};
        double lb_A[] = {0.0}, ub_A[] = {10.0 * scale};
        double expected[] = {1.0};
        int ret = numerical_affine(sparse, 1, D, d, A, NULL, NULL, lb_A, ub_A,
                                   x, config);
        if (numerical_solution(ret, 1, x, expected, 1e-6))
          return -10;
      } else {
        double D[] = {1.0, 0.0, 0.0, 2.0}, d[] = {1.0, 1.0};
        double A[] = {scale, scale}, lb[] = {0.0, 0.0}, ub[] = {0.5, 1.0};
        double lb_A[] = {scale}, ub_A[] = {scale};
        double expected[] = {0.5, 0.5};
        int ret =
            numerical_affine(sparse, 2, D, d, A, lb, ub, lb_A, ub_A, x, config);
        if (numerical_solution(ret, 2, x, expected, 1e-6) || x[0] < lb[0] ||
            x[0] > ub[0] || x[1] < lb[1] || x[1] > ub[1] ||
            !(fabs(x[0] + x[1] - 1.0) <= 5e-7))
          return -10;
      }
    }
    return 0;
  } else if (test_id == 45 || test_id == 46) {
    double D[] = {1.0}, d[] = {1.0}, expected[] = {1.0};
    config.strategy = NANOQSP_STRATEGY_ADMM;
    config.tolerance = 1e-14;
    int ret = numerical_box(sparse, 1, D, d, NULL, NULL, x, config);
    return numerical_solution(ret, 1, x, expected, 2.1e-14);
  } else if (test_id == 47) {
    double scales[] = {1.0, 1.0001e-6, 1e-6};
    double expected[] = {1.0 / (1.0 + 1e-8)};
    config.tolerance = 1e-20;
    int size = nanoqsp_workspace_size_least_squares(1, config.strategy);
    double *ws = malloc((size_t)size * sizeof(double));
    if (!ws)
      return -10;
    config.workspace = ws;
    config.workspace_size = size;
    int result = 0;
    for (int k = 0; k < 3; k++) {
      double A[] = {scales[k]}, b[] = {scales[k]};
      x[0] = 0.0;
      int ret = nanoqsp_solve_least_squares(1, 1, A, b, NULL, NULL, x, &config);
      if (numerical_solution(ret, 1, x, expected, 1e-7)) {
        result = -10;
        break;
      }
    }
    free(ws);
    return result;
  } else if (test_id == 48 || test_id == 49) {
    /* small iterate change must not masquerade as affine convergence */
    double D[] = {1.0}, d[] = {1.0}, A[] = {1e6};
    double lb_A[] = {0.0}, ub_A[] = {1e7};
    config.strategy = NANOQSP_STRATEGY_ADMM;
    config.max_iterations = 1;
    config.tolerance = 1e-6;
    int ret = numerical_affine(test_id == 49, 1, D, d, A, NULL, NULL, lb_A,
                               ub_A, x, config);
    return ret == NANOQSP_ERR_MAX_ITERATIONS ? 0 : -10;
  } else if (test_id == 50) {
    /* non-finite vectors, degenerate infinite bounds, and bad configs */
    double D[] = {1.0};
    double d_ok[] = {1.0};
    double d_inf[] = {INFINITY};
    double x_inf[] = {-INFINITY};
    double lb_inf[] = {INFINITY};
    double ub_inf[] = {INFINITY};
    double lb_ninf[] = {-INFINITY};
    double ub_ninf[] = {-INFINITY};
    x[0] = 0.0;
    if (nanoqsp_solve_box(1, D, d_inf, NULL, NULL, x, &config) !=
            NANOQSP_ERR_INVALID_ARG ||
        nanoqsp_solve_box(1, D, d_ok, NULL, NULL, x_inf, &config) !=
            NANOQSP_ERR_INVALID_ARG ||
        nanoqsp_solve_box(1, D, d_ok, lb_inf, ub_inf, x, &config) !=
            NANOQSP_ERR_INVALID_ARG ||
        nanoqsp_solve_box(1, D, d_ok, lb_ninf, ub_ninf, x, &config) !=
            NANOQSP_ERR_INVALID_ARG)
      return -10;

    NanoqspConfig bad = config;
    bad.tolerance = -1e-6;
    if (nanoqsp_solve_box(1, D, d_ok, NULL, NULL, x, &bad) !=
        NANOQSP_ERR_INVALID_ARG)
      return -10;
    bad = config;
    bad.max_iterations = -5;
    if (nanoqsp_solve_box(1, D, d_ok, NULL, NULL, x, &bad) !=
        NANOQSP_ERR_INVALID_ARG)
      return -10;
    bad = config;
    bad.workspace = NULL;
    bad.workspace_size = 4;
    if (nanoqsp_solve_box(1, D, d_ok, NULL, NULL, x, &bad) !=
        NANOQSP_ERR_INVALID_ARG)
      return -10;
    return 0;
  } else if (test_id == 51 || test_id == 52) {
    /* large ||A|| must not relax O(1) box primal accuracy */
    double D[] = {1.0, 0.0, 0.0, 1.0};
    double d[] = {1.0, 0.5};
    double A[] = {1e6, 0.0};
    double lb[] = {-10.0, 0.0}, ub[] = {10.0, 1.0};
    double lb_A[] = {0.0}, ub_A[] = {2e6};
    double expected[] = {1.0, 0.5};
    config.tolerance = 1e-6;
    x[0] = 0.0;
    x[1] = 1.0;
    int ret = numerical_affine(test_id == 52, 2, D, d, A, lb, ub, lb_A, ub_A, x,
                               config);
    return numerical_solution(ret, 2, x, expected, 1e-5);
  } else if (test_id == 53 || test_id == 54) {
    /* multiscale pose (1e6) + global (scale, bias ~ 1) affine QP */
    enum { N = 16, M = 2 };
    double D[N * N];
    double d[N], lb[N], ub[N], A[M * N], sol[N], ref[N];
    memset(D, 0, sizeof(D));
    memset(A, 0, sizeof(A));
    for (int i = 0; i < N; i++) {
      double w = (i < N - 2) ? 1e6 : 1.0;
      D[i * N + i] = w;
      if (i + 1 < N - 2) {
        D[i * N + (i + 1)] = -0.1 * w;
        D[(i + 1) * N + i] = -0.1 * w;
      }
      d[i] = (i < N - 2) ? (w * (0.2 * (double)((i % 5) - 2)))
                         : (i == N - 2 ? 1.5 : -0.8);
      lb[i] = (i < N - 2) ? -0.25 : -2.0;
      ub[i] = (i < N - 2) ? 0.25 : 2.0;
      sol[i] = 0.0;
      ref[i] = 0.0;
      A[0 * N + i] = 1.0;
      A[1 * N + i] = (i % 2 == 0) ? 1.0 : -1.0;
    }
    double lb_A[M] = {0.4, -0.5};
    double ub_A[M] = {0.4, 0.3};

    NanoqspConfig ref_cfg = {.strategy = NANOQSP_STRATEGY_SPECTRAL_GRADIENT,
                             .max_iterations = 500,
                             .tolerance = 1e-12};
    if (nanoqsp_solve_affine(N, M, D, d, A, lb, ub, lb_A, ub_A, ref, &ref_cfg) <
        0)
      return -10;

    config.max_iterations = 200;
    config.tolerance = 1e-10;
    int ret = -1;
    if (!sparse) {
      ret =
          nanoqsp_solve_affine(N, M, D, d, A, lb, ub, lb_A, ub_A, sol, &config);
    } else {
      double d_vals[3 * N], a_vals[M * N];
      int d_cols[3 * N], d_rows[N + 1], a_cols[M * N], a_rows[M + 1];
      int dnnz = 0, annz = 0;
      for (int i = 0; i < N; i++) {
        d_rows[i] = dnnz;
        for (int j = 0; j < N; j++) {
          if (D[i * N + j] != 0.0) {
            d_vals[dnnz] = D[i * N + j];
            d_cols[dnnz++] = j;
          }
        }
      }
      d_rows[N] = dnnz;
      for (int i = 0; i < M; i++) {
        a_rows[i] = annz;
        for (int j = 0; j < N; j++) {
          a_vals[annz] = A[i * N + j];
          a_cols[annz++] = j;
        }
      }
      a_rows[M] = annz;
      NanoqspCSR csr_D = {N, dnnz, d_vals, d_cols, d_rows};
      NanoqspCSR csr_A = {M, annz, a_vals, a_cols, a_rows};
      ret = nanoqsp_solve_affine_sparse(N, M, &csr_D, d, &csr_A, lb, ub, lb_A,
                                        ub_A, sol, &config);
    }
    if (ret < 0 || ret > 120)
      return -10;
    return numerical_solution(ret, N, sol, ref, 1e-4);
  } else if (test_id == 55 || test_id == 56) {
    /* high-scale ||D|| ~ 1e6 must seed rho_0 near ||D|| without 200-step lag */
    double D[] = {2e6, 0.5e6, 0.5e6, 1.5e6};
    double d[] = {3e6, -1e6};
    double A[] = {1.0, 1.0};
    double lb[] = {-2.0, -2.0}, ub[] = {2.0, 2.0};
    double lb_A[] = {-0.5}, ub_A[] = {0.5};
    double ref[2] = {0.0, 0.0};
    NanoqspConfig ref_cfg = {.strategy = NANOQSP_STRATEGY_SPECTRAL_GRADIENT,
                             .max_iterations = 200,
                             .tolerance = 1e-10};
    if (nanoqsp_solve_affine(2, 1, D, d, A, lb, ub, lb_A, ub_A, ref, &ref_cfg) <
        0)
      return -10;
    config.max_iterations = 50;
    config.tolerance = 1e-8;
    x[0] = x[1] = 0.0;
    int ret =
        numerical_affine(sparse, 2, D, d, A, lb, ub, lb_A, ub_A, x, config);
    if (ret < 0 || ret > 40)
      return -10;
    return numerical_solution(ret, 2, x, ref, 1e-6);
  } else if (test_id == 57 || test_id == 58) {
    /* multi-row affine inequalities + equality cross-check (ADMM vs ALM-SPG) */
    enum { N = 8, M = 3 };
    double D[N * N], d[N], lb[N], ub[N], A[M * N], s_admm[N], s_spg[N];
    rng_state = 123;
    generate_random_qp(N, D, d, lb, ub, 0);
    for (int i = 0; i < M * N; i++)
      A[i] = rand_uniform(-1.5, 1.5);
    double lb_A[M] = {0.2, -0.5, -INFINITY};
    double ub_A[M] = {0.2, 0.4, 0.3};
    for (int i = 0; i < N; i++) {
      s_admm[i] = -10.0; /* infeasible warm start */
      s_spg[i] = 10.0;
    }
    int ws_sz = nanoqsp_workspace_size_affine(N, M);
    double *ws = malloc((size_t)ws_sz * sizeof(double));
    if (!ws)
      return -10;

    NanoqspConfig c_admm = {.strategy = NANOQSP_STRATEGY_ADMM,
                            .max_iterations = 500,
                            .tolerance = 1e-8,
                            .workspace = ws,
                            .workspace_size = ws_sz};
    NanoqspConfig c_spg = c_admm;
    c_spg.strategy = NANOQSP_STRATEGY_SPECTRAL_GRADIENT;

    int r_admm, r_spg;
    if (!sparse) {
      r_admm = nanoqsp_solve_affine(N, M, D, d, A, lb, ub, lb_A, ub_A, s_admm,
                                    &c_admm);
      r_spg = nanoqsp_solve_affine(N, M, D, d, A, lb, ub, lb_A, ub_A, s_spg,
                                   &c_spg);
    } else {
      double d_vals[N * N], a_vals[M * N];
      int d_cols[N * N], d_rows[N + 1], a_cols[M * N], a_rows[M + 1];
      for (int i = 0; i < N; i++) {
        d_rows[i] = i * N;
        for (int j = 0; j < N; j++) {
          d_vals[i * N + j] = D[i * N + j];
          d_cols[i * N + j] = j;
        }
      }
      d_rows[N] = N * N;
      for (int i = 0; i < M; i++) {
        a_rows[i] = i * N;
        for (int j = 0; j < N; j++) {
          a_vals[i * N + j] = A[i * N + j];
          a_cols[i * N + j] = j;
        }
      }
      a_rows[M] = M * N;
      NanoqspCSR csr_D = {N, N * N, d_vals, d_cols, d_rows};
      NanoqspCSR csr_A = {M, M * N, a_vals, a_cols, a_rows};
      r_admm = nanoqsp_solve_affine_sparse(N, M, &csr_D, d, &csr_A, lb, ub,
                                           lb_A, ub_A, s_admm, &c_admm);
      r_spg = nanoqsp_solve_affine_sparse(N, M, &csr_D, d, &csr_A, lb, ub, lb_A,
                                          ub_A, s_spg, &c_spg);
    }
    free(ws);
    if (r_admm < 0 || r_spg < 0)
      return -10;
    return numerical_solution(r_spg, N, s_spg, s_admm, 1e-6);
  } else if (test_id == 59 || test_id == 60) {
    /* unconstrained multiscale QP fast-path solves Dx = d in <= N PCG steps */
    enum { N = 12 };
    double D[N * N], x_true[N], d[N], lb_inf[N], ub_inf[N];
    double s1[N], s2[N], s3[N];
    memset(D, 0, sizeof(D));
    for (int i = 0; i < N; i++) {
      double w = (i < N - 2) ? 1e6 : 1.0;
      D[i * N + i] = 2.0 * w;
      if (i + 1 < N) {
        double c = -0.25 * sqrt(w * ((i + 1 < N - 2) ? 1e6 : 1.0));
        D[i * N + (i + 1)] = c;
        D[(i + 1) * N + i] = c;
      }
      x_true[i] = 0.3 * (double)((i % 5) - 2);
      lb_inf[i] = -INFINITY;
      ub_inf[i] = INFINITY;
      s1[i] = s2[i] = s3[i] = 0.0;
    }
    for (int i = 0; i < N; i++) {
      double acc = 0.0;
      for (int j = 0; j < N; j++)
        acc += D[i * N + j] * x_true[j];
      d[i] = acc;
    }

    double vals[N * N];
    int cols[N * N], rows[N + 1];
    int nnz = 0;
    for (int i = 0; i < N; i++) {
      rows[i] = nnz;
      for (int j = 0; j < N; j++) {
        if (D[i * N + j] != 0.0) {
          vals[nnz] = D[i * N + j];
          cols[nnz] = j;
          nnz++;
        }
      }
    }
    rows[N] = nnz;
    NanoqspCSR csr_D = {N, nnz, vals, cols, rows};

    int ws_box =
        sparse ? nanoqsp_workspace_size_box_sparse(N, NANOQSP_STRATEGY_ADMM)
               : nanoqsp_workspace_size_box(N, NANOQSP_STRATEGY_ADMM);
    int ws_aff = nanoqsp_workspace_size_affine(N, 0);
    int ws_max = ws_box > ws_aff ? ws_box : ws_aff;
    double *ws = malloc((size_t)ws_max * sizeof(double));
    if (!ws)
      return -10;

    config.strategy = NANOQSP_STRATEGY_ADMM;
    config.max_iterations = N;
    config.tolerance = 1e-12;
    config.workspace = ws;
    config.workspace_size = ws_box;

    int r1 =
        sparse ? nanoqsp_solve_box_sparse(N, &csr_D, d, NULL, NULL, s1, &config)
               : nanoqsp_solve_box(N, D, d, NULL, NULL, s1, &config);
    int r2 = sparse ? nanoqsp_solve_box_sparse(N, &csr_D, d, lb_inf, ub_inf, s2,
                                               &config)
                    : nanoqsp_solve_box(N, D, d, lb_inf, ub_inf, s2, &config);
    config.workspace_size = ws_aff;
    int r3 = sparse ? nanoqsp_solve_affine_sparse(N, 0, &csr_D, d, NULL, NULL,
                                                  NULL, NULL, NULL, s3, &config)
                    : nanoqsp_solve_affine(N, 0, D, d, NULL, NULL, NULL, NULL,
                                           NULL, s3, &config);
    free(ws);

    /* singular unbounded unconstrained QP must still fail cleanly */
    double D_zero[4] = {0.0, 0.0, 0.0, 0.0};
    double d_pos[2] = {1.0, 1.0};
    double x_bad[2] = {0.0, 0.0};
    NanoqspConfig c_bad = {.strategy = NANOQSP_STRATEGY_ADMM,
                           .max_iterations = 50,
                           .tolerance = 1e-8};
    int r_bad =
        numerical_box(sparse, 2, D_zero, d_pos, NULL, NULL, x_bad, c_bad);

    if (r1 <= 0 || r1 > N || r2 <= 0 || r2 > N || r3 <= 0 || r3 > N ||
        r_bad != NANOQSP_ERR_MAX_ITERATIONS)
      return -10;
    if (numerical_solution(r1, N, s1, x_true, 1e-10) ||
        numerical_solution(r2, N, s2, x_true, 1e-10) ||
        numerical_solution(r3, N, s3, x_true, 1e-10))
      return -10;
    x[0] = s1[0];
    x[1] = s1[N - 1];
    return 0;
  } else if (test_id == 61 || test_id == 62) {
    /* non-convex [0, 1] CLIPPER max-clique QP with negative curvature */
    enum { N = 6 };
    const int adj[N][N] = {
        {1, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 0, 0},
        {1, 1, 1, 1, 1, 0}, {0, 0, 0, 1, 1, 1}, {0, 0, 0, 0, 1, 1},
    };
    double D[N * N], d[N], lb[N], ub[N], sol[N];
    const double expected[N] = {1.0, 1.0, 1.0, 1.0, 0.0, 0.0};
    double vals[N * N];
    int cols[N * N], rows[N + 1];
    int nnz = 0;

    for (int i = 0; i < N; i++) {
      rows[i] = nnz;
      for (int j = 0; j < N; j++) {
        double m_ij = adj[i][j] ? 1.0 : -3.0;
        D[i * N + j] = -1e4 * m_ij;
        vals[nnz] = D[i * N + j];
        cols[nnz] = j;
        nnz++;
      }
      d[i] = 0.0;
      lb[i] = 0.0;
      ub[i] = 1.0;
      sol[i] = 0.5;
    }
    rows[N] = nnz;
    NanoqspCSR csr_D = {N, nnz, vals, cols, rows};

    int ws_sz =
        sparse
            ? nanoqsp_workspace_size_box_sparse(
                  N, NANOQSP_STRATEGY_SPECTRAL_GRADIENT)
            : nanoqsp_workspace_size_box(N, NANOQSP_STRATEGY_SPECTRAL_GRADIENT);
    double *ws = malloc((size_t)ws_sz * sizeof(double));
    if (!ws)
      return -10;

    config.strategy = NANOQSP_STRATEGY_SPECTRAL_GRADIENT;
    config.max_iterations = 200;
    config.tolerance = 1e-9;
    config.workspace = ws;
    config.workspace_size = ws_sz;

    int ret = sparse
                  ? nanoqsp_solve_box_sparse(N, &csr_D, d, lb, ub, sol, &config)
                  : nanoqsp_solve_box(N, D, d, lb, ub, sol, &config);
    free(ws);
    if (numerical_solution(ret, N, sol, expected, 1e-9))
      return -10;
    x[0] = sol[0];
    x[1] = sol[4];
    return 0;
  }
  return -99;
}

static int run_test(int test_id, int strategy_id, double *out_x) {
  if (test_id >= 33)
    return run_numerical_test(test_id, strategy_id, out_x);

  NanoqspConfig config = {
      .strategy = (NanoqspStrategy)strategy_id,
      .max_iterations = 2000,
      .tolerance = 1e-7,
  };

  if (test_id == 1) {
    int n = 2;
    double D[] = {1.0, 0.0, 0.0, 1.0};
    double d[] = {0.5, 0.5};
    double lb[] = {0.0, 0.0};
    double ub[] = {1.0, 1.0};
    out_x[0] = 0.0;
    out_x[1] = 0.0;
    int ret = nanoqsp_solve_box(n, D, d, lb, ub, out_x, &config);
    if (ret >= 0 && !verify_kkt_dense(n, D, d, lb, ub, out_x, 1e-5))
      return -10;
    return ret;
  } else if (test_id == 2) {
    int n = 2;
    double D[] = {1.0, 0.0, 0.0, 1.0};
    double d[] = {2.0, -1.0};
    double lb[] = {0.0, 0.0};
    double ub[] = {1.0, 1.0};
    out_x[0] = 0.5;
    out_x[1] = 0.5;
    int ret = nanoqsp_solve_box(n, D, d, lb, ub, out_x, &config);
    if (ret >= 0 && !verify_kkt_dense(n, D, d, lb, ub, out_x, 1e-5))
      return -10;
    return ret;
  } else if (test_id == 3) {
    return nanoqsp_solve_box(0, NULL, NULL, NULL, NULL, out_x, &config);
  } else if (test_id == 4) {
    int n = 2;
    double D[] = {0.0, 0.0, 0.0, 0.0};
    double d[] = {1.0, 1.0};
    double lb[] = {0.0, 0.0};
    double ub[] = {10.0, 10.0};
    out_x[0] = 0.0;
    out_x[1] = 0.0;
    return nanoqsp_solve_box(n, D, d, lb, ub, out_x, &config);
  } else if (test_id == 10) {
    /* bounded linear least squares across all solver strategies */
    double A[] = {5.0, 10.0, 3.0, 0.0, 5.0, 10.0, 0.0, 6.0,
                  0.0, 10.0, 0.0, 0.0, 0.0, 0.0,  3.0, 6.0,
                  5.0, 0.0,  0.0, 6.0, 5.0, 10.0, 3.0, 6.0};
    double b[] = {10.50, 15.00, 5.20, 5.40, 9.50, 15.50};
    double lb[] = {0.05, 0.05, 0.05, 0.05};
    double ub[] = {5.00, 5.00, 5.00, 5.00};
    double ref[4] = {1.0, 1.0, 1.0, 1.0};
    NanoqspConfig ref_cfg = {.strategy = NANOQSP_STRATEGY_SPECTRAL_GRADIENT,
                             .max_iterations = 50000,
                             .tolerance = 1e-9};
    if (nanoqsp_solve_least_squares(6, 4, A, b, lb, ub, ref, &ref_cfg) < 0)
      return -10;
    double x_ls[4] = {1.0, 1.0, 1.0, 1.0};
    config.max_iterations = 50000;
    config.tolerance = 1e-7;
    int ret = nanoqsp_solve_least_squares(6, 4, A, b, lb, ub, x_ls, &config);
    out_x[0] = x_ls[0];
    out_x[1] = x_ls[1];
    return numerical_solution(ret, 4, x_ls, ref, 1e-4) ? -10 : ret;
  } else if (test_id == 11) {
    int n = 2;
    double values[] = {1.0, 1.0};
    int col_indices[] = {0, 1};
    int row_ptr[] = {0, 1, 2};
    NanoqspCSR D = {n, 2, values, col_indices, row_ptr};
    double d[] = {0.5, 0.5};
    double lb[] = {0.0, 0.0};
    double ub[] = {1.0, 1.0};
    out_x[0] = 0.0;
    out_x[1] = 0.0;
    int ret = nanoqsp_solve_box_sparse(n, &D, d, lb, ub, out_x, &config);
    if (ret >= 0 && !verify_kkt_sparse(n, &D, d, lb, ub, out_x, 1e-5))
      return -10;
    return ret;
  } else if (test_id == 12) {
    int n = 2;
    double values[] = {1.0, 1.0};
    int col_indices[] = {0, 1};
    int row_ptr[] = {0, 1, 2};
    NanoqspCSR D = {n, 2, values, col_indices, row_ptr};
    double d[] = {2.0, -1.0};
    double lb[] = {0.0, 0.0};
    double ub[] = {1.0, 1.0};
    out_x[0] = 0.5;
    out_x[1] = 0.5;
    int ret = nanoqsp_solve_box_sparse(n, &D, d, lb, ub, out_x, &config);
    if (ret >= 0 && !verify_kkt_sparse(n, &D, d, lb, ub, out_x, 1e-5))
      return -10;
    return ret;
  } else if (test_id == 13) {
    return nanoqsp_solve_box_sparse(0, NULL, NULL, NULL, NULL, out_x, &config);
  } else if (test_id == 14) {
    int n = 2;
    int row_ptr[] = {0, 0, 0};
    NanoqspCSR D = {n, 0, NULL, NULL, row_ptr};
    double d[] = {1.0, 1.0};
    double lb[] = {0.0, 0.0};
    double ub[] = {10.0, 10.0};
    out_x[0] = 0.0;
    out_x[1] = 0.0;
    return nanoqsp_solve_box_sparse(n, &D, d, lb, ub, out_x, &config);
  } else if (test_id == 15) {
    int n = 2;
    int m = 1;
    double D[] = {4.0, 1.0, 1.0, 2.0};
    double d[] = {1.0, 1.0};
    double A[] = {1.0, 2.0};
    double lb_x[] = {0.0, 0.0};
    double ub_x[] = {1.0, 1.0};
    double lb_A[] = {1.5};
    double ub_A[] = {2.0};
    out_x[0] = 0.0;
    out_x[1] = 0.0;
    return nanoqsp_solve_affine(n, m, D, d, A, lb_x, ub_x, lb_A, ub_A, out_x,
                                &config);
  } else if (test_id == 16) {
    int n = 2;
    int m = 1;
    double D_vals[] = {4.0, 1.0, 1.0, 2.0};
    int D_cols[] = {0, 1, 0, 1};
    int D_rows[] = {0, 2, 4};
    NanoqspCSR D = {n, 4, D_vals, D_cols, D_rows};
    double d[] = {1.0, 1.0};
    double A_vals[] = {1.0, 2.0};
    int A_cols[] = {0, 1};
    int A_rows[] = {0, 2};
    NanoqspCSR A = {m, 2, A_vals, A_cols, A_rows};
    double lb_x[] = {0.0, 0.0};
    double ub_x[] = {1.0, 1.0};
    double lb_A[] = {1.5};
    double ub_A[] = {2.0};
    out_x[0] = 0.0;
    out_x[1] = 0.0;
    return nanoqsp_solve_affine_sparse(n, m, &D, d, &A, lb_x, ub_x, lb_A, ub_A,
                                       out_x, &config);
  } else if (test_id == 17) {
    /* infeasible dense SPG warm start must be projected before KKT checks */
    double D[] = {1.0};
    double d[] = {-2.0};
    double lb[] = {0.0};
    double ub[] = {1.0};
    out_x[0] = -1.0;
    config.strategy = NANOQSP_STRATEGY_SPECTRAL_GRADIENT;
    return nanoqsp_solve_box(1, D, d, lb, ub, out_x, &config);
  } else if (test_id == 18) {
    /* infeasible sparse SPG warm start must be projected before KKT checks */
    double values[] = {1.0};
    int col_indices[] = {0};
    int row_ptr[] = {0, 1};
    NanoqspCSR D = {1, 1, values, col_indices, row_ptr};
    double d[] = {-2.0};
    double lb[] = {0.0};
    double ub[] = {1.0};
    out_x[0] = -1.0;
    config.strategy = NANOQSP_STRATEGY_SPECTRAL_GRADIENT;
    return nanoqsp_solve_box_sparse(1, &D, d, lb, ub, out_x, &config);
  } else if (test_id == 19) {
    double D[] = {1.0, 0.0, 0.0, 100.0};
    double d[] = {1.0, 100.0};
    out_x[0] = 0.0;
    out_x[1] = 0.0;
    config.strategy = NANOQSP_STRATEGY_PROJECTED_GRADIENT;
    config.max_iterations = 1;
    config.tolerance = 1e-12;
    return nanoqsp_solve_box(2, D, d, NULL, NULL, out_x, &config);
  } else if (test_id == 20) {
    int col_indices[] = {0};
    int row_ptr[] = {0, 1};
    NanoqspCSR D = {1, 1, NULL, col_indices, row_ptr};
    double d[] = {0.0};
    out_x[0] = 0.0;
    return nanoqsp_solve_box_sparse(1, &D, d, NULL, NULL, out_x, &config);
  } else if (test_id == 21) {
    double D[] = {1.0};
    double d[] = {0.0};
    out_x[0] = 0.0;
    return nanoqsp_solve_affine(1, -1, D, d, NULL, NULL, NULL, NULL, NULL,
                                out_x, NULL);
  } else if (test_id == 22) {
    double D[] = {1.0};
    double d[] = {10.0};
    double lb[] = {0.0};
    double ub[] = {1.0};
    out_x[0] = 0.0;
    config.strategy = NANOQSP_STRATEGY_ADMM;
    config.tolerance = 0.1;
    return nanoqsp_solve_affine(1, 0, D, d, NULL, lb, ub, NULL, NULL, out_x,
                                &config);
  } else if (test_id == 23) {
    double D[] = {1.0};
    double d[] = {0.0};
    double lb[] = {2.0};
    double ub[] = {1.0};
    out_x[0] = 0.0;
    return nanoqsp_solve_box(1, D, d, lb, ub, out_x, &config);
  } else if (test_id == 24) {
    double D[] = {1.0};
    double d[] = {0.0};
    out_x[0] = 0.0;
    config.strategy = NANOQSP_STRATEGY_PROJECTED_GRADIENT;
    return nanoqsp_solve_affine(1, 0, D, d, NULL, NULL, NULL, NULL, NULL, out_x,
                                &config);
  } else if (test_id == 25) {
    double values[] = {10000.0, 9999.0, 9999.0, 10000.0};
    int col_indices[] = {0, 1, 0, 1};
    int row_ptr[] = {0, 2, 4};
    NanoqspCSR sparse_D = {2, 4, values, col_indices, row_ptr};
    double d[] = {1.0, -1.0};
    double lb[] = {-10.0, -10.0};
    double ub[] = {10.0, 10.0};
    double dense_x[] = {0.0, 0.0};
    out_x[0] = 0.0;
    out_x[1] = 0.0;
    config.strategy = NANOQSP_STRATEGY_COORDINATE_DESCENT;
    config.max_iterations = 10;
    config.tolerance = 1e-3;
    int dense_ret = nanoqsp_solve_box(2, values, d, lb, ub, dense_x, &config);
    int sparse_ret =
        nanoqsp_solve_box_sparse(2, &sparse_D, d, lb, ub, out_x, &config);
    if (dense_ret != NANOQSP_ERR_MAX_ITERATIONS ||
        sparse_ret != NANOQSP_ERR_MAX_ITERATIONS ||
        fabs(dense_x[0] - out_x[0]) > 1e-12 ||
        fabs(dense_x[1] - out_x[1]) > 1e-12)
      return -10;
    return 0;
  } else if (test_id == 26) {
    if (nanoqsp_workspace_size_box(10, NANOQSP_STRATEGY_COORDINATE_DESCENT) !=
            10 ||
        nanoqsp_workspace_size_box(10, NANOQSP_STRATEGY_PROJECTED_GRADIENT) !=
            10 ||
        nanoqsp_workspace_size_box(10, NANOQSP_STRATEGY_ACCELERATED_GRADIENT) !=
            20 ||
        nanoqsp_workspace_size_box(10, NANOQSP_STRATEGY_SPECTRAL_GRADIENT) !=
            30 ||
        nanoqsp_workspace_size_box(10, NANOQSP_STRATEGY_ADMM) != 70 ||
        nanoqsp_workspace_size_box_sparse(
            10, NANOQSP_STRATEGY_COORDINATE_DESCENT) != 20 ||
        nanoqsp_workspace_size_box_sparse(
            10, NANOQSP_STRATEGY_PROJECTED_GRADIENT) != 10 ||
        nanoqsp_workspace_size_box_sparse(
            10, NANOQSP_STRATEGY_ACCELERATED_GRADIENT) != 20 ||
        nanoqsp_workspace_size_box_sparse(10, NANOQSP_STRATEGY_ADMM) != 70 ||
        nanoqsp_workspace_size_least_squares(
            10, NANOQSP_STRATEGY_COORDINATE_DESCENT) != 120 ||
        nanoqsp_workspace_size_least_squares(
            10, NANOQSP_STRATEGY_SPECTRAL_GRADIENT) != 140 ||
        nanoqsp_workspace_size_affine(10, 3) != 79 ||
        nanoqsp_workspace_size_box(10, (NanoqspStrategy)99) !=
            NANOQSP_ERR_INVALID_STRATEGY ||
        nanoqsp_workspace_size_least_squares(10, (NanoqspStrategy)-1) !=
            NANOQSP_ERR_INVALID_STRATEGY ||
        nanoqsp_workspace_size_box(0, NANOQSP_STRATEGY_ADMM) !=
            NANOQSP_ERR_INVALID_ARG)
      return -10;
    return 0;
  } else if (test_id == 27) {
    /* duplicate CSR diagonal entries must be summed in sparse CD */
    double values[] = {2.0, 2.0};
    int col_indices[] = {0, 0};
    int row_ptr[] = {0, 2};
    NanoqspCSR D = {1, 2, values, col_indices, row_ptr};
    double d[] = {2.0};
    out_x[0] = 0.0;
    config.strategy = NANOQSP_STRATEGY_COORDINATE_DESCENT;
    return nanoqsp_solve_box_sparse(1, &D, d, NULL, NULL, out_x, &config);
  } else if (test_id == 28) {
    /* NaN in matrix or config tolerance must not cause false convergence */
    double D[] = {NAN};
    double d[] = {1.0};
    out_x[0] = 0.0;
    int r1 = nanoqsp_solve_box(1, D, d, NULL, NULL, out_x, &config);
    double D_ok[] = {1.0};
    config.tolerance = NAN;
    int r2 = nanoqsp_solve_box(1, D_ok, d, NULL, NULL, out_x, &config);
    if (r1 != NANOQSP_ERR_MAX_ITERATIONS || r2 != NANOQSP_ERR_INVALID_ARG)
      return -10;
    return 0;
  } else if (test_id == 29 || test_id == 30) {
    /* small steps can fail KKT before FISTA converges */
    double D[] = {1e8, 0.0, 0.0, 1e12};
    double values[] = {1e8, 1e12};
    int col_indices[] = {0, 1};
    int row_ptr[] = {0, 1, 2};
    NanoqspCSR sparse_D = {2, 2, values, col_indices, row_ptr};
    double d[] = {1e6, 1e6};
    out_x[0] = out_x[1] = 0.0;
    config.strategy = NANOQSP_STRATEGY_ACCELERATED_GRADIENT;
    config.max_iterations = 1000;
    config.tolerance = 1e-6;
    int ret = test_id == 29
                  ? nanoqsp_solve_box(2, D, d, NULL, NULL, out_x, &config)
                  : nanoqsp_solve_box_sparse(2, &sparse_D, d, NULL, NULL, out_x,
                                             &config);
    if (ret < 0 || !(fabs(out_x[0] - 0.01) < 1.1e-8) ||
        !(fabs(out_x[1] - 1e-6) < 1e-12))
      return -10;
    return 0;
  } else if (test_id == 31) {
    return test_simd_backends();
  } else if (test_id == 32) {
    /* a one-entry CSR row must retain its norm on vector backends */
    double values[] = {100.0};
    int col_indices[] = {0};
    int row_ptr[] = {0, 1};
    NanoqspCSR D = {1, 1, values, col_indices, row_ptr};
    double d[] = {100.0};
    config.max_iterations = 10;
    for (int strategy = NANOQSP_STRATEGY_PROJECTED_GRADIENT;
         strategy <= NANOQSP_STRATEGY_ACCELERATED_GRADIENT; strategy++) {
      config.strategy = (NanoqspStrategy)strategy;
      out_x[0] = 0.0;
      int ret = nanoqsp_solve_box_sparse(1, &D, d, NULL, NULL, out_x, &config);
      if (ret < 0 || !(fabs(out_x[0] - 1.0) < 1e-6))
        return -10;
    }
    return 0;
  }
  return -99;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <test_id> [strategy_id]\n", argv[0]);
    return 1;
  }
  int test_id = atoi(argv[1]);

  if (test_id >= 5 && test_id <= 9) {
    int ret = test_cross_verify(test_id);
    printf("RET:%d\n", ret);
    return 0;
  }

  if (argc < 3) {
    fprintf(stderr, "Usage: %s <test_id> <strategy_id>\n", argv[0]);
    return 1;
  }
  int strategy_id = atoi(argv[2]);
  double x[2] = {0.0, 0.0};
  int ret = run_test(test_id, strategy_id, x);
  printf("RET:%d X0:%.6f X1:%.6f\n", ret, x[0], x[1]);
  return 0;
}
