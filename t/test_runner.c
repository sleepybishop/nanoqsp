#include "nanoqsp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long int next = 1;
static int my_rand(void) {
  next = next * 1103515245 + 12345;
  return (unsigned int)(next / 65536) % 32768;
}

static double my_rand_double(double min, double max) {
  double r = (double)my_rand() / 32768.0;
  return min + r * (max - min);
}

static int verify_kkt_dense(int n, const double *D, const double *d,
                            const double *lb, const double *ub, const double *x,
                            double tol) {
  for (int i = 0; i < n; i++) {
    double grad_i = -d[i];
    for (int j = 0; j < n; j++) {
      grad_i += D[i * n + j] * x[j];
    }
    double lower = lb ? lb[i] : -INFINITY;
    double upper = ub ? ub[i] : INFINITY;
    double viol = 0.0;
    if (x[i] <= lower + 1e-9) {
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
              "KKT Violation at index %d: x=%f, grad=%f, viol=%e (tol=%e)\n", i,
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
    for (int idx = start; idx < end; idx++) {
      grad_i += D->values[idx] * x[D->col_indices[idx]];
    }
    double lower = lb ? lb[i] : -INFINITY;
    double upper = ub ? ub[i] : INFINITY;
    double viol = 0.0;
    if (x[i] <= lower + 1e-9) {
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
              "KKT Violation at index %d: x=%f, grad=%f, viol=%e (tol=%e)\n", i,
              x[i], grad_i, viol, tol);
      return 0;
    }
  }
  return 1;
}

static void generate_random_qp(int n, double *D, double *d, double *lb,
                               double *ub, int type) {
  next = 42; /* Fixed seed for cross-platform determinism */

  /* Generate random matrix M */
  double *M = malloc(n * n * sizeof(double));
  for (int i = 0; i < n * n; i++) {
    M[i] = my_rand_double(-1.0, 1.0);
  }

  /* D = M^T * M */
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++) {
        sum += M[k * n + i] * M[k * n + j];
      }
      D[i * n + j] = sum;
    }
  }
  free(M);

  /* Diagonal shift to make it strictly positive-definite */
  double shift = 0.5;
  if (type == 3) {
    shift = 1e-4; /* Weak diagonal shift to allow ill-conditioning */
  }
  for (int i = 0; i < n; i++) {
    D[i * n + i] += shift;
    if (type == 3) {
      /* scale diagonals by power of 10 to create ill-conditioning */
      double scale = pow(10.0, my_rand_double(-3.0, 3.0));
      D[i * n + i] *= scale;
    }
  }

  /* Generate linear cost vector d */
  for (int i = 0; i < n; i++) {
    d[i] = my_rand_double(-2.0, 2.0);
  }

  /* Generate bounds */
  for (int i = 0; i < n; i++) {
    if (type == 1) {
      /* Unconstrained */
      lb[i] = -INFINITY;
      ub[i] = INFINITY;
    } else if (type == 2) {
      /* Semi-constrained */
      lb[i] = 0.0;
      ub[i] = INFINITY;
    } else {
      /* Standard bounded QP */
      lb[i] = my_rand_double(-1.0, 0.0);
      ub[i] = my_rand_double(0.1, 2.0);
    }
  }
}

int test_cross_verify(int test_id) {
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

  double *D = malloc(n * n * sizeof(double));
  double *d = malloc(n * sizeof(double));
  double *lb = malloc(n * sizeof(double));
  double *ub = malloc(n * sizeof(double));
  generate_random_qp(n, D, d, lb, ub, type);

  /* Allocate memory for solutions of all 5 strategies */
  double **solutions = malloc(5 * sizeof(double *));
  int *status = malloc(5 * sizeof(int));
  for (int s = 0; s < 5; s++) {
    solutions[s] = malloc(n * sizeof(double));
    memset(solutions[s], 0, n * sizeof(double));

    NanoqspConfig config = {
        .strategy = (NanoqspStrategy)s,
        .max_iterations = (type == 3) ? 100000 : 5000,
        .tolerance = 1e-6,
    };

    /* Allocate injected workspace */
    int ws_size = 6 * n;
    double *ws = malloc(ws_size * sizeof(double));
    config.workspace = ws;
    config.workspace_size = ws_size;

    status[s] = nanoqsp_solve_box(n, D, d, lb, ub, solutions[s], &config);
    free(ws);
  }

  /* Verify status and KKT of all solvers */
  int match = 1;
  for (int s = 0; s < 5; s++) {
    if (status[s] < 0) {
      fprintf(stderr, "Strategy %d failed with status %d\n", s, status[s]);
      match = 0;
      break;
    }
    /* Verify KKT optimality directly */
    double kkt_tol = (type == 3) ? 1e-1 : 1e-2;
    if (!verify_kkt_dense(n, D, d, lb, ub, solutions[s], kkt_tol)) {
      fprintf(stderr, "Strategy %d failed KKT verification\n", s);
      match = 0;
      break;
    }
  }

  /* Cross-compare all solutions against SPG (Strategy 3) */
  if (match) {
    int ref_strat = 3;
    for (int s = 0; s < 5; s++) {
      if (s == ref_strat)
        continue;
      double max_diff = 0.0;
      for (int i = 0; i < n; i++) {
        double diff = fabs(solutions[s][i] - solutions[ref_strat][i]);
        if (diff > max_diff) {
          max_diff = diff;
        }
      }

      /* Coordinate Descent (0) or ill-conditioned problems might diverge
       * slightly */
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

  /* Clean up */
  for (int s = 0; s < 5; s++) {
    free(solutions[s]);
  }
  free(solutions);
  free(status);
  free(D);
  free(d);
  free(lb);
  free(ub);

  return match ? 0 : -2;
}

int run_test(int test_id, int strategy_id, double *out_x) {
  NanoqspConfig config = {
      .strategy = (NanoqspStrategy)strategy_id,
      .max_iterations = 2000,
      .tolerance = 1e-7,
  };

  if (test_id == 1) {
    /* Test 1: Unconstrained min within bounds */
    int n = 2;
    double D[] = {1.0, 0.0, 0.0, 1.0};
    double d[] = {0.5, 0.5};
    double lb[] = {0.0, 0.0};
    double ub[] = {1.0, 1.0};
    out_x[0] = 0.0;
    out_x[1] = 0.0;
    int ret = nanoqsp_solve_box(n, D, d, lb, ub, out_x, &config);
    if (ret >= 0 && !verify_kkt_dense(n, D, d, lb, ub, out_x, 1e-5)) {
      return -10;
    }
    return ret;
  } else if (test_id == 2) {
    /* Test 2: Constrained min at bounds */
    int n = 2;
    double D[] = {1.0, 0.0, 0.0, 1.0};
    double d[] = {2.0, -1.0};
    double lb[] = {0.0, 0.0};
    double ub[] = {1.0, 1.0};
    out_x[0] = 0.5;
    out_x[1] = 0.5;
    int ret = nanoqsp_solve_box(n, D, d, lb, ub, out_x, &config);
    if (ret >= 0 && !verify_kkt_dense(n, D, d, lb, ub, out_x, 1e-5)) {
      return -10;
    }
    return ret;
  } else if (test_id == 3) {
    /* Test 3: Invalid arguments check */
    return nanoqsp_solve_box(0, NULL, NULL, NULL, NULL, out_x, &config);
  } else if (test_id == 4) {
    /* Test 4: Singular matrix regularization */
    int n = 2;
    double D[] = {0.0, 0.0, 0.0, 0.0};
    double d[] = {1.0, 1.0};
    double lb[] = {0.0, 0.0};
    double ub[] = {10.0, 10.0};
    out_x[0] = 0.0;
    out_x[1] = 0.0;
    return nanoqsp_solve_box(n, D, d, lb, ub, out_x, &config);
  } else if (test_id == 11) {
    /* Test 11: Sparse unconstrained min within bounds */
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
    if (ret >= 0 && !verify_kkt_sparse(n, &D, d, lb, ub, out_x, 1e-5)) {
      return -10;
    }
    return ret;
  } else if (test_id == 12) {
    /* Test 12: Sparse constrained min at bounds */
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
    if (ret >= 0 && !verify_kkt_sparse(n, &D, d, lb, ub, out_x, 1e-5)) {
      return -10;
    }
    return ret;
  } else if (test_id == 13) {
    /* Test 13: Sparse invalid arguments check */
    return nanoqsp_solve_box_sparse(0, NULL, NULL, NULL, NULL, out_x, &config);
  } else if (test_id == 14) {
    /* Test 14: Sparse singular matrix */
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
    /* Test 15: Dense affine QP with general linear constraints */
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
    NanoqspConfig admm_config = config;
    admm_config.strategy = NANOQSP_STRATEGY_ADMM;
    return nanoqsp_solve_affine(n, m, D, d, A, lb_x, ub_x, lb_A, ub_A, out_x,
                                &admm_config);
  } else if (test_id == 16) {
    /* Test 16: Sparse affine QP with general linear constraints */
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
    NanoqspConfig admm_config = config;
    admm_config.strategy = NANOQSP_STRATEGY_ADMM;
    return nanoqsp_solve_affine_sparse(n, m, &D, d, &A, lb_x, ub_x, lb_A, ub_A,
                                       out_x, &admm_config);
  } else if (test_id == 17) {
    /* Infeasible dense SPG warm starts must be projected before KKT checks. */
    double D[] = {1.0};
    double d[] = {-2.0};
    double lb[] = {0.0};
    double ub[] = {1.0};
    out_x[0] = -1.0;
    config.strategy = NANOQSP_STRATEGY_SPECTRAL_GRADIENT;
    return nanoqsp_solve_box(1, D, d, lb, ub, out_x, &config);
  } else if (test_id == 18) {
    /* Infeasible sparse SPG warm starts must be projected before KKT checks. */
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
    /* Iteration exhaustion must not be reported as success. */
    double D[] = {1.0, 0.0, 0.0, 100.0};
    double d[] = {1.0, 100.0};
    out_x[0] = 0.0;
    out_x[1] = 0.0;
    config.strategy = NANOQSP_STRATEGY_PROJECTED_GRADIENT;
    config.max_iterations = 1;
    config.tolerance = 1e-12;
    return nanoqsp_solve_box(2, D, d, NULL, NULL, out_x, &config);
  } else if (test_id == 20) {
    /* Malformed CSR input must return INVALID_ARG instead of crashing. */
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
    /* Dense affine output must be the projected z variable. */
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
    /* Sparse coordinate descent must not overwrite its diagonal during KKT. */
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
    if (nanoqsp_workspace_size_box(10, NANOQSP_STRATEGY_ADMM) != 60 ||
        nanoqsp_workspace_size_box_sparse(
            10, NANOQSP_STRATEGY_COORDINATE_DESCENT) != 20 ||
        nanoqsp_workspace_size_least_squares(
            10, NANOQSP_STRATEGY_SPECTRAL_GRADIENT) != 140 ||
        nanoqsp_workspace_size_affine(10, 3) != 79)
      return -10;
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
