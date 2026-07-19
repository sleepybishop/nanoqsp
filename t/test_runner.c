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
        .max_iterations = 5000,
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

  /* Verify status of all solvers */
  int match = 1;
  for (int s = 0; s < 5; s++) {
    if (status[s] < 0) {
      fprintf(stderr, "Strategy %d failed with status %d\n", s, status[s]);
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
    return nanoqsp_solve_box(n, D, d, lb, ub, out_x, &config);
  } else if (test_id == 2) {
    /* Test 2: Constrained min at bounds */
    int n = 2;
    double D[] = {1.0, 0.0, 0.0, 1.0};
    double d[] = {2.0, -1.0};
    double lb[] = {0.0, 0.0};
    double ub[] = {1.0, 1.0};
    out_x[0] = 0.5;
    out_x[1] = 0.5;
    return nanoqsp_solve_box(n, D, d, lb, ub, out_x, &config);
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
