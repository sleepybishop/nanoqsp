#define _POSIX_C_SOURCE 199309L
#include "nanoqsp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double get_time_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
  int n = 512;
  int runs = 20;
  double density = 0.05; /* 5% non-zero density */
  NanoqspStrategy strategy = NANOQSP_STRATEGY_SPECTRAL_GRADIENT;

  if (argc > 1) {
    int s = atoi(argv[1]);
    if (s >= 0 && s <= 4) {
      strategy = (NanoqspStrategy)s;
    }
  }

  srand(42);

  /* Allocate dense matrix D_dense and zero it */
  double *D_dense = calloc(n * n, sizeof(double));
  double *d = malloc(n * sizeof(double));
  double *x_dense = malloc(n * sizeof(double));
  double *x_sparse = malloc(n * sizeof(double));
  double *lb = malloc(n * sizeof(double));
  double *ub = malloc(n * sizeof(double));

  if (!D_dense || !d || !x_dense || !x_sparse || !lb || !ub) {
    fprintf(stderr, "Out of memory\n");
    return 1;
  }

  /* Generate a diagonally dominant symmetric sparse matrix */
  double diag_value = n * density + 5.0;
  for (int i = 0; i < n; i++) {
    D_dense[i * n + i] = diag_value;
    d[i] = (double)rand() / RAND_MAX - 0.5;
    lb[i] = -0.2;
    ub[i] = 0.2;
  }

  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      if ((double)rand() / RAND_MAX < density) {
        double val = ((double)rand() / RAND_MAX - 0.5) * 2.0;
        D_dense[i * n + j] = val;
        D_dense[j * n + i] = val;
      }
    }
  }

  /* Convert to CSR format */
  int nnz = 0;
  for (int i = 0; i < n * n; i++) {
    if (D_dense[i] != 0.0)
      nnz++;
  }

  double *val = malloc(nnz * sizeof(double));
  int *col_ind = malloc(nnz * sizeof(int));
  int *row_ptr = malloc((n + 1) * sizeof(int));

  if (!val || !col_ind || !row_ptr) {
    fprintf(stderr, "Out of memory for CSR conversion\n");
    return 1;
  }

  int idx = 0;
  for (int i = 0; i < n; i++) {
    row_ptr[i] = idx;
    for (int j = 0; j < n; j++) {
      if (D_dense[i * n + j] != 0.0) {
        val[idx] = D_dense[i * n + j];
        col_ind[idx] = j;
        idx++;
      }
    }
  }
  row_ptr[n] = idx;

  NanoqspCSR D_sparse = {n, nnz, val, col_ind, row_ptr};

  /* Workspace allocation */
  int required_ws = 6 * n;
  double *ws = malloc(required_ws * sizeof(double));
  if (!ws) {
    fprintf(stderr, "Out of memory for workspace\n");
    return 1;
  }

  NanoqspConfig config = {.strategy = strategy,
                          .max_iterations = 2000,
                          .tolerance = 1e-9,
                          .workspace = ws,
                          .workspace_size = required_ws};

  const char *names[] = {"Coordinate Descent", "Projected Gradient",
                         "Accelerated Gradient", "Spectral Projected Gradient",
                         "ADMM"};
  printf("Sparse vs Dense Benchmark [%s]: N=%d, density=%.1f%%, runs=%d\n\n",
         names[strategy], n, density * 100.0, runs);

  /* 1. DENSE SOLVER */
  double start_dense = get_time_sec();
  int dense_iters = 0;
  for (int r = 0; r < runs; r++) {
    for (int i = 0; i < n; i++)
      x_dense[i] = 0.0;
    dense_iters += nanoqsp_solve_box(n, D_dense, d, lb, ub, x_dense, &config);
  }
  double end_dense = get_time_sec();
  double time_dense = end_dense - start_dense;

  /* 2. SPARSE SOLVER */
  double start_sparse = get_time_sec();
  int sparse_iters = 0;
  for (int r = 0; r < runs; r++) {
    for (int i = 0; i < n; i++)
      x_sparse[i] = 0.0;
    sparse_iters +=
        nanoqsp_solve_box_sparse(n, &D_sparse, d, lb, ub, x_sparse, &config);
  }
  double end_sparse = get_time_sec();
  double time_sparse = end_sparse - start_sparse;

  /* Check solutions are identical */
  double diff = 0.0;
  for (int i = 0; i < n; i++) {
    diff += fabs(x_dense[i] - x_sparse[i]);
  }

  printf("Dense Solver  : %.4f seconds (Total iterations: %d)\n", time_dense,
         dense_iters);
  printf("Sparse Solver : %.4f seconds (Total iterations: %d)\n", time_sparse,
         sparse_iters);
  printf("Speedup       : %.2fx\n", time_dense / time_sparse);
  printf("Solution diff : %.6e\n", diff);

  free(D_dense);
  free(d);
  free(x_dense);
  free(x_sparse);
  free(lb);
  free(ub);
  free(val);
  free(col_ind);
  free(row_ptr);
  free(ws);
  return 0;
}
