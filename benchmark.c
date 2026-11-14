#define _POSIX_C_SOURCE 199309L
#include "nanoqsp.h"
#include "qsp_blas.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double get_time_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int run_dense_bench(int n, int runs, NanoqspStrategy strategy) {
  static const char *names[] = {"Coordinate Descent", "Projected Gradient",
                                "Accelerated Gradient",
                                "Spectral Projected Gradient", "ADMM"};

  srand(42);
  double *D = malloc((size_t)n * (size_t)n * sizeof(double));
  double *A = malloc((size_t)n * (size_t)n * sizeof(double));
  double *d = malloc((size_t)n * sizeof(double));
  double *x = malloc((size_t)n * sizeof(double));
  double *lb = malloc((size_t)n * sizeof(double));
  double *ub = malloc((size_t)n * sizeof(double));
  int ws_size = nanoqsp_workspace_size_box(n, strategy);
  double *ws = ws_size > 0 ? malloc((size_t)ws_size * sizeof(double)) : NULL;

  if (!D || !A || !d || !x || !lb || !ub || (ws_size > 0 && !ws)) {
    fprintf(stderr, "Out of memory\n");
    free(D);
    free(A);
    free(d);
    free(x);
    free(lb);
    free(ub);
    free(ws);
    return 1;
  }

  /* generate SPD matrix D = A^T A + I */
  for (int i = 0; i < n * n; i++)
    A[i] = (double)rand() / RAND_MAX - 0.5;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++)
        sum += A[k * n + i] * A[k * n + j];
      D[i * n + j] = sum;
    }
    D[i * n + i] += 1.0;
    d[i] = (double)rand() / RAND_MAX - 0.5;
    lb[i] = -0.2;
    ub[i] = 0.2;
  }
  free(A);

  NanoqspConfig config = {.strategy = strategy,
                          .max_iterations = 2000,
                          .tolerance = 1e-9,
                          .workspace = ws,
                          .workspace_size = ws_size};

  printf("Dense Benchmark [%s]: N=%d, Runs=%d\n", names[strategy], n, runs);
  for (int k = 0; k < QSP_SIMD_COUNT; k++) {
    if (!nanoqsp_blas_select((qsp_simd_kind)k))
      continue;
    double start = get_time_sec();
    int total_iters = 0;
    for (int r = 0; r < runs; r++) {
      for (int i = 0; i < n; i++)
        x[i] = 0.0;
      int iters = nanoqsp_solve_box(n, D, d, lb, ub, x, &config);
      if (iters < 0) {
        fprintf(stderr, "Solver failed on %s: %d\n", nanoqsp_blas.name, iters);
        free(D);
        free(d);
        free(x);
        free(lb);
        free(ub);
        free(ws);
        return 1;
      }
      total_iters += iters;
    }
    double elapsed = get_time_sec() - start;
    printf("  %-10s : %.4f s (total iterations: %d)\n", nanoqsp_blas.name,
           elapsed, total_iters);
  }
  printf("\n");

  free(D);
  free(d);
  free(x);
  free(lb);
  free(ub);
  free(ws);
  return 0;
}

static int run_sparse_bench(int n, int runs, double density,
                            NanoqspStrategy strategy) {
  static const char *names[] = {"Coordinate Descent", "Projected Gradient",
                                "Accelerated Gradient",
                                "Spectral Projected Gradient", "ADMM"};

  srand(42);
  double *D_dense = calloc((size_t)n * (size_t)n, sizeof(double));
  double *d = malloc((size_t)n * sizeof(double));
  double *x_dense = malloc((size_t)n * sizeof(double));
  double *x_sparse = malloc((size_t)n * sizeof(double));
  double *lb = malloc((size_t)n * sizeof(double));
  double *ub = malloc((size_t)n * sizeof(double));
  if (!D_dense || !d || !x_dense || !x_sparse || !lb || !ub) {
    free(D_dense);
    free(d);
    free(x_dense);
    free(x_sparse);
    free(lb);
    free(ub);
    return 1;
  }

  /* diagonally dominant symmetric sparse matrix */
  double diag_value = (double)n * density + 5.0;
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

  int nnz = 0;
  for (int i = 0; i < n * n; i++) {
    if (D_dense[i] != 0.0)
      nnz++;
  }

  double *val = malloc((size_t)nnz * sizeof(double));
  int *col_ind = malloc((size_t)nnz * sizeof(int));
  int *row_ptr = malloc((size_t)(n + 1) * sizeof(int));
  int ws_dense = nanoqsp_workspace_size_box(n, strategy);
  int ws_sparse = nanoqsp_workspace_size_box_sparse(n, strategy);
  int ws_size = ws_dense > ws_sparse ? ws_dense : ws_sparse;
  double *ws = malloc((size_t)ws_size * sizeof(double));
  if (!val || !col_ind || !row_ptr || !ws) {
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
  NanoqspConfig config = {.strategy = strategy,
                          .max_iterations = 2000,
                          .tolerance = 1e-9,
                          .workspace = ws,
                          .workspace_size = ws_size};

  /* reset to best host SIMD backend */
  for (int k = QSP_SIMD_COUNT - 1; k >= 0; k--) {
    if (nanoqsp_blas_select((qsp_simd_kind)k))
      break;
  }

  double t0 = get_time_sec();
  int dense_iters = 0;
  for (int r = 0; r < runs; r++) {
    for (int i = 0; i < n; i++)
      x_dense[i] = 0.0;
    dense_iters += nanoqsp_solve_box(n, D_dense, d, lb, ub, x_dense, &config);
  }
  double time_dense = get_time_sec() - t0;

  t0 = get_time_sec();
  int sparse_iters = 0;
  for (int r = 0; r < runs; r++) {
    for (int i = 0; i < n; i++)
      x_sparse[i] = 0.0;
    sparse_iters +=
        nanoqsp_solve_box_sparse(n, &D_sparse, d, lb, ub, x_sparse, &config);
  }
  double time_sparse = get_time_sec() - t0;

  double diff = 0.0;
  for (int i = 0; i < n; i++)
    diff += fabs(x_dense[i] - x_sparse[i]);

  printf("Sparse vs Dense [%s, %s]: N=%d, density=%.1f%%, runs=%d\n",
         names[strategy], nanoqsp_blas.name, n, density * 100.0, runs);
  printf("  Dense  : %.4f s (total iterations: %d)\n", time_dense, dense_iters);
  printf("  Sparse : %.4f s (total iterations: %d)\n", time_sparse,
         sparse_iters);
  printf("  Speedup: %.2fx (L1 diff: %.3e)\n\n", time_dense / time_sparse,
         diff);

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

int main(int argc, char **argv) {
  int n = 512;
  int runs = 50;

  if (argc > 1) {
    int s = atoi(argv[1]);
    if (s >= 0 && s <= 4) {
      if (run_dense_bench(n, runs, (NanoqspStrategy)s))
        return 1;
      return run_sparse_bench(n, 20, 0.05, (NanoqspStrategy)s);
    }
  }

  NanoqspStrategy suite[] = {NANOQSP_STRATEGY_COORDINATE_DESCENT,
                             NANOQSP_STRATEGY_ACCELERATED_GRADIENT,
                             NANOQSP_STRATEGY_SPECTRAL_GRADIENT,
                             NANOQSP_STRATEGY_ADMM};
  for (size_t i = 0; i < sizeof(suite) / sizeof(suite[0]); i++) {
    if (run_dense_bench(n, runs, suite[i]))
      return 1;
  }
  return run_sparse_bench(n, 20, 0.05, NANOQSP_STRATEGY_SPECTRAL_GRADIENT);
}
