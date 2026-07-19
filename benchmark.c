#define _POSIX_C_SOURCE 199309L
#include "nanoqsp.h"
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
  int runs = 50;
  NanoqspStrategy strategy = NANOQSP_STRATEGY_SPECTRAL_GRADIENT;

  if (argc > 1) {
    int s = atoi(argv[1]);
    if (s >= 0 && s <= 4) {
      strategy = (NanoqspStrategy)s;
    }
  }

  srand(42);
  double *D = malloc(n * n * sizeof(double));
  double *d = malloc(n * sizeof(double));
  double *x = malloc(n * sizeof(double));
  double *lb = malloc(n * sizeof(double));
  double *ub = malloc(n * sizeof(double));

  if (!D || !d || !x || !lb || !ub) {
    fprintf(stderr, "Out of memory\n");
    return 1;
  }

  /* Generate a positive-definite matrix D = A^T A + I */
  double *A = malloc(n * n * sizeof(double));
  if (!A)
    return 1;
  for (int i = 0; i < n * n; i++) {
    A[i] = (double)rand() / RAND_MAX - 0.5;
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++) {
        sum += A[k * n + i] * A[k * n + j];
      }
      D[i * n + j] = sum;
    }
    D[i * n + i] += 1.0; /* Ensure strict positive definiteness */
    d[i] = (double)rand() / RAND_MAX - 0.5;
    lb[i] = -0.2;
    ub[i] = 0.2;
  }
  free(A);

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
  printf("Running benchmark [%s]: N=%d, Runs=%d\n", names[strategy], n, runs);

  double start = get_time_sec();
  int total_iters = 0;
  for (int r = 0; r < runs; r++) {
    for (int i = 0; i < n; i++)
      x[i] = 0.0;
    int iters = nanoqsp_solve_box(n, D, d, lb, ub, x, &config);
    if (iters < 0) {
      fprintf(stderr, "Solver failed: %d\n", iters);
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
  double end = get_time_sec();

  double elapsed = end - start;
  printf("Time: %.4f seconds (Total iterations: %d)\n", elapsed, total_iters);

  free(D);
  free(d);
  free(x);
  free(lb);
  free(ub);
  free(ws);
  return 0;
}
