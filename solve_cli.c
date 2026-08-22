#include "nanoqsp.h"
#include <stdio.h>
#include <stdlib.h>

#define CLI_MAX_ELEMENTS 268435455LL

int main(int argc, char **argv) {
  double *A = NULL;
  double *b = NULL;
  double *p = NULL;
  double *x = NULL;
  double *lb_arr = NULL;
  double *ub_arr = NULL;
  int exit_code = 1;
  int strategy = 0;
  if (argc > 1) {
    strategy = atoi(argv[1]);
  }

  int m, n;
  if (scanf("%d %d", &m, &n) != 2)
    goto cleanup;
  if (m <= 0 || n <= 0 || (long long)m * n > CLI_MAX_ELEMENTS)
    goto cleanup;

  double lb, ub;
  if (scanf("%lf %lf", &lb, &ub) != 2)
    goto cleanup;
  if (lb > ub)
    goto cleanup;

  A = malloc((size_t)m * (size_t)n * sizeof(double));
  b = malloc((size_t)m * sizeof(double));
  if (!A || !b)
    goto cleanup;

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (scanf("%lf", &A[i * n + j]) != 1)
        goto cleanup;
    }
    if (scanf("%lf", &b[i]) != 1)
      goto cleanup;
  }

  p = malloc((size_t)n * sizeof(double));
  if (!p)
    goto cleanup;
  for (int j = 0; j < n; j++) {
    if (scanf("%lf", &p[j]) != 1)
      goto cleanup;
  }

  x = calloc((size_t)n, sizeof(double));
  if (!x)
    goto cleanup;

  NanoqspConfig config = {.strategy = (NanoqspStrategy)strategy,
                          .max_iterations = 50000,
                          .tolerance = 1e-6,
                          .workspace = NULL,
                          .workspace_size = 0};

  lb_arr = malloc((size_t)n * sizeof(double));
  ub_arr = malloc((size_t)n * sizeof(double));
  if (!lb_arr || !ub_arr)
    goto cleanup;
  for (int j = 0; j < n; j++) {
    lb_arr[j] = lb;
    ub_arr[j] = ub;
  }

  int iters =
      nanoqsp_solve_least_squares(m, n, A, b, lb_arr, ub_arr, x, &config);

  printf("Iterations: %d\n", iters);
  for (int j = 0; j < n; j++) {
    printf("x[%d] = %.4f\n", j, x[j]);
  }

  double prediction = nanoqsp_predict(n, x, p);
  printf("Prediction = %.4f\n", prediction);

  exit_code = (iters >= 0) ? 0 : 2;

cleanup:
  free(A);
  free(b);
  free(p);
  free(x);
  free(lb_arr);
  free(ub_arr);
  return exit_code;
}
