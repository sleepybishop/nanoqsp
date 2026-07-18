#include "nanoqsp.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  int strategy = 0;
  if (argc > 1) {
    strategy = atoi(argv[1]);
  }

  int m, n;
  if (scanf("%d %d", &m, &n) != 2)
    return 1;

  double lb, ub;
  if (scanf("%lf %lf", &lb, &ub) != 2)
    return 1;

  double *A = malloc(m * n * sizeof(double));
  double *b = malloc(m * sizeof(double));
  if (!A || !b)
    return 1;

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (scanf("%lf", &A[i * n + j]) != 1)
        return 1;
    }
    if (scanf("%lf", &b[i]) != 1)
      return 1;
  }

  double *p = malloc(n * sizeof(double));
  if (!p)
    return 1;
  for (int j = 0; j < n; j++) {
    if (scanf("%lf", &p[j]) != 1)
      return 1;
  }

  double *x = calloc(n, sizeof(double));
  if (!x)
    return 1;

  NanoqspConfig config = {.strategy = (NanoqspStrategy)strategy,
                          .max_iterations = 50000,
                          .tolerance = 1e-6,
                          .workspace = NULL,
                          .workspace_size = 0};

  double *lb_arr = malloc(n * sizeof(double));
  double *ub_arr = malloc(n * sizeof(double));
  if (!lb_arr || !ub_arr)
    return 1;
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

  free(A);
  free(b);
  free(p);
  free(x);
  free(lb_arr);
  free(ub_arr);
  return (iters >= 0) ? 0 : 2;
}
