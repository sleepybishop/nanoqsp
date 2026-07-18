#include "nanoqsp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int run_test(int test_id, int strategy_id, double *out_x) {
  NanoqspConfig config = {.strategy = (NanoqspStrategy)strategy_id,
                          .max_iterations = 2000,
                          .tolerance = 1e-7};

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
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <test_id> <strategy_id>\n", argv[0]);
    return 1;
  }
  int test_id = atoi(argv[1]);
  int strategy_id = atoi(argv[2]);
  double x[2] = {0.0, 0.0};
  int ret = run_test(test_id, strategy_id, x);
  printf("RET:%d X0:%.6f X1:%.6f\n", ret, x[0], x[1]);
  return 0;
}
