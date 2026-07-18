#include "nanoqsp.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  printf("NanoQSP - Complex Least Squares Example\n");
  printf("---------------------------------------\n");
  printf(
      "Scenario: Estimating individual item prices from grocery receipts.\n");
  printf("Variables (Unit Prices):\n");
  printf("  x0: Apple\n");
  printf("  x1: Banana\n");
  printf("  x2: Cherry\n");
  printf("  x3: Date\n\n");

  int m = 6; /* 6 receipts */
  int n = 4; /* 4 items */

  double A[] = {5.0, 10.0, 3.0, 0.0, 5.0, 10.0, 0.0, 6.0, 0.0, 10.0, 0.0, 0.0,
                0.0, 0.0,  3.0, 6.0, 5.0, 0.0,  0.0, 6.0, 5.0, 10.0, 3.0, 6.0};

  double b[] = {10.50, 15.00, 5.20, 5.40, 9.50, 15.50};

  double lb[] = {0.05, 0.05, 0.05, 0.05};
  double ub[] = {5.00, 5.00, 5.00, 5.00};

  NanoqspStrategy strategies[] = {
      NANOQSP_STRATEGY_COORDINATE_DESCENT, NANOQSP_STRATEGY_PROJECTED_GRADIENT,
      NANOQSP_STRATEGY_ACCELERATED_GRADIENT, NANOQSP_STRATEGY_SPECTRAL_GRADIENT,
      NANOQSP_STRATEGY_ADMM};
  const char *names[] = {"Coordinate Descent", "Projected Gradient",
                         "Accelerated Gradient (FISTA w/ Restarts)",
                         "Spectral Projected Gradient (SPG)", "Nano ADMM"};

  for (int s = 0; s < 5; s++) {
    double x[4] = {1.0, 1.0, 1.0, 1.0};
    NanoqspConfig config = {
        .strategy = strategies[s], .max_iterations = 50000, .tolerance = 1e-7};

    int iters = nanoqsp_solve_least_squares(m, n, A, b, lb, ub, x, &config);

    printf("Strategy: %s\n", names[s]);
    if (iters >= 0) {
      printf("  Converged in %d iterations.\n", iters);
      printf("  Estimated Item Prices:\n");
      printf("    Apple:  $%.2f\n", x[0]);
      printf("    Banana: $%.2f\n", x[1]);
      printf("    Cherry: $%.2f\n", x[2]);
      printf("    Date:   $%.2f\n", x[3]);

      double test_receipt[] = {1.0, 1.0, 1.0, 1.0};
      double pred = nanoqsp_predict(n, x, test_receipt);
      printf("  Prediction for 1 of each item: $%.2f\n\n", pred);
    } else {
      printf("  Solver returned error: %d\n\n", iters);
    }
  }

  return 0;
}
