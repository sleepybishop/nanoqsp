#include "nanoqsp.h"
#include <float.h>
#include <math.h>
#include <stdio.h>

static int test_count;
static int failures;

static void check(int ok, const char *name, int sparse, int strategy,
                  int status, const double *x) {
  printf("%s %d - %s (%s, strategy %d)\n", ok ? "ok" : "not ok", ++test_count,
         name, sparse ? "CSR" : "dense", strategy);
  if (!ok) {
    printf("# status=%d x=(%.17g, %.17g)\n", status, x[0], x[1]);
    failures++;
  }
}

static int solve(int sparse, int n, const double *D, const double *d,
                 const double *lb, const double *ub, double *x,
                 NanoqspConfig config) {
  /* tiny fixtures store every matrix entry, including zeros */
  int rows[] = {0, n, 2 * n};
  int cols[] = {0, 1, 0, 1};
  NanoqspCSR csr = {n, n * n, D, cols, rows};
  if (sparse)
    return nanoqsp_solve_box_sparse(n, &csr, d, lb, ub, x, &config);
  return nanoqsp_solve_box(n, D, d, lb, ub, x, &config);
}

int main(void) {
  for (int sparse = 0; sparse < 2; sparse++) {
    NanoqspConfig config = {
        .strategy = NANOQSP_STRATEGY_COORDINATE_DESCENT,
        .max_iterations = 1,
        .tolerance = 1e-8,
    };
    {
      double D[] = {1.0}, d[] = {1.0}, x[] = {0.0, 0.0};
      int status = solve(sparse, 1, D, d, NULL, NULL, x, config);
      check(status == 1 && x[0] == 1.0,
            "CD recognizes an optimum reached on its final iteration", sparse,
            config.strategy, status, x);
    }
    {
      double D[] = {4.0, 1.0, 1.0, 2.0}, d[] = {1.0, 1.0};
      double x[] = {0.0, 0.0};
      int status = solve(sparse, 2, D, d, NULL, NULL, x, config);
      check(status == NANOQSP_ERR_MAX_ITERATIONS && x[0] == 0.25 &&
                x[1] == 0.375,
            "CD still reports exhaustion when the final iterate fails KKT",
            sparse, config.strategy, status, x);
    }

    for (int strategy = NANOQSP_STRATEGY_COORDINATE_DESCENT;
         strategy <= NANOQSP_STRATEGY_SPECTRAL_GRADIENT; strategy++) {
      config.strategy = (NanoqspStrategy)strategy;
      config.max_iterations = 1000;
      {
        double D[] = {1e308, 0.0, 0.0, 1.0}, d[] = {0.0, 1.0};
        double lb[] = {2.0, -2.0}, ub[] = {2.0, 2.0};
        double x[] = {2.0, 0.0};
        int status = solve(sparse, 2, D, d, lb, ub, x, config);
        check(status == NANOQSP_ERR_MAX_ITERATIONS,
              "an overflowing fixed-coordinate gradient cannot signal success",
              sparse, strategy, status, x);
      }
      {
        double D[] = {2.0, 1.0, 1.0, 2.0}, d[] = {2.0, 2.0};
        double lb[] = {0.0, -10.0}, ub[] = {0.0, 10.0};
        double x[] = {0.0, 0.0};
        int status = solve(sparse, 2, D, d, lb, ub, x, config);
        check(status >= 0 && x[0] == 0.0 && fabs(x[1] - 1.0) < 1e-7,
              "finite fixed-coordinate gradients still allow convergence",
              sparse, strategy, status, x);
      }
      {
        double D[] = {DBL_MAX}, d[] = {0.0};
        double lb[] = {1.0}, ub[] = {1.0}, x[] = {1.0, 0.0};
        int status = solve(sparse, 1, D, d, lb, ub, x, config);
        check(status >= 0 && x[0] == 1.0,
              "a finite gradient near DBL_MAX is accepted", sparse, strategy,
              status, x);
      }
    }
  }
  for (int strategy = NANOQSP_STRATEGY_COORDINATE_DESCENT;
       strategy <= NANOQSP_STRATEGY_ADMM; strategy++) {
    double values[] = {1e5, -1e5, 2.0};
    int cols[] = {0, 0, 0}, rows[] = {0, 3};
    NanoqspCSR csr = {1, 3, values, cols, rows};
    double d[] = {1.0}, lb[] = {-2.0}, ub[] = {2.0}, x[] = {0.0, 0.0};
    double ws[7];
    int ws_size =
        nanoqsp_workspace_size_box_sparse(1, (NanoqspStrategy)strategy);
    NanoqspConfig config = {
        .strategy = (NanoqspStrategy)strategy,
        .max_iterations = 1000,
        .tolerance = 1e-8,
        .workspace = ws,
        .workspace_size = ws_size,
    };
    int status = nanoqsp_solve_box_sparse(1, &csr, d, lb, ub, x, &config);
    check(status >= 0 && fabs(x[0] - 0.5) < 1e-7,
          "CSR duplicate cancellation in D preserves convergence", 1, strategy,
          status, x);
  }
  printf("1..%d\n", test_count);
  return failures ? 1 : 0;
}
