#ifndef NANOQSP_H
#define NANOQSP_H

#ifdef __cplusplus
extern "C" {
#endif

/* Standard error and status codes */
typedef enum {
  NANOQSP_ERR_OUT_OF_MEMORY = -3,
  NANOQSP_ERR_INVALID_STRATEGY = -2,
  NANOQSP_ERR_INVALID_ARG = -1,
  NANOQSP_SUCCESS = 0
} NanoqspStatus;

/* Supported solver strategies */
typedef enum {
  /* Projected Gauss-Seidel / Coordinate Descent */
  NANOQSP_STRATEGY_COORDINATE_DESCENT,
  /* Projected Gradient Descent with Gerschgorin step size */
  NANOQSP_STRATEGY_PROJECTED_GRADIENT,
  /* Nesterov Accelerated Projected Gradient Descent with Adaptive Restarts */
  NANOQSP_STRATEGY_ACCELERATED_GRADIENT,
  /* Spectral Projected Gradient (Barzilai-Borwein step size) */
  NANOQSP_STRATEGY_SPECTRAL_GRADIENT,
  /* Alternating Direction Method of Multipliers (ADMM) */
  NANOQSP_STRATEGY_ADMM
} NanoqspStrategy;

/* Configuration parameters for the solver */
typedef struct {
  NanoqspStrategy strategy;
  int max_iterations;
  double tolerance;
  double *workspace;  /* Optional: Pre-allocated workspace buffer */
  int workspace_size; /* Size of workspace buffer in terms of number of doubles
                       */
} NanoqspConfig;

/**
 * Solves the box-constrained quadratic programming (QP) problem:
 *   minimize    0.5 * x^T * D * x - d^T * x
 *   subject to  lb <= x_i <= ub    for all i
 *
 * Parameters:
 *   n:           Number of variables
 *   D:           n x n symmetric positive-definite matrix (row-major, size n*n)
 *   d:           n-dimensional vector (size n)
 *   lb:          Lower bounds array (size n). If NULL, treated as -INFINITY
 *   ub:          Upper bounds array (size n). If NULL, treated as INFINITY
 *   x:           Input/Output vector (size n). Can be pre-initialized for a
 * warm start. config:      Configuration options (if NULL, defaults are used)
 *
 * Returns:
 *   Number of iterations performed on success, or a NanoqspStatus error code.
 */
int nanoqsp_solve_box(int n, const double *D, const double *d, const double *lb,
                      const double *ub, double *x, const NanoqspConfig *config);

/**
 * High-level API: Solves a least squares problem (min ||Ax - b||^2) subject to
 * bounds. Automatically handles matrix multiplication (D = A^T A, d = A^T b)
 * and applies Tikhonov regularization to prevent singular matrix errors.
 *
 * Parameters:
 *   m: Number of rows (equations)
 *   n: Number of columns (variables)
 *   A: Matrix A in row-major order (size m * n)
 *   b: Vector b (size m)
 *   lb: Lower bounds array for x (size n). If NULL, treated as -INFINITY.
 *   ub: Upper bounds array for x (size n). If NULL, treated as INFINITY.
 *   x: Output vector (size n). Can be pre-initialized for a warm start.
 *   config: Solver configuration (NULL for defaults)
 *
 * Returns:
 *   Number of iterations performed on success, or a NanoqspStatus error code.
 */
int nanoqsp_solve_least_squares(int m, int n, const double *A, const double *b,
                                const double *lb, const double *ub, double *x,
                                const NanoqspConfig *config);

/**
 * High-level API: Generates a prediction (dot product) given a solved model x
 * and a new feature vector.
 */
double nanoqsp_predict(int n, const double *x, const double *feature_vector);

#ifdef __cplusplus
}
#endif

#endif /* NANOQSP_H */
