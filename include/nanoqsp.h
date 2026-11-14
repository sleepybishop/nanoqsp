#ifndef NANOQSP_H
#define NANOQSP_H

#ifdef __cplusplus
#define NANOQSP_RESTRICT
extern "C" {
#else
#define NANOQSP_RESTRICT restrict
#endif

/* standard error and status codes */
typedef enum {
  NANOQSP_ERR_MAX_ITERATIONS = -4,
  NANOQSP_ERR_OUT_OF_MEMORY = -3,
  NANOQSP_ERR_INVALID_STRATEGY = -2,
  NANOQSP_ERR_INVALID_ARG = -1,
  NANOQSP_SUCCESS = 0
} NanoqspStatus;

/* supported solver strategies */
typedef enum {
  /* projected gauss-seidel / coordinate descent */
  NANOQSP_STRATEGY_COORDINATE_DESCENT,
  /* projected gradient descent with gershgorin step size */
  NANOQSP_STRATEGY_PROJECTED_GRADIENT,
  /* nesterov accelerated projected gradient (FISTA) with adaptive restarts */
  NANOQSP_STRATEGY_ACCELERATED_GRADIENT,
  /* spectral projected gradient (BB steps with GLL non-monotone safeguard) */
  NANOQSP_STRATEGY_SPECTRAL_GRADIENT,
  /* alternating direction method of multipliers (ADMM) */
  NANOQSP_STRATEGY_ADMM
} NanoqspStrategy;

/* solver configuration parameters */
typedef struct {
  NanoqspStrategy strategy;
  int max_iterations;
  double tolerance;
  /* optional caller-supplied scratch buffer (measured in doubles) */
  double *NANOQSP_RESTRICT workspace;
  int workspace_size;
} NanoqspConfig;

/* sparse matrix in compressed sparse row (CSR) format */
typedef struct {
  int n;   /* number of rows */
  int nnz; /* number of non-zero elements */
  const double *NANOQSP_RESTRICT values;
  const int *col_indices; /* length nnz */
  const int *row_ptr;     /* length n + 1 */
} NanoqspCSR;

/*
 * minimum workspace size in doubles required for allocation-free solves.
 * returns a negative NanoqspStatus code on invalid dimensions or strategies.
 */
int nanoqsp_workspace_size_box(int n, NanoqspStrategy strategy);
int nanoqsp_workspace_size_box_sparse(int n, NanoqspStrategy strategy);
int nanoqsp_workspace_size_least_squares(int n, NanoqspStrategy strategy);
int nanoqsp_workspace_size_affine(int n, int m);

/*
 * solve dense box-constrained QP:
 *   minimize    0.5 * x^T D x - d^T x
 *   subject to  lb <= x <= ub
 *
 * D is an n x n row-major SPD matrix. lb/ub may be NULL (+/-INFINITY).
 * x is warm-started and projected into [lb, ub] before solving.
 * returns iteration count on convergence, or a negative NanoqspStatus code.
 */
int nanoqsp_solve_box(int n, const double *NANOQSP_RESTRICT D,
                      const double *NANOQSP_RESTRICT d,
                      const double *NANOQSP_RESTRICT lb,
                      const double *NANOQSP_RESTRICT ub,
                      double *NANOQSP_RESTRICT x, const NanoqspConfig *config);

/*
 * solve sparse box-constrained QP with D in CSR format.
 */
int nanoqsp_solve_box_sparse(int n, const NanoqspCSR *D,
                             const double *NANOQSP_RESTRICT d,
                             const double *NANOQSP_RESTRICT lb,
                             const double *NANOQSP_RESTRICT ub,
                             double *NANOQSP_RESTRICT x,
                             const NanoqspConfig *config);

/*
 * solve bounded linear least squares:
 *   minimize    0.5 * ||A x - b||_2^2
 *   subject to  lb <= x <= ub
 *
 * forms normal equations D = A^T A + eps * I, d = A^T b in workspace.
 */
int nanoqsp_solve_least_squares(int m, int n, const double *NANOQSP_RESTRICT A,
                                const double *NANOQSP_RESTRICT b,
                                const double *NANOQSP_RESTRICT lb,
                                const double *NANOQSP_RESTRICT ub,
                                double *NANOQSP_RESTRICT x,
                                const NanoqspConfig *config);

/*
 * solve dense QP with box and general linear constraints (ADMM or SPG):
 *   minimize    0.5 * x^T D x - d^T x
 *   subject to  lb_x <= x <= ub_x
 *               lb_A <= A x <= ub_A
 */
int nanoqsp_solve_affine(
    int n, int m, const double *NANOQSP_RESTRICT D,
    const double *NANOQSP_RESTRICT d, const double *NANOQSP_RESTRICT A,
    const double *NANOQSP_RESTRICT lb_x, const double *NANOQSP_RESTRICT ub_x,
    const double *NANOQSP_RESTRICT lb_A, const double *NANOQSP_RESTRICT ub_A,
    double *NANOQSP_RESTRICT x, const NanoqspConfig *config);

/*
 * solve sparse QP with box and general linear constraints in CSR format.
 */
int nanoqsp_solve_affine_sparse(
    int n, int m, const NanoqspCSR *D, const double *NANOQSP_RESTRICT d,
    const NanoqspCSR *A, const double *NANOQSP_RESTRICT lb_x,
    const double *NANOQSP_RESTRICT ub_x, const double *NANOQSP_RESTRICT lb_A,
    const double *NANOQSP_RESTRICT ub_A, double *NANOQSP_RESTRICT x,
    const NanoqspConfig *config);

#ifdef __cplusplus
}
#endif

#undef NANOQSP_RESTRICT

#endif /* NANOQSP_H */
