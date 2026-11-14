# nanoqsp

[![CI](https://github.com/sleepybishop/nanoqsp/actions/workflows/ci.yml/badge.svg)](https://github.com/sleepybishop/nanoqsp/actions/workflows/ci.yml)

`nanoqsp` (Nano Quadratic Solver Programming) is a lightweight, zero-dependency, allocation-free C99 library for solving dense and sparse quadratic programs (QPs) in deterministic $O(n + m)$ workspace without matrix factorizations:

$$\begin{aligned}
\text{minimize} \quad & \frac{1}{2} x^T D x - d^T x \\
\text{subject to} \quad & l_x \le x \le u_x \\
& l_A \le A x \le u_A
\end{aligned}$$

Handy when Ceres, OSQP, or SuiteSparse are too heavy—especially when sparsity patterns change at runtime or heap allocation is off the table.

## Features

- **Zero-factorization, $O(n + m)$ memory:** Operates directly on row-major arrays or read-only CSR (`NanoqspCSR`) matrices with no symbolic analysis, no fill-in, and no internal matrix copies.
- **Allocation-free & freestanding (`-DNANOQSP_NO_LIBC`):** Caller-supplied scratch buffers eliminate `malloc`/`free` on hot paths; compiling with `-DNANOQSP_NO_LIBC` strips all libc allocator/assert symbols while exporting only `nanoqsp_*` symbols.
- **Five solver strategies behind one header (`include/nanoqsp.h`):**
  - `NANOQSP_STRATEGY_COORDINATE_DESCENT` — Projected Gauss-Seidel / coordinate descent.
  - `NANOQSP_STRATEGY_PROJECTED_GRADIENT` — Projected gradient descent with Gershgorin step scaling.
  - `NANOQSP_STRATEGY_ACCELERATED_GRADIENT` — Nesterov FISTA with adaptive gradient restarts.
  - `NANOQSP_STRATEGY_SPECTRAL_GRADIENT` — Barzilai-Borwein (BB1) Spectral Projected Gradient with GLL non-monotone safeguards, exact 1D kink-aware segment line search, `fabs` curvature scaling for indefinite/non-convex box QPs, and a proximal Augmented Lagrangian (`ALM-SPG`) outer loop for affine constraints.
  - `NANOQSP_STRATEGY_ADMM` — Equilibrated ADMM with adaptive-tolerance Jacobi-preconditioned CG, plus a direct 1-shot PCG fast-path for unconstrained solves.
- **Runtime SIMD dispatch:** Single-binary CPU detection on x86 (`AVX-512`, `AVX2+FMA`, `SSE2`), compile-time `ARM NEON` and `RISC-V Vector (RVV)` backends, and portable scalar fallback.
- **Scale-invariant KKT stopping criteria:** Relative primal and dual residuals normalized against problem data and iterate scales, with warm-start support via `x`.

## API Overview

| Entry Point | Problem | Supported Strategies | Workspace (doubles) |
| :--- | :--- | :--- | :--- |
| `nanoqsp_solve_box` | Dense box QP ($l_x \le x \le u_x$) | All 5 strategies | `n` (CD, PG), `2n` (AG), `3n` (SPG), `7n` (ADMM) |
| `nanoqsp_solve_box_sparse` | Sparse CSR box QP | All 5 strategies | `n` (PG), `2n` (CD, AG), `3n` (SPG), `7n` (ADMM) |
| `nanoqsp_solve_least_squares` | Bounded least squares ($\frac{1}{2}\|Ax - b\|_2^2$) | All 5 strategies | `n^2 + n + box` |
| `nanoqsp_solve_affine` | Dense box + linear constraints ($l_A \le Ax \le u_A$) | `ADMM`, `SPECTRAL_GRADIENT` | `7n + 3m` |
| `nanoqsp_solve_affine_sparse` | Sparse CSR box + linear constraints | `ADMM`, `SPECTRAL_GRADIENT` | `7n + 3m` |

Pass `NULL` for `lb_x` or `ub_x` when variables are unbounded (`-INFINITY` / `+INFINITY` per coordinate are also accepted).

## Quick Start

```c
#include "nanoqsp.h"

double D[4] = {4.0, 1.0, 1.0, 2.0};
double d[2] = {1.0, 1.0};
double A[2] = {1.0, 1.0};
double lb_x[2] = {0.0, 0.0};
double ub_x[2] = {0.7, 0.7};
double lb_A[1] = {1.0}, ub_A[1] = {1.0};
double x[2] = {0.0, 0.0};

/* 7*n + 3*m = 17 doubles for allocation-free affine solve */
double ws[17];
NanoqspConfig cfg = {
    .strategy = NANOQSP_STRATEGY_ADMM, /* or NANOQSP_STRATEGY_SPECTRAL_GRADIENT */
    .max_iterations = 1000,
    .tolerance = 1e-8,
    .workspace = ws,
    .workspace_size = 17,
};

int iters = nanoqsp_solve_affine(2, 1, D, d, A, lb_x, ub_x, lb_A, ub_A, x, &cfg);
```

## Build and Verification

```bash
make                  # build libnanoqsp.a, examples, test runner, and benchmark
make check            # run full TAP test suite (prove -v t/)
make check-asan       # rebuild and run tests under AddressSanitizer + UBSan
make check-embedded   # verify -DNANOQSP_NO_LIBC build has zero allocator/non-prefixed symbols
make bench            # run solver benchmarks
```

For RISC-V Vector builds, pass the target ISA flags directly (e.g. `make CC=riscv64-linux-gnu-gcc CFLAGS="-Wall -Wextra -O3 -std=c99 -march=rv64gcv"`). On x86 and ARM64, SIMD acceleration is enabled out of the box with a standard `make`.

## Workspace and Status Handling

By default, solvers dynamically allocate temporary workspace for the duration of a call when `NanoqspConfig.workspace` is `NULL` (or smaller than required). To run completely allocation-free—or when compiled with `-DNANOQSP_NO_LIBC`, where missing workspace returns `NANOQSP_ERR_OUT_OF_MEMORY`—query the exact buffer length in `double`s via:

- `nanoqsp_workspace_size_box(n, strategy)`
- `nanoqsp_workspace_size_box_sparse(n, strategy)`
- `nanoqsp_workspace_size_least_squares(n, strategy)`
- `nanoqsp_workspace_size_affine(n, m)`

and supply that buffer through `NanoqspConfig.workspace` and `workspace_size`.

Successful solves return the iteration count ($\ge 0$). Failure returns a negative `NanoqspStatus` code (`NANOQSP_ERR_INVALID_ARG`, `NANOQSP_ERR_INVALID_STRATEGY`, `NANOQSP_ERR_OUT_OF_MEMORY`, or `NANOQSP_ERR_MAX_ITERATIONS` if the iteration cap is exhausted before meeting tolerance).

## Examples

- `examples/least_squares` — Bounded non-negative/box least squares across all five box strategies.
- `examples/svm/svm_train` & `examples/svm/svm_predict` — LIBSVM-compatible dual C-SVC trainer and predictor supporting sparse text datasets, linear and RBF kernels, regularized box-QP formulations (`-s 0`), and exact equality-constrained affine-QP formulations (`-s 1` with `ADMM` or `SPECTRAL_GRADIENT`).
