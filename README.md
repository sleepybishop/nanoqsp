# nanoqsp

[![CI](https://github.com/sleepybishop/nanoqsp/actions/workflows/ci.yml/badge.svg)](https://github.com/sleepybishop/nanoqsp/actions/workflows/ci.yml)

`nanoqsp` (Nano Quadratic Solver Programming) is a lightweight, allocation-free, and high-performance C99 library for solving box-constrained quadratic programming (QP) problems:

$$\text{minimize} \quad \frac{1}{2} x^T D x - d^T x$$
$$\text{subject to} \quad lb \le x \le ub$$

Handy when Ceres is a bit too heavy.

## Features
- Allocation-free solve paths when a caller-provided workspace is used.
- Multiple solver strategies: Coordinate Descent, Projected Gradient, Accelerated Gradient, SPG, and ADMM.
- Dense & Sparse (CSR) matrix solver interfaces.
- SIMD Accelerations on AVX, SSE3, ARM NEON, and RISC-V Vector backends.
- KKT Stopping Criteria with scale-invariant relative tolerances.

## Build
```bash
make [CLASSIC=1 | SSE=1 | AVX=1 | NEON=1 | RVV=1]
make check
```

Changing the SIMD selection automatically rebuilds the backend; a manual
`make clean` is not required.

## Workspace and status handling

By default, strategies that need scratch storage allocate it for the duration
of a solve. To keep repeated solves allocation-free, query the required number
of doubles with `nanoqsp_workspace_size_box`,
`nanoqsp_workspace_size_box_sparse`,
`nanoqsp_workspace_size_least_squares`, or
`nanoqsp_workspace_size_affine`, then provide that buffer through
`NanoqspConfig.workspace` and `workspace_size`.

Successful solves return the number of iterations performed. Exhausting the
configured iteration limit returns `NANOQSP_ERR_MAX_ITERATIONS`; it is not
reported as a successful solution.

General affine constraints are currently solved by ADMM only. When passing a
configuration to an affine solve, set its strategy to
`NANOQSP_STRATEGY_ADMM`.
