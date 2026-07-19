# nanoqsp

[![CI](https://github.com/sleepybishop/nanoqsp/actions/workflows/ci.yml/badge.svg)](https://github.com/sleepybishop/nanoqsp/actions/workflows/ci.yml)

`nanoqsp` (Nano Quadratic Solver Programming) is a lightweight, allocation-free, and high-performance C99 library for solving box-constrained quadratic programming (QP) problems:

$$\text{minimize} \quad \frac{1}{2} x^T D x - d^T x$$
$$\text{subject to} \quad lb \le x \le ub$$

Handy when Ceres is a bit too heavy.

## Features
- Zero dynamic memory allocation on hot solve paths.
- Multiple solver strategies: Coordinate Descent, Projected Gradient, Accelerated Gradient, SPG, and ADMM.
- Dense & Sparse (CSR) matrix solver interfaces.
- SIMD Accelerations on AVX, SSE3, ARM NEON, and RISC-V Vector backends.
- KKT Stopping Criteria with scale-invariant relative tolerances.

## Build
```bash
make [CLASSIC=1 | SSE=1 | AVX=1 | NEON=1 | RVV=1]
make check
```
