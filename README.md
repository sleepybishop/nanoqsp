# nanoqsp

[![CI](https://github.com/sleepybishop/nanoqsp/actions/workflows/ci.yml/badge.svg)](https://github.com/sleepybishop/nanoqsp/actions/workflows/ci.yml)

`nanoqsp` (Nano Quadratic Solver Programming) is a lightweight, allocation-free C library for solving box-constrained quadratic programming (QP) problems:

$$\text{minimize} \quad \frac{1}{2} x^T D x - d^T x$$
$$\text{subject to} \quad lb \le x_i \le ub \quad \forall i$$

Handy when ceres is bit too heavy

