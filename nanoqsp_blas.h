#ifndef NANOQSP_BLAS_H
#define NANOQSP_BLAS_H

#include <stddef.h>

void v_copy(double *restrict dest, const double *restrict src, int n);
void v_zero(double *restrict dest, int n);
double v_dot(const double *restrict x, const double *restrict y, int n);
double v_norm1(const double *restrict x, int n);
void v_axpy(double *restrict y, const double *restrict x, double alpha, int n);

#endif /* NANOQSP_BLAS_H */
