#include "nanoqsp_blas.h"
#include <riscv_vector.h>
#include <math.h>

void v_copy(double *restrict dest, const double *restrict src, int n) {
  size_t vl;
  for (int i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e64m1(n - i);
    vfloat64m1_t s = __riscv_vle64_v_f64m1(&src[i], vl);
    __riscv_vse64_v_f64m1(&dest[i], s, vl);
  }
}

void v_zero(double *restrict dest, int n) {
  size_t vl;
  for (int i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e64m1(n - i);
    vfloat64m1_t zero = __riscv_vfmv_v_f_f64m1(0.0, vl);
    __riscv_vse64_v_f64m1(&dest[i], zero, vl);
  }
}

double v_dot(const double *restrict x, const double *restrict y, int n) {
  size_t vlmax = __riscv_vsetvlmax_e64m1();
  vfloat64m1_t sum_vec = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
  size_t vl;
  for (int i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e64m1(n - i);
    vfloat64m1_t xi = __riscv_vle64_v_f64m1(&x[i], vl);
    vfloat64m1_t yi = __riscv_vle64_v_f64m1(&y[i], vl);
    sum_vec = __riscv_vfmacc_vv_f64m1(sum_vec, xi, yi, vl);
  }
  
  vfloat64m1_t red = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
  red = __riscv_vfredusum_vs_f64m1_f64m1(sum_vec, red, vlmax);
  return __riscv_vfmv_f_s_f64m1_f64(red);
}

double v_norm1(const double *restrict x, int n) {
  size_t vlmax = __riscv_vsetvlmax_e64m1();
  vfloat64m1_t sum_vec = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
  size_t vl;
  for (int i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e64m1(n - i);
    vfloat64m1_t xi = __riscv_vle64_v_f64m1(&x[i], vl);
    xi = __riscv_vfabs_v_f64m1(xi, vl);
    sum_vec = __riscv_vfadd_vv_f64m1(sum_vec, xi, vl);
  }
  
  vfloat64m1_t red = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
  red = __riscv_vfredusum_vs_f64m1_f64m1(sum_vec, red, vlmax);
  return __riscv_vfmv_f_s_f64m1_f64(red);
}

void v_axpy(double *restrict y, const double *restrict x, double alpha, int n) {
  if (alpha == 0.0) return;
  size_t vl;
  for (int i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e64m1(n - i);
    vfloat64m1_t xi = __riscv_vle64_v_f64m1(&x[i], vl);
    vfloat64m1_t yi = __riscv_vle64_v_f64m1(&y[i], vl);
    yi = __riscv_vfmacc_vf_f64m1(yi, alpha, xi, vl);
    __riscv_vse64_v_f64m1(&y[i], yi, vl);
  }
}
