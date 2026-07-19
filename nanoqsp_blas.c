#include "nanoqsp_blas.h"

#ifdef NANOQSP_AVX
#include "nanoqsp_blas_avx.c"
#elif defined(NANOQSP_SSE)
#include "nanoqsp_blas_sse.c"
#elif defined(NANOQSP_NEON)
#include "nanoqsp_blas_neon.c"
#elif defined(NANOQSP_RVV)
#include "nanoqsp_blas_rvv.c"
#else
#include "nanoqsp_blas_classic.c"
#endif
