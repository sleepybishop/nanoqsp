#include "nanoqsp_blas.h"

#ifdef NANOQSP_AVX
#include "nanoqsp_blas_avx.c"
#else
#ifdef NANOQSP_SSE
#include "nanoqsp_blas_sse.c"
#else
#include "nanoqsp_blas_classic.c"
#endif
#endif
