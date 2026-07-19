CC = gcc
CFLAGS = -Wall -Wextra -O3 -std=c99 -I.
LDFLAGS = -lm

# SIMD configuration
ifeq ($(CLASSIC), 1)
  DEFAULT_SIMD_FLAGS =
else ifeq ($(AVX), 1)
  DEFAULT_SIMD_FLAGS = -DNANOQSP_AVX -mavx
else ifeq ($(NEON), 1)
  DEFAULT_SIMD_FLAGS = -DNANOQSP_NEON
else ifeq ($(RVV), 1)
  DEFAULT_SIMD_FLAGS = -DNANOQSP_RVV
else
  DEFAULT_SIMD_FLAGS = -DNANOQSP_SSE -msse3
endif

LIB_NAME = libnanoqsp.a
OBJS = nanoqsp.o nanoqsp_blas.o
EXAMPLE = example
TEST_RUNNER = test_runner
SOLVE_CLI = solve_cli

all: $(LIB_NAME) $(EXAMPLE) $(TEST_RUNNER) $(SOLVE_CLI)

$(LIB_NAME): $(OBJS)
	ar rcs $@ $^

nanoqsp.o: nanoqsp.c nanoqsp.h nanoqsp_blas.h
	$(CC) $(CFLAGS) $(DEFAULT_SIMD_FLAGS) -c $< -o $@

nanoqsp_blas.o: nanoqsp_blas.c nanoqsp_blas.h nanoqsp_blas_classic.c nanoqsp_blas_sse.c nanoqsp_blas_avx.c nanoqsp_blas_neon.c nanoqsp_blas_rvv.c
	$(CC) $(CFLAGS) $(DEFAULT_SIMD_FLAGS) -c $< -o $@


$(EXAMPLE): example.c $(LIB_NAME)
	$(CC) $(CFLAGS) $(DEFAULT_SIMD_FLAGS) -o $@ example.c -L. -lnanoqsp $(LDFLAGS)

$(TEST_RUNNER): t/test_runner.c $(LIB_NAME)
	$(CC) $(CFLAGS) $(DEFAULT_SIMD_FLAGS) -o $@ t/test_runner.c -L. -lnanoqsp $(LDFLAGS)

$(SOLVE_CLI): solve_cli.c $(LIB_NAME)
	$(CC) $(CFLAGS) $(DEFAULT_SIMD_FLAGS) -o $@ solve_cli.c -L. -lnanoqsp $(LDFLAGS)

# Benchmark Targets (Classic vs SSE vs AVX comparison)
benchmark.o: benchmark.c nanoqsp.h
	$(CC) $(CFLAGS) -c $< -o $@

nanoqsp_blas_c.o: nanoqsp_blas.c nanoqsp_blas.h nanoqsp_blas_classic.c
	$(CC) $(CFLAGS) -c $< -o $@

nanoqsp_blas_s.o: nanoqsp_blas.c nanoqsp_blas.h nanoqsp_blas_sse.c
	$(CC) $(CFLAGS) -DNANOQSP_SSE -msse3 -c $< -o $@

nanoqsp_blas_a.o: nanoqsp_blas.c nanoqsp_blas.h nanoqsp_blas_avx.c
	$(CC) $(CFLAGS) -DNANOQSP_AVX -mavx -c $< -o $@

bench_classic: benchmark.o nanoqsp.o nanoqsp_blas_c.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

bench_sse: benchmark.o nanoqsp.o nanoqsp_blas_s.o
	$(CC) $(CFLAGS) -DNANOQSP_SSE -msse3 -o $@ $^ $(LDFLAGS)

bench_avx: benchmark.o nanoqsp.o nanoqsp_blas_a.o
	$(CC) $(CFLAGS) -DNANOQSP_AVX -mavx -o $@ $^ $(LDFLAGS)

bench: bench_classic bench_sse bench_avx
	@echo "=================================================="
	@echo "  Running Benchmarks: Classic vs SSE3 vs AVX  "
	@echo "=================================================="
	@echo ""
	./bench_classic 3
	@echo ""
	./bench_sse 3
	@echo ""
	./bench_avx 3
	@echo ""
	@echo "=================================================="

check: all
	prove -v t/

indent:
	clang-format -i nanoqsp.h nanoqsp.c example.c t/test_runner.c solve_cli.c nanoqsp_blas.h nanoqsp_blas.c nanoqsp_blas_classic.c nanoqsp_blas_sse.c nanoqsp_blas_avx.c benchmark.c

clean:
	rm -f $(OBJS) $(LIB_NAME) $(EXAMPLE) $(TEST_RUNNER) $(SOLVE_CLI) benchmark.o nanoqsp_blas_c.o nanoqsp_blas_s.o nanoqsp_blas_a.o bench_classic bench_sse bench_avx

.PHONY: all clean indent check bench
