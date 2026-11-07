CC = gcc
CFLAGS = -Wall -Wextra -O3 -std=c99 -I. -MMD -MP
LDFLAGS = -lm

# SIMD configuration
ifeq ($(CLASSIC), 1)
  SIMD_BACKEND = classic
  DEFAULT_SIMD_FLAGS =
else ifeq ($(AVX), 1)
  SIMD_BACKEND = avx
  DEFAULT_SIMD_FLAGS = -DNANOQSP_AVX -mavx
else ifeq ($(NEON), 1)
  SIMD_BACKEND = neon
  DEFAULT_SIMD_FLAGS = -DNANOQSP_NEON
else ifeq ($(RVV), 1)
  SIMD_BACKEND = rvv
  DEFAULT_SIMD_FLAGS = -DNANOQSP_RVV
else
  SIMD_BACKEND = sse
  DEFAULT_SIMD_FLAGS = -DNANOQSP_SSE
endif

BACKEND_STAMP = .nanoqsp-backend-$(SIMD_BACKEND)

LIB_NAME = libnanoqsp.a
OBJS = nanoqsp.o nanoqsp_blas.o
DEPS = $(OBJS:.o=.d) benchmark.d
EXAMPLE = example
TEST_RUNNER = test_runner
SOLVE_CLI = solve_cli
BENCHMARK_SPARSE = benchmark_sparse

-include $(DEPS)

all: $(LIB_NAME) $(EXAMPLE) $(TEST_RUNNER) $(SOLVE_CLI) $(BENCHMARK_SPARSE)

$(LIB_NAME): $(OBJS)
	$(AR) rcs $@ $^

nanoqsp.o: nanoqsp.c $(BACKEND_STAMP)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEFAULT_SIMD_FLAGS) -c $< -o $@

$(BACKEND_STAMP):
	$(RM) .nanoqsp-backend-*
	touch $@

nanoqsp_blas.o: nanoqsp_blas.c $(BACKEND_STAMP)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEFAULT_SIMD_FLAGS) -c $< -o $@

$(EXAMPLE): example.c $(LIB_NAME)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEFAULT_SIMD_FLAGS) -o $@ example.c -L. -lnanoqsp $(LDFLAGS)

$(TEST_RUNNER): t/test_runner.c $(LIB_NAME)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEFAULT_SIMD_FLAGS) -o $@ t/test_runner.c -L. -lnanoqsp $(LDFLAGS)

$(SOLVE_CLI): solve_cli.c $(LIB_NAME)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEFAULT_SIMD_FLAGS) -o $@ solve_cli.c -L. -lnanoqsp $(LDFLAGS)

# Benchmark Targets (Classic vs SSE vs AVX comparison)
benchmark.o: benchmark.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

nanoqsp_blas_c.o: nanoqsp_blas.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

nanoqsp_blas_s.o: nanoqsp_blas.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -DNANOQSP_SSE -c $< -o $@

nanoqsp_blas_a.o: nanoqsp_blas.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -DNANOQSP_AVX -mavx -c $< -o $@

bench_classic: benchmark.o nanoqsp.o nanoqsp_blas_c.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

bench_sse: benchmark.o nanoqsp.o nanoqsp_blas_s.o
	$(CC) $(CFLAGS) -DNANOQSP_SSE -o $@ $^ $(LDFLAGS)

bench_avx: benchmark.o nanoqsp.o nanoqsp_blas_a.o
	$(CC) $(CFLAGS) -DNANOQSP_AVX -mavx -o $@ $^ $(LDFLAGS)

bench: bench_classic bench_sse bench_avx
	@echo "=================================================="
	@echo "  Running Benchmarks: Classic vs SSE2 vs AVX  "
	@echo "=================================================="
	@echo ""
	./bench_classic 3
	@echo ""
	./bench_sse 3
	@echo ""
	./bench_avx 3
	@echo ""
	@echo "=================================================="

$(BENCHMARK_SPARSE): benchmark_sparse.c $(LIB_NAME)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ benchmark_sparse.c -L. -lnanoqsp $(LDFLAGS)

bench_sparse: $(BENCHMARK_SPARSE)
	@echo "=================================================="
	@echo "  Running Benchmarks: Sparse vs Dense Solvers    "
	@echo "=================================================="
	@echo ""
	./$(BENCHMARK_SPARSE) 3
	@echo ""
	@echo "=================================================="

check: all
	prove -v t/

check-asan: CFLAGS += -fsanitize=address,undefined -fno-omit-frame-pointer
check-asan: LDFLAGS += -fsanitize=address,undefined
check-asan: clean check

check-embedded: clean
	$(MAKE) $(LIB_NAME) CPPFLAGS="$(CPPFLAGS) -DNANOQSP_NO_LIBC"
	@echo "--- Checking undefined allocator symbols in $(LIB_NAME) ---"
	@if nm -u $(LIB_NAME) | grep -E '\b(malloc|calloc|realloc|free)\b'; then \
		echo "FAIL: allocator symbols found in embedded build"; exit 1; \
	else \
		echo "PASS: no allocator symbols found"; \
	fi
	$(MAKE) clean

indent:
	find . -name '*.[h,c]' -not -path './deps/*' | xargs clang-format -i

clean:
	$(RM) *.o *.d $(LIB_NAME) $(EXAMPLE) $(TEST_RUNNER) $(SOLVE_CLI) bench_classic bench_sse bench_avx $(BENCHMARK_SPARSE) .nanoqsp-backend-*

.PHONY: all clean indent check check-asan check-embedded bench bench_sparse
