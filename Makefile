.DEFAULT_GOAL := all

CC = gcc
CFLAGS = -Wall -Wextra -O3 -std=c99
LDFLAGS = -lm
INCS = -Iinclude -MMD -MP

LIB_NAME = libnanoqsp.a

OBJ = \
	lib/nanoqsp.o \
	lib/qsp_blas.o \
	lib/qsp_box.o \
	lib/qsp_admm.o \
	lib/qsp_alm.o

EXAMPLES = \
	examples/least_squares \
	examples/svm/svm_train \
	examples/svm/svm_predict

TEST_UTILS = \
	t/00util/test_runner

BENCHMARK = benchmark

PROG_OBJS = \
	examples/least_squares.o \
	examples/svm/svm.o \
	examples/svm/svm_train.o \
	examples/svm/svm_predict.o \
	t/00util/test_runner.o \
	benchmark.o

DEPS = $(OBJ:.o=.d) $(PROG_OBJS:.o=.d)

-include $(DEPS)

all: $(LIB_NAME) $(EXAMPLES) $(TEST_UTILS) $(BENCHMARK)

$(LIB_NAME): $(OBJ)
	$(AR) rcs $@ $^

lib/%.o: lib/%.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(INCS) -c $< -o $@

examples/%.o: examples/%.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(INCS) -c $< -o $@

examples/svm/%.o: examples/svm/%.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(INCS) -c $< -o $@

t/00util/%.o: t/00util/%.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(INCS) -c $< -o $@

benchmark.o: benchmark.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(INCS) -c $< -o $@

examples/least_squares: examples/least_squares.o $(LIB_NAME)
	$(CC) $(CFLAGS) -o $@ $< $(LIB_NAME) $(LDFLAGS)

examples/svm/svm_train: examples/svm/svm_train.o examples/svm/svm.o $(LIB_NAME)
	$(CC) $(CFLAGS) -o $@ examples/svm/svm_train.o examples/svm/svm.o $(LIB_NAME) $(LDFLAGS)

examples/svm/svm_predict: examples/svm/svm_predict.o examples/svm/svm.o $(LIB_NAME)
	$(CC) $(CFLAGS) -o $@ examples/svm/svm_predict.o examples/svm/svm.o $(LIB_NAME) $(LDFLAGS)

t/00util/test_runner: t/00util/test_runner.o $(LIB_NAME)
	$(CC) $(CFLAGS) -o $@ $< $(LIB_NAME) $(LDFLAGS)

$(BENCHMARK): benchmark.o $(LIB_NAME)
	$(CC) $(CFLAGS) -o $@ $< $(LIB_NAME) $(LDFLAGS)

bench: $(BENCHMARK)
	./$(BENCHMARK)

check: all
	NANOQSP_CC="$(CC)" \
	NANOQSP_CPPFLAGS="$(CPPFLAGS)" \
	NANOQSP_CFLAGS="$(CFLAGS)" \
	NANOQSP_LDFLAGS="$(LDFLAGS)" \
	prove -v t/

check-asan:
	$(MAKE) clean
	$(MAKE) check \
		CFLAGS="$(CFLAGS) -fsanitize=address,undefined -fno-omit-frame-pointer" \
		LDFLAGS="$(LDFLAGS) -fsanitize=address,undefined"

check-embedded:
	$(MAKE) clean
	$(MAKE) $(LIB_NAME) CPPFLAGS="$(CPPFLAGS) -DNANOQSP_NO_LIBC"
	@echo "--- Checking undefined allocator/assert symbols in $(LIB_NAME) ---"
	@if nm -u $(LIB_NAME) | grep -E '\b(malloc|calloc|realloc|free|posix_memalign|__assert_fail)\b'; then \
		echo "FAIL: forbidden libc symbols found in embedded build"; exit 1; \
	else \
		echo "PASS: no allocator/assert symbols found"; \
	fi
	@echo "--- Checking exported symbol namespace in $(LIB_NAME) ---"
	@if nm -g --defined-only $(LIB_NAME) | awk 'NF >= 2 && $$1 !~ /:$$/ && $$NF !~ /^_?nanoqsp_/ { print; found = 1 } END { exit !found }'; then \
		echo "FAIL: non-namespaced global symbols exported"; exit 1; \
	else \
		echo "PASS: all exported symbols use nanoqsp_ prefix"; \
	fi
	$(MAKE) clean

indent:
	clang-format -i lib/*.c include/*.h examples/*.c examples/svm/*.c examples/svm/*.h t/00util/*.c benchmark.c

clean:
	$(RM) $(OBJ) $(PROG_OBJS) $(DEPS) $(LIB_NAME) $(EXAMPLES) $(TEST_UTILS) $(BENCHMARK)

.PHONY: all bench check check-asan check-embedded indent clean
