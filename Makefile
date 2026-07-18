CC = gcc
CFLAGS = -Wall -Wextra -O3 -std=c99 -I.
LDFLAGS = -lm

LIB_NAME = libnanoqsp.a
OBJS = nanoqsp.o
EXAMPLE = example
TEST_RUNNER = test_runner
SOLVE_CLI = solve_cli

all: $(LIB_NAME) $(EXAMPLE) $(TEST_RUNNER) $(SOLVE_CLI)

$(LIB_NAME): $(OBJS)
	ar rcs $@ $^

%.o: %.c nanoqsp.h
	$(CC) $(CFLAGS) -c $< -o $@

$(EXAMPLE): example.c $(LIB_NAME)
	$(CC) $(CFLAGS) -o $@ example.c -L. -lnanoqsp $(LDFLAGS)

$(TEST_RUNNER): t/test_runner.c $(LIB_NAME)
	$(CC) $(CFLAGS) -o $@ t/test_runner.c -L. -lnanoqsp $(LDFLAGS)

$(SOLVE_CLI): solve_cli.c $(LIB_NAME)
	$(CC) $(CFLAGS) -o $@ solve_cli.c -L. -lnanoqsp $(LDFLAGS)

check: all
	prove -v t/

indent:
	clang-format -i nanoqsp.h nanoqsp.c example.c t/test_runner.c solve_cli.c

clean:
	rm -f $(OBJS) $(LIB_NAME) $(EXAMPLE) $(TEST_RUNNER) $(SOLVE_CLI)

.PHONY: all clean indent check
