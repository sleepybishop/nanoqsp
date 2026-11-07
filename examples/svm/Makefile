CC ?= gcc
CFLAGS ?= -Wall -Wextra -O3 -std=c99 -I../..

LIB_NANOQSP = ../../libnanoqsp.a
OBJS = svm.o

all: $(LIB_NANOQSP) svm_train svm_predict

$(LIB_NANOQSP): FORCE
	$(MAKE) -C ../.. libnanoqsp.a

FORCE:

svm.o: svm.c svm.h ../../nanoqsp.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c svm.c -o svm.o

svm_train: svm_train.c $(OBJS) $(LIB_NANOQSP)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ svm_train.c $(OBJS) $(LIB_NANOQSP) -lm

svm_predict: svm_predict.c $(OBJS) $(LIB_NANOQSP)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ svm_predict.c $(OBJS) $(LIB_NANOQSP) -lm

check: all
	prove -v t/

indent:
	clang-format -i svm.h svm.c svm_train.c svm_predict.c

clean:
	rm -f $(OBJS) *.d svm_train svm_predict

.PHONY: all check indent clean FORCE
