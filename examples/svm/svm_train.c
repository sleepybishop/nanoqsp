#include "svm.h"
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int parse_cli_int(const char *arg, int *out) {
  if (!arg || *arg == '\0')
    return -1;
  char *end = NULL;
  errno = 0;
  long v = strtol(arg, &end, 10);
  if (end == arg || *end != '\0' || errno == ERANGE || v < INT_MIN ||
      v > INT_MAX)
    return -1;
  *out = (int)v;
  return 0;
}

static int parse_cli_double(const char *arg, double *out) {
  if (!arg || *arg == '\0')
    return -1;
  char *end = NULL;
  errno = 0;
  double v = strtod(arg, &end);
  if (end == arg || *end != '\0' || (errno == ERANGE && v == 0.0))
    return -1;
  *out = v;
  return 0;
}

static void print_usage(const char *prog) {
  printf("Usage: %s [options] training_set_file [model_file]\n", prog);
  printf("Options:\n");
  printf("  -s solver   : 0 -- Box QP (default, fast regularized bias)\n");
  printf(
      "                1 -- Affine QP (equality constraint sum y_i alpha_i = "
      "0)\n");
  printf("  -t kernel   : 0 -- Linear (default)\n");
  printf("                1 -- RBF (exp(-gamma * |u-v|^2))\n");
  printf("  -m strategy : nanoqsp strategy (0: CD, 1: PG, 2: AG, 3: SPG, 4: "
         "ADMM)\n");
  printf("  -c cost     : set the parameter C (default 1.0)\n");
  printf("  -g gamma    : set gamma in kernel function (default "
         "1/num_features)\n");
  printf(
      "  -b bias     : set regularized bias for box solver (default 1.0, 0 to "
      "disable)\n");
  printf("  -e epsilon  : set tolerance of termination criterion (default "
         "1e-4)\n");
  printf("  -i max_iter : set max iterations (default 1000)\n");
}

int main(int argc, char **argv) {
  SvmParam param;
  svm_param_init(&param);

  int i = 1;
  while (i < argc) {
    if (argv[i][0] != '-')
      break;
    if (argv[i][1] == '\0' || argv[i][2] != '\0' || ++i >= argc) {
      print_usage(argv[0]);
      return 1;
    }
    int parsed_int = 0;
    int bad_opt = 0;
    switch (argv[i - 1][1]) {
    case 's':
      bad_opt = parse_cli_int(argv[i], &parsed_int);
      param.solver = (SvmSolverType)parsed_int;
      break;
    case 't':
      bad_opt = parse_cli_int(argv[i], &parsed_int);
      param.kernel = (SvmKernelType)parsed_int;
      break;
    case 'm':
      bad_opt = parse_cli_int(argv[i], &parsed_int);
      param.strategy = (NanoqspStrategy)parsed_int;
      break;
    case 'c':
      bad_opt = parse_cli_double(argv[i], &param.C);
      break;
    case 'g':
      bad_opt = parse_cli_double(argv[i], &param.gamma);
      break;
    case 'b':
      bad_opt = parse_cli_double(argv[i], &param.bias);
      break;
    case 'e':
      bad_opt = parse_cli_double(argv[i], &param.tol);
      break;
    case 'i':
      bad_opt = parse_cli_int(argv[i], &param.max_iter);
      break;
    default:
      fprintf(stderr, "Unknown option: -%c\n", argv[i - 1][1]);
      print_usage(argv[0]);
      return 1;
    }
    if (bad_opt) {
      fprintf(stderr, "Invalid value for -%c: %s\n", argv[i - 1][1], argv[i]);
      return 1;
    }
    i++;
  }

  if (i >= argc || i + 2 < argc) {
    print_usage(argv[0]);
    return 1;
  }

  const char *train_file = argv[i++];
  char *default_model_file = NULL;
  const char *model_file = NULL;

  if (i < argc) {
    model_file = argv[i++];
  } else {
    size_t train_file_len = strlen(train_file);
    if (train_file_len > (size_t)-1 - sizeof(".model")) {
      fprintf(stderr, "Error: training filename is too long\n");
      return 1;
    }
    default_model_file = (char *)malloc(train_file_len + sizeof(".model"));
    if (!default_model_file) {
      fprintf(stderr, "Error: cannot allocate model filename\n");
      return 1;
    }
    memcpy(default_model_file, train_file, train_file_len);
    memcpy(default_model_file + train_file_len, ".model", sizeof(".model"));
    model_file = default_model_file;
  }

  SvmProblem prob;
  if (svm_read_problem(train_file, &prob) != 0) {
    fprintf(stderr, "Error: cannot read training dataset %s\n", train_file);
    free(default_model_file);
    return 1;
  }

  printf("Loaded %d instances with max dimension %d from %s\n", prob.l,
         prob.max_dim, train_file);
  printf("Training with solver=%s, kernel=%s, C=%.4f, bias=%.2f\n",
         param.solver == SVM_SOLVER_BOX ? "box" : "affine",
         param.kernel == SVM_KERNEL_LINEAR ? "linear" : "rbf", param.C,
         param.bias);

  clock_t t0 = clock();
  SvmModel *model = svm_train(&prob, &param);
  clock_t t1 = clock();

  if (!model) {
    fprintf(stderr, "Training failed!\n");
    svm_free_problem(&prob);
    free(default_model_file);
    return 1;
  }

  double elapsed_ms = (double)(t1 - t0) * 1000.0 / CLOCKS_PER_SEC;
  printf("Training completed in %.2f ms (support vectors: %d)\n", elapsed_ms,
         model->nr_sv);

  /* evaluate training accuracy */
  int correct = 0;
  for (int r = 0; r < prob.l; r++) {
    double pred = svm_predict(model, prob.x[r]);
    if (pred == prob.y[r])
      correct++;
  }
  printf("Training accuracy: %.2f%% (%d/%d)\n",
         (double)correct * 100.0 / prob.l, correct, prob.l);

  int status = 0;
  if (svm_save_model(model_file, model) != 0) {
    fprintf(stderr, "Error: cannot save model to %s\n", model_file);
    status = 1;
  } else {
    printf("Model saved to %s\n", model_file);
  }

  svm_free_model(model);
  svm_free_problem(&prob);
  free(default_model_file);
  return status;
}
