#include "svm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char *prog) {
  printf("Usage: %s test_file model_file [output_file]\n", prog);
}

int main(int argc, char **argv) {
  if (argc < 3) {
    print_usage(argv[0]);
    return 1;
  }

  const char *test_file = argv[1];
  const char *model_file = argv[2];
  const char *out_file = (argc >= 4) ? argv[3] : NULL;

  SvmModel *model = svm_load_model(model_file);
  if (!model) {
    fprintf(stderr, "Error: cannot load model from %s\n", model_file);
    return 1;
  }

  SvmProblem prob;
  if (svm_read_problem(test_file, &prob) != 0) {
    fprintf(stderr, "Error: cannot read test file %s\n", test_file);
    svm_free_model(model);
    return 1;
  }

  FILE *out_fp = NULL;
  if (out_file) {
    out_fp = fopen(out_file, "w");
    if (!out_fp) {
      fprintf(stderr, "Error: cannot open output file %s for writing\n",
              out_file);
      svm_free_problem(&prob);
      svm_free_model(model);
      return 1;
    }
  }

  int correct = 0;
  int total = prob.l;
  int status = 0;

  for (int i = 0; i < total; i++) {
    double dec_val = 0.0;
    double pred = svm_predict_values(model, prob.x[i], &dec_val);
    if (pred == prob.y[i])
      correct++;
    if (out_fp && fprintf(out_fp, "%.17g %.17g\n", pred, dec_val) < 0) {
      status = 1;
      break;
    }
  }

  if (out_fp && fclose(out_fp) != 0)
    status = 1;
  if (status) {
    fprintf(stderr, "Error: cannot write output file %s\n", out_file);
  } else {
    printf("Accuracy = %.2f%% (%d/%d)\n", (double)correct * 100.0 / total,
           correct, total);
  }

  svm_free_problem(&prob);
  svm_free_model(model);
  return status;
}
