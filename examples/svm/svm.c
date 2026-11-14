#include "svm.h"
#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void svm_param_init(SvmParam *param) {
  if (!param)
    return;
  param->solver = SVM_SOLVER_BOX;
  param->kernel = SVM_KERNEL_LINEAR;
  param->strategy = NANOQSP_STRATEGY_COORDINATE_DESCENT;
  param->C = 1.0;
  param->gamma = 0.0;
  param->bias = 1.0;
  param->tol = 1e-4;
  param->max_iter = 1000;
}

/* return 1 for a line, 0 for EOF, and -1 for a read/allocation error */
static int read_line(FILE *fp, char **buf, size_t *cap) {
  if (!*buf) {
    *cap = 1024;
    *buf = (char *)malloc(*cap);
    if (!*buf)
      return -1;
  }

  size_t len = 0;
  int ch;
  while ((ch = fgetc(fp)) != EOF) {
    if (ch == '\0')
      return -1;
    if (len + 1 >= *cap) {
      if (*cap > SIZE_MAX / 2)
        return -1;
      size_t new_cap = *cap * 2;
      char *new_buf = (char *)realloc(*buf, new_cap);
      if (!new_buf)
        return -1;
      *buf = new_buf;
      *cap = new_cap;
    }
    (*buf)[len++] = (char)ch;
    if (ch == '\n')
      break;
  }
  if (ferror(fp))
    return -1;
  (*buf)[len] = '\0';
  return len > 0;
}

static int parse_double(const char *token, double *value) {
  if (!token)
    return -1;
  char *end = NULL;
  errno = 0;
  double parsed = strtod(token, &end);
  if (end == token || *end != '\0' || !isfinite(parsed) ||
      (errno == ERANGE && parsed == 0.0))
    return -1;
  *value = parsed;
  return 0;
}

static int parse_nonnegative_int(const char *token, int *value) {
  if (!token)
    return -1;
  char *end = NULL;
  errno = 0;
  long parsed = strtol(token, &end, 10);
  if (end == token || *end != '\0' || errno == ERANGE || parsed < 0 ||
      parsed > INT_MAX)
    return -1;
  *value = (int)parsed;
  return 0;
}

static int parse_node(char *token, int prev_index, SvmNode *node) {
  char *colon = strchr(token, ':');
  if (!colon)
    return -1;
  *colon = '\0';
  if (parse_nonnegative_int(token, &node->index) != 0 ||
      node->index <= prev_index || parse_double(colon + 1, &node->value) != 0)
    return -1;
  return 0;
}

static double dot_sparse(const SvmNode *px, const SvmNode *py) {
  double sum = 0.0;
  while (px->index != -1 && py->index != -1) {
    if (px->index == py->index) {
      sum += px->value * py->value;
      px++;
      py++;
    } else if (px->index > py->index) {
      py++;
    } else {
      px++;
    }
  }
  return sum;
}

static double norm_sq_sparse(const SvmNode *px) {
  double sum = 0.0;
  while (px->index != -1) {
    sum += px->value * px->value;
    px++;
  }
  return sum;
}

static double distance_sq_sparse(const SvmNode *px, const SvmNode *py) {
  double sum = 0.0;
  while (px->index != -1 && py->index != -1) {
    if (px->index == py->index) {
      double diff = px->value - py->value;
      sum += diff * diff;
      px++;
      py++;
    } else if (px->index > py->index) {
      sum += py->value * py->value;
      py++;
    } else {
      sum += px->value * px->value;
      px++;
    }
  }
  return sum + norm_sq_sparse(px) + norm_sq_sparse(py);
}

static double kernel_eval(const SvmNode *x1, const SvmNode *x2,
                          const SvmParam *param) {
  if (param->kernel == SVM_KERNEL_RBF) {
    double dist_sq = distance_sq_sparse(x1, x2);
    return exp(-param->gamma * dist_sq);
  }
  return dot_sparse(x1, x2);
}

static SvmNode *clone_sparse(const SvmNode *src) {
  int n = 0;
  while (src[n].index != -1)
    n++;
  SvmNode *dst = (SvmNode *)malloc((size_t)(n + 1) * sizeof(SvmNode));
  if (!dst)
    return NULL;
  memcpy(dst, src, (size_t)(n + 1) * sizeof(SvmNode));
  return dst;
}

int svm_read_problem(const char *filename, SvmProblem *prob) {
  if (!prob)
    return -1;
  memset(prob, 0, sizeof(*prob));
  if (!filename)
    return -1;

  FILE *fp = fopen(filename, "r");
  if (!fp)
    return -1;

  size_t elements = 0;
  char *line = NULL;
  size_t line_cap = 0;
  SvmNode *space = NULL;
  int rc;

  while ((rc = read_line(fp, &line, &line_cap)) > 0) {
    char *p = strtok(line, " \t\n\r");
    if (!p)
      continue;
    double label;
    if (parse_double(p, &label) != 0 || prob->l == INT_MAX)
      goto fail;
    prob->l++;
    int prev_idx = 0;
    while ((p = strtok(NULL, " \t\n\r")) != NULL) {
      SvmNode node;
      if (parse_node(p, prev_idx, &node) != 0 ||
          elements == SIZE_MAX / sizeof(SvmNode))
        goto fail;
      prev_idx = node.index;
      if (node.index > prob->max_dim)
        prob->max_dim = node.index;
      elements++;
    }
    if (elements == SIZE_MAX / sizeof(SvmNode))
      goto fail;
    elements++; /* end-of-row sentinel */
  }

  if (rc < 0 || prob->l == 0 || fseek(fp, 0, SEEK_SET) != 0 ||
      (size_t)prob->l > SIZE_MAX / sizeof(double) ||
      (size_t)prob->l > SIZE_MAX / sizeof(SvmNode *))
    goto fail;

  prob->y = (double *)malloc((size_t)prob->l * sizeof(double));
  prob->x = (SvmNode **)calloc((size_t)prob->l, sizeof(SvmNode *));
  space = (SvmNode *)malloc(elements * sizeof(SvmNode));
  if (!prob->y || !prob->x || !space)
    goto fail;

  size_t j = 0;
  int row = 0;
  while ((rc = read_line(fp, &line, &line_cap)) > 0) {
    char *p = strtok(line, " \t\n\r");
    if (!p)
      continue;
    if (row >= prob->l || parse_double(p, &prob->y[row]) != 0)
      goto fail;
    prob->x[row] = &space[j];
    int prev_idx = 0;
    while ((p = strtok(NULL, " \t\n\r")) != NULL) {
      SvmNode node;
      if (parse_node(p, prev_idx, &node) != 0 || j >= elements ||
          node.index > prob->max_dim)
        goto fail;
      prev_idx = node.index;
      space[j++] = node;
    }
    if (j >= elements)
      goto fail;
    space[j].index = -1;
    space[j].value = 0.0;
    j++;
    row++;
  }
  if (rc < 0 || row != prob->l || j != elements)
    goto fail;

  free(line);
  fclose(fp);
  return 0;

fail:
  free(line);
  free(space);
  free(prob->x);
  free(prob->y);
  memset(prob, 0, sizeof(*prob));
  fclose(fp);
  return -1;
}

void svm_free_problem(SvmProblem *prob) {
  if (!prob)
    return;
  if (prob->y) {
    free(prob->y);
    prob->y = NULL;
  }
  if (prob->x) {
    if (prob->x[0])
      free(prob->x[0]);
    free(prob->x);
    prob->x = NULL;
  }
  prob->l = 0;
  prob->max_dim = 0;
}

static int validate_param(const SvmParam *param) {
  if (!param)
    return 0;
  if (param->solver != SVM_SOLVER_BOX && param->solver != SVM_SOLVER_AFFINE)
    return 0;
  if (param->kernel != SVM_KERNEL_LINEAR && param->kernel != SVM_KERNEL_RBF)
    return 0;
  if (param->strategy < NANOQSP_STRATEGY_COORDINATE_DESCENT ||
      param->strategy > NANOQSP_STRATEGY_ADMM)
    return 0;
  if (!isfinite(param->C) || param->C <= 0.0)
    return 0;
  if (!isfinite(param->gamma) || param->gamma < 0.0)
    return 0;
  if (!isfinite(param->bias) || param->bias < 0.0 ||
      (param->solver == SVM_SOLVER_BOX && !isfinite(param->bias * param->bias)))
    return 0;
  if (!isfinite(param->tol) || param->tol <= 0.0)
    return 0;
  if (param->max_iter <= 0)
    return 0;
  return 1;
}

SvmModel *svm_train(const SvmProblem *prob, const SvmParam *param) {
  if (!prob || !prob->y || !prob->x || prob->l <= 0 || prob->max_dim < 0 ||
      !validate_param(param))
    return NULL;

  int l = prob->l;
  if ((size_t)l > SIZE_MAX / sizeof(double) / (size_t)l)
    return NULL;

  /* detect binary labels */
  double label_pos = prob->y[0];
  double label_neg = 0.0;
  int found_second = 0;

  for (int i = 1; i < prob->l; i++) {
    if (prob->y[i] != label_pos) {
      if (!found_second) {
        label_neg = prob->y[i];
        found_second = 1;
      } else if (prob->y[i] != label_neg) {
        /* only binary classification supported currently */
        return NULL;
      }
    }
  }

  if (!found_second)
    return NULL;

  /* ensure positive class label convention */
  if (label_pos < label_neg) {
    double tmp = label_pos;
    label_pos = label_neg;
    label_neg = tmp;
  }

  SvmParam eff_param = *param;
  if (eff_param.kernel == SVM_KERNEL_RBF && eff_param.gamma == 0.0) {
    eff_param.gamma = prob->max_dim > 0 ? (1.0 / (double)prob->max_dim) : 1.0;
  }

  double *y_bin = (double *)malloc((size_t)l * sizeof(double));
  double *D = (double *)malloc((size_t)l * (size_t)l * sizeof(double));
  double *d_vec = (double *)malloc((size_t)l * sizeof(double));
  double *lb = (double *)malloc((size_t)l * sizeof(double));
  double *ub = (double *)malloc((size_t)l * sizeof(double));
  double *alpha = (double *)calloc((size_t)l, sizeof(double));

  if (!y_bin || !D || !d_vec || !lb || !ub || !alpha) {
    free(y_bin);
    free(D);
    free(d_vec);
    free(lb);
    free(ub);
    free(alpha);
    return NULL;
  }

  for (int i = 0; i < l; i++) {
    y_bin[i] = (prob->y[i] == label_pos) ? +1.0 : -1.0;
    d_vec[i] = 1.0;
    lb[i] = 0.0;
    ub[i] = eff_param.C;
  }

  /* compute kernel gram matrix */
  double bias_sq = 0.0;
  if (eff_param.solver == SVM_SOLVER_BOX && eff_param.bias > 0.0) {
    bias_sq = eff_param.bias * eff_param.bias;
  }

  double max_diag = 0.0;
  for (int i = 0; i < l; i++) {
    for (int j = i; j < l; j++) {
      double k_val = kernel_eval(prob->x[i], prob->x[j], &eff_param) + bias_sq;
      double q_ij = y_bin[i] * y_bin[j] * k_val;
      if (!isfinite(q_ij)) {
        free(y_bin);
        free(D);
        free(d_vec);
        free(lb);
        free(ub);
        free(alpha);
        return NULL;
      }
      D[(size_t)i * (size_t)l + (size_t)j] = q_ij;
      D[(size_t)j * (size_t)l + (size_t)i] = q_ij;
    }
    double diag_i = D[(size_t)i * (size_t)l + (size_t)i];
    if (diag_i > max_diag)
      max_diag = diag_i;
  }

  /* relative diagonal jitter ensures strict SPD across kernel scales */
  double jitter = (max_diag > 0.0) ? (1e-9 * max_diag) : 1e-9;
  for (int i = 0; i < l; i++)
    D[(size_t)i * (size_t)l + (size_t)i] += jitter;

  NanoqspConfig config;
  memset(&config, 0, sizeof(config));
  config.max_iterations = eff_param.max_iter;
  config.tolerance = eff_param.tol;

  int status = 0;
  if (eff_param.solver == SVM_SOLVER_BOX) {
    config.strategy = eff_param.strategy;
    status = nanoqsp_solve_box(l, D, d_vec, lb, ub, alpha, &config);
  } else {
    /* affine equality constraint: sum y_i alpha_i = 0 */
    config.strategy = NANOQSP_STRATEGY_ADMM;
    double zero = 0.0;
    status = nanoqsp_solve_affine(l, 1, D, d_vec, y_bin, lb, ub, &zero, &zero,
                                  alpha, &config);
  }

  if (status < 0 && status != NANOQSP_ERR_MAX_ITERATIONS) {
    free(y_bin);
    free(D);
    free(d_vec);
    free(lb);
    free(ub);
    free(alpha);
    return NULL;
  }

  /* assemble trained model */
  SvmModel *model = (SvmModel *)calloc(1, sizeof(SvmModel));
  if (!model) {
    free(y_bin);
    free(D);
    free(d_vec);
    free(lb);
    free(ub);
    free(alpha);
    return NULL;
  }

  model->param = eff_param;
  model->labels[0] = label_pos;
  model->labels[1] = label_neg;
  model->max_dim = prob->max_dim;

  double max_alpha = 0.0;
  for (int i = 0; i < l; i++) {
    if (alpha[i] > max_alpha)
      max_alpha = alpha[i];
  }
  double sv_tol = 1e-10 * (max_alpha > 0.0 ? max_alpha : eff_param.C);
  int nr_sv = 0;
  for (int i = 0; i < l; i++) {
    if (alpha[i] > sv_tol)
      nr_sv++;
  }
  model->nr_sv = nr_sv;

  /* calculate bias */
  if (eff_param.solver == SVM_SOLVER_BOX) {
    if (eff_param.bias > 0.0) {
      double sum_ya = 0.0;
      for (int i = 0; i < l; i++) {
        sum_ya += y_bin[i] * alpha[i];
      }
      model->bias = bias_sq * sum_ya;
    } else {
      model->bias = 0.0;
    }
  } else {
    /* affine solver: compute bias from free SVs or KKT bound interval */
    double bias_sum = 0.0;
    int free_sv_count = 0;
    double b_low = -DBL_MAX;
    double b_up = DBL_MAX;
    double tol_lo = 1e-4 * (max_alpha > 0.0 ? max_alpha : eff_param.C);
    double tol_hi = 1e-4 * eff_param.C;

    for (int i = 0; i < l; i++) {
      double d_alpha_i = -jitter * alpha[i];
      const double *D_row = &D[(size_t)i * (size_t)l];
      for (int j = 0; j < l; j++) {
        d_alpha_i += D_row[j] * alpha[j];
      }
      /* since D_ij = y_i y_j K(x_i, x_j), y_i - f_i = y_i (1 - (D alpha)_i) */
      double r_i = y_bin[i] * (1.0 - d_alpha_i);

      if (alpha[i] > tol_lo && alpha[i] < eff_param.C - tol_hi) {
        bias_sum += r_i;
        free_sv_count++;
      } else {
        if ((alpha[i] <= tol_lo && y_bin[i] < 0.0) ||
            (alpha[i] >= eff_param.C - tol_hi && y_bin[i] > 0.0)) {
          if (r_i < b_up)
            b_up = r_i;
        }
        if ((alpha[i] <= tol_lo && y_bin[i] > 0.0) ||
            (alpha[i] >= eff_param.C - tol_hi && y_bin[i] < 0.0)) {
          if (r_i > b_low)
            b_low = r_i;
        }
      }
    }
    if (free_sv_count > 0) {
      model->bias = bias_sum / (double)free_sv_count;
    } else if (b_low > -DBL_MAX / 2.0 && b_up < DBL_MAX / 2.0) {
      model->bias = 0.5 * (b_low + b_up);
    } else {
      model->bias = 0.0;
    }
  }

  if (eff_param.kernel == SVM_KERNEL_LINEAR) {
    /* condense primal weight vector for linear kernel */
    model->w = (double *)calloc((size_t)prob->max_dim + 1, sizeof(double));
    if (!model->w) {
      svm_free_model(model);
      free(y_bin);
      free(D);
      free(d_vec);
      free(lb);
      free(ub);
      free(alpha);
      return NULL;
    }
    for (int i = 0; i < l; i++) {
      if (alpha[i] > sv_tol) {
        double ya = y_bin[i] * alpha[i];
        const SvmNode *p = prob->x[i];
        while (p->index != -1) {
          if (p->index >= 1 && p->index <= prob->max_dim) {
            model->w[p->index] += ya * p->value;
          }
          p++;
        }
      }
    }
  } else if (nr_sv > 0) {
    /* store support vectors for non-linear kernel */
    model->alpha = (double *)malloc((size_t)nr_sv * sizeof(double));
    model->SV = (SvmNode **)calloc((size_t)nr_sv, sizeof(SvmNode *));
    if (!model->alpha || !model->SV) {
      svm_free_model(model);
      free(y_bin);
      free(D);
      free(d_vec);
      free(lb);
      free(ub);
      free(alpha);
      return NULL;
    }
    int sv_idx = 0;
    for (int i = 0; i < l; i++) {
      if (alpha[i] > sv_tol) {
        model->alpha[sv_idx] = y_bin[i] * alpha[i];
        model->SV[sv_idx] = clone_sparse(prob->x[i]);
        if (!model->SV[sv_idx]) {
          svm_free_model(model);
          free(y_bin);
          free(D);
          free(d_vec);
          free(lb);
          free(ub);
          free(alpha);
          return NULL;
        }
        sv_idx++;
      }
    }
  }

  free(y_bin);
  free(D);
  free(d_vec);
  free(lb);
  free(ub);
  free(alpha);
  return model;
}

double svm_predict_values(const SvmModel *model, const SvmNode *x,
                          double *dec_value) {
  if (!model || !x)
    return 0.0;
  double f = model->bias;

  if (model->w && model->param.kernel == SVM_KERNEL_LINEAR) {
    /* condensed linear model dot product */
    const SvmNode *p = x;
    while (p->index != -1) {
      if (p->index >= 1 && p->index <= model->max_dim) {
        f += model->w[p->index] * p->value;
      }
      p++;
    }
  } else if (model->SV && model->alpha) {
    /* kernel support vector evaluation */
    for (int i = 0; i < model->nr_sv; i++) {
      double k_val = kernel_eval(model->SV[i], x, &model->param);
      f += model->alpha[i] * k_val;
    }
  }

  if (dec_value)
    *dec_value = f;
  return (f >= 0.0) ? model->labels[0] : model->labels[1];
}

double svm_predict(const SvmModel *model, const SvmNode *x) {
  return svm_predict_values(model, x, NULL);
}

void svm_free_model(SvmModel *model) {
  if (!model)
    return;
  free(model->w);
  free(model->alpha);
  if (model->SV) {
    for (int i = 0; i < model->nr_sv; i++) {
      free(model->SV[i]);
    }
    free(model->SV);
  }
  free(model);
}

int svm_save_model(const char *filename, const SvmModel *model) {
  if (!filename || !model)
    return -1;
  FILE *fp = fopen(filename, "w");
  if (!fp)
    return -1;

  fprintf(fp, "solver %d\n", model->param.solver);
  fprintf(fp, "kernel %d\n", model->param.kernel);
  fprintf(fp, "gamma %.17g\n", model->param.gamma);
  fprintf(fp, "bias %.17g\n", model->bias);
  fprintf(fp, "labels %.17g %.17g\n", model->labels[0], model->labels[1]);
  fprintf(fp, "nr_sv %d\n", model->nr_sv);
  fprintf(fp, "max_dim %d\n", model->max_dim);

  if (model->w) {
    fprintf(fp, "w");
    for (int i = 1; i <= model->max_dim; i++) {
      fprintf(fp, " %.17g", model->w[i]);
    }
    fprintf(fp, "\n");
  }

  if (model->SV && model->alpha) {
    fprintf(fp, "SV\n");
    for (int i = 0; i < model->nr_sv; i++) {
      fprintf(fp, "%.17g", model->alpha[i]);
      const SvmNode *p = model->SV[i];
      while (p->index != -1) {
        fprintf(fp, " %d:%.17g", p->index, p->value);
        p++;
      }
      fprintf(fp, "\n");
    }
  }

  int failed = ferror(fp);
  if (fclose(fp) != 0)
    failed = 1;
  return failed ? -1 : 0;
}

SvmModel *svm_load_model(const char *filename) {
  if (!filename)
    return NULL;
  FILE *fp = fopen(filename, "r");
  if (!fp)
    return NULL;

  SvmModel *model = (SvmModel *)calloc(1, sizeof(SvmModel));
  if (!model) {
    fclose(fp);
    return NULL;
  }

  enum {
    HEADER_SOLVER = 1 << 0,
    HEADER_KERNEL = 1 << 1,
    HEADER_GAMMA = 1 << 2,
    HEADER_BIAS = 1 << 3,
    HEADER_LABELS = 1 << 4,
    HEADER_NR_SV = 1 << 5,
    HEADER_MAX_DIM = 1 << 6,
    HEADER_ALL = (1 << 7) - 1
  };
  unsigned headers = 0;
  int body = 0; /* 1: linear weights, 2: support vectors */
  int loaded_sv = 0;
  char *line = NULL;
  size_t line_cap = 0;
  SvmNode *buf = NULL;
  size_t buf_cap = 0;
  int rc;

  while ((rc = read_line(fp, &line, &line_cap)) > 0) {
    char *header = strtok(line, " \t\n\r");
    if (!header)
      continue;
    unsigned field = 0;
    int number;
    if (strcmp(header, "solver") == 0) {
      field = HEADER_SOLVER;
      if (parse_nonnegative_int(strtok(NULL, " \t\n\r"), &number) != 0 ||
          (number != SVM_SOLVER_BOX && number != SVM_SOLVER_AFFINE))
        goto fail;
      model->param.solver = (SvmSolverType)number;
    } else if (strcmp(header, "kernel") == 0) {
      field = HEADER_KERNEL;
      if (parse_nonnegative_int(strtok(NULL, " \t\n\r"), &number) != 0 ||
          (number != SVM_KERNEL_LINEAR && number != SVM_KERNEL_RBF))
        goto fail;
      model->param.kernel = (SvmKernelType)number;
    } else if (strcmp(header, "gamma") == 0) {
      field = HEADER_GAMMA;
      if (parse_double(strtok(NULL, " \t\n\r"), &model->param.gamma) != 0)
        goto fail;
    } else if (strcmp(header, "bias") == 0) {
      field = HEADER_BIAS;
      if (parse_double(strtok(NULL, " \t\n\r"), &model->bias) != 0)
        goto fail;
    } else if (strcmp(header, "labels") == 0) {
      field = HEADER_LABELS;
      if (parse_double(strtok(NULL, " \t\n\r"), &model->labels[0]) != 0 ||
          parse_double(strtok(NULL, " \t\n\r"), &model->labels[1]) != 0)
        goto fail;
    } else if (strcmp(header, "nr_sv") == 0) {
      field = HEADER_NR_SV;
      if (parse_nonnegative_int(strtok(NULL, " \t\n\r"), &model->nr_sv) != 0)
        goto fail;
    } else if (strcmp(header, "max_dim") == 0) {
      field = HEADER_MAX_DIM;
      if (parse_nonnegative_int(strtok(NULL, " \t\n\r"), &model->max_dim) != 0)
        goto fail;
    } else if (strcmp(header, "w") == 0) {
      if (headers != HEADER_ALL || model->param.kernel != SVM_KERNEL_LINEAR ||
          (size_t)model->max_dim + 1 > SIZE_MAX / sizeof(double))
        goto fail;
      model->w = (double *)calloc((size_t)model->max_dim + 1, sizeof(double));
      if (!model->w)
        goto fail;
      for (size_t i = 1; i <= (size_t)model->max_dim; i++) {
        if (parse_double(strtok(NULL, " \t\n\r"), &model->w[i]) != 0)
          goto fail;
      }
      if (strtok(NULL, " \t\n\r"))
        goto fail;
      body = 1;
      break;
    } else if (strcmp(header, "SV") == 0) {
      if (headers != HEADER_ALL || model->param.kernel != SVM_KERNEL_RBF ||
          strtok(NULL, " \t\n\r"))
        goto fail;
      body = 2;
      break;
    } else {
      goto fail;
    }
    if ((headers & field) || strtok(NULL, " \t\n\r"))
      goto fail;
    headers |= field;
  }

  if (rc < 0 || headers != HEADER_ALL || model->labels[0] == model->labels[1] ||
      (model->param.kernel == SVM_KERNEL_LINEAR && body != 1) ||
      (model->param.kernel == SVM_KERNEL_RBF &&
       (model->param.gamma <= 0.0 || (model->nr_sv > 0 && body != 2))))
    goto fail;

  if (body == 2 && model->nr_sv > 0) {
    if ((size_t)model->nr_sv > SIZE_MAX / sizeof(double) ||
        (size_t)model->nr_sv > SIZE_MAX / sizeof(SvmNode *))
      goto fail;
    model->alpha = (double *)malloc((size_t)model->nr_sv * sizeof(double));
    model->SV = (SvmNode **)calloc((size_t)model->nr_sv, sizeof(SvmNode *));
    if (!model->alpha || !model->SV)
      goto fail;

    buf_cap = 64;
    buf = (SvmNode *)malloc(buf_cap * sizeof(SvmNode));
    if (!buf)
      goto fail;
    for (int i = 0; i < model->nr_sv; i++) {
      if (read_line(fp, &line, &line_cap) != 1 ||
          parse_double(strtok(line, " \t\n\r"), &model->alpha[i]) != 0)
        goto fail;
      size_t n_nodes = 0;
      int prev_idx = 0;
      char *p;
      while ((p = strtok(NULL, " \t\n\r")) != NULL) {
        SvmNode node;
        if (parse_node(p, prev_idx, &node) != 0 || node.index > model->max_dim)
          goto fail;
        prev_idx = node.index;
        if (n_nodes + 1 >= buf_cap) {
          if (buf_cap > SIZE_MAX / sizeof(SvmNode) / 2)
            goto fail;
          size_t new_cap = buf_cap * 2;
          SvmNode *new_buf = (SvmNode *)realloc(buf, new_cap * sizeof(SvmNode));
          if (!new_buf)
            goto fail;
          buf = new_buf;
          buf_cap = new_cap;
        }
        buf[n_nodes++] = node;
      }
      buf[n_nodes].index = -1;
      buf[n_nodes].value = 0.0;
      n_nodes++;
      model->SV[i] = (SvmNode *)malloc(n_nodes * sizeof(SvmNode));
      if (!model->SV[i])
        goto fail;
      loaded_sv++;
      memcpy(model->SV[i], buf, n_nodes * sizeof(SvmNode));
    }
  }

  /* no metadata or additional data is allowed after the model body */
  while ((rc = read_line(fp, &line, &line_cap)) > 0) {
    if (strtok(line, " \t\n\r"))
      goto fail;
  }
  if (rc < 0)
    goto fail;

  free(line);
  free(buf);
  fclose(fp);
  return model;

fail:
  free(line);
  free(buf);
  fclose(fp);
  if (model->SV)
    model->nr_sv = loaded_sv;
  svm_free_model(model);
  return NULL;
}
