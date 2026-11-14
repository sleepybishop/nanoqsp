#ifndef SVM_H
#define SVM_H

#include "nanoqsp.h"

#ifdef __cplusplus
extern "C" {
#endif

/* sparse feature node: index is 1-based, index -1 marks end */
typedef struct {
  int index;
  double value;
} SvmNode;

/* training dataset */
typedef struct {
  int l;
  int max_dim;
  double *y;
  SvmNode **x;
} SvmProblem;

/* solver choice */
typedef enum { SVM_SOLVER_BOX = 0, SVM_SOLVER_AFFINE = 1 } SvmSolverType;

/* kernel choice */
typedef enum { SVM_KERNEL_LINEAR = 0, SVM_KERNEL_RBF = 1 } SvmKernelType;

/* training parameters */
typedef struct {
  SvmSolverType solver;
  SvmKernelType kernel;
  NanoqspStrategy strategy;
  double C;
  double gamma;
  double bias;
  double tol;
  int max_iter;
} SvmParam;

/* trained model */
typedef struct {
  SvmParam param;
  double labels[2];
  int nr_sv;
  int max_dim;
  double bias;
  double *w;
  double *alpha;
  SvmNode **SV;
} SvmModel;

/* initialize default parameters */
void svm_param_init(SvmParam *param);

/* load problem from libsvm/liblinear sparse text format */
int svm_read_problem(const char *filename, SvmProblem *prob);
void svm_free_problem(SvmProblem *prob);

/* train model using nanoqsp */
SvmModel *svm_train(const SvmProblem *prob, const SvmParam *param);

/* predict label and decision value */
double svm_predict(const SvmModel *model, const SvmNode *x);
double svm_predict_values(const SvmModel *model, const SvmNode *x,
                          double *dec_value);

/* model serialization */
int svm_save_model(const char *filename, const SvmModel *model);
SvmModel *svm_load_model(const char *filename);
void svm_free_model(SvmModel *model);

#ifdef __cplusplus
}
#endif

#endif /* SVM_H */
