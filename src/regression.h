typedef struct {
  double *betas;
  int *indices;
  int count;
} Sparse_Betas;

// TODO: maybe this should be sparse?
typedef struct {
  long count;
  Sparse_Betas *betas;
  double *lambdas;
  long vec_length;
} Beta_Sequence;

typedef struct {
  double actual_diff;
  double pre_lambda_diff;
} Changes;

double *simple_coordinate_descent_lasso(
    XMatrix X, double *Y, int n, int p, long max_interaction_distance,
    double lambda_min, double lambda_max, int max_iter, int VERBOSE,
    double frac_overlap_allowed, double halt_beta_diff,
    enum LOG_LEVEL log_level, char **job_args, int job_args_num,
    int use_adaptive_calibration, int max_nz_beta);
double update_intercept_cyclic(double intercept, int **X, double *Y,
                               double *beta, int n, int p);
// Changes update_beta_cyclic(XMatrixSparse xmatrix_sparse, double *Y,
Changes update_beta_cyclic(S8bCol col, double *Y, double *rowsum, int n, int p,
                           double lambda, double *beta, long k,
                           double intercept, int_pair *precalc_get_num,
                           int *column_cache);
Changes update_beta_cyclic_old(XMatrixSparse xmatrix_sparse, double *Y,
                               double *rowsum, int n, int p, double lambda,
                               double *beta, long k, double intercept,
                               int_pair *precalc_get_num, int *column_cache);
double soft_threshold(double z, double gamma);