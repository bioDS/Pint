typedef struct {
  ska::flat_hash_map<long, float> betas;
  int *indices;
  int count;
} Sparse_Betas;

// TODO: maybe this should be sparse?
typedef struct {
  long count;
  Sparse_Betas *betas;
  float *lambdas;
  long vec_length;
} Beta_Sequence;

typedef struct {
  float actual_diff;
  float pre_lambda_diff;
} Changes;

ska::flat_hash_map<long, float> simple_coordinate_descent_lasso(
    XMatrix X, float *Y, int n, int p, long max_interaction_distance,
    float lambda_min, float lambda_max, int max_iter, int VERBOSE,
    float frac_overlap_allowed, float halt_beta_diff,
    enum LOG_LEVEL log_level, char **job_args, int job_args_num,
    int use_adaptive_calibration, int max_nz_beta);
float update_intercept_cyclic(float intercept, int **X, float *Y,
                               ska::flat_hash_map<long, float> beta, int n, int p);
// Changes update_beta_cyclic(XMatrixSparse xmatrix_sparse, float *Y,
Changes update_beta_cyclic(S8bCol col, float *Y, float *rowsum, int n, int p,
                           float lambda, ska::flat_hash_map<long, float> beta, long k,
                           float intercept, int_pair *precalc_get_num,
                           int *column_cache);
Changes update_beta_cyclic_old(XMatrixSparse xmatrix_sparse, float *Y,
                               float *rowsum, int n, int p, float lambda,
                               ska::flat_hash_map<long, float> beta, long k, float intercept,
                               int_pair *precalc_get_num, int *column_cache);
float soft_threshold(float z, float gamma);