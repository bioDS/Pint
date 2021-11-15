//TODO: no reason this can't be an array
typedef struct {
    // robin_hood::unordered_flat_map<long, float> betas;
    // Beta_Value_Sets *beta_sets;
    long* indices;
    float* values;
    long count;
} Sparse_Betas;

// TODO: maybe this should be sparse?
typedef struct {
    long count;
    Sparse_Betas* betas;
    // long* values;
    float* lambdas;
    long vec_length;
} Beta_Sequence;

typedef struct {
    float actual_diff;
    float pre_lambda_diff;
    bool added;
    bool removed;
} Changes;

Lasso_Result simple_coordinate_descent_lasso(
    XMatrix X, float* Y, long n, long p, long max_interaction_distance,
    float lambda_min, float lambda_max, long max_iter, long VERBOSE,
    float frac_overlap_allowed, float halt_beta_diff,
    enum LOG_LEVEL log_level, const char** job_args, long job_args_num,
    long use_adaptive_calibration, long max_nz_beta, const char* log_filename, long depth, char estimate_unbiased, char use_intercept);
float update_intercept_cyclic(float intercept, long** X, float* Y,
    robin_hood::unordered_flat_map<long, float>* beta, long n, long p);
// Changes update_beta_cyclic(XMatrixSparse xmatrix_sparse, float *Y,
Changes update_beta_cyclic(S8bCol col, float* Y, float* rowsum, long n, long p,
    float lambda, robin_hood::unordered_flat_map<long, float>* beta, long k,
    float intercept, long* column_cache);
Changes update_beta_cyclic_old(XMatrixSparse xmatrix_sparse, float* Y,
    float* rowsum, long n, long p, float lambda,
    robin_hood::unordered_flat_map<long, float>* beta, long k, float intercept,
    int_pair* precalc_get_num, long* column_cache);

float soft_threshold(float z, float gamma);

float calculate_error(float* Y, float* rowsum, long n);

long adaptive_calibration_check_beta(float c_bar, float lambda_1,
    Sparse_Betas* beta_1, float lambda_2,
    Sparse_Betas* beta_2, long n);
