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

Beta_Value_Sets simple_coordinate_descent_lasso(
    XMatrix X, float* Y, int n, int p, long max_interaction_distance,
    float lambda_min, float lambda_max, int max_iter, int VERBOSE,
    float frac_overlap_allowed, float halt_beta_diff,
    enum LOG_LEVEL log_level, char** job_args, int job_args_num,
    int use_adaptive_calibration, int max_nz_beta, char* log_filename, int depth);
float update_intercept_cyclic(float intercept, int** X, float* Y,
    robin_hood::unordered_flat_map<long, float>* beta, int n, int p);
// Changes update_beta_cyclic(XMatrixSparse xmatrix_sparse, float *Y,
Changes update_beta_cyclic(S8bCol col, float* Y, float* rowsum, int n, int p,
    float lambda, robin_hood::unordered_flat_map<long, float>* beta, long k,
    float intercept, int_pair* precalc_get_num,
    int* column_cache);
Changes update_beta_cyclic_old(XMatrixSparse xmatrix_sparse, float* Y,
    float* rowsum, int n, int p, float lambda,
    robin_hood::unordered_flat_map<long, float>* beta, long k, float intercept,
    int_pair* precalc_get_num, int* column_cache);
float soft_threshold(float z, float gamma);

int adaptive_calibration_check_beta(float c_bar, float lambda_1,
    Sparse_Betas* beta_1, float lambda_2,
    Sparse_Betas* beta_2, int n);
