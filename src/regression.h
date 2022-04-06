#ifndef reg_h
#define reg_h
//TODO: no reason this can't be an array
typedef struct {
    // robin_hood::unordered_flat_map<int_fast64_t, float> betas;
    // Beta_Value_Sets *beta_sets;
    int_fast64_t* indices;
    float* values;
    int_fast64_t count;
} Sparse_Betas;

// TODO: maybe this should be sparse?
typedef struct {
    int_fast64_t count;
    Sparse_Betas* betas;
    // int_fast64_t* values;
    float* lambdas;
    int_fast64_t vec_length;
} Beta_Sequence;

typedef struct {
    float actual_diff;
    float pre_lambda_diff;
    bool added;
    bool removed;
} Changes;

Lasso_Result simple_coordinate_descent_lasso(
    XMatrix X, float* Y, int_fast64_t n, int_fast64_t p, int_fast64_t max_interaction_distance,
    float lambda_min, float lambda_max, int_fast64_t max_iter, const bool VERBOSE,
    float halt_beta_diff,
    enum LOG_LEVEL log_level, const char** job_args, int_fast64_t job_args_num,
    int_fast64_t max_nz_beta, const char* log_filename, int_fast64_t depth, const bool estimate_unbiased, const bool use_intercept, const bool check_duplicates, struct continuous_info* cont_inf);
float update_intercept_cyclic(float intercept, int_fast64_t** X, float* Y,
    robin_hood::unordered_flat_map<int_fast64_t, float>* beta, int_fast64_t n, int_fast64_t p);
// Changes update_beta_cyclic(XMatrixSparse xmatrix_sparse, float *Y,
Changes update_beta_cyclic(AS_Entry* entry, float* Y, float* rowsum, int_fast64_t n, int_fast64_t p,
    float lambda, robin_hood::unordered_flat_map<int_fast64_t, float>* beta, int_fast64_t k,
    float intercept, int_fast64_t* column_cache, struct continuous_info* ci);
Changes update_beta_cyclic_old(XMatrixSparse xmatrix_sparse, float* Y,
    float* rowsum, int_fast64_t n, int_fast64_t p, float lambda,
    robin_hood::unordered_flat_map<int_fast64_t, float>* beta, int_fast64_t k, float intercept,
    int_pair* precalc_get_num, int_fast64_t* column_cache);

float soft_threshold(float z, float gamma);

float calculate_error(float* Y, float* rowsum, int_fast64_t n);

int_fast64_t adaptive_calibration_check_beta(float c_bar, float lambda_1,
    Sparse_Betas* beta_1, float lambda_2,
    Sparse_Betas* beta_2, int_fast64_t n);

#endif