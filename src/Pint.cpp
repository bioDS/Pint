#include "liblasso.h"
#include "robin_hood.h"
#include <cstdint>
#include <vector>

struct C_Beta_Sets {
    int_fast64_t main_len;
    int_fast64_t* main_effects;
    int_fast64_t* num_indist_from_main;
    int_fast64_t** indist_from_main;
    float* main_strength;
    int_fast64_t int_len;
    int_fast64_t* int_i;
    int_fast64_t* int_j;
    int_fast64_t* num_indist_from_int;
    int_fast64_t** indist_from_int;
    float* int_strength;
    int_fast64_t trip_len;
    int_fast64_t* trip_a;
    int_fast64_t* trip_b;
    int_fast64_t* trip_c;
    int_fast64_t* num_indist_from_trip;
    int_fast64_t** indist_from_trip;
    float* trip_strength;
};

struct C_Beta_Sets cpp_bs_to_c(
    Beta_Value_Sets* beta_sets,
    robin_hood::unordered_flat_map<int_fast64_t, std::vector<int_fast64_t>>* indist_from_val)
{
    int_fast64_t main_len = beta_sets->beta1.size();
    int_fast64_t int_len = beta_sets->beta2.size();
    int_fast64_t trip_len = beta_sets->beta3.size();
    int_fast64_t p = beta_sets->p;

    int_fast64_t* main_effects = new int_fast64_t[main_len];
    int_fast64_t* num_indist_from_main = new int_fast64_t[main_len];
    int_fast64_t** indist_from_main = new int_fast64_t*[main_len];
    float* main_strength = new float[main_len];
    int_fast64_t* int_i = new int_fast64_t[int_len];
    int_fast64_t* int_j = new int_fast64_t[int_len];
    int_fast64_t* num_indist_from_int = new int_fast64_t[int_len];
    int_fast64_t** indist_from_int = new int_fast64_t*[int_len];
    float* int_strength = new float[int_len];
    int_fast64_t* trip_a = new int_fast64_t[trip_len];
    int_fast64_t* trip_b = new int_fast64_t[trip_len];
    int_fast64_t* trip_c = new int_fast64_t[trip_len];
    int_fast64_t* num_indist_from_trip = new int_fast64_t[trip_len];
    int_fast64_t** indist_from_trip = new int_fast64_t*[trip_len];
    float* trip_strength = new float[trip_len];

    int_fast64_t main_offset = 0;
    for (auto it : beta_sets->beta1) {
        // for (auto it = beta_sets->beta1.begin(); it != beta_sets->beta1.end();
        // it++) {
        std::vector<int_fast64_t> indist_vec;
        if (NULL != indist_from_val)
            indist_vec = (*indist_from_val)[it.first];
        int_fast64_t num_indist = indist_vec.size();
        int_fast64_t* indist_arr = new int_fast64_t[num_indist];
        if (NULL != indist_from_val)
            for (int i = 0; i < num_indist; i++) {
                indist_arr[i] = indist_vec[i];
            }
        num_indist_from_main[main_offset] = num_indist;
        indist_from_main[main_offset] = indist_arr;

        main_effects[main_offset] = it.first + 1;
        main_strength[main_offset] = it.second;
        main_offset++;
    }
    int_fast64_t int_offset = 0;
    for (auto it = beta_sets->beta2.begin(); it != beta_sets->beta2.end(); it++) {
        std::vector<int_fast64_t> indist_vec;
        if (NULL != indist_from_val)
            indist_vec = (*indist_from_val)[it->first];
        int_fast64_t num_indist = indist_vec.size();
        int_fast64_t* indist_arr = new int_fast64_t[num_indist];
        if (NULL != indist_from_val)
            for (int i = 0; i < num_indist; i++) {
                indist_arr[i] = indist_vec[i];
            }
        num_indist_from_int[int_offset] = num_indist;
        indist_from_int[int_offset] = indist_arr;

        int_fast64_t val = it->first;
        std::tuple<int_fast64_t, long> ij = val_to_pair(val, p);
        int_i[int_offset] = std::get<0>(ij) + 1;
        int_j[int_offset] = std::get<1>(ij) + 1;
        int_strength[int_offset] = it->second;
        int_offset++;
    }
    int_fast64_t trip_offset = 0;
    for (auto it = beta_sets->beta3.begin(); it != beta_sets->beta3.end(); it++) {
        std::vector<int_fast64_t> indist_vec;
        if (NULL != indist_from_val)
            indist_vec = (*indist_from_val)[it->first];
        int_fast64_t num_indist = indist_vec.size();
        int_fast64_t* indist_arr = new int_fast64_t[num_indist];
        if (NULL != indist_from_val) {
            for (int i = 0; i < num_indist; i++) {
                indist_arr[i] = indist_vec[i];
            }
        }
        num_indist_from_trip[trip_offset] = num_indist;
        indist_from_trip[trip_offset] = indist_arr;

        int_fast64_t val = it->first;
        std::tuple<int_fast64_t, int_fast64_t, long> ij = val_to_triplet(val, p);
        trip_a[trip_offset] = std::get<0>(ij) + 1;
        trip_b[trip_offset] = std::get<1>(ij) + 1;
        trip_c[trip_offset] = std::get<2>(ij) + 1;
        trip_strength[trip_offset] = it->second;
        trip_offset++;
    }

    return C_Beta_Sets { main_len,
        main_effects,
        num_indist_from_main,
        indist_from_main,
        main_strength,
        int_len,
        int_i,
        int_j,
        num_indist_from_int,
        indist_from_int,
        int_strength,
        trip_len,
        trip_a,
        trip_b,
        trip_c,
        num_indist_from_trip,
        indist_from_trip,
        trip_strength };
}

extern "C" {
#include <R.h>
#include <Rinternals.h>

struct effect {
    int_fast64_t i, j;
    float strength;
};

SEXP process_beta(
    Beta_Value_Sets* beta_sets, float f_intercept,
    robin_hood::unordered_flat_map<int_fast64_t, std::vector<int_fast64_t>>* indist_from_val)
{
    struct C_Beta_Sets cbs = cpp_bs_to_c(beta_sets, indist_from_val);
    int_fast64_t main_count = 0, int_count = 0;
    int_fast64_t total_main_count = cbs.main_len;
    int_fast64_t total_int_count = cbs.int_len;
    int_fast64_t total_trip_count = cbs.trip_len;

    SEXP intercept = PROTECT(allocVector(REALSXP, 1));
    SEXP main_i = PROTECT(allocVector(INTSXP, total_main_count));
    SEXP main_indist = PROTECT(allocVector(VECSXP, total_main_count));
    SEXP main_strength = PROTECT(allocVector(REALSXP, total_main_count));
    SEXP int_i = PROTECT(allocVector(INTSXP, total_int_count));
    SEXP int_j = PROTECT(allocVector(INTSXP, total_int_count));
    SEXP int_indist = PROTECT(allocVector(VECSXP, total_int_count));
    SEXP int_strength = PROTECT(allocVector(REALSXP, total_int_count));
    SEXP trip_a = PROTECT(allocVector(INTSXP, total_trip_count));
    SEXP trip_b = PROTECT(allocVector(INTSXP, total_trip_count));
    SEXP trip_c = PROTECT(allocVector(INTSXP, total_trip_count));
    SEXP trip_indist = PROTECT(allocVector(VECSXP, total_trip_count));
    SEXP trip_strength = PROTECT(allocVector(REALSXP, total_trip_count));
    SEXP all_effects = PROTECT(allocVector(VECSXP, 13));

    for (int_fast64_t i = 0; i < cbs.main_len; i++) {
        SEXP indist = PROTECT(allocVector(INTSXP, cbs.num_indist_from_main[i]));
        for (int j = 0; j < cbs.num_indist_from_main[i]; j++) {
            INTEGER(indist)
            [j] = cbs.indist_from_main[i][j];
        }
        SET_VECTOR_ELT(main_indist, i, indist);
        INTEGER(main_i)
        [i] = cbs.main_effects[i];
        REAL(main_strength)
        [i] = cbs.main_strength[i];
        UNPROTECT(1);
    }
    for (int_fast64_t i = 0; i < cbs.int_len; i++) {
        SEXP indist = PROTECT(allocVector(INTSXP, cbs.num_indist_from_int[i]));
        for (int j = 0; j < cbs.num_indist_from_int[i]; j++) {
            INTEGER(indist)
            [j] = cbs.indist_from_int[i][j];
        }
        SET_VECTOR_ELT(int_indist, i, indist);
        INTEGER(int_i)
        [i] = cbs.int_i[i];
        INTEGER(int_j)
        [i] = cbs.int_j[i];
        REAL(int_strength)
        [i] = cbs.int_strength[i];
        UNPROTECT(1);
    }
    for (int_fast64_t i = 0; i < cbs.trip_len; i++) {
        SEXP indist = PROTECT(allocVector(INTSXP, cbs.num_indist_from_trip[i]));
        for (int j = 0; j < cbs.num_indist_from_trip[i]; j++) {
            INTEGER(indist)
            [j] = cbs.indist_from_trip[i][j];
        }
        SET_VECTOR_ELT(trip_indist, i, indist);
        INTEGER(trip_a)
        [i] = cbs.trip_a[i];
        INTEGER(trip_b)
        [i] = cbs.trip_b[i];
        INTEGER(trip_c)
        [i] = cbs.trip_c[i];
        REAL(trip_strength)
        [i] = cbs.trip_strength[i];
        UNPROTECT(1);
    }

    for (int i = 0; i < cbs.main_len; i++)
        free(cbs.indist_from_main[i]);
    for (int i = 0; i < cbs.int_len; i++)
        free(cbs.indist_from_int[i]);
    for (int i = 0; i < cbs.trip_len; i++)
        free(cbs.indist_from_trip[i]);
    free(cbs.indist_from_main);
    free(cbs.num_indist_from_main);
    free(cbs.indist_from_int);
    free(cbs.num_indist_from_int);
    free(cbs.indist_from_trip);
    free(cbs.num_indist_from_trip);
    free(cbs.int_i);
    free(cbs.int_j);
    free(cbs.trip_a);
    free(cbs.trip_b);
    free(cbs.trip_c);

    REAL(intercept)
    [0] = f_intercept;
    SET_VECTOR_ELT(all_effects, 0, main_i);
    SET_VECTOR_ELT(all_effects, 1, main_strength);
    SET_VECTOR_ELT(all_effects, 2, main_indist);
    SET_VECTOR_ELT(all_effects, 3, int_i);
    SET_VECTOR_ELT(all_effects, 4, int_j);
    SET_VECTOR_ELT(all_effects, 5, int_strength);
    SET_VECTOR_ELT(all_effects, 6, int_indist);
    SET_VECTOR_ELT(all_effects, 7, trip_a);
    SET_VECTOR_ELT(all_effects, 8, trip_b);
    SET_VECTOR_ELT(all_effects, 9, trip_c);
    SET_VECTOR_ELT(all_effects, 10, trip_strength);
    SET_VECTOR_ELT(all_effects, 11, trip_indist);
    SET_VECTOR_ELT(all_effects, 12, intercept);

    UNPROTECT(14);
    return all_effects;
}

SEXP read_log_(SEXP log_filename_)
{
    const char* log_filename = CHAR(STRING_ELT(log_filename_, 0));

    int_fast64_t restored_iter = -1;
    int_fast64_t restored_lambda_count = -1;
    float restored_lambda_value = -1.0;
    Beta_Value_Sets restored_beta_sets;

    restore_from_log(log_filename, false, 0, 0, 0, 0, &restored_iter,
        &restored_lambda_count, &restored_lambda_value,
        &restored_beta_sets);

    // intercept isn't in log at the moment
    SEXP all_effects = process_beta(&restored_beta_sets, 0.0, NULL);

    return all_effects;
}

SEXP lasso_(SEXP X_, SEXP Y_, SEXP lambda_min_, SEXP lambda_max_,
    SEXP halt_error_diff_, SEXP max_interaction_distance_, SEXP max_nz_beta_,
    SEXP max_lambdas_, SEXP verbose_, SEXP log_filename_, SEXP depth_, SEXP log_level_,
    SEXP estimate_unbiased_, SEXP use_intercept_, SEXP use_cores_,
    SEXP check_duplicates_, SEXP continuous_X_)
{
    double* x = REAL(X_);
    double* y = REAL(Y_);
    SEXP dim = getAttrib(X_, R_DimSymbol);
    int_fast64_t n = INTEGER(dim)[0];
    int_fast64_t p = INTEGER(dim)[1];
    int_fast64_t max_interaction_distance = asInteger(max_interaction_distance_);
    int_fast64_t p_int = get_p_int(p, max_interaction_distance);
    int_fast64_t max_nz_beta = asInteger(max_nz_beta_);
    bool verbose = asLogical(verbose_);
    int_fast64_t max_lambdas = asInteger(max_lambdas_);
    const char* log_filename = CHAR(STRING_ELT(log_filename_, 0));
    int_fast64_t depth = asInteger(depth_);
    int_fast64_t log_level_enum = asInteger(log_level_);
    int_fast64_t use_cores = asInteger(use_cores_);
    bool check_duplicates = asLogical(check_duplicates_);
    bool continuous_X = asLogical(continuous_X_);

    VERBOSE = verbose;

    initialise_static_resources(use_cores);

    enum LOG_LEVEL log_level = NONE;
    switch (log_level_enum) {
    case 1:
        log_level = LAMBDA;
    case 2:
        log_level = ITER;
    }

    if (log_level != NONE) {
        printf("using log file %s, ", log_filename);
    }
    if (log_level == LAMBDA) {
        printf("updating once per lambda value\n");

    } else if (log_level == ITER) {
        printf("updating once per iteration\n");
    }

    bool estimate_unbiased = asLogical(estimate_unbiased_);
    bool use_intercept = asLogical(use_intercept_);

    float halt_error_diff = asReal(halt_error_diff_);

    int_fast64_t** X = (int_fast64_t**)malloc(p * sizeof *X);
    for (int_fast64_t i = 0; i < p; i++)
        X[i] = (int_fast64_t*)malloc(n * sizeof *X[i]);

    for (int_fast64_t i = 0; i < p; i++) {
        for (int_fast64_t j = 0; j < n; j++) {
            X[i][j] = (int)(x[j + i * n]);
        }
    }
    float* Y = (float*)malloc(n * sizeof(float));
    for (int_fast64_t i = 0; i < n; i++) {
        Y[i] = (float)y[i];
    }

    XMatrix xmatrix;
    xmatrix.actual_cols = n;
    xmatrix.X = X;

    bool use_adaptive_calibration = false; // option has been removed.
    Lasso_Result lasso_result = simple_coordinate_descent_lasso(
        xmatrix, Y, n, p, max_interaction_distance, asReal(lambda_min_),
        asReal(lambda_max_), max_lambdas, verbose, halt_error_diff, log_level, NULL, 0,
        max_nz_beta, log_filename, depth, estimate_unbiased, use_intercept, check_duplicates, continuous_X);
    float final_lambda = lasso_result.final_lambda;
    float regularized_intercept = lasso_result.regularized_intercept;
    float unbiased_intercept = lasso_result.unbiased_intercept;

    SEXP regularized_effects = PROTECT(
        process_beta(&lasso_result.regularized_result, regularized_intercept,
            &lasso_result.indist_from_val));
    SEXP unbiased_effects = PROTECT(process_beta(&lasso_result.unbiased_result, unbiased_intercept,
        &lasso_result.indist_from_val));

    SEXP results = PROTECT(allocVector(VECSXP, 3));
    SEXP final_lambda_sexp = PROTECT(allocVector(REALSXP, 1));
    REAL(final_lambda_sexp)
    [0] = final_lambda;

    SET_VECTOR_ELT(results, 0, regularized_effects);
    SET_VECTOR_ELT(results, 1, unbiased_effects);
    SET_VECTOR_ELT(results, 2, final_lambda_sexp);

    free_static_resources();

    for (int_fast64_t i = 0; i < p; i++)
        free(X[i]);
    free(X);
    free(Y);

    UNPROTECT(4);
    free_lasso_result(lasso_result);
    return results;
}

static const R_CallMethodDef CallEntries[] = {
    { "lasso_", (DL_FUNC)&lasso_, 17 },
    { "read_log_", (DL_FUNC)&read_log_, 1 },
    { NULL, NULL, 0 }
};

void R_init_Pint(DllInfo* info)
{
    R_registerRoutines(info, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(info, (Rboolean)FALSE);
}
}
