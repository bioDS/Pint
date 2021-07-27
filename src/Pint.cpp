#include "liblasso.h"

struct C_Beta_Sets {
    long   main_len;
    long*  main_effects;
    float* main_strength;
    long   int_len;
    long*  int_i;
    long*  int_j;
    float* int_strength;
    long   trip_len;
    long*  trip_a;
    long*  trip_b;
    long*  trip_c;
    float* trip_strength;
};

struct C_Beta_Sets cpp_bs_to_c(Beta_Value_Sets *beta_sets) {
    long main_len = beta_sets->beta1.size();
    long int_len = beta_sets->beta2.size();
    long trip_len = beta_sets->beta3.size();
    long p = beta_sets->p;

    long* main_effects = new long[main_len];
    float* main_strength = new float[main_len];
    long* int_i = new long[int_len];
    long* int_j = new long[int_len];
    float* int_strength = new float[int_len];
    long* trip_a = new long[trip_len];
    long* trip_b = new long[trip_len];
    long* trip_c = new long[trip_len];
    float* trip_strength = new float[trip_len];

    long main_offset = 0;
    for (auto it = beta_sets->beta1.begin(); it != beta_sets->beta1.end(); it++) {
        main_effects[main_offset] = it->first+1;
        main_strength[main_offset] = it->second;
        main_offset++;
    }
    long int_offset = 0;
    for (auto it = beta_sets->beta2.begin(); it != beta_sets->beta2.end(); it++) {
        long val = it->first;
        std::tuple<long,long> ij = val_to_pair(val, p);
        int_i[int_offset] = std::get<0>(ij)+1;
        int_j[int_offset] = std::get<1>(ij)+1;
        int_strength[int_offset] = it->second;
        int_offset++;
    }
    long trip_offset = 0;
    for (auto it = beta_sets->beta3.begin(); it != beta_sets->beta3.end(); it++) {
        long val = it->first;
        std::tuple<long,long,long> ij = val_to_triplet(val, p);
        trip_a[trip_offset] = std::get<0>(ij)+1;
        trip_b[trip_offset] = std::get<1>(ij)+1;
        trip_c[trip_offset] = std::get<2>(ij)+1;
        trip_strength[trip_offset] = it->second;
        trip_offset++;
    }

    return C_Beta_Sets {main_len, main_effects, main_strength,
        int_len, int_i, int_j, int_strength,
        trip_len, trip_a, trip_b, trip_c, trip_strength};
}

extern "C" {
#include <R.h>
#include <Rinternals.h>

struct effect {
    long i, j;
    float strength;
};

SEXP process_beta(Beta_Value_Sets* beta_sets)
{
    struct C_Beta_Sets cbs = cpp_bs_to_c(beta_sets);
    long main_count = 0, int_count = 0;
    long total_main_count = cbs.main_len;
    long total_int_count = cbs.int_len;
    long total_trip_count = cbs.trip_len;

    SEXP main_i = PROTECT(allocVector(INTSXP, total_main_count));
    SEXP main_strength = PROTECT(allocVector(REALSXP, total_main_count));
    SEXP int_i = PROTECT(allocVector(INTSXP, total_int_count));
    SEXP int_j = PROTECT(allocVector(INTSXP, total_int_count));
    SEXP int_strength = PROTECT(allocVector(REALSXP, total_int_count));
    SEXP trip_a = PROTECT(allocVector(INTSXP, total_trip_count));
    SEXP trip_b = PROTECT(allocVector(INTSXP, total_trip_count));
    SEXP trip_c = PROTECT(allocVector(INTSXP, total_trip_count));
    SEXP trip_strength = PROTECT(allocVector(REALSXP, total_trip_count));
    SEXP all_effects = PROTECT(allocVector(VECSXP, 9));

    for (long i = 0; i < cbs.main_len; i++) {
        INTEGER(main_i)[i] = cbs.main_effects[i];
        REAL(main_strength)[i] = cbs.main_strength[i];
    }
    for (long i = 0; i < cbs.int_len; i++) {
        INTEGER(int_i)[i] = cbs.int_i[i];
        INTEGER(int_j)[i] = cbs.int_j[i];
        REAL(int_strength)[i] = cbs.int_strength[i];
    }
    for (long i = 0; i < cbs.trip_len; i++) {
        INTEGER(trip_a)[i] = cbs.trip_a[i];
        INTEGER(trip_b)[i] = cbs.trip_b[i];
        INTEGER(trip_c)[i] = cbs.trip_c[i];
        REAL(trip_strength)[i] = cbs.trip_strength[i];
    }

    free(cbs.int_i);
    free(cbs.int_j);
    free(cbs.trip_a);
    free(cbs.trip_b);
    free(cbs.trip_c);

    SET_VECTOR_ELT(all_effects, 0, main_i);
    SET_VECTOR_ELT(all_effects, 1, main_strength);
    SET_VECTOR_ELT(all_effects, 2, int_i);
    SET_VECTOR_ELT(all_effects, 3, int_j);
    SET_VECTOR_ELT(all_effects, 4, int_strength);
    SET_VECTOR_ELT(all_effects, 5, trip_a);
    SET_VECTOR_ELT(all_effects, 6, trip_b);
    SET_VECTOR_ELT(all_effects, 7, trip_c);
    SET_VECTOR_ELT(all_effects, 8, trip_strength);

    UNPROTECT(10);
    return all_effects;
}

SEXP read_log_(SEXP log_filename_)
{
    char* log_filename = CHAR(STRING_ELT(log_filename_, 0));

    long restored_iter = -1;
    long restored_lambda_count = -1;
    float restored_lambda_value = -1.0;
    Beta_Value_Sets restored_beta_sets;

    restore_from_log(log_filename, false, 0, 0, 0, 0, &restored_iter, &restored_lambda_count, &restored_lambda_value, &restored_beta_sets);

    SEXP all_effects = process_beta(&restored_beta_sets);

    return all_effects;
}

SEXP lasso_(SEXP X_, SEXP Y_, SEXP lambda_min_, SEXP lambda_max_,
    SEXP frac_overlap_allowed_, SEXP halt_error_diff_,
    SEXP max_interaction_distance_, SEXP use_adaptive_calibration_,
    SEXP max_nz_beta_, SEXP max_lambdas_, SEXP verbose_, SEXP log_filename_, SEXP depth_, SEXP log_level_)
{
    double* x = REAL(X_);
    double* y = REAL(Y_);
    SEXP dim = getAttrib(X_, R_DimSymbol);
    long n = INTEGER(dim)[0];
    long p = INTEGER(dim)[1];
    float frac_overlap_allowed = asReal(frac_overlap_allowed_);
    long max_interaction_distance = asInteger(max_interaction_distance_);
    long p_int = get_p_int(p, max_interaction_distance);
    long max_nz_beta = asInteger(max_nz_beta_);
    bool verbose = asLogical(verbose_);
    long max_lambdas = asInteger(max_lambdas_);
    char* log_filename = CHAR(STRING_ELT(log_filename_, 0));
    long depth = asInteger(depth_);
    long log_level_enum = asInteger(log_level_);

    initialise_static_resources();

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

    long use_adaptive_calibration = asLogical(use_adaptive_calibration_);

    float halt_error_diff = asReal(halt_error_diff_);

    long** X = (long**)malloc(p * sizeof *X);
    for (long i = 0; i < p; i++)
        X[i] = (long*)malloc(n * sizeof *X[i]);

    for (long i = 0; i < p; i++) {
        for (long j = 0; j < n; j++) {
            X[i][j] = (int)(x[j + i * n]);
        }
    }
    float* Y = (float*)malloc(n * sizeof(float));
    for (long i = 0; i < n; i++) {
        Y[i] = (float)y[i];
    }

    XMatrix xmatrix;
    xmatrix.actual_cols = n;
    xmatrix.X = X;


    Rprintf("limiting interaction distance to %ld\n", max_interaction_distance);

    Beta_Value_Sets beta_sets = simple_coordinate_descent_lasso(
        xmatrix, Y, n, p, max_interaction_distance, asReal(lambda_min_),
        asReal(lambda_max_), max_lambdas, verbose, frac_overlap_allowed,
        halt_error_diff, log_level, NULL, 0, use_adaptive_calibration,
        max_nz_beta, log_filename, depth);

    SEXP all_effects = process_beta(&beta_sets);

    free_static_resources();

    for (long i = 0; i < p; i++)
        free(X[i]);
    free(X);
    free(Y);

    return all_effects;
}

static const R_CallMethodDef CallEntries[] = { { "lasso_", (DL_FUNC)&lasso_, 7 },
    { "read_log_", (DL_FUNC)&read_log_, 1 },
    { NULL, NULL, 0 } };

void R_init_Pint(DllInfo* info)
{
    // R_RegisterCCallable("Pint", "lasso_", (DL_FUNC) &lasso_);
    R_registerRoutines(info, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(info, FALSE);
}
}