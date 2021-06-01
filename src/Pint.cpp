#include "liblasso.h"
extern "C" {
#include <R.h>
#include <Rinternals.h>

struct effect {
    int i, j;
    float strength;
};

SEXP lasso_(SEXP X_, SEXP Y_, SEXP lambda_min_, SEXP lambda_max_,
    SEXP frac_overlap_allowed_, SEXP halt_error_diff_,
    SEXP max_interaction_distance_, SEXP use_adaptive_calibration_,
    SEXP max_nz_beta_, SEXP max_lambdas_, SEXP verbose_)
{
    double* x = REAL(X_);
    double* y = REAL(Y_);
    SEXP dim = getAttrib(X_, R_DimSymbol);
    int n = INTEGER(dim)[0];
    int p = INTEGER(dim)[1];
    float frac_overlap_allowed = asReal(frac_overlap_allowed_);
    // int p_int = p*(p+1)/2;
    int max_interaction_distance = asInteger(max_interaction_distance_);
    int p_int = get_p_int(p, max_interaction_distance);
    int max_nz_beta = asInteger(max_nz_beta_);
    bool verbose = asLogical(verbose_);
    int max_lambdas = asInteger(max_lambdas_);
    initialise_static_resources();

    int use_adaptive_calibration = asLogical(use_adaptive_calibration_);

    float halt_error_diff = asReal(halt_error_diff_);

    int** X = (int**)malloc(p * sizeof(int*));
    for (int i = 0; i < p; i++)
        X[i] = (int*)malloc(n * sizeof(int));

    for (int i = 0; i < p; i++) {
        for (int j = 0; j < n; j++) {
            X[i][j] = (int)(x[j + i * n]);
        }
    }
    float* Y = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        Y[i] = (float)y[i];
    }

    XMatrix xmatrix;
    xmatrix.actual_cols = n;
    xmatrix.X = X;

    enum LOG_LEVEL log_level = NONE;

    Rprintf("limiting interaction distance to %d\n", max_interaction_distance);

    robin_hood::unordered_flat_map<long, float> beta = simple_coordinate_descent_lasso(
        xmatrix, Y, n, p, max_interaction_distance, asReal(lambda_min_),
        asReal(lambda_max_), max_lambdas, verbose, frac_overlap_allowed, halt_error_diff,
        log_level, NULL, 0, use_adaptive_calibration, max_nz_beta);
    int main_count = 0, int_count = 0;
    int total_main_count = 0, total_int_count = 0;

    for (int i = 0; i < p_int; i++) {
        if (beta[i] != 0) {
            int_pair ip = get_num(i, p);
            if (ip.i == ip.j) {
                total_main_count++;
            } else {
                total_int_count++;
            }
        }
    }
    SEXP main_i = PROTECT(allocVector(REALSXP, total_main_count));
    SEXP main_strength = PROTECT(allocVector(REALSXP, total_main_count));
    SEXP int_i = PROTECT(allocVector(REALSXP, total_int_count));
    SEXP int_j = PROTECT(allocVector(REALSXP, total_int_count));
    SEXP int_strength = PROTECT(allocVector(REALSXP, total_int_count));
    SEXP all_effects = PROTECT(allocVector(VECSXP, 5));
    // int protected = 6;
    for (long i = 0; i < p_int; i++) {
        if (beta[i] != 0) {
            int_pair ip = get_num(i, p);
            if (ip.i == ip.j) {
                REAL(main_i)
                [main_count] = ip.i + 1;
                REAL(main_strength)
                [main_count] = beta[i];
                main_count++;
            } else {
                REAL(int_i)
                [int_count] = ip.i + 1;
                REAL(int_j)
                [int_count] = ip.j + 1;
                REAL(int_strength)
                [int_count] = beta[i];
                int_count++;
            }
        }
    }

    free_static_resources();

    SET_VECTOR_ELT(all_effects, 0, main_i);
    SET_VECTOR_ELT(all_effects, 1, main_strength);
    SET_VECTOR_ELT(all_effects, 2, int_i);
    SET_VECTOR_ELT(all_effects, 3, int_j);
    SET_VECTOR_ELT(all_effects, 4, int_strength);

    for (int i = 0; i < p; i++)
        free(X[i]);
    free(X);
    free(Y);
    free(beta);
    UNPROTECT(6);
    return all_effects;
}

static const R_CallMethodDef CallEntries[] = { { "lasso_", (DL_FUNC)&lasso_, 7 },
    { NULL, NULL, 0 } };

void R_init_Pint(DllInfo* info)
{
    // R_RegisterCCallable("Pint", "lasso_", (DL_FUNC) &lasso_);
    R_registerRoutines(info, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(info, FALSE);
}
}