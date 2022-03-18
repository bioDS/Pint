#include "liblasso.h"
#include <R.h>
#include <Rinternals.h>

SEXP lasso_(SEXP X_, SEXP Y_, SEXP lambda_min_, SEXP lambda_max_,
    SEXP frac_overlap_allowed_, SEXP halt_error_diff_,
    SEXP max_interaction_distance_, SEXP use_adaptive_calibration_,
    SEXP max_nz_beta_, SEXP max_lambdas_, SEXP verbose_, SEXP log_filename_, SEXP depth_, SEXP log_level_, SEXP estimate_unbiased_, SEXP use_intercept_, SEXP use_cores_);