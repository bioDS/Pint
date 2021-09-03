#' @title Cyclic Lasso Function
#'
#' @name interaction_lasso
#' @description Performas lasso regression on all pairwise combinations of columns
#' @param X_filename, Y_filename, lambda, n, p
#' @export
#' @examples
#' interaction_lasso(X, Y, n = dim(X)[1], p = dim(X)[2])
#' @useDynLib Pint

process_result <- function(result) {
    #i <- sapply(result[[1]], `[`, 1)
    #strength <- sapply(result[[1]], `[`, 2)
    i <- result[[1]]
    strength <- result[[2]]
    df_main <- data.frame(i,strength)

    i <- result[[3]]
    j <- result[[4]]
    strength <- result[[5]]
    df_int <- data.frame(i,j,strength)

    i <- result[[6]]
    j <- result[[7]]
    k <- result[[8]]
    strength <- result[[9]]
    df_trip <- data.frame(i,j,k,strength)

    intercept <- result[[10]]

    rm(result)

    return (list(intercept=intercept, main_effects = df_main, pairwise_effects = df_int, triple_effects = df_trip))
}

read_log <- function(log_filename="regression.log") {
    result = .Call(read_log_, log_filename);
    return(process_result(result))
}

interaction_lasso <- function(X, Y, n = dim(X)[1], p = dim(X)[2], lambda_min = -1, frac_overlap_allowed = -1, halt_error_diff=1.01, max_interaction_distance=-1, use_adaptive_calibration=FALSE, max_nz_beta=-1, max_lambdas=200, verbose=FALSE, log_filename="regression.log", depth=2, log_level="none", estimate_unbiased=FALSE, use_intercept=TRUE) {
    Ym = as.matrix(Y)
    if (!dim(Ym)[1] == n) {
        stop("Y does not have the same number of rows as X, or the format is wrong")
    }

    log_level_enum = 0;
    if (log_level == "lambda") {
        log_level_enum = 1;
    }

    p <- ncol(X)
    if (depth == 2 && p > 2^31) {
        stop(sprtinf("cannot consider %d^%d interactions, consider reducing depth or reducing the number of columns of X", p, depth))
    }
    if (depth == 3 && p > 2^20) {
        stop(sprtinf("cannot consider %d^%d interactions, consider reducing depth or reducing the number of columns of X", p, depth))
    }

    tmp = apply(X,2, `%*%`, Y)
    lambda_max = max(abs(tmp))
    if (is.na(lambda_max)) {
        stop(sprintf("cannot start at lambda_max of %d, ensure input does not contain NA\n", lambda_max))
    }
    rm(tmp)

    result = .Call(lasso_, X, Ym, lambda_min, lambda_max, frac_overlap_allowed, halt_error_diff, max_interaction_distance, use_adaptive_calibration, max_nz_beta, max_lambdas, verbose, log_filename, depth, log_level_enum, estimate_unbiased, use_intercept)

    rm(Ym)

    result_regularized <- process_result(result[[1]])
    if (estimate_unbiased) {
        result_unbiased <- process_result(result[[2]])
    }
    final_lambda <- result[[3]]


    all_stats <- c(
        list("final_lambda" = final_lambda),
        result_regularized)
    if (estimate_unbiased) {
        all_stats <- c(all_stats, list("estimate_unbiased" = result_unbiased))
    }

    return(all_stats)
}
