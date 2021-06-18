#' @title Cyclic Lasso Function
#'
#' @name pairwise_lasso
#' @description Performas lasso regression on all pairwise combinations of columns
#' @param X_filename, Y_filename, lambda, n, p
#' @export
#' @examples
#' pairwise_lasso(X, Y, n = dim(X)[1], p = dim(X)[2], lambda = p, frac_overlap_allowed = 0.05)
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

    a <- result[[6]]
    b <- result[[7]]
    c <- result[[8]]
    strength <- result[[9]]
    df_trip <- data.frame(a,b,c,strength)

    rm(result)

    return (list(main_effects = df_main, pairwise_effects = df_int, triple_effects = df_trip))
}

read_log <- function(log_filename="regression.log") {
    result = .Call(read_log_, log_filename);
    return(process_result(result))
}

pairwise_lasso <- function(X, Y, n = dim(X)[1], p = dim(X)[2], lambda_min = 0.05, frac_overlap_allowed = -1, halt_error_diff=1.01, max_interaction_distance=-1, use_adaptive_calibration=FALSE, max_nz_beta=-1, max_lambdas=200, verbose=FALSE, log_filename="regression.log", depth=2) {
    Ym = as.matrix(Y)
    if (!dim(Ym)[1] == n) {
        stop("Y does not have the same number of rows as X, or the format is wrong")
    }

    tmp = apply(X,2, `%*%`, Y)
    lambda_max = max(abs(tmp))
    rm(tmp)

    result = .Call(lasso_, X, Ym, lambda_min, lambda_max, frac_overlap_allowed, halt_error_diff, max_interaction_distance, use_adaptive_calibration, max_nz_beta, max_lambdas, verbose, log_filename, depth)

    rm(Ym)

    return(process_result(result))
}
