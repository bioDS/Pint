#' @title Cyclic Lasso Function
#'
#' @name pairwise_lasso
#' @description Performas lasso regression on all pairwise combinations of columns
#' @param X_filename, Y_filename, lambda, n, p
#' @export
#' @examples
#' pairwise_lasso(X, Y, n = dim(X)[1], p = dim(X)[2], lambda = p, frac_overlap_allowed = 0.05)
#' @useDynLib Pint

pairwise_lasso <- function(X, Y, n = dim(X)[1], p = dim(X)[2], lambda_min = 0.05, frac_overlap_allowed = 0.05, halt_error_diff=1.0001, max_interaction_distance=-1, use_adaptive_calibration=FALSE, max_nz_beta=-1) {
    Ym = as.matrix(Y)
    if (!dim(Ym)[1] == n) {
        stop("Y does not have the same number of rows as X, or the format is wrong")
    }

    tmp = apply(X,2, `%*%`, Y)
    lambda_max = max(abs(tmp))
    rm(tmp)

    result = .Call(lasso_, X, Ym, lambda_min, lambda_max, frac_overlap_allowed, halt_error_diff, max_interaction_distance, use_adaptive_calibration, max_nz_beta)

    #i <- sapply(result[[1]], `[`, 1)
    #strength <- sapply(result[[1]], `[`, 2)
    i <- result[[1]]
    strength <- result[[2]]
    df_main <- data.frame(i,strength)

    i <- result[[3]]
    j <- result[[4]]
    strength <- result[[5]]
    df_int <- data.frame(i,j,strength)
    rm(result)
    rm(Ym)

    return (list(main_effects = df_main, interaction_effects = df_int))
}
