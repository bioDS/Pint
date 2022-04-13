#' @title Cyclic Lasso Function
#'
#' @name interaction_lasso
#' @description Performas lasso regression on all pairwise combinations of columns
#' @param X_filename, Y_filename, lambda, n, p
#' @export
#' @examples
#' interaction_lasso(X, Y, depth=2)
#' @useDynLib Pint

# converts interactions to tuple representation and adjusts for 0/1 start.
val_to_list_name <- function(val, X) {
    # integers are only 32 bits. Doubles at least give us 53.
    val <- as.numeric(val)
    range <- as.numeric(ncol(X))
    names <- as.numeric(colnames(X))
    if (val < range) {
        return(names[val + 1])
    } else if (val < range * range) {
        a <- val %/% range; # no +1 here
        b <- val %% range + 1;
        return(c(names[a], names[b]))
    } else {
        a <- val %/% (range * range); # no +1 here
        b <- (val - (a * range * range)) %/% (range) + 1;
        c <- val %% range + 1;
        return(c(names[a], names[b], names[c]))
    }
}

all_vals_to_list_name <- function(val_seq, X) {
    return(sapply(val_seq, val_to_list_name, X))
}

process_result <- function(X, result) {
    i <- colnames(X)[result[[1]]]
    strength <- result[[2]]
    equiv_list <- c()
    if (length(result[[3]][1]) > 0 && !is.null(unlist(result[[3]][1]))) {
        equiv_list <- lapply(result[[3]], all_vals_to_list_name, X)
        names(equiv_list) <- i
    }
    df_main <- list(effects=data.frame(i=i,strength=strength),equivalent=equiv_list)

    i <- colnames(X)[result[[4]]]
    j <- colnames(X)[result[[5]]]
    strength <- result[[6]]
    equiv_list <- c()
    if (length(result[[7]][1]) > 0 && !is.null(unlist(result[[7]][1]))) {
        equiv_list <- lapply(result[[7]], all_vals_to_list_name, X)
        names(equiv_list) <- paste0(i, ",", j)
    }
    df_int <- list(effects=data.frame(i=i,j=j,strength=strength),equivalent=equiv_list)

    i <- colnames(X)[result[[8]]]
    j <- colnames(X)[result[[9]]]
    k <- colnames(X)[result[[10]]]
    equiv_list <- c()
    if (length(result[[12]][1]) > 0 && !is.null(unlist(result[[12]][1]))) {
        equiv_list <- lapply(result[[12]], all_vals_to_list_name, X)
        names(equiv_list) <- paste0(i, ",", j, ",", k)
    }
    strength <- result[[11]]
    df_trip <- list(effects=data.frame(i=i,j=j,k=k,strength=strength),equivalent=equiv_list)

    intercept <- result[[13]]

    rm(result)

    return (list(intercept=intercept, main = df_main, pairwise = df_int, triple = df_trip))
}

read_log <- function(log_filename="regression.log") {
    result <- .Call(read_log_, log_filename);
    return(process_result(result))
}

interaction_lasso <- function(X, Y, n = dim(X)[1], p = dim(X)[2], lambda_min = -1, halt_error_diff=1.01, max_interaction_distance=-1, max_nz_beta=-1, max_lambdas=200, verbose=FALSE, log_filename="regression.log", depth=2, log_level="none", estimate_unbiased=FALSE, use_intercept=TRUE, num_threads=-1, approximate_hierarchy=FALSE, check_duplicates=FALSE, continuous_X=FALSE) {
    Ym = as.matrix(Y)
    if (!dim(Ym)[1] == n) {
        stop("Y does not have the same number of rows as X, or the format is wrong")
    }

    # combination currently not implemented
    if (continuous_X) {
        check_duplicates <- FALSE
    } else {
        # binarise X
        X[X != 0] <- 1
    }
    if (is.null(colnames(X))) {
        colnames(X) <- seq(ncol(X))
    }

    log_level_enum <- 0;
    if (log_level == "lambda") {
        log_level_enum <- 1;
    }

    if (length(colnames(X)) == 0) {
        colnames(X) <- seq(ncol(X))
    }

    p <- ncol(X)
    if (depth == 2 && p > 2^31) {
        stop(sprtinf("cannot consider %d^%d interactions, consider reducing depth or reducing the number of columns of X", p, depth))
    }
    if (depth == 3 && p > 2^20) {
        stop(sprtinf("cannot consider %d^%d interactions, consider reducing depth or reducing the number of columns of X", p, depth))
    }

    tmp <- apply(X, 2, `%*%`, Y)
    lambda_max <- max(abs(tmp))
    if (is.na(lambda_max)) {
        stop(sprintf("cannot start at lambda_max of %d, ensure input does not contain NA\n", lambda_max))
    }
    rm(tmp)



    result <- .Call(lasso_, X, Ym, lambda_min, lambda_max, halt_error_diff, max_interaction_distance, max_nz_beta, max_lambdas, verbose, log_filename, depth, log_level_enum, estimate_unbiased, use_intercept, num_threads, check_duplicates, continuous_X, approximate_hierarchy)

    rm(Ym)

    result_regularized <- process_result(X, result[[1]])
    if (estimate_unbiased) {
        result_unbiased <- process_result(X, result[[2]])
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
