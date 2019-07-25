#' @title Cyclic Lasso Function
#'
#' @name cyclic_lasso
#' @description Performas a cyclic lasso
#' @param X_filename, Y_filename, lambda, n, p
#' @export
#' @examples
#' cyclic_lasso(xName = "X.csv", yName = "Y.csv", lambda = 20, n = 1000, p = 100)
#' @useDynLib LassoTesting

cyclic_lasso <- function(xName = "X.csv", yName = "Y.csv", lambda = 20, n = 1000, p = 100) {
    #dyn.load(LassoTesting)
    Ret = .Call(lasso_, as.character(xName), as.character(yName), as.numeric(lambda), as.integer(n), as.integer(p))
    #Ret = "test"
    return(Ret)
}