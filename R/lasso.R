CyclicLasso <- function(xName, yName, lambda, n, p) {
    Ret = .C("lasso", as.character(xName), as.character(yName), as.numeric(lambda), as.integer(n), as.integer(p))
    return(Ret)
}