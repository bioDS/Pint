#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
frac <- as.numeric(args[1])

if (args[2] == "reinstall") {
	install.packages(repos=NULL, pkgs="./")
}

large <- FALSE
if (args[3] == "large") {
	large <- TRUE
}


library(LassoTesting)


if (large) {
	cyclic_lasso("./X_nlethals50_v15803.csv", "./Y_nlethals50_v15803.csv", frac_overlap_allowed = frac, n=10000, p=1000)
} else {
	cyclic_lasso("./testX.csv", "./testY.csv", frac_overlap_allowed = frac, n=1000, p=100)
}
