#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
frac <- as.numeric(args[1])

if (args[2] == "reinstall") {
	install.packages(repos=NULL, pkgs="./")
}

library(LassoTesting)


cyclic_lasso("./X_nlethals50_v15803.csv", "./Y_nlethals50_v15803.csv", 1000, frac_overlap_allowed = frac, n=10000, p=1000)
#cyclic_lasso("./testX.csv", "./testY.csv", 100, frac_overlap_allowed = frac, n=1000, p=100)
