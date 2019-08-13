#!/usr/bin/env Rscript

library(dplyr)

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
	d <- readRDS("../xyz-simulation/simulated_large_data/n10000_p1000_SNR5_nbi10_nbij500_nlethals50_viol0_15803.rds")
} else {
	d <- readRDS("../xyz-simulation/simulated_data/n1000_p100_SNR5_nbi10_nbij50_nlethals5_viol0_23649.rds")
}
X <- d$X
Y <- d$Y

result <- overlap_lasso(X, Y, frac_overlap_allowed = frac)
result$main_effects %>% filter(strength < -5)
result$interaction_effects %>% filter(strength < -20)
