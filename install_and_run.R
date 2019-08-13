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

obs <- d$obs
bij_ind <- d$bij_ind
lethal_ind <- d$lethal_ind
lethal_coef <- -1000

fx_int <- data.frame(gene_i = result$interaction_effects$i, gene_j = result$interaction_effects$j,
                     effect = result$interaction_effects$strength %>% unlist) %>%
  filter(effect < -500) %>%
  arrange(gene_i) %>%
  left_join(., obs, by = c("gene_i", "gene_j")) %>%
  mutate(type = "interaction") %>%
  rowwise %>%
  left_join(., merge(bij_ind, lethal_ind, all=T), by = c("gene_i", "gene_j")) %>%
  ungroup %>%
  mutate(TP = !is.na(coef)) %>%
  mutate(lethal = (coef == lethal_coef)) %>%
  arrange(desc(TP)) %>%
  arrange(desc(lethal)) %>%
  select(gene_i, gene_j, type, TP, lethal) %>%
  distinct(gene_i, gene_j, .keep_all=TRUE) %>%
  tbl_df

fx_int %>% data.frame
