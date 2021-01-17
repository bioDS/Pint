#!/usr/bin/env Rscript

library(dplyr)

args <- commandArgs(trailingOnly = TRUE)

if (length(args >= 2)) {
	f <- (args[2])
} else {
  #f <- "../simulated_data/simulated_data_small_repeat/n1000_p100_SNR5_nbi100_nbij50_nlethals0_viol0_3231.rds"
  #f <- "../xyz-simulation/simulated_lethal_data/n1000_p100_SNR5_nbi10_nbij50_nlethals10_viol0_78568.rds"
  #f <- "../xyz-simulation/simulated_data/n1000_p100_SNR5_nbi10_nbij50_nlethals5_viol0_23649.rds"
  #f <- "../xyz-simulation/simulated_data/n10000_p1000_SNR10_nbi0_nbij1000_nlethals0_viol0_11504.rds"
  #f <- "../data/simulated_small_data/n1000_p100_SNR5_nbi100_nbij50_nlethals0_viol0_28462.rds"
  f <- "../data/simulated_large_data_sample/n10000_p1000_SNR5_nbi500_nbij500_nlethals0_viol0_50884.rds"
  #f <- "../data/simulated_large_data_sample/n10000_p1000_SNR10_nbi1000_nbij50_nlethals0_viol0_9312.rds"
}

if (length(args) >= 1) {
	if (args[1] == "reinstall") {
		install.packages(repos=NULL, pkgs="./")
	}
}

large <- FALSE

library(LassoTesting)

d <- readRDS(f)

X <- d$X
Y <- d$Y

result <- overlap_lasso(X, Y, lambda_min = 0.004, max_interaction_distance=-1, use_adaptive_calibration=FALSE)

obs <- d$obs
bij_ind <- d$bij_ind
lethal_ind <- d$lethal_ind
lethal_coef <- -1000
large_coef <- 2

fx_int <- data.frame(gene_i = result$interaction_effects$i, gene_j = result$interaction_effects$j,
                     effect = result$interaction_effects$strength %>% unlist) %>%
  #filter(abs(effect) > 0.1) %>%
  arrange(gene_i) %>%
  left_join(., obs, by = c("gene_i", "gene_j")) %>%
  mutate(type = "interaction") %>%
  rowwise %>%
  left_join(., merge(bij_ind, lethal_ind, all=T), by = c("gene_i", "gene_j")) %>%
  ungroup %>%
  mutate(TP = !is.na(coef)) %>%
  mutate(lethal = (coef == lethal_coef)) %>%
  mutate(large = (abs(coef) >= large_coef)) %>%
  arrange(desc(TP)) %>%
  arrange(desc(lethal)) %>%
  arrange(desc(large)) %>%
  select(gene_i, gene_j, type, TP, large, lethal) %>%
  distinct(gene_i, gene_j, .keep_all=TRUE) %>%
  tbl_df

fx_int %>% data.frame

print("main effects:")
result$main_effects

count(result$interaction_effects)
count(result$main_effects)
