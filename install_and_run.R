#!/usr/bin/env Rscript

library(dplyr)

args <- commandArgs(trailingOnly = TRUE)

if (length(args >= 2)) {
	f <- (args[2])
} else {
	f <- "../simulated_data/simulated_data_small_repeat/n1000_p100_SNR5_nbi100_nbij50_nlethals0_viol0_3231.rds"
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

result <- overlap_lasso(X, Y, lambda_min = 0.05)

obs <- d$obs
bij_ind <- d$bij_ind
lethal_ind <- d$lethal_ind
lethal_coef <- -1000

fx_int <- data.frame(gene_i = result$interaction_effects$i, gene_j = result$interaction_effects$j,
                     effect = result$interaction_effects$strength %>% unlist) %>%
  filter(abs(effect) > 0.5) %>%
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
