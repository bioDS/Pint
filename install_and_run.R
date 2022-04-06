#!/usr/bin/env Rscript

library(dplyr)

args <- commandArgs(trailingOnly = TRUE)

if (length(args >= 2)) {
  f <- (args[2])
} else {
  # f <- "../simulated_data/simulated_data_small_repeat/n1000_p100_SNR5_nbi100_nbij50_nlethals0_viol0_3231.rds"
  # f <- "../xyz-simulation/simulated_lethal_data/n1000_p100_SNR5_nbi10_nbij50_nlethals10_viol0_78568.rds"
  # f <- "../xyz-simulation/simulated_data/n1000_p100_SNR5_nbi10_nbij50_nlethals5_viol0_23649.rds"
  #  f <- "../xyz-simulation/simulated_data/n10000_p1000_SNR10_nbi0_nbij1000_nlethals0_viol0_11504.rds"
  # f <- "../data/simulated_small_data/n1000_p100_SNR5_nbi100_nbij50_nlethals0_viol0_28462.rds"
  # f <- "../data/simulated_large_data/n10000_p1000_SNR5_nbi500_nbij500_nlethals0_viol0_50884.rds"
  # f <- "../data/simulated_8k/n8000_p4000_SNR5_nbi40_nbij800_nlethals200_viol0_91159.rds"
  # f <- "../data/simulated_small_data/n1000_p100_SNR5_nbi0_nbij100_nlethals0_viol0_11754.rds"
  # f <- "../data/simulated_large_data/n10000_p1000_SNR10_nbi0_nbij1000_nlethals0_viol0_11504.rds"
  # f <- "../data/simulated_8k/n2000_p1000_SNR5_nbi10_nbij200_nlethals50_viol0_11057.rds"
  f <- "../data/simulated_8k/n8000_p4000_SNR5_nbi40_nbij800_nlethals200_viol0_78715.rds"
  # f <- "../infx_lasso_data/3way_data_to_run/n1000_p100_SNR4_nbi10_nbij252_nbijk1666_nlethals0_70443.rds"
  # f <- "./weirdly_slow_case/n1000_p100_SNR10_nbi0_nbij100_nlethals0_viol0_33859.rds"
  # f <- "./antibio_data.rds"
}

if (length(args) >= 1) {
  if (args[1] == "reinstall") {
    install.packages(repos = NULL, pkgs = "./")
  }
}

large <- FALSE

library(Pint)

d <- readRDS(f)

X <- d$X
Y <- d$Y

# result <- interaction_lasso(X, Y, lambda_min = 0.0001, max_interaction_distance=-1, use_adaptive_calibration=TRUE, max_nz_beta=500, depth=2)
## result <- interaction_lasso(X, Y, lambda_min = -1, max_interaction_distance = -1, use_adaptive_calibration = FALSE, max_nz_beta = 200, depth = 3)
# result <- interaction_lasso(X, Y, depth = 3)
# result <- interaction_lasso(X, Y, depth = 2)
# result <- interaction_lasso(X, Y, depth = 2, max_nz_beta = 150, estimate_unbiased = TRUE, num_threads = 4, verbose=TRUE, strong_hierarchy = TRUE, check_duplicates = TRUE, continuous_X = TRUE)
result <- interaction_lasso(X, Y, depth = 2, max_nz_beta = 150, estimate_unbiased = TRUE, num_threads = 4, verbose=TRUE, approximate_hierarchy = FALSE, check_duplicates = TRUE, continuous_X = TRUE)
# print(result)

# q()

# result

obs <- d$obs %>% mutate(gene_i = as.character(gene_i), gene_j = as.character(gene_j))
# N.B. components of pairwise effects generally are also main effects in the simulated data.
bi_ind <- d$bi_ind
bi_ind$gene_i <- as.character(bi_ind$gene_i)
bij_ind <- d$bij_ind
bij_ind$gene_i <- as.character(bij_ind$gene_i)
bij_ind$gene_j <- as.character(bij_ind$gene_j)
lethal_ind <- d$lethal_ind
lethal_ind$gene_i <- as.character(lethal_ind$gene_i)
lethal_ind$gene_j <- as.character(lethal_ind$gene_j)
lethal_coef <- -1000
large_coef <- 2

fx_main <- data.frame(gene_i = result$main_effects$i) %>%
  arrange(gene_i) %>%
  mutate(type = "main", gene_j = NA, TP = (gene_i %in% bi_ind[["gene_i"]] || gene_i %in% lethal_ind[["gene_i"]])) %>%
  mutate(lethal = gene_i %in% lethal_ind[["gene_i"]]) %>%
  select(gene_i, gene_j, type, TP, lethal) %>%
  arrange(desc(TP)) %>%
  arrange(desc(lethal)) %>%
  tbl_df()

fx_int <- data.frame(
  gene_i = result$pairwise_effects$i, gene_j = result$pairwise_effects$j,
  effect = result$pairwise_effects$strength %>% unlist()
) %>%
  # filter(abs(effect) > 0.1) %>%
  arrange(as.numeric(gene_i)) %>%
  left_join(., obs, by = c("gene_i", "gene_j")) %>%
  mutate(type = "interaction") %>%
  rowwise() %>%
  left_join(., merge(bij_ind, lethal_ind, all = T), by = c("gene_i", "gene_j")) %>%
  ungroup() %>%
  mutate(TP = !is.na(coef)) %>%
  mutate(lethal = (coef == lethal_coef)) %>%
  mutate(large = (abs(coef) >= large_coef)) %>%
  arrange(desc(TP)) %>%
  arrange(desc(lethal)) %>%
  arrange(desc(large)) %>%
  select(gene_i, gene_j, type, TP, large, lethal) %>%
  distinct(gene_i, gene_j, .keep_all = TRUE) %>%
  tbl_df()

# fx_main %>% data.frame
# fx_int %>% data.frame

print(fx_int %>% filter(TP == TRUE), n = 200)
print("main effect summary")
print(summary(fx_main$TP))
print("pairwise effect summary")
print(summary(fx_int$TP))

# print("equivalent effect summary")
# print("main")
# print(result$main_effects$equivalent)
# print("pairs")
# print(result$pairwise_effects$equivalent)
# print("triples")
# print(result$triple_effects$equivalent)

length(result$main_effects$strength)
length(result$pairwise_effects$strength)

print(sprintf("first main effect (%s) was indistinguishable from:", result$main_effects$i[1]))
print(result$main_effects$equivalent[1])
