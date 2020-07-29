#!/usr/bin/env Rscript
require(Matrix)
require(dplyr)
require(LassoTesting)

## Uncomment to use modified xyz
#library(Rcpp)
#source('./xyz/R/regression.R')
#source('./xyz/R/search.R')
#source('./xyz/R/xyz.R')
#sourceCpp('./xyz/src/core.cpp')

verbose <- TRUE
lethal_coef <- -1000
lambda_min_ratio = 5e-2

args <- commandArgs(trailingOnly = TRUE)

print(args)
f <- args[1]
#f <- "simulated_data/n1000_p100_SNR2_nbi0_nbij5_nlethals0_viol0_6493.rds"
#L <- args[2] %>% as.numeric
write_out <- args[2] == 'write'
regression_alpha <- 0.9

n <- regmatches(x = f, m = regexpr(f, pattern = "(?<=n)\\d+(?=_)", perl = TRUE)) %>% as.numeric
p <- regmatches(x = f, m = regexpr(f, pattern = "(?<=_p)\\d+(?=_)", perl = TRUE)) %>% as.numeric
SNR <- regmatches(x = f, m = regexpr(f, pattern = "(?<=_SNR)\\d+(?=_)", perl = TRUE)) %>% as.numeric
num_bi <- regmatches(x = f, m = regexpr(f, pattern = "(?<=_nbi)\\d+(?=_)", perl = TRUE)) %>% as.numeric
num_bij <- regmatches(x = f, m = regexpr(f, pattern = "(?<=_nbij)\\d+(?=_)", perl = TRUE)) %>% as.numeric
num_lethals <- regmatches(x = f, m = regexpr(f, pattern = "(?<=_nlethals)\\d+(?=_)", perl = TRUE)) %>% as.numeric
perc_viol <- regmatches(x = f, m = regexpr(f, pattern = "(?<=_viol)\\d+(?=_)", perl = TRUE)) %>% as.numeric
ID <- regmatches(x = f, m = regexpr(f, pattern = "(?<=_)\\d[0-9a-z_]+(?=\\.rds)", perl = TRUE))


## not really necessary, but X is neater than data$X
#data <- readRDS(paste("simulated_lethal_data/", f, sep=''))
data <- readRDS(f)
X <- data$X
Y <- data$Y
obs <- data$obs
bi_ind <- data$bi_ind
bij_ind <- data$bij_ind
lethal_ind <- data$lethal_ind

data <- NULL
gc()

## Fit model using LassoTesting
if (verbose) cat("Fitting model\n")
if (verbose) cat("Fitting model\n")

time <- system.time(fit <- overlap_lasso(X, Y, use_adaptive_calibration=FALSE))



if (verbose) cat("Collecting stats\n")
cf <- coef(fit, lambdaType = "lambdaHat") #lambdaIndex = 50)#

## Collect coefficients
fx_main <- data.frame(gene_i = fit$main_effects$i,
                     effect = fit$main_effects$strength %>% unlist) %>%
  arrange(gene_i) %>%
  mutate(type = "main", gene_j = NA, TP = (gene_i %in% bi_ind[["gene_i"]] || gene_i %in% lethal_ind[["gene_i"]]))  %>%
  mutate(lethal=gene_i %in% lethal_ind[["gene_i"]]) %>%
  select(gene_i, gene_j, type, TP, lethal) %>%
  arrange(desc(TP)) %>%
  arrange(desc(lethal)) %>%
  tbl_df
print("fx_main")
fx_main

print("interaction_effects")
fit$interaction_effects

fx_int <- data.frame(gene_i = fit$interaction_effects$i, gene_j = fit$interaction_effects$j,
                     effect = fit$interaction_effects$strength %>% unlist) %>%
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

## Statistical test if b_i and b_ij are sig. > 0
Z <- cbind(X[,fx_main[["gene_i"]]])
for (i in 1:nrow(fx_int)) {
  Z <- cbind(Z, X[,fx_int[i,][["gene_i"]], drop = FALSE] * X[,fx_int[i,][["gene_j"]], drop = FALSE])
}
Z <- as.matrix(Z)
colnames(Z) <- rownames(Z) <- NULL
Ynum <- as.numeric(Y)
ols-time = system.time(fit_red <- lm(Ynum ~ Z))


pvals <- data.frame(id = 1:ncol(Z), coef = coef(fit_red)[-1]) %>%
  filter(!is.na(coef)) %>%
  data.frame(., pval = summary(fit_red)$coef[-1,4]) %>%
  tbl_df
  
smry <- left_join(rbind(fx_main, fx_int) %>% data.frame(id = 1:nrow(.), .), pvals, by = "id") %>%
  mutate(pval = ifelse(is.na(pval), 1, pval)) %>%
  rename(coef.est = coef) %>%
  left_join(., obs, by = c("gene_i", "gene_j")) 

# Print time taken to actuall run xyz
if (verbose)
    time

## Write out
if (write_out) {
    if (verbose) cat("Saving\n")
    saveRDS(list(fit = fit,
                 bij = bij_ind,
                 bi = bi_ind,
                 obs = obs,
                 fx_int = fx_int,
                 fx_main = fx_main,
                 fit_red = fit_red,
                 time = time,
                 ols_time = ols_time,
                 smry = smry),
            file = sprintf("./fits_testing/n%d_p%d_SNR%d_nbi%d_nbij%d_nlethals%d_viol%d_%s.rds",
                       n, p, SNR, num_bi, num_bij, num_lethals, perc_viol, ID))
} else {
    cat("Not saving\n")
}
