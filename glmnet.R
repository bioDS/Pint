#! /usr/bin/env Rscript
library(glmnet)
library(dplyr)
X <- read.csv('testX2.csv', row.names=1, header = FALSE)
Y <- read.csv('testY.csv', row.names=1, header = FALSE)
result <- glmnet(X %>% as.matrix, Y[,1] %>% as.numeric, , alpha=0.9, family="gaussian")
beta = result[["beta"]]
final_beta = beta[,100]
final_beta[final_beta < -700]
