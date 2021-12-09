#!/usr/bin/env Rscript
require(Matrix)
require(dplyr)
require(Pint)

## X should be a matrix, Y a vector
# X <- ...
# Y <- ...

# find non-zero interactions
fit <- interaction_lasso(X, Y, max_nz_beta=2000)

# matrix containing only non-zero effects
Z <- X[,fit$main_effects$i]
if (length(fit$interaction_effects$i) > 0) {
	Z <- cbind(Z, X[,fit$interaction_effects$i, drop=FALSE] * X[,fit$interaction_effects$j, drop=FALSE])
}

# fit with standard regression for a better estimate
Z <- as.matrix(Z)

fit_red <- lm(Ynum ~ Z)

summary(fit_red)
