# R package

This repo is an installable R package, you can install a locally cloned copy with `R CMD INSTALL ./cloned-location`.

This library provides a single function that performs lasso regularised linear regression on all pairs of columns in the input matrix X, otherwise modelling Y ~ X.

```
pairwise_lasso <- function(X, Y, n = dim(X)[1], p = dim(X)[2], lambda_min = 0.05, frac_overlap_allowed = 0.05, halt_error_diff=1.0001, max_interaction_distance=-1, use_adaptive_calibration=FALSE, max_nz_beta=-1)
```
A list of non-zero pairwise/interaction and main effects is returned.

For an example that finds non-zero interactions with pint, before finding a more accurate estimate of effect strengths and various summary statistics with lm() see `lm_example.R`

# Utils
There is an executable version (primarily for testing) that can be run on X/Y as .csv files.

### Build Utils
```
meson --buildtype release build
ninja -C build
```

### Usage
```
./build/utils/src/lasso_exe X.csv Y.csv [main/int] verbose=T/F [max lambda] N P [max interaction distance] [frac overlap allowed] [q/t/filename] [log_level [i]ter/[l]ambda/[n]one]
```

All arguments must be supplied.

Argument | Use
--- | ---
X.csv			| Path to X matrix in .csv format (see testX.csv for an example)
Y.csv			| Path to Y matrix in .csv format (see testY.csv for an example)
main/int:		| Find only main effects, or interactions. Main effects only intended for testing and may be broken.
verbose:		| For debugging purposes.
max lambda:	| 	Initial lambda value for regression, must be > 0.
N:			| 	Number of rows of X/Y  (e.g. no. fitness scores)
P:			| 	Number of columns of X (e.g. no. genes)
max interaction distance:	| Only columns within this distance in X will be considered. -1 to use all pairs.
frac overlap:	| fraction of columns being updated at the same time that is allowed to overlap. No longer used.
q/t/filename: | output mode. [q]uit immediately without printing output, [t]erminal: prints first 10 values < -500 to terminal, [filename]: prints all non-zero effects to the given file.
log_level:	| 	Whether and how to log partial results. iter -> every iteration, lambda -> every new lambda, none -> do not log.
