![tests](test-badge.svg)
![cod cov](coverage-badge.svg)
# R package

This repo is an installable R package, you can install a locally cloned copy with `R CMD INSTALL ./cloned-location`.

Alternatively, install directly from github with:
```
install.packages("https://github.com/bioDS/Pint/archive/refs/heads/main.tar.gz", repos=NULL)
```

This library provides a single function that performs square root lasso regularised linear regression on all pairs of columns in the input matrix X, otherwise modelling Y ~ X. The primary function (including default arguments) is:

```
output <- interaction_lasso(X, Y, n = dim(X)[1], p = dim(X)[2], lambda_min = -1, halt_error_diff=1.01, max_interaction_distance=-1, max_nz_beta=-1, max_lambdas=200, verbose=FALSE, log_filename="regression.log", depth=2, log_level="none", estimate_unbiased=FALSE, use_intercept=TRUE, num_threads=-1, approximate_hierarchy=FALSE, check_duplicates=FALSE, continuous_X=FALSE)
```

## Arguments:

`X` : A binary $n \times p$ matrix.

`Y` : A vector of $n$ real values.

`lambda_min` : optionally set the final value of lambda. If $ < 0$ the default value of $ϕ⁻¹(\frac{0.95}{2 \times p})$ is used.

`halt_error_diff` : The loss-threshold to determine when an iteration is complete.

`max_interaction_distance` : The maximum distance between any two components of an interaction effect. Set to '-1' for no limit (default).

`max_nz_beta` : If >=0, halt after this many $\beta$ values are non-zero (note the the current $\lambda$ iteration will be completed first, so more values may be set). '-1' implies no limit.

`max_lambdas` : maximum number of iterations (i.e. number of $\lambda$ values). Initial iterations in which no $\beta$ values are changed do not count.

`depth` : Maximum number of columns that may be included in an interaction. If depth=1, only main effects (columns on their own) are included. If depth=2, pairwise interactions are also included. If depth=3 main effects, pairwise and three-way interactions are included.

`estimate_unbiased` : once the non-zero $\beta$ values have been determined, optionally re-fit with $\lambda=0$ to avoid the minimising effect on $\beta$ values, while still keeping the result sparse.

`use_intercept` : If true, allow a non-zero intercept.

`approximate_hierarchy` : Approximates a strong hierarchy by only allowing interactions between columns that are (or were at a larger $\lambda$ value) non-zero. Note that a main effect may still be set to zero after the interactions is included, so this does not strictly enforce either a strong or weak hierarchy. This can considerably speed up fitting interactions on large data sets.

`check_duplicates` : Identify and report any duplicate columns or interactions, and only assign an effect to the one of them.

`num_threads` : Number of threads to use, use '-1' (default) to use all available CPU cores.

### Experimental Features
A number of options have been implemented, but not thoroughly tested. These are:

`continuous_X` : If true, use floating point values for X. If false, all non-zero values in X are treated as 1. Note that this currently disables duplicate column detection.

`log_filename` : name of file to save current progress in case the process needs to be interrupted and resumed.

`log_level` : options are 'none' (no logging), and 'lambda' where progress is saved after each $\lambda$ iteration is completed.

## Return Values

A list of non-zero pairwise/interaction and main effects is returned.

More precisely:

`final_lambda` : the final value of $\lambda$.

`intercept` : (if `use_intercept=TRUE`) the intercept value.

`main` : A data frame `effects` containing $i, \beta_i$ for individual columns $X_i$, and a list `eqiuvalent` of the columns/interactions that were indistinguishable from each (if check_duplicates was enabled).

`pairwise` (if `depth` $\geq 2$) A data frame `effects` containing $i,j, \beta_{i,j}$ for $X_i \circ X_j$ and a list `equivalent` of the columns/interactions that were indistinguishable from each (if check_duplicates was enabled).


`triple` (if `depth` $\geq 3$) A data frame `effects` containing $i,j,k, \beta_{i,j,k}$ for $X_i \circ X_j \circ X_k$ and a list `equivalent` of the columns/interactions that were indistinguishable from each (if check_duplicates was enabled).


`estimate_unbiased` : (if `estimate_unbiased=TRUE`) $\beta_i, \beta_{i,j}, \beta_{i,j,k}$ fit with $\lambda = 0$, including only the effects that are non-zero for $lambda = $ final_lambda.
For an estimate of the best fit, while excluding columns lasso regression sets to zero.

For an example that finds non-zero interactions with pint, before finding a more accurate estimate of effect strengths and various summary statistics with lm() see `lm_example.R`

## Build Requirements

Compiling on Ubuntu 22.04 requires the following package:

```
libxxhash-dev
```

Additionally, the following are required for the standalone executable and/or running tests:

```
libgsl-dev
ninja-build
libglib2.0-dev
meson
gcovr
```

# Standalone Executable
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

# Acknowledgements

This project includes the following work:

- [xxHash](https://github.com/Cyan4973/xxHash) (for identifying identical columns).
- Malte Skarupke's [flat hash map](https://github.com/skarupke/flat_hash_map).
