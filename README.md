# Current state:

As of [15/04](https://github.com/bioDS/lasso_testing/commit/6c1bbdc4a80c7079a5cc8cafee96223b1b94843) running with \lambda = 20 finds 3/5 lethal interactions correctly, and one wrong result (counting strengths <-500 as lethal).
- 31/07 update: descending lambda values finds 4/5.

to make an installable R package:
./autogen.sh


# Some implementation notes:
- Using X[i][j]?beta[j]:0.0 rather than multiplication actually slows down computation.
- removing the j \neq k requirement definitely breaks things

# Utils
### Build Utils
```
meson build
ninja -C build
```

### Usage
```
./build/utils/src/lasso-testing X.csv Y.csv [main/int] verbose=T/F [max lambda] N P [frac overlap allowed] [q/t/filename]
```

main/int:		Find only main effects, or interactions. Main effects only is probably broken at the moment.
verbose:		Strongly not recommended
max lambda:		In general = P seems to be a good choice.
N:				Number of rows of X/Y  (e.g. no. fitness scores)
P:				Number of columns of X (e.g. no. genes)
frac overlap:	fraction of columns being updated at the same time that is allowed to overlap. 0 will give the same results as running on a single thread. ~0.05 appears to make good use of multiple threads without a significant impact to accuracy. (N.B. this is something of a work in progress)
q/t/filename: output mode. [q]uit immediately without printing output, [t]erminal: prints first 10 values < -500 to terminal, [filename]: prints all non-zero effects to the given file.
