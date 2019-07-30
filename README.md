# Current state:

As of [15/04](https://github.com/bioDS/lasso_testing/commit/6c1bbdc4a80c7079a5cc8cafee96223b1b94843) running with \lambda = 20 finds 3/5 lethal interactions correctly, and one wrong result (counting strengths <-500 as lethal).
- 31/07 update: descending lambda values finds 4/5.

to make an installable R package:
automake --add-missing --copy
autoreconf -fi


# Some implementation notes:
- Using X[i][j]?beta[j]:0.0 rather than multiplication actually slows down computation.
- removing the j \neq k requirement definitely breaks things
