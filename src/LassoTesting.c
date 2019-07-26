#include "liblasso.h"
#include <R.h>
#include <Rinternals.h>

SEXP lasso_(SEXP x_fname, SEXP y_fname, SEXP lambda_, SEXP N_, SEXP P_, SEXP frac_overlap_allowed_) {
	char *method = "cyclic";
	char *scale = "int";
	char *verbose = "F";

	int USE_INT=0; // main effects only by default
	if (strcmp(scale, "int") == 0)
		USE_INT=1;

	//VERBOSE = "T";
	//if (strcmp(verbose, "T") == 0)
		VERBOSE = 1;

	double lambda = asReal(lambda_);
	if (lambda == 0)
		lambda = 3.604;
	int N = asInteger(N_);
	int P = asInteger(P_);

	double frac_overlap_allowed = asReal(frac_overlap_allowed_);

	// testing: wip
	XMatrix xmatrix = read_x_csv(CHAR(asChar(x_fname)), N, P);
	double *Y = read_y_csv(CHAR(asChar(y_fname)), N);

	int **X2;
	int nbeta;
	nbeta = xmatrix.actual_cols;
	X2 = xmatrix.X;

	if (xmatrix.X == NULL) {
		fprintf(stderr, "failed to read X\n");
		return 1;
	}
	if (Y == NULL) {
		fprintf(stderr, "failed to read Y\n");
		return 1;
	}

	double *beta = simple_coordinate_descent_lasso(xmatrix, Y, N, nbeta, lambda, method, 100, USE_INT, VERBOSE, frac_overlap_allowed);
	int nbeta_int = nbeta;
	if (USE_INT) {
		nbeta_int = nbeta*(nbeta+1)/2;
	}
	if (beta == NULL) {
		fprintf(stderr, "failed to estimate beta values\n");
		return 1;
	}

	int sig_beta_count = 0;
	for (int i = 0; i < nbeta_int; i++) {
		if (beta[i] < -500) {
			sig_beta_count++;
			int_pair ip = get_num(i, nbeta);
			if (ip.i == ip.j)
				Rprintf("main: %d (%d):     %f\n", i, ip.i + 1, beta[i]);
			else
				Rprintf("int: %d  (%d, %d): %f\n", i, ip.i + 1, ip.j + 1, beta[i]);
		}
	}
	return ScalarReal(1);
}

static const R_CallMethodDef CallEntries[] ={
	{"lasso_", (DL_FUNC) &lasso_, 5},
	{NULL, NULL, 0}
};

void R_init_LassoTesting(DllInfo *info) {
	//R_RegisterCCallable("LassoTesting", "lasso_", (DL_FUNC) &lasso_);
	R_registerRoutines(info, NULL, CallEntries, NULL, NULL);
	R_useDynamicSymbols(info, FALSE);
}
