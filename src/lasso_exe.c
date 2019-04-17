#include <gsl/gsl_vector.h>
#include "lasso_lib.h"

int main(int argc, char** argv) {
	if (argc != 9) {
		fprintf(stderr, "usage: ./lasso-testing X.csv Y.csv [greedy/cyclic] [main/int] verbose=T/F [lambda] N P\n");
		printf("actual args(%d): '", argc);
		for (int i = 0; i < argc; i++) {
			printf("%s ", argv[i]);
		}
		printf("\n");
		return 1;
	}

	char *method = argv[3];
	char *scale = argv[4];
	char *verbose = argv[5];

	int USE_INT=0; // main effects only by default
	if (strcmp(scale, "int") == 0)
		USE_INT=1;

	VERBOSE = 0;
	if (strcmp(verbose, "T") == 0)
		VERBOSE = 1;

	double lambda;

	if ((lambda = strtod(argv[6], NULL)) == 0)
		lambda = 3.604;
	printf("using lambda = %f\n", lambda);


	int N = atoi(argv[7]);
	int P = atoi(argv[8]);
	printf("using N = %d, P = %d\n", N, P);

	gsl_vector *v = gsl_vector_alloc(3);
	gsl_vector *w = gsl_vector_alloc(3);
	gsl_vector_set_zero(v);
	gsl_vector_set(v, 1, 2);
	gsl_vector_set(v, 2, 3);
	gsl_vector_memcpy(w, v);
	gsl_vector_set(w,2,1);

	int result = gsl_vector_mul(v,w);
	printf("result: %d\n", result);
	printf("v: ");
	for (int i = 0; i < VECTOR_SIZE; i++) {
		printf("%f ", gsl_vector_get(v, i));
	}
	printf("\n");


	// testing: wip
	XMatrix xmatrix = read_x_csv(argv[1], N, P);
	double *Y = read_y_csv(argv[2], N);

	int **X2;
	int nbeta;
	if (USE_INT) {
		printf("converting to X2\n");
		X2 = X2_from_X(xmatrix.X, N, xmatrix.actual_cols);
		nbeta = (xmatrix.actual_cols*(xmatrix.actual_cols+1))/2;
	} else {
		nbeta = xmatrix.actual_cols;
		X2 = xmatrix.X;
	}
	printf("using nbeta = %d\n", nbeta);

	if (xmatrix.X == NULL) {
		fprintf(stderr, "failed to read X\n");
		return 1;
	}
	if (Y == NULL) {
		fprintf(stderr, "failed to read Y\n");
		return 1;
	}

	printf("begginning coordinate descent\n");
	double *beta = simple_coordinate_descent_lasso(X2, Y, N, nbeta, lambda, method, 10);
	printf("done coordinate descent lasso, printing (%d) beta values:\n", nbeta);
	if (beta == NULL) {
		fprintf(stderr, "failed to estimate beta values\n");
		return 1;
	}
	for (int i = 0; i < nbeta; i++) {
		printf("%f ", beta[i]);
	}
	printf("\n");

	printf("indices significantly negative (-500):\n");
	for (int i = 0; i < nbeta; i++) {
		if (beta[i] < -500)
			printf("%d: %f\n", i, beta[i]);
	}

	printf("freeing X/Y\n");
	free(xmatrix.X);
	free(Y);
	return 0;
}
