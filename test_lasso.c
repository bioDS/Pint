#include<stdio.h>
#include <gsl/gsl_vector.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <math.h>

#define VECTOR_SIZE 3
// are all ids the same size?
#define ID_LEN 20
#define BUF_SIZE 4096
//#define N 30856
#define N 1000
//#define N 30
//#define P 21110
#define P 100
#define HALT_BETA_DIFF 50

static int VERBOSE;

typedef struct XMatrix {
	int **X;
	int actual_cols;
} XMatrix;

XMatrix read_x_csv(char *fn, int n, int p) {
	char *buf = NULL;
	size_t line_size = 0;
	int **X = malloc(n*sizeof(int*));
	for (int i = 0; i < n; i++)
		X[i] = malloc(p*sizeof(int));

	printf("reading X from: \"%s\"\n", fn);

	FILE *fp = fopen(fn, "r");
	if (fp == NULL) {
		perror("opening failed");
	}

	int col = 0, row = 0, actual_cols = p;
	int readline_result = 0;
	while((readline_result = getline(&buf, &line_size, fp)) > 0) {
		// remove name from beginning (for the moment)
		int i = 1;
		while (buf[i] != '"')
			i++;
		i++;
		// read to the end of the line
		while (buf[i] != '\n' && i < line_size) {
			if (buf[i] == ',')
				{i++; continue;}
			if (buf[i] == '0')
				X[row][col] = 0;
			else if (buf[i] == '1')
				X[row][col] = 1;
			else {
				fprintf(stderr, "format error reading X from %s at row: %d, col: %d\n", fn, row, col);
				exit(0);
			}
			i++;
			if (++col >= p)
				break;
		}
		if (buf[i] != '\n')
			fprintf(stderr, "reached end of file without a newline\n");
		if (col < actual_cols)
			actual_cols = col;
		col = 0;
		if (++row >= n)
			break;
	}
	if (readline_result == -1)
		fprintf(stderr, "failed to read line, errno %d\n", errno);

	if (actual_cols < p)
		printf("number of columns < p, should p have been %d?\n", actual_cols);
	printf("read %dx%d, freeing stuff\n", row, actual_cols);
	free(buf);
	XMatrix xmatrix;
	xmatrix.X = X;
	xmatrix.actual_cols = actual_cols;
	return xmatrix;
}

double *read_y_csv(char *fn, int n) {
	char *buf = malloc(BUF_SIZE);
	char *temp = malloc(BUF_SIZE);
	memset(buf, 0, BUF_SIZE);
	double *Y = malloc(n*sizeof(double));

	printf("reading Y from: \"%s\"\n", fn);
	FILE *fp = fopen(fn, "r");
	if (fp == NULL) {
		perror("opening failed");
	}

	int col = 0, i = 0;
	// drop the first line
	if (fgets(buf, BUF_SIZE, fp) == NULL)
		fprintf(stderr, "failed to read first line of Y from \"%s\"\n", fn);
	while(fgets(buf, BUF_SIZE, fp) != NULL) {
		i = 1;
		// skip the name
		while(buf[i] != '"')
			i++;
		i++;
		if (buf[i] == ',')
			i++;
		// read the rest of the line as a float
		memset(temp, 0, BUF_SIZE);
		int j = 0;
		while(buf[i] != '\n')
			temp[j++] = buf[i++];
		Y[col] = atof(temp);
		col++;
	}

	printf("read %d lines, freeing stuff\n", col + 1);
	free(buf);
	free(temp);
	return Y;
}

double update_beta_cyclic(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax, double intercept) {
	double derivative = 0.0;
	double sumk = 0.0;
	double sumn = 0.0;
	double sump;

	for (int i = 0; i < n; i++) {
		sump = 0.0;
		for (int j = 0; j < p; j++) {
			// if j != k ?
				//sump += X[i][j] * beta[j];
				sump += X[i][j]?beta[j]:0.0;
		}
		//sumn += (Y[i] - sump)*(double)X[i][k];
		sumn += X[i][k]?(Y[i] - intercept - sump):0.0;
		sumk += X[i][k] * X[i][k];
	}
	derivative = -sumn;

	// TODO: This is probably slower than necessary.
	double Bkn = fmin(0.0, beta[k] - (derivative - lambda)/(sumk));
	double Bkp = fmax(0.0, beta[k] - (derivative + lambda)/(sumk));
	double Bk_diff = beta[k];
	if (Bkn < 0.0)
		beta[k] = Bkn;
	else if (Bkp > 0.0)
		beta[k] = Bkp;
	else {
		beta[k] = 0.0;
		if (VERBOSE)
			fprintf(stderr, "both \\Beta_k- (%f) and \\Beta_k+ (%f) were invalid\n", Bkn, Bkp);
	}
	Bk_diff = fabs(beta[k] - Bk_diff);
	Bk_diff *= Bk_diff;
	if (Bk_diff > dBMax)
		dBMax = Bk_diff;
	if (VERBOSE)
		printf("beta_%d is now %f\n", k, beta[k]);
	return dBMax;
}

double update_intercept_cyclic(double intercept, int **X, double *Y, double *beta, int n, int p) {
	double new_intercept = 0.0;
	double sumn = 0.0, sumx = 0.0;

	for (int i = 0; i < n; i++) {
		sumx = 0.0;
		for (int j = 0; j < p; j++) {
			sumx += X[i][j] * beta[j];
		}
		sumn += Y[i] - sumx;
	}
	new_intercept = sumn / n;
	return new_intercept;
}

double update_beta_greedy_l1(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax) {
	double derivative = 0.0;
	double sumk = 0.0;
	double sumn = 0.0;
	double sump;

	for (int i = 0; i < n; i++) {
		sump = 0.0;
		for (int j = 0; j < p; j++) {
			// if j != k ?
				//sump += X[i][j] * beta[j];
				sump += X[i][j]?beta[j]:0.0;
		}
		//sumn += (Y[i] - sump)*(double)X[i][k];
		sumn += X[i][k]?(Y[i] - sump):0.0;
		sumk += X[i][k] * X[i][k];
	}
	derivative = -sumn;

	// TODO: This is probably slower than necessary.
	double Bkn = fmin(0.0, beta[k] - (derivative - lambda)/(sumk));
	double Bkp = fmax(0.0, beta[k] - (derivative + lambda)/(sumk));
	double Bk_diff = beta[k];
	if (Bkn < 0.0)
		beta[k] = Bkn;
	else if (Bkp > 0.0)
		beta[k] = Bkp;
	else {
		beta[k] = 0.0;
		if (VERBOSE)
			fprintf(stderr, "both \\Beta_k- (%f) and \\Beta_k+ (%f) were invalid\n", Bkn, Bkp);
	}
	Bk_diff = fabs(beta[k] - Bk_diff);
	Bk_diff *= Bk_diff;
	if (Bk_diff > dBMax)
		dBMax = Bk_diff;
	if (VERBOSE)
		printf("beta_%d is now %f\n", k, beta[k]);
	return dBMax;
}

/* Edgeworths's algorithm:
 * \mu is zero for the moment, since the intercept (where no effects occurred)
 * would have no effect on fitness, so 1x survived. log(1) = 0.
 * This is probably assuming that the population doesn't grow, which we may
 * not want.
 * TODO: add an intercept
 */
double *simple_coordinate_descent_lasso(int **X, double *Y, int n, int p, double lambda) {
	// TODO: until converged
		// TODO: for each main effect x_i or interaction x_ij
			// TODO: choose index i to update uniformly at random
			// TODO: update x_i in the direction -(dF(x)/de_i / B)
	//TODO: free
	double *beta = malloc(p*sizeof(double)); // probably too big in most cases.
	memset(beta, 0, p*sizeof(double));

	int max_iter = 100;

	double error = 0, prev_error;
	double intercept = 0.0;

	for (int iter = 0; iter < max_iter; iter++) {
		prev_error = error;
		error = 0;
		double dBMax = 0.0; // largest beta diff this cycle

		// update intercept
		intercept = update_intercept_cyclic(intercept, X, Y, beta, n, p);

		for (int k = 0; k < p; k++) {
			// update the predictor \Beta_k
			//dBMax = update_beta_greedy_l1(X, Y, n, p, lambda, beta, k, dBMax);
			dBMax = update_beta_cyclic(X, Y, n, p, lambda, beta, k, dBMax, intercept);
		}
		// caculate cumulative error after update
		for (int row = 0; row < n; row++) {
			double sum = 0;
			for (int k = 0; k < p; k++) {
				sum += X[row][k]*beta[k];
			}
			double e_diff = Y[row] - intercept - sum;
			e_diff *= e_diff;
			error += e_diff;
		}
		error /= n;
		printf("mean squared error is now %f, w/ intercept %f\n", error, intercept);
		// Be sure to clean up anything extra we allocate
		// TODO: don't actually do this, see glmnet convergence conditions for a more detailed approach.
		if (dBMax < HALT_BETA_DIFF) {
			printf("largest change (%f) was less than %d, halting\n", dBMax, HALT_BETA_DIFF);
			return beta;
		}

		printf("done iteration %d\n", iter);
	}

	return beta;
}

// assumes p is even
int **X2_from_X(int **X, int n, int p) {
	int **X2 = malloc(n*sizeof(int*));
	for (int row = 0; row < n; row++) {
		X2[row] = malloc(((p*p)/2 + p/2)*sizeof(int));
		int offset = 0;
		for (int i = 0; i < p; i++) {
			for (int j = i; j < p; j++) {
				X2[row][offset++] = X[row][i] * X[row][j];
			}
		}
	}
	return X2;
}

int main(int argc, char** argv) {
	if (argc != 3 && argc != 4) {
		fprintf(stderr, "usage: ./lasso-testing X.csv Y.csv [optional: verbose=T/F]\n");
		return 1;
	}

	double lambda;

	VERBOSE = 0;
	if (argc == 4) {
		if (strcmp(argv[3], "T") == 0) {
			printf("verbose = True\n");
			VERBOSE=1;
		}
		else if ((lambda = strtod(argv[3], NULL)) == 0)
			lambda = 3.604;
	}
	printf("using lambda = %f\n", lambda);

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

	printf("converting to X2\n");
	int **X2 = X2_from_X(xmatrix.X, N, xmatrix.actual_cols);
	int nbeta = (xmatrix.actual_cols*(xmatrix.actual_cols - 1))/2;
	printf("done converting to X2\n");

	if (xmatrix.X == NULL) {
		fprintf(stderr, "failed to read X\n");
		return 1;
	}
	if (Y == NULL) {
		fprintf(stderr, "failed to read Y\n");
		return 1;
	}

	printf("begginning coordinate descent\n");
	double *beta = simple_coordinate_descent_lasso(X2, Y, N, nbeta, lambda);
	printf("done coordinate descent lasso, printing (%d) beta values:\n", nbeta);
	for (int i = 0; i < nbeta; i++) {
		printf("%f ", beta[i]);
	}
	printf("\n");

	printf("freeing X/Y\n");
	free(xmatrix.X);
	free(Y);
	return 0;
}
