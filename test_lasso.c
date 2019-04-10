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
#define P 5050
//#define P 35
#define HALT_BETA_DIFF 0

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

	if (actual_cols < p) {
		printf("number of columns < p, should p have been %d?\n", actual_cols);
		p = actual_cols;
	}
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
	//if (fgets(buf, BUF_SIZE, fp) == NULL)
	//	fprintf(stderr, "failed to read first line of Y from \"%s\"\n", fn);
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
		//printf("temp '%s' set %d: %f\n", temp, col, Y[col]);
		col++;
	}

	printf("normalising y values\n");
	double mean = 0.0;
	for (int i = 0; i < n; i++) {
		mean += Y[i];
	}
	mean /= n;
	for (int i = 0; i < n; i++) {
		Y[i] -= mean;
	}

	printf("read %d lines, freeing stuff\n", col);
	free(buf);
	free(temp);
	return Y;
}

// n.b.: for glmnet gamma should be lambda * [alpha=1] = lambda
double soft_threshold(double z, double gamma) {
	if (fabs(z) < gamma)
		return 0.0;
	double val = fabs(z) - gamma;
	if (signbit(z))
		return -val;
	else
		return val;
}

//TODO: applies when the x variables are standardized to have unit variance, is this the case?
//TODO: glmnet also standardizes Y before computing its lambda sequence.
double update_beta_glmnet(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax, double intercept) {
	double derivative = 0.0;
	double sumk = 0.0;
	double sumn = 0.0;
	double sump;
	double new_beta;

	for (int i = 0; i < n; i++) {
		sump = 0.0;
		for (int j = 0; j < p; j++) {
			if (j != k)
				sump += X[i][j]?beta[j]:0.0;
		}
		//sumn += (Y[i] - sump)*(double)X[i][k];
		sumn += X[i][k]?(Y[i] - intercept - sump):0.0;
		sumk += X[i][k] * X[i][k];
	}

	new_beta = soft_threshold(sumn/n, lambda);
	// soft thresholding of n, lambda*[alpha=1]

	if (fabs(beta[k] - new_beta) > dBMax)
		dBMax = fabs(beta[k] - new_beta);
	beta[k] = new_beta;
	if (VERBOSE)
		printf("beta_%d is now %f\n", k, beta[k]);
	return dBMax;
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
			if (j != k)
				sump += X[i][j]?beta[j]:0.0;
		}
		if (VERBOSE)
			printf("sump: %f, Y[%d]: %f, intercept: %f\n", sump, i, Y[i], intercept);
		//sumn += (Y[i] - sump)*(double)X[i][k];
		sumn += X[i][k]?(Y[i] - intercept - sump):0.0;
		if (VERBOSE)
			printf("adding %f\n", X[i][k]?(Y[i] - intercept - sump):0.0);
		sumk += X[i][k] * X[i][k];
	}
	if (VERBOSE)
		printf("sumn: %f\n", sumn);
	derivative = -sumn;

	// TODO: This is probably slower than necessary.
	double Bk_diff = beta[k];
	if (sumk == 0.0) {
		beta[k] = 0.0;
	} else {
		double Bkn = fmin(0.0, -(derivative + lambda)/(sumk));
		double Bkp = fmax(0.0, -(derivative - lambda)/(sumk));
		if (Bkn < 0.0)
			beta[k] = Bkn;
		else if (Bkp > 0.0)
			beta[k] = Bkp;
		else {
			beta[k] = 0.0;
			//if (VERBOSE)
			//	fprintf(stderr, "both \\Beta_k- (%f) and \\Beta_k+ (%f) were invalid\n", Bkn, Bkp);
		}
	}
	Bk_diff = fabs(beta[k] - Bk_diff);
	Bk_diff *= Bk_diff;
	if (Bk_diff > dBMax)
		dBMax = Bk_diff;
	if (VERBOSE)
		printf("beta_%d is now %f\n", k, beta[k]);
	return dBMax;
}

void update_beta_shoot() {
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
double *simple_coordinate_descent_lasso(int **X, double *Y, int n, int p, double lambda, char *method) {
	// TODO: until converged
		// TODO: for each main effect x_i or interaction x_ij
			// TODO: choose index i to update uniformly at random
			// TODO: update x_i in the direction -(dF(x)/de_i / B)
	//TODO: free
	double *beta = malloc(p*sizeof(double)); // probably too big in most cases.
	memset(beta, 0, p*sizeof(double));

	int max_iter = 3;

	double error = 0, prev_error;
	double intercept = 0.0;
	double iter_lambda;
	int use_cyclic = 0, use_greedy = 0;

	printf("original lambda: %f n: %d ", lambda, n);
	lambda = lambda*n/2.0;
	printf("effective lambda is %f\n", lambda);

	if (strcmp(method,"cyclic") == 0) {
		printf("using cyclic descent\n");
		use_cyclic = 1;
	} else if (strcmp(method, "greedy") == 0) {
		printf("using greedy descent\n");
		use_greedy = 1;
	}

	if (use_greedy == 0 && use_cyclic == 0) {
		fprintf(stderr, "exactly one of cyclic/greedy must be specified\n");
		return NULL;
	}

	for (int iter = 0; iter < max_iter; iter++) {
		prev_error = error;
		error = 0;
		double dBMax = 0.0; // largest beta diff this cycle

		// update intercept
		//intercept = update_intercept_cyclic(intercept, X, Y, beta, n, p);
		//iter_lambda = lambda*(max_iter-iter)/max_iter;
		//printf("using lambda = %f\n", iter_lambda);

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

int **X2_from_X(int **X, int n, int p) {
	int **X2 = malloc(n*sizeof(int*));
	for (int row = 0; row < n; row++) {
		X2[row] = malloc(((p*(p+1))/2)*sizeof(int));
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
	if (argc != 6 && argc != 7) {
		fprintf(stderr, "usage: ./lasso-testing X.csv Y.csv [greedy/cyclic] [main/int] verbose=T/F [optional: lambda]\n");
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

	if (argc == 7) {
		if ((lambda = strtod(argv[6], NULL)) == 0)
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
	double *beta = simple_coordinate_descent_lasso(X2, Y, N, nbeta, lambda, method);
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
