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

static int VERBOSE;

int **read_x_csv(char *fn, int n, int p) {
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
	return X;
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

/* Edgeworths's algorithm:
 * \mu is zero for the moment, since the intercept (where no effects occurred)
 * would have no effect on fitness, so 1x survived. log(1) = 0.
 * This is probably assuming that the population doesn't grow, which we may
 * not want.
 */
double *simple_coordinate_descent_lasso(int **X, double *Y, int n, int p) {
	// TODO: until converged
		// TODO: for each main effect x_i or interaction x_ij
			// TODO: choose index i to update uniformly at random
			// TODO: update x_i in the direction -(dF(x)/de_i / B)
	//TODO: free
	double *beta = malloc(p*p*sizeof(double)); // probably too big in most cases.
	memset(beta, 0, p*p*sizeof(double));

	// Zhiyi's numbers
	int max_iter = 10;
	int lambda = 6.46;
	double sump, sumn, sumk;


	for (int iter = 0; iter < max_iter; iter++) {
			for (int k = 0; k < p; k++) {
			// update the predictor \Beta_k
				double derivative = 0.0;
				sumk = 0.0;
				sumn = 0.0;
				for (int i = 0; i < n; i++) {
					sump = 0.0;
					for (int j = 0; j < p; j++) {
						sump += X[i][j] * beta[j];
					}
					sumn += (Y[i] - sump)*(double)X[i][k];
					sumk += X[i][k] * X[i][k];
				}
				derivative = -sumn;

				// TODO: This is probably slower than necessary.
				double Bkn = fmin(0.0, beta[k] - (derivative - lambda)/(sumk));
				double Bkp = fmax(0.0, beta[k] - (derivative + lambda)/(sumk));
				if (Bkn < 0.0)
					beta[k] += Bkn;
				else if (Bkp > 0.0)
					beta[k] += Bkp;
				else
					fprintf(stderr, "both \\Beta_k- (%f) and \\Beta_k+ (%f) were invalid\n", Bkn, Bkp);
				if (VERBOSE)
					printf("beta_k is now %f\n", beta[k]);
			}
		printf("done iteration %d\n", iter);
	}

	return beta;
}

int main(int argc, char** argv) {
	if (argc != 3 && argc != 4) {
		fprintf(stderr, "usage: ./lasso-testing X.csv Y.csv [optional: verbose=T/F]\n");
		return 1;
	}

	VERBOSE = 0;
	if (argc == 4)
		if (strcmp(argv[3], "T") == 0) {
			printf("verbose = True\n");
			VERBOSE=1;
		}

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
	int **X = read_x_csv(argv[1], N, P);
	double *Y = read_y_csv(argv[2], N);

	if (X == NULL) {
		fprintf(stderr, "failed to read X\n");
		return 1;
	}
	if (Y == NULL) {
		fprintf(stderr, "failed to read Y\n");
		return 1;
	}

	printf("begginning coordinate descent\n");
	double *beta = simple_coordinate_descent_lasso(X, Y, N, P);
	printf("done coordinate descent lasso, printing beta values:\n");
	for (int i = 0; i < P; i++) {
		printf("%f ", beta[i]);
	}
	printf("\n");

	printf("freeing X/Y\n");
	free(X);
	free(Y);
	return 0;
}
