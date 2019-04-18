#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <math.h>

#define VECTOR_SIZE 3
// are all ids the same size?
#define ID_LEN 20
#define BUF_SIZE 16384
//#define N 30856
//#define N 10000
//#define N 30
//#define P 1000
//#define P 35
#define HALT_BETA_DIFF 0

static int VERBOSE;

typedef struct XMatrix {
	int **X;
	int actual_cols;
} XMatrix;

int **X2_from_X(int **X, int n, int p);
double *simple_coordinate_descent_lasso(int **X, double *Y, int n, int p, double lambda, char *method, int max_iter, int USE_INT, int VERBOSE);
double update_beta_greedy_l1(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax);
double update_intercept_cyclic(double intercept, int **X, double *Y, double *beta, int n, int p);
double update_beta_cyclic(int **X, double *Y, double *rowsum, int n, int p, double lambda, double *beta, int k, double dBMax, double intercept, int USE_INT);
double update_beta_glmnet(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax, double intercept);
double soft_threshold(double z, double gamma);
double *read_y_csv(char *fn, int n);
XMatrix read_x_csv(char *fn, int n, int p);
