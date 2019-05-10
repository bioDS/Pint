#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_spmatrix.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <math.h>
#include <glib-2.0/glib.h>

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
	gsl_spmatrix *X_sparse;
	int actual_cols;
} XMatrix;

typedef struct XMatrix_sparse {
	int **col_nz_indices;
	int *col_nz;
} XMatrix_sparse;

typedef struct XMatrix_sparse_row {
	int **row_nz_indices;
	int *row_nz;
} XMatrix_sparse_row;

typedef struct {
	int i; int j;
} int_pair;

struct Beta_Set {
	GSList *set;
	int set_size;
};

typedef struct Beta_Sets {
	struct Beta_Set *sets;
	int number_of_sets;
} Beta_Sets;

int **X2_from_X(int **X, int n, int p);
XMatrix_sparse sparse_X2_from_X(int **X, int n, int p, int USE_INT);
XMatrix_sparse_row sparse_horizontal_X2_from_X(int **X, int n, int p, int USE_INT);
double *simple_coordinate_descent_lasso(XMatrix X, double *Y, int n, int p, double lambda, char *method, int max_iter, int USE_INT, int VERBOSE);
double update_beta_greedy_l1(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax);
double update_intercept_cyclic(double intercept, int **X, double *Y, double *beta, int n, int p);
double update_beta_cyclic(XMatrix X, XMatrix_sparse xmatrix_sparse, double *Y, double *rowsum, int n, int p, double lambda, double *beta, int k, double dBMax, double intercept, int USE_INT, int_pair *precalc_get_num);
double update_beta_glmnet(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax, double intercept);
double soft_threshold(double z, double gamma);
double *read_y_csv(char *fn, int n);
XMatrix read_x_csv(char *fn, int n, int p);
int_pair get_num(int num, int p);
Beta_Sets find_beta_sets(XMatrix_sparse x2col, XMatrix_sparse_row x2row, int actual_p_int, int n);
