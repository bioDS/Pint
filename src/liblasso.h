#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <math.h>
#include <gsl/gsl_permutation.h>

#include "s8b.h"

#define VECTOR_SIZE 3
// are all ids the same size?
#define ID_LEN 20
#define BUF_SIZE 16384

static int VERBOSE;

enum LOG_LEVEL {
	ITER,
	LAMBDA,
	NONE,
};


typedef struct XMatrix {
	int **X;
	int actual_cols;
} XMatrix;

typedef struct column_set_entry {
	int value;
	int nextEntry;
} ColEntry;

typedef struct Column_Set {
	int size;
	ColEntry *cols;
} Column_Set;

typedef struct XMatrix_sparse {
	int *col_nz;
	int *col_nwords;
	unsigned short **col_nz_indices;
	gsl_permutation *permutation;
	S8bWord **compressed_indices;
} XMatrix_sparse;

typedef struct XMatrix_sparse_row {
	unsigned short **row_nz_indices;
	int *row_nz;
} XMatrix_sparse_row;

typedef struct {
	int i; int j;
} int_pair;

typedef struct {
	double *betas;
	int *indices;
	int count;
} Sparse_Betas;

//TODO: maybe this should be sparse?
typedef struct {
	long count;
	Sparse_Betas *betas;
	double *lambdas;
	long vec_length;
} Beta_Sequence;

int **X2_from_X(int **X, int n, int p);
XMatrix_sparse sparse_X2_from_X(int **X, int n, int p, long max_interaction_distance, int shuffle);
double *simple_coordinate_descent_lasso(XMatrix X, double *Y, int n, int p, long max_interaction_distance,
		double lambda_min, double lambda_max, int max_iter, int VERBOSE, double frac_overlap_allowed,
		double halt_beta_diff, enum LOG_LEVEL log_level, char **job_args, int job_args_num, int use_adaptive_calibration, int max_nz_beta);
double update_intercept_cyclic(double intercept, int **X, double *Y, double *beta, int n, int p);
double update_beta_cyclic(XMatrix X, XMatrix_sparse xmatrix_sparse, double *Y, double *rowsum, int n, int p, double lambda,
						  double *beta, long k, double intercept, int_pair *precalc_get_num, int *column_cache);
double soft_threshold(double z, double gamma);
double *read_y_csv(char *fn, int n);
XMatrix read_x_csv(char *fn, int n, int p);
int_pair get_num(long num, long p);
void free_static_resources();
void initialise_static_resources();
void parallel_shuffle(gsl_permutation* permutation, long split_size, long final_split_size, long splits);
long get_p_int(long p, long max_interaction_distance);

#define TRUE 1
#define FALSE 0
