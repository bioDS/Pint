#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <math.h>
#include <gsl/gsl_permutation.h>

#define p4ENC p4ndenc32
#define p4DEC p4nddec32

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

typedef struct S8bWord {
	unsigned int selector : 4;
	unsigned long values: 60;
} S8bWord;

static int item_width[16] = {0,   0,   1,  2,  3,  4,  5,  6,  7, 8, 10, 12, 15, 20, 30, 60};
static int group_size[16] = {240, 120, 60, 30, 20, 15, 12, 10, 8, 7, 6,  5,  4,  3,  2,  1};
static long masks[16] = {0, 0, (1<<1)-1,(1<<2)-1,(1<<3)-1,(1<<4)-1,(1<<5)-1,(1<<6)-1,(1<<7)-1,(1<<8)-1,(1<<10)-1,(1<<12)-1,(1<<15)-1,(1<<20)-1,(1<<30)-1,((long)1<<60)-1};

S8bWord to_s8b(int count, int *vals);

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
	unsigned **col_nz_indices;
	gsl_permutation *permutation;
	unsigned char**compressed_indices;
} XMatrix_sparse;

typedef struct XMatrix_sparse_row {
	unsigned short **row_nz_indices;
	int *row_nz;
} XMatrix_sparse_row;

typedef struct {
	int i; int j;
} int_pair;

int **X2_from_X(int **X, int n, int p);
XMatrix_sparse sparse_X2_from_X(int **X, int n, int p, int permute, int max_interaction_distance);
double *simple_coordinate_descent_lasso(XMatrix X, double *Y, int n, int p, int max_interaction_distance, 
		double lambda_min, double lambda_max, char *method, int max_iter, int VERBOSE, double frac_overlap_allowed, 
		double halt_beta_diff, enum LOG_LEVEL log_level, char **job_args, int job_args_num);
double update_intercept_cyclic(double intercept, int **X, double *Y, double *beta, int n, int p);
double update_beta_cyclic(XMatrix X, XMatrix_sparse xmatrix_sparse, double *Y, double *rowsum, int n, int p, double lambda, double *beta, int k, double dBMax, double intercept, int_pair *precalc_get_num, int *column_cache);
double update_beta_glmnet(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax, double intercept);
double soft_threshold(double z, double gamma);
double *read_y_csv(char *fn, int n);
XMatrix read_x_csv(char *fn, int n, int p);
int_pair get_num(int num, int p);
void free_static_resources();
void initialise_static_resources();

#define TRUE 1
#define FALSE 0
