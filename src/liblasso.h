#include <stdio.h>
#include <gsl/gsl_vector.h>
//#include <gsl/gsl_spmatrix.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <math.h>
#include <glib-2.0/glib.h>
#include <gsl/gsl_permutation.h>

#define VECTOR_SIZE 3
// are all ids the same size?
#define ID_LEN 20
#define BUF_SIZE 16384
//#define N 30856
//#define N 10000
//#define N 30
//#define P 1000
//#define P 35
#define HALT_BETA_DIFF 1.000001

static int VERBOSE;

typedef struct S8bWord {
	unsigned int selector : 4;
	unsigned long values: 60;
} S8bWord;
//typedef struct XMatrix {
//	int **X;
//	int actual_cols;
//} XMatrix;

static int item_width[16] = {0,   0,   1,  2,  3,  4,  5,  6,  7, 8, 10, 12, 15, 20, 30, 60};
//static int max_size_given_entries[60] = { 60, 30, 20, 15, 12, 10, 8, 7, 6, 6, 5, 5, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
//}
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

struct Beta_Set {
	int *set;
	int set_size;
};

typedef struct Beta_Sets {
	struct Beta_Set *sets;
	int number_of_sets;
} Beta_Sets;

typedef struct Mergeset {
	int size; //number of rows/entries in supercolumns (should really be renamed)
	unsigned short *entries;
	int ncols;
	int *cols;
} Mergeset;



int **X2_from_X(int **X, int n, int p);
XMatrix_sparse sparse_X2_from_X(int **X, int n, int p, int USE_INT, int permute);
//XMatrix_sparse_row sparse_horizontal_X2_from_X(int **X, int n, int p, int USE_INT);
double *simple_coordinate_descent_lasso(XMatrix X, double *Y, int n, int p, double lambda_min, double lambda_max, char *method, int max_iter, int USE_INT, int VERBOSE, double frac_overlap_allowed);
double update_beta_greedy_l1(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax);
double update_intercept_cyclic(double intercept, int **X, double *Y, double *beta, int n, int p);
double update_beta_cyclic(XMatrix X, XMatrix_sparse xmatrix_sparse, double *Y, double *rowsum, int n, int p, double lambda, double *beta, int k, double dBMax, double intercept, int USE_INT, int_pair *precalc_get_num);
double update_beta_glmnet(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax, double intercept);
double soft_threshold(double z, double gamma);
double *read_y_csv(char *fn, int n);
XMatrix read_x_csv(char *fn, int n, int p);
int_pair get_num(int num, int p);
Beta_Sets find_beta_sets(XMatrix_sparse x2col, int actual_p_int, int n, double frac_overlap_allowed);
Column_Set copy_column_set(Column_Set from);
void fancy_col_remove(Column_Set set, int entry);
int fancy_col_find_entry_value_or_next(Column_Set colset, int value);
void merge_sets(Mergeset *all_sets, int i, int j);
int can_merge(Mergeset *all_sets, int i, int j, double frac_overlap_allowed);
int compare_n(Mergeset *all_sets, int *valid_mergesets, int **set_bins_of_size, int *num_bins_of_size, int *sets_to_merge, int small, int large, int n, int small_offset, int large_offset, double frac_overlap_allowed);
void merge_n(Mergeset *all_sets, int **set_bins_of_size, int *num_bins_of_size, int *valid_mergesets, int *sets_to_merge, int small, int large, int n, int small_offset, int large_offset, int num_bins_to_merge);
void free_static_resources();
void initialise_static_resources();