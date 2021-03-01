#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>

#include <src/config.h>
#ifdef NOT_R
	#define Rprintf(args...) printf (args);
#else
	#include <R.h>
#endif

typedef struct {
	int i; int j;
} int_pair;

enum LOG_LEVEL {
	ITER,
	LAMBDA,
	NONE,
};

#include "s8b.h"
#include "sparse_matrix.h"
#include "queue.h"
#include "regression.h"
#include "log.h"
#include  "csv.h"

int **X2_from_X(int **X, int n, int p);
double *read_y_csv(char *fn, int n);
XMatrix read_x_csv(char *fn, int n, int p);
int_pair get_num(long num, long p);
void free_static_resources();
void initialise_static_resources();
void parallel_shuffle(gsl_permutation* permutation, long split_size, long final_split_size, long splits);
long get_p_int(long p, long max_interaction_distance);

#define TRUE 1
#define FALSE 0

#define VECTOR_SIZE 3
// are all ids the same size?
#define ID_LEN 20
#define BUF_SIZE 16384

extern int NumCores;
extern long permutation_splits;
extern long permutation_split_size;
extern long final_split_size;
extern const int NORMALISE_Y;
extern int skipped_updates;
extern int total_updates;
extern int skipped_updates_entries;
extern int total_updates_entries;
extern int zero_updates;
extern int zero_updates_entries;
extern int VERBOSE;
extern int *colsum;
extern double *col_ysum;
extern int max_size_given_entries[61];

#define NUM_MAX_ROWSUMS 1
extern double max_rowsums[NUM_MAX_ROWSUMS];
extern double max_cumulative_rowsums[NUM_MAX_ROWSUMS];
extern gsl_permutation *global_permutation;
extern gsl_permutation *global_permutation_inverse;
extern int_pair *cached_nums;
extern int VERBOSE;