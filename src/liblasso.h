// #define interesting_col 58
#define interesting_col 59
// #define interesting_col 0
#include <errno.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <math.h>
#include <stdalign.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <config.h>
#ifdef NOT_R
#define Rprintf(args...) printf(args);
#else
#include <R.h>
#endif

typedef struct {
  long val;
  // disable padding for now, pretty sure we don't need it.
  // alignas(64) long val;
  // char padding[64 - sizeof(long)];
} pad_int;

typedef struct {
  int i;
  int j;
} int_pair;

enum LOG_LEVEL {
  ITER,
  LAMBDA,
  NONE,
};

#include "log.h"
#include "s8b.h"
#include "sparse_matrix.h"

#include "csv.h"
#include "pruning.h"
#include "queue.h"
#include "regression.h"

int **X2_from_X(int **X, int n, int p);
float *read_y_csv(char *fn, int n);
XMatrix read_x_csv(char *fn, int n, int p);
int_pair get_num(long num, long p);
void free_static_resources();
void initialise_static_resources();
void parallel_shuffle(gsl_permutation *permutation, long split_size,
                      long final_split_size, long splits);
long get_p_int(long p, long max_interaction_distance);
int_pair *get_all_nums(int p, int max_interaction_distance);

#define TRUE 1
#define FALSE 0

#define VECTOR_SIZE 3
// are all ids the same size?
#define ID_LEN 20
#define BUF_SIZE 16384

// pad by an entire cache line, just to be safe.
#define PADDING 64

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
extern float *col_ysum;
extern int max_size_given_entries[61];

#define NUM_MAX_ROWSUMS 1
extern float max_rowsums[NUM_MAX_ROWSUMS];
extern float max_cumulative_rowsums[NUM_MAX_ROWSUMS];
extern gsl_permutation *global_permutation;
extern gsl_permutation *global_permutation_inverse;
extern int_pair *cached_nums;
extern int VERBOSE;