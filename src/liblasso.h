// #define interesting_col 58
#include "flat_hash_map.hpp"
#define interesting_col 54
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
#include <limits.h>

// #include <CL/opencl.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

// #define NOT_R

#include <config.h>
#ifdef NOT_R
#define Rprintf(args...) printf(args);
#else
extern "C" {
#include <R.h>
}
#endif

typedef struct {
  int *col_i;
  int *col_j;
  ska::flat_hash_map<long, float> lf_map;
} Thread_Cache;
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

struct X_uncompressed {
  int* host_X;
  int* host_col_nz;
  int* host_col_offsets;
  int* host_X_row;
  int* host_row_nz;
  int* host_row_offsets;
  size_t total_size;
};
struct AS_Properties {
  int was_present : 1;
  int present : 1;
};

struct OpenCL_Setup {
};
//struct OpenCL_Setup {
//    cl_context context;
//    cl_kernel kernel;
//    cl_command_queue command_queue;
//    cl_program program;
//    cl_mem target_X;
//    cl_mem target_col_nz;
//    cl_mem target_col_offsets;
//    cl_mem target_wont_update;
//    cl_mem target_rowsum;
//    cl_mem target_beta;
//    cl_mem target_updateable_items;
//    cl_mem target_append;
//    cl_mem target_last_max;
//};

/*
 * Fits 6 to a cache line. As long as schedule is static, this should be fine.
 */

#include "log.h"
#include "s8b.h"
#include "sparse_matrix.h"

typedef struct {
  // int *entries;
  // struct AS_Properties *properties;
  struct AS_Entry *entries;
  int length;
  int max_length;
  gsl_permutation *permutation;
  // S8bCol *compressed_cols;
} Active_Set;

struct AS_Entry {
  long val : 62;
  int was_present : 1;
  int present : 1;
  S8bCol col;
  // TODO: shouldn't need this
  // char padding[39];
};

typedef struct {
  XMatrixSparse Xc;
  float **last_rowsum;
  Thread_Cache *thread_caches;
  int n;
  float *beta;
  float *last_max;
  bool *wont_update;
  int p;
  long p_int;
  XMatrixSparse X2c;
  float *Y;
  float *max_int_delta;
  int_pair *precalc_get_num;
  gsl_permutation *iter_permutation;
  struct X_uncompressed Xu;
} Iter_Vars;

#include "csv.h"
#include "pruning.h"
#include "queue.h"
#include "regression.h"
#include "update_working_set.h"


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

extern double pruning_time;
extern double working_set_update_time;
extern double subproblem_time;

extern long used_branches;
extern long pruned_branches;

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