#ifndef LIBLASSO_H
#define LIBLASSO_h

// #define interesting_col 58
#include "flat_hash_map.hpp"
#include "robin_hood.h"
// #define interesting_col 54
// #define interesting_col 2
//#define interesting_col 101
//#define interesting_val 4194
//#define interesting_col1 4  -1
//#define interesting_col2 195 -1
#include <errno.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <limits.h>
#include <math.h>
#include <stdalign.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    long* col_i;
    long* col_j;
    robin_hood::unordered_flat_map<long, float> lf_map;
} Thread_Cache;
typedef struct {
    long val;
    // disable padding for now, pretty sure we don't need it.
    // alignas(64) long val;
    // char padding[64 - sizeof(long)];
} pad_int;

typedef struct {
    long i;
    long j;
} int_pair;

enum LOG_LEVEL {
    ITER,
    LAMBDA,
    NONE,
};

struct X_uncompressed {
    long* host_X;
    long* host_col_nz;
    long* host_col_offsets;
    long* host_X_row;
    long* host_row_nz;
    long* host_row_offsets;
    long n;
    long p;
    size_t total_size;
};
struct AS_Properties {
    long was_present : 1;
    long present : 1;
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

#include "s8b.h"
#include "sparse_matrix.h"
#include "tuple_val.h"

typedef struct {
    // long *entries;
    // struct AS_Properties *properties;
    // struct AS_Entry *entries;
    robin_hood::unordered_flat_map<long, struct AS_Entry> entries1;
    robin_hood::unordered_flat_map<long, struct AS_Entry> entries2;
    robin_hood::unordered_flat_map<long, struct AS_Entry> entries3;
    long length;
    long max_length;
    gsl_permutation* permutation;
    long p;
    // S8bCol *compressed_cols;
} Active_Set;

struct AS_Entry {
    long val : 62;
    long was_present : 1;
    long present : 1;
    S8bCol col;
    float *last_rowsum;
    float last_max;
};

typedef struct {
    robin_hood::unordered_flat_map<long, float> beta1;
    robin_hood::unordered_flat_map<long, float> beta2;
    robin_hood::unordered_flat_map<long, float> beta3;
    long p;
} Beta_Value_Sets;

typedef struct {
    Beta_Value_Sets regularized_result;
    Beta_Value_Sets unbiased_result;
    float final_lambda;
} Lasso_Result;

typedef struct {
    XMatrixSparse Xc;
    float** last_rowsum;
    Thread_Cache* thread_caches;
    long n;
    // robin_hood::unordered_flat_map<long, float> *beta;
    Beta_Value_Sets* beta_sets;
    float* last_max;
    bool* wont_update;
    long p;
    long p_int;
    XMatrixSparse X2c;
    float* Y;
    float* max_int_delta;
    int_pair* precalc_get_num;
    gsl_permutation* iter_permutation;
    struct X_uncompressed Xu;
} Iter_Vars;

#include "csv.h"
#include "pruning.h"
#include "queue.h"
#include "regression.h"
#include "update_working_set.h"
#include "log.h"

long** X2_from_X(long** X, long n, long p);
int_pair get_num(long num, long p);
void free_static_resources();
void initialise_static_resources();
void parallel_shuffle(gsl_permutation* permutation, long split_size,
    long final_split_size, long splits);
long get_p_int(long p, long max_interaction_distance);
int_pair* get_all_nums(long p, long max_interaction_distance);

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

extern long NumCores;
extern long permutation_splits;
extern long permutation_split_size;
extern long final_split_size;
extern const long NORMALISE_Y;
extern long skipped_updates;
extern long total_updates;
extern long skipped_updates_entries;
extern long total_updates_entries;
extern long zero_updates;
extern long zero_updates_entries;
extern long* colsum;
extern float* col_ysum;
extern long total_beta_updates;
extern long total_beta_nz_updates;
extern float halt_error_diff;

#define NUM_MAX_ROWSUMS 1
extern float max_rowsums[NUM_MAX_ROWSUMS];
extern float max_cumulative_rowsums[NUM_MAX_ROWSUMS];
extern gsl_permutation* global_permutation;
extern gsl_permutation* global_permutation_inverse;
extern int_pair* cached_nums;
extern long VERBOSE;
extern float total_sqrt_error;

#endif