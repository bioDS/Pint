#ifndef LIBLASSO_H
#define LIBLASSO_H

#include "flat_hash_map.hpp"
#include "robin_hood.h"
#include <cstdint>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdalign.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

// #include <CL/opencl.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <xxhash.h>

#ifdef R_PACKAGE
extern "C" {
#include <R.h>
}
#else
#define Rprintf(args...) printf(args);
#endif

#ifdef __WIN32
#define CLOCK_MONOTONIC_RAW CLOCK_MONOTONIC
#endif

typedef struct {
    int_fast64_t* col_i;
    int_fast64_t* col_j;
    robin_hood::unordered_flat_map<int_fast64_t, float> lf_map;
    robin_hood::unordered_flat_map<int_fast64_t, XXH3_state_t*> hash_with_col;
} Thread_Cache;
typedef struct {
    int_fast64_t val;
    // disable padding for now, pretty sure we don't need it.
    // alignas(64) int_fast64_t val;
    // char padding[64 - sizeof(long)];
} pad_int;

typedef struct {
    int_fast64_t i;
    int_fast64_t j;
} int_pair;

enum LOG_LEVEL {
    ITER,
    LAMBDA,
    NONE,
};

typedef struct {
    int_fast64_t* host_X;
    int_fast64_t* host_col_nz;
    int_fast64_t* host_col_offsets;
    int_fast64_t* host_X_row;
    int_fast64_t* host_row_nz;
    int_fast64_t* host_row_offsets;
    int_fast64_t n;
    int_fast64_t p;
    size_t total_size;
} X_uncompressed;
struct AS_Properties {
    int_fast64_t was_present : 1;
    int_fast64_t present : 1;
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
 * Fits 6 to a cache line. As int_fast64_t as schedule is static, this should be fine.
 */
// typedef struct {
//     int_fast64_t a : 64;
//     int_fast64_t b : 64;
// } int_128;

typedef struct {
    robin_hood::unordered_flat_map<XXH64_hash_t, robin_hood::unordered_flat_set<int_fast64_t>> cols_for_hash;
    // robin_hood::unordered_flat_map<int64_t, std::vector<int64_t>> defining_co;
    robin_hood::unordered_flat_set<int_fast64_t> main_col_hashes;
    robin_hood::unordered_flat_set<int_fast64_t> pair_col_hashes;
    // robin_hood::unordered_flat_set<int_fast32_t> skip_main_col_ids;
    std::vector<bool> skip_main_col_ids;
    robin_hood::unordered_flat_set<int_fast64_t> skip_pair_ids;
    robin_hood::unordered_flat_set<int_fast64_t> skip_triple_ids;
    robin_hood::unordered_flat_set<int_fast64_t> seen_together;
    robin_hood::unordered_flat_set<uint_fast32_t>* seen_with_main;
    robin_hood::unordered_flat_set<uint_fast64_t>* seen_pair_with_main;
    // robin_hood::unordered_flat_set<int_fast64_t> found_hashes;
    // int_fast64_t total_found_hash_count;
} IndiCols;

#include "s8b.h"
#include "sparse_matrix.h"
#include "tuple_val.h"

typedef struct {
    // int_fast64_t *entries;
    // struct AS_Properties *properties;
    // struct AS_Entry *entries;
    robin_hood::unordered_flat_map<int_fast64_t, struct AS_Entry> entries1;
    robin_hood::unordered_flat_map<int_fast64_t, struct AS_Entry> entries2;
    robin_hood::unordered_flat_map<int_fast64_t, struct AS_Entry> entries3;
    int_fast64_t length;
    int_fast64_t max_length;
    int_fast64_t p;
    // S8bCol *compressed_cols;
} Active_Set;

struct AS_Entry {
    int_fast64_t val : 62;
    int_fast64_t was_present : 1;
    int_fast64_t present : 1;
    S8bCol col;
    float *last_rowsum;
    float last_max;
};

typedef struct {
    robin_hood::unordered_flat_map<int_fast64_t, float> beta1;
    robin_hood::unordered_flat_map<int_fast64_t, float> beta2;
    robin_hood::unordered_flat_map<int_fast64_t, float> beta3;
    int_fast64_t p;
} Beta_Value_Sets;

typedef struct {
    Beta_Value_Sets regularized_result;
    Beta_Value_Sets unbiased_result;
    float final_lambda;
    float regularized_intercept;
    float unbiased_intercept;
    IndiCols indi;
    robin_hood::unordered_flat_map<int_fast64_t, std::vector<int_fast64_t>> indist_from_val;
} Lasso_Result;

typedef struct {
    XMatrixSparse Xc;
    float** last_rowsum;
    Thread_Cache* thread_caches;
    int_fast64_t n;
    // robin_hood::unordered_flat_map<int_fast64_t, float> *beta;
    Beta_Value_Sets* beta_sets;
    float* last_max;
    bool* wont_update;
    std::vector<bool>* seen_before;
    int_fast64_t p;
    int_fast64_t p_int;
    XMatrixSparse X2c;
    float* Y;
    float* max_int_delta;
    X_uncompressed Xu;
    float intercept;
    int_fast64_t max_interaction_distance;
} Iter_Vars;

#include "csv.h"
#include "pruning.h"
#include "queue.h"
#include "regression.h"
#include "update_working_set.h"
#include "log.h"

int_fast64_t** X2_from_X(int_fast64_t** X, int_fast64_t n, int_fast64_t p);
int_pair get_num(int_fast64_t num, int_fast64_t p);
void free_static_resources();
void initialise_static_resources(int_fast64_t num_cores);
int_fast64_t get_p_int(int_fast64_t p, int_fast64_t max_interaction_distance);
int_pair* get_all_nums(int_fast64_t p, int_fast64_t max_interaction_distance);

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

extern int_fast64_t used_branches;
extern int_fast64_t pruned_branches;

extern int_fast64_t NumCores;
extern int_fast64_t permutation_splits;
extern int_fast64_t permutation_split_size;
extern int_fast64_t final_split_size;
extern const int_fast64_t NORMALISE_Y;
extern int_fast64_t skipped_updates;
extern int_fast64_t total_updates;
extern int_fast64_t skipped_updates_entries;
extern int_fast64_t total_updates_entries;
extern int_fast64_t zero_updates;
extern int_fast64_t zero_updates_entries;
extern int_fast64_t* colsum;
extern float* col_ysum;
extern int_fast64_t total_beta_updates;
extern int_fast64_t total_beta_nz_updates;
extern float halt_error_diff;

#define NUM_MAX_ROWSUMS 1
extern float max_rowsums[NUM_MAX_ROWSUMS];
extern float max_cumulative_rowsums[NUM_MAX_ROWSUMS];
extern int_pair* cached_nums;
extern bool VERBOSE;
extern float total_sqrt_error;

#endif