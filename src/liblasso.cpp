#include "liblasso.h"
#include <errno.h>
#include <limits.h>
#include <omp.h>

int_fast64_t NumCores = 1;
int_fast64_t permutation_splits = 1;
int_fast64_t permutation_split_size;
int_fast64_t final_split_size;

const int_fast64_t NORMALISE_Y = 0;
int_fast64_t skipped_updates = 0;
int_fast64_t total_updates = 0;
int_fast64_t skipped_updates_entries = 0;
int_fast64_t total_updates_entries = 0;
int_fast64_t zero_updates = 0;
int_fast64_t zero_updates_entries = 0;

int_fast64_t VERBOSE = 1;
int_fast64_t* colsum;
float* col_ysum;

float max_rowsums[NUM_MAX_ROWSUMS];
float max_cumulative_rowsums[NUM_MAX_ROWSUMS];

double pruning_time = 0.0;
double working_set_update_time = 0.0;
double subproblem_time = 0.0;

int_fast64_t used_branches = 0;
int_fast64_t pruned_branches = 0;

int_fast64_t min(int_fast64_t a, int_fast64_t b)
{
    if (a < b)
        return a;
    return b;
}

int_pair* cached_nums = NULL;

// TODO: the compiler should really do this
void initialise_static_resources()
{
    NumCores = omp_get_num_procs();
    printf("using %ld cores\n", NumCores);
}

void free_static_resources()
{
    if (cached_nums != NULL)
        free(cached_nums);
}

int_fast64_t get_p_int(int_fast64_t p, int_fast64_t dist)
{
    int_fast64_t p_int = 0;
    if (dist <= 0 || dist >= p / 2)
        p_int = (p * (p + 1)) / 2;
    else {
        // everything short of p-dist will interact with dist items to the right
        p_int = (p - dist) * (dist)
            // the rightmost items will all interact
            + dist * (dist + 1) / 2;
    }

    printf("p: %ld, dist: %ld, interactions = %ld\n", p, dist, p_int);
    return p_int;
}

int_fast64_t max(int_fast64_t a, int_fast64_t b)
{
    if (a > b)
        return a;
    return b;
}

// n.b.: for glmnet gamma should be lambda * [alpha=1] = lambda
float soft_threshold(float z, float gamma)
{
    float abs = fabs(z);
    if (abs < gamma)
        return 0.0;
    float val = abs - gamma;
    if (signbit(z))
        return -val;
    else
        return val;
}

float get_sump(int_fast64_t p, int_fast64_t k, int_fast64_t i, robin_hood::unordered_flat_map<int_fast64_t, float> beta, int_fast64_t** X)
{
    float sump = 0;
    for (int_fast64_t j = 0; j < p; j++) {
        if (j != k)
            sump += X[i][j] * beta[j];
    }
    return sump;
}

int_pair get_num(int_fast64_t num, int_fast64_t p)
{
    return cached_nums[num];
}

int_pair* get_all_nums(int_fast64_t p, int_fast64_t max_interaction_distance)
{
    int_fast64_t p_int = get_p_int(p, max_interaction_distance);
    if (max_interaction_distance == -1)
        max_interaction_distance = p_int / 2 + 1;
    int_pair* nums = (int_pair*)malloc(p_int * sizeof(int_pair));
    int_fast64_t offset = 0;
    for (int_fast64_t i = 0; i < p; i++) {
        for (int_fast64_t j = i; j < min(p, i + max_interaction_distance); j++) {
            int_pair ip;
            ip.i = i;
            ip.j = j;
            nums[offset] = ip;
            offset++;
        }
    }
    return nums;
}

/* Edgeworths's algorithm:
 * \mu is zero for the moment, since the intercept (where no effects occurred)
 * would have no effect on fitness, so 1x survived. log(1) = 0.
 * This is probably assuming that the population doesn't grow, which we may
 * not want.
 * TODO: add an intercept
 * TODO: haschanged can only have an effect if an entire iteration does nothing.
 * This should never happen.
 */

int_fast64_t** X2_from_X(int_fast64_t** X, int_fast64_t n, int_fast64_t p)
{
    int_fast64_t** X2 = (int_fast64_t**)malloc(n * sizeof *X2);
    for (int_fast64_t row = 0; row < n; row++) {
        X2[row] = (int_fast64_t*)malloc(((p * (p + 1)) / 2) * sizeof *X2[row]);
        int_fast64_t offset = 0;
        for (int_fast64_t i = 0; i < p; i++) {
            for (int_fast64_t j = i; j < p; j++) {
                X2[row][offset++] = X[row][i] * X[row][j];
            }
        }
    }
    return X2;
}