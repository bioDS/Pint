#include "liblasso.h"
#include <errno.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <limits.h>
#include <omp.h>

int NumCores = 1;
long permutation_splits = 1;
long permutation_split_size;
long final_split_size;

const int NORMALISE_Y = 0;
int skipped_updates = 0;
int total_updates = 0;
int skipped_updates_entries = 0;
int total_updates_entries = 0;
int zero_updates = 0;
int zero_updates_entries = 0;

int VERBOSE = 1;
int *colsum;
float *col_ysum;
int max_size_given_entries[61];

float max_rowsums[NUM_MAX_ROWSUMS];
float max_cumulative_rowsums[NUM_MAX_ROWSUMS];

gsl_permutation *global_permutation;
gsl_permutation *global_permutation_inverse;

int min(int a, int b) {
  if (a < b)
    return a;
  return b;
}

static gsl_rng **thread_r;
int_pair *cached_nums = NULL;

// TODO: the compiler should really do this
void initialise_static_resources() {
  const gsl_rng_type *T = gsl_rng_default;
  NumCores = omp_get_num_procs();
  printf("using %d cores\n", NumCores);
  thread_r = malloc(NumCores * sizeof(gsl_rng *));
  for (int i = 0; i < NumCores; i++)
    thread_r[i] = gsl_rng_alloc(T);

  for (int i = 0; i < 60; i++) {
    max_size_given_entries[i] = 60 / (i + 1);
  }
  max_size_given_entries[60] = 0;
}

void free_static_resources() {
  if (global_permutation != NULL)
    gsl_permutation_free(global_permutation);
  if (global_permutation_inverse != NULL)
    gsl_permutation_free(global_permutation_inverse);
  if (cached_nums != NULL)
    free(cached_nums);
  for (int i = 0; i < NumCores; i++)
    gsl_rng_free(thread_r[i]);
}

// #define UINT_MAX  (__INT_MAX__  *2U +1U)
/* generate two random ints to get a long (if necessary)
 * Assumes rng is capable of 2^32-1 values, it should be one of
 * gsl_rng_{taus,mt19937,ran1xd1}
 */
long rand_long(gsl_rng *thread_rng, long max) {
  long r = -UINT_MAX;
  long rng_max = gsl_rng_max(thread_rng);
  if (rng_max != UINT_MAX) {
    fprintf(stderr, "Chosen psrng cannot generale all ints. This is most "
                    "likely a mistake.\n");
  }
  if (max > gsl_rng_max(thread_rng)) {
    long lower = gsl_rng_uniform_int(thread_rng, max % rng_max);
    long upper = gsl_rng_uniform_int(thread_rng, max / rng_max);
    upper = upper << 32;
    r = lower & upper;
  } else {
    r = gsl_rng_uniform_int(thread_rng, max);
  }
  if (r > max) {
    fprintf(stderr, "%d > %d, something went wrong in rand_long\n", r, max);
  }
  return r;
}

/* fisher yates algorithm for randomly permuting an array.
 * thread_rng should be local to the current thread.
 */
void fisher_yates(size_t *arr, long len, gsl_rng *thread_rng) {
  for (long i = len - 1; i > 0; i--) {
    long j = rand_long(thread_rng, i);
    size_t tmp = arr[j];
    arr[j] = arr[i];
    arr[i] = tmp;
  }
}

void parallel_shuffle(gsl_permutation *permutation, long split_size,
                      long final_split_size, long splits) {
#pragma omp parallel for
  for (int i = 0; i < splits; i++) {
    // printf("%p, %p, %d, %d\n", permutation->data,
    // &permutation->data[i*split_size], i, split_size); printf("range %p-%p\n",
    // &permutation->data[i*split_size], &permutation->data[i*split_size] +
    // split_size);
    fisher_yates(&permutation->data[i * split_size], split_size,
                 thread_r[omp_get_thread_num()]);
  }
  if (final_split_size > 0) {
    fisher_yates(&permutation->data[permutation->size - 1 - final_split_size],
                 final_split_size, thread_r[omp_get_thread_num()]);
  }
}

long get_p_int(long p, long dist) {
  long p_int = 0;
  if (dist <= 0 || dist >= p / 2)
    p_int = (p * (p + 1)) / 2;
  else {
    // everything short of p-dist will interact with dist items to the right
    p_int = (p - dist) * (dist)
            // the rightmost items will all interact
            + dist * (dist + 1) / 2;
  }

  printf("p: %d, dist: %d, interactions = %d\n", p, dist, p_int);
  return p_int;
}

int max(int a, int b) {
  if (a > b)
    return a;
  return b;
}

// n.b.: for glmnet gamma should be lambda * [alpha=1] = lambda
float soft_threshold(float z, float gamma) {
  float abs = fabs(z);
  if (abs < gamma)
    return 0.0;
  float val = abs - gamma;
  if (signbit(z))
    return -val;
  else
    return val;
}

float get_sump(int p, int k, int i, float *beta, int **X) {
  float sump = 0;
  for (int j = 0; j < p; j++) {
    if (j != k)
      sump += X[i][j] * beta[j];
  }
  return sump;
}

int_pair get_num(long num, long p) {
  size_t num_post_permutation = gsl_permutation_get(global_permutation, num);
  return cached_nums[num_post_permutation];
}

int_pair *get_all_nums(int p, int max_interaction_distance) {
  long p_int = get_p_int(p, max_interaction_distance);
  if (max_interaction_distance == -1)
    max_interaction_distance = p_int / 2 + 1;
  int_pair *nums = malloc(p_int * sizeof(int_pair));
  long offset = 0;
  for (int i = 0; i < p; i++) {
    for (int j = i; j < min(p, i + max_interaction_distance); j++) {
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

int **X2_from_X(int **X, int n, int p) {
  int **X2 = malloc(n * sizeof(int *));
  for (int row = 0; row < n; row++) {
    X2[row] = malloc(((p * (p + 1)) / 2) * sizeof(int));
    int offset = 0;
    for (int i = 0; i < p; i++) {
      for (int j = i; j < p; j++) {
        X2[row][offset++] = X[row][i] * X[row][j];
      }
    }
  }
  return X2;
}