#include "liblasso.h"
#include <stdalign.h>
#define verbose FALSE
// #define verbose TRUE

// Force all paramaters for this function onto a single cache line.
struct pe_params {
  alignas(64) long n;
  long colsize;
  float pos_max;
  float neg_max;
  float estimate;
  long i;
  float diff_i;
  long ind;
  // char padding[] __attribute__((alignment(16)));
};

// max of either positive or negative contributions to rowsum sum.
// TODO: we know every interaction after the first iter, can we give a better
// estimate?
float pessimistic_estimate(float alpha, float *last_rowsum, float *rowsum,
                            XMatrixSparse X, int k, int *column_cache) {
  struct pe_params p = {X.n, X.cols[k].nz, 0.0, 0.0, 0.0, 0, 0.0};
  for (p.ind = 0; p.ind < p.colsize; p.ind++) {
    p.i = column_cache[p.ind];
    p.diff_i = rowsum[p.i] - alpha * last_rowsum[p.i];
    if (p.diff_i > 0) {
      p.pos_max += (p.diff_i);
    } else {
      p.neg_max += p.diff_i;
    }
  }
  p.estimate = fmaxf(p.pos_max, fabs(p.neg_max));
  return p.estimate;
}

float exact_multiple() {}

// the worst case effect is \leq last_max * alpha + pessimistic_estimate()
float l2_combined_estimate(XMatrixSparse X, float lambda, int k,
                            float last_max, float *last_rowsum,
                            float *rowsum, int *column_cache) {
  float alpha = 0.0;
  // read through the compressed column
  // TODO: make these an aligned struct?
  float estimate_squared = 0.0;
  float real_squared = 0.0;
  int entry = -1;
  int col_entry_pos = 0;
  // forcing alignment puts values on it's own cache line, so which seems to
  // help.
  for (int i = 0; i < X.cols[k].nwords; i++) {
    alignas(64) S8bWord word = X.cols[k].compressed_indices[i];
    alignas(64) long values = word.values;
    for (alignas(64) int j = 0; j <= group_size[word.selector]; j++) {
      long diff = values & masks[word.selector];
      if (diff != 0) {
        entry += diff;
        column_cache[col_entry_pos] = entry;
        col_entry_pos++;

        // do whatever we need here with the index below:
        estimate_squared += rowsum[entry] * last_rowsum[entry];
        real_squared += last_rowsum[entry] * last_rowsum[entry];
      }
      values >>= item_width[word.selector];
    }
  }
  if (real_squared != 0.0)
    alpha = fabs(estimate_squared / real_squared);
  else
    alpha = 0.0;
  if (verbose && k == interesting_col)
    printf("alpha: %f = %f/%f\n", alpha, estimate_squared, real_squared);

  float remainder =
      pessimistic_estimate(alpha, last_rowsum, rowsum, X, k, column_cache);

  float total_estimate = fabs(last_max * alpha) + remainder;
  if (verbose && k == interesting_col)
    printf("effect %d total estimate: %f = %f*%f + %f\n", k, total_estimate,
           fabs(last_max), alpha, remainder);
  return total_estimate;
}

/**
 * Branch pruning condition from Morvin & Vert
 * returns True if the branch k should be pruned
 * w is the maximum sum over i of y[i] - rowsum[i]
 *      for any interaction with effect k at the last check.
 * last_rowsums is R^n, containing the rowsums for which last_max was checked.
 * rowsums is R^n, the current rowsums
 * column_cache should be a reusable allocated array of size >= X.col_nz[k],
 * somewhere convenient for this thread.
 */
// TODO: should beta[k] be in here?
bool wont_update_effect(XMatrixSparse X, float lambda, int k, float last_max,
                       float *last_rowsum, float *rowsum, int *column_cache,
                       float *beta) {
  int *cache = malloc(X.n * sizeof *column_cache);
  float upper_bound =
      l2_combined_estimate(X, lambda, k, last_max, last_rowsum, rowsum, cache);
  if (verbose && k == interesting_col) {
    printf("beta[%d] = %f\n", k, beta[k]);
    printf("%d: upper bound: %f < lambda: %f?\n", k, upper_bound,
           lambda * (X.n / 2));
           if (upper_bound <= lambda * (X.n/2)) {
             printf("may update %d\n", k);
           }
  }
  free(cache);
  return upper_bound <= lambda * (X.n / 2);
}