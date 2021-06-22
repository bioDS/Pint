#include "liblasso.h"
#include <stdalign.h>
#define verbose FALSE
// #define verbose TRUE
#ifdef NOT_R
#include <glib-2.0/glib.h>
#endif

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
float pessimistic_estimate(float alpha, float* last_rowsum, float* rowsum,
    int* col, long col_nz)
{
    // int *col = &X.host_X[X.host_col_offsets[k]];
    float pos_max = 0.0, neg_max = 0.0;
    for (int ind = 0; ind < col_nz; ind++) {
        int i = col[ind];
        #ifdef NOT_R
            g_assert_true(i >= 0);
        #endif
        float diff_i = rowsum[i] - alpha * last_rowsum[i];
        if (diff_i > 0) {
            pos_max += diff_i;
        } else {
            neg_max += diff_i;
        }
    }
    float estimate = fmaxf(pos_max, fabs(neg_max));
    return estimate;
}

float exact_multiple() {}

// the worst case effect is \leq last_max * alpha + pessimistic_estimate()
float l2_combined_estimate(X_uncompressed X, float lambda, int k,
    float last_max, float* last_rowsum,
    float* rowsum)
{
    float alpha = 0.0;
    // read through the compressed column
    // TODO: make these an aligned struct?
    float estimate_squared = 0.0;
    float real_squared = 0.0;
    int entry = -1;
    int col_entry_pos = 0;
    // forcing alignment puts values on it's own cache line, so which seems to
    // help.
    int *col = &X.host_X[X.host_col_offsets[k]];
    for (int i = 0; i < X.host_col_nz[k]; i++) {
        int entry = col[i];
        estimate_squared += rowsum[entry] * last_rowsum[entry];
        real_squared += last_rowsum[entry] * last_rowsum[entry];
    }
    if (real_squared != 0.0)
        alpha = fabs(estimate_squared / real_squared);
    else
        alpha = 0.0;
    if (verbose && k == interesting_col)
        printf("alpha: %f = %f/%f\n", alpha, estimate_squared, real_squared);

    float remainder = pessimistic_estimate(alpha, last_rowsum, rowsum, col, X.host_col_nz[k]);

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
bool wont_update_effect(X_uncompressed X, float lambda, int k, float last_max,
    float* last_rowsum, float* rowsum, int* column_cache)
{
    // int* cache = malloc(X.n * sizeof *column_cache);
    float upper_bound = l2_combined_estimate(X, lambda, k, last_max, last_rowsum, rowsum);
    if (verbose && k == interesting_col) {
        // printf("beta[%d] = %f\n", k, beta[k]);
        printf("%d: upper bound: %f < lambda: %f?\n", k, upper_bound,
            lambda);
        if (upper_bound <= lambda) {
            printf("may update %d\n", k);
        }
    }
    // free(cache);
    return upper_bound <= lambda;
}

float as_combined_estimate(float lambda, float last_max, float* last_rowsum, float* rowsum, S8bCol col, int *cache)
{
    float alpha = 0.0;
    // read through the compressed column
    // TODO: make these an aligned struct?
    float estimate_squared = 0.0;
    float real_squared = 0.0;
    int col_entry_pos = 0;

    int entry = -1;
    for (int i = 0; i < col.nwords; i++) {
      S8bWord word = col.compressed_indices[i];
      unsigned long values = word.values;
      for (int j = 0; j <= group_size[word.selector]; j++) {
        int diff = values & masks[word.selector];
        if (diff != 0) {
            entry += diff;

            // do thing here
            cache[col_entry_pos] = entry;
            estimate_squared += rowsum[entry] * last_rowsum[entry];
            real_squared += last_rowsum[entry] * last_rowsum[entry];
            col_entry_pos++;
        }
        values >>= item_width[word.selector];
      }
    }

    if (real_squared != 0.0)
        alpha = fabs(estimate_squared / real_squared);
    else
        alpha = 0.0;

    float remainder = pessimistic_estimate(alpha, last_rowsum, rowsum, cache, col_entry_pos);

    float total_estimate = fabs(last_max * alpha) + remainder;
    return total_estimate;
}

bool as_wont_update(X_uncompressed Xu, float lambda, float last_max, float* last_rowsum, float* rowsum, S8bCol col, int* column_cache) {
    float upper_bound = as_combined_estimate(lambda, last_max, last_rowsum, rowsum, col, column_cache);
    return upper_bound <= lambda;
}

bool as_pessimistic_est(float lambda, float* rowsum, S8bCol col) {
    int entry = -1;
    float pos_max = 0.0, neg_max = 0.0;
    for (int i = 0; i < col.nwords; i++) {
      S8bWord word = col.compressed_indices[i];
      unsigned long values = word.values;
      for (int j = 0; j <= group_size[word.selector]; j++) {
        int diff = values & masks[word.selector];
        if (diff != 0) {
            entry += diff;

            // do thing here
            float diff_i = rowsum[i];
            if (diff_i > 0) {
                pos_max += diff_i;
            } else {
                neg_max += diff_i;
            }
        }
        values >>= item_width[word.selector];
      }
    }

    float upper_bound = fmaxf(pos_max, fabs(neg_max));
    return upper_bound <= lambda;
}