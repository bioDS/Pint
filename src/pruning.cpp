#include "liblasso.h"
#include "robin_hood.h"
#include <stdalign.h>
#define verbose FALSE
// #define verbose TRUE
#ifdef NOT_R
#include <glib-2.0/glib.h>
#endif

// Force all paramaters for this function onto a single cache line.
struct pe_params {
    alignas(64) int_fast64_t n;
    int_fast64_t colsize;
    float pos_max;
    float neg_max;
    float estimate;
    int_fast64_t i;
    float diff_i;
    int_fast64_t ind;
    // char padding[] __attribute__((alignment(16)));
};

// max of either positive or negative contributions to rowsum sum.
// TODO: we know every interaction after the first iter, can we give a better
// estimate?
float pessimistic_estimate(float alpha, float* last_rowsum, float* rowsum,
    int_fast64_t* col, int_fast64_t col_nz)
{
    // int_fast64_t *col = &X.host_X[X.host_col_offsets[k]];
    float pos_max = 0.0, neg_max = 0.0;
    for (int_fast64_t ind = 0; ind < col_nz; ind++) {
        int_fast64_t i = col[ind];
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

// the worst case effect is \leq last_max * alpha + pessimistic_estimate()
float l2_combined_estimate(X_uncompressed X, float lambda, int_fast64_t k,
    float last_max, float* last_rowsum,
    float* rowsum, struct continuous_info* ci)
{
    float alpha = 0.0;
    // read through the compressed column
    // TODO: make these an aligned struct?
    float estimate_squared = 0.0;
    float real_squared = 0.0;
    int_fast64_t entry = -1;
    int_fast64_t col_entry_pos = 0;
    // forcing alignment puts values on it's own cache line, so which seems to
    // help.
    int_fast64_t* col = &X.host_X[X.host_col_offsets[k]];
    for (int_fast64_t i = 0; i < X.host_col_nz[k]; i++) {
        int_fast64_t entry = col[i];
        estimate_squared += rowsum[entry] * last_rowsum[entry];
        real_squared += last_rowsum[entry] * last_rowsum[entry];
    }
    if (real_squared != 0.0)
        alpha = fabs(estimate_squared / real_squared);
    else
        alpha = 0.0;

    float remainder = pessimistic_estimate(alpha, last_rowsum, rowsum, col, X.host_col_nz[k]);
    if (ci->use_cont)
        remainder *= fabs(ci->col_max_vals[k]);

    float total_estimate = fabs(last_max * alpha) + remainder;
    return total_estimate;
}

/**
 * Branch pruning condition from Morvin & Vert
 * returns True if the branch k should be pruned
 * w is the maximum sum over i of y[i] - rowsum[i]
 *      for any interaction with effect k at the last check.
 * last_rowsums is R^n, containing the rowsums for which last_max was checked.
 * rowsums is R^n, the current rowsums
//  * column_cache should be a reusable allocated array of size >= X.col_nz[k],
 * somewhere convenient for this thread.
 */
// TODO: should beta[k] be in here?
bool wont_update_effect(X_uncompressed X, float lambda, int_fast64_t k, float last_max,
    float* last_rowsum, float* rowsum, int_fast64_t* column_cache, struct continuous_info* ci)
{
    // int_fast64_t* cache = malloc(X.n * sizeof *column_cache);
    float upper_bound = l2_combined_estimate(X, lambda, k, last_max, last_rowsum, rowsum, ci);
    return upper_bound <= lambda * total_sqrt_error;
}

// float as_pessimistic_estimate(float alpha, robin_hood::unordered_flat_map<int_fast64_t, float>* last_rowsum, float* rowsum,
float as_pessimistic_estimate(float alpha, float* last_rowsum, float* rowsum,
    int_fast64_t* col, int_fast64_t col_nz)
{
    float pos_max = 0.0, neg_max = 0.0;
    for (int_fast64_t ind = 0; ind < col_nz; ind++) {
        int_fast64_t i = col[ind];
#ifdef NOT_R
        g_assert_true(i >= 0);
#endif
        float diff_i = rowsum[i] - alpha * last_rowsum[ind];
        if (diff_i > 0) {
            pos_max += diff_i;
        } else {
            neg_max += diff_i;
        }
    }
    float estimate = fmaxf(pos_max, fabs(neg_max));
    return estimate;
}

float as_combined_estimate(float lambda, float last_max, float* last_rowsum, float* rowsum, S8bCol col, int_fast64_t* cache, float col_max, bool use_cont)
{
    float alpha = 0.0;
    // read through the compressed column
    // TODO: make these an aligned struct?
    float estimate_squared = 0.0;
    float real_squared = 0.0;
    int_fast64_t col_entry_pos = 0;

    int_fast64_t entry = -1;
    for (int_fast64_t i = 0; i < col.nwords; i++) { //TODO: broken
        S8bWord word = col.compressed_indices[i];
        int_fast64_t values = word.values;
        for (int_fast64_t j = 0; j <= group_size[word.selector]; j++) {
            int_fast64_t diff = values & masks[word.selector];
            if (diff != 0) {
                entry += diff;

                // do thing here
                cache[col_entry_pos] = entry;
                estimate_squared += rowsum[entry] * last_rowsum[col_entry_pos];
                real_squared += last_rowsum[col_entry_pos] * last_rowsum[col_entry_pos];
                col_entry_pos++;
            }
            values >>= item_width[word.selector];
        }
    }

    if (real_squared != 0.0)
        alpha = fabs(estimate_squared / real_squared);
    else
        alpha = 0.0;

    float remainder = as_pessimistic_estimate(alpha, last_rowsum, rowsum, cache, col_entry_pos);
    if (use_cont)
        remainder *= fabs(col_max);

    float total_estimate = fabs(last_max * alpha) + remainder;
    return total_estimate;
}
bool as_wont_update(X_uncompressed Xu, float lambda, float last_max, float* last_rowsum, float* rowsum, S8bCol col, int_fast64_t* column_cache, float col_max, bool use_cont)
{
    float upper_bound = as_combined_estimate(lambda, last_max, last_rowsum, rowsum, col, column_cache, col_max, use_cont);
    return upper_bound <= lambda * total_sqrt_error;
}
