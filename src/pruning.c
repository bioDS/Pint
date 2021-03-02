#include "liblasso.h"

// max of either positive or negative contributions to rowsum sum.
double pessimistic_estimate(double alpha, double *last_rowsum, double *rowsum, XMatrixSparse X, int k, int *column_cache) {
    int n = X.n;
    int colsize = X.col_nz[k];
    double pos_max = 0.0, neg_max = 0.0;
    for (int ind = 0; ind < colsize; ind++) {
        int i = column_cache[ind];
        double diff_i = rowsum[i] - alpha*last_rowsum[i];
        printf("i: %d, diff_i: %f\n", i, diff_i);
        if (diff_i > 0) {
            pos_max += (diff_i);
        } else if (diff_i < 0) {
            neg_max += fabs(diff_i);
        }
    }
    double estimate = fmaxf(pos_max, neg_max);
    printf("pressimistic remainder estimate: %f\n", estimate);
    return estimate;
}

double exact_multiple() {
}

// the worst case effect is \leq last_max * alpha + pessimistic_estimate()
double l2_combined_estimate(XMatrixSparse X, double lambda, int k, double last_max, double *last_rowsum, double *rowsum,
    int *column_cache) {
    double alpha = 0.0;
    // read through the compressed column
    double estimate_squared = 0.0;
    double real_squared = 0.0;
    int entry = 0;
    int col_entry_pos = 0;
    // printf("X col %d contains %d entries\n", k, X.col_nz[k]);
	for (int i = 0; i < X.col_nwords[k]; i++) {
		S8bWord word = X.compressed_indices[k][i];
		unsigned long values = word.values;
		for (int j = 0; j < group_size[word.selector]; j++) {
			int diff = values & masks[word.selector];
			if (diff != 0) {
				entry += diff;
                // printf("entry %d\n", entry);
				column_cache[col_entry_pos] = entry; //TODO: do we need this here
				col_entry_pos++;

                // do whatever we need here with the index below:
                estimate_squared += rowsum[entry]*last_rowsum[entry];
                real_squared     += last_rowsum[entry]*last_rowsum[entry];
			}
			values >>= item_width[word.selector];
		}
	}
    // printf("col_entry_pos: %d\n", col_entry_pos);
    if (real_squared != 0.0)
        alpha = fabs(estimate_squared/real_squared);
    else
        alpha = 0.0;
    //TODO: remove this
    // printf("alpha: %f = %f/%f\n", alpha, estimate_squared, real_squared);

    double remainder = pessimistic_estimate(alpha, last_rowsum, rowsum, X, k, column_cache);

    double total_estimate = last_max*alpha + remainder;
    // printf("total estimate: %f = %f*%f + %f\n", total_estimate, last_max, alpha, remainder);
    return total_estimate;
}

/**
 *  Branch pruning condition from Morvin & Vert
 * returns True if the branch k should be pruned
 * w is the maximum sum over i of y[i] - rowsum[i]
 *      for any interaction with effect k at the last check.
 * last_rowsums is R^n, containing the rowsums for which last_max was checked.
 * rowsums is R^n, the current rowsums
 * column_cache should be a reusable allocated array of size >= X.col_nz[k], somewhere convenient for this thread.
 */
int wont_update_effect(XMatrixSparse X, double lambda, int k, double last_max, double *last_rowsum, double *rowsum, int *column_cache) {
    double upper_bound = l2_combined_estimate(X, lambda, k, last_max, last_rowsum, rowsum, column_cache);
    //printf("upper bound: %f < lambda: %f?\n", upper_bound, lambda);
    return  upper_bound <= lambda;
}