#include "liblasso.h"
#define verbose FALSE
// #define verbose TRUE

// max of either positive or negative contributions to rowsum sum.
double pessimistic_estimate(double alpha, double *last_rowsum, double *rowsum, XMatrixSparse X, int k, int *column_cache) {
    int n = X.n;
    int colsize = X.col_nz[k];
    double pos_max = 0.0, neg_max = 0.0;
    int count = 0;
    for (int ind = 0; ind < colsize; ind++) {
        int i = column_cache[ind];
        double diff_i = rowsum[i] - alpha*last_rowsum[i];
        // if (verbose && k==interesting_col) {
            // printf("rowsum[%d] = %f\n", i, rowsum[i]);
            // printf("i: %d, diff_i = %f:  %f - %f\n", i, diff_i, rowsum[i], alpha*last_rowsum[i]);
        // }
        if (diff_i > 0) {
            pos_max += (diff_i);
        } else {
            neg_max += diff_i;
        }
        count++;
    }
    double estimate = fmaxf(pos_max, fabs(neg_max));
    if (verbose && k==interesting_col) {
        printf("added %d entries\n", count);
        printf("pos_max: %f, neg_max: %f\n", pos_max, neg_max);
        printf("pressimistic remainder estimate: %f\n", estimate);
    }
    return estimate;
}

double exact_multiple() {
}

// the worst case effect is \leq last_max * alpha + pessimistic_estimate()
//TODO: exclude interactions already in the working set
double l2_combined_estimate(XMatrixSparse X, double lambda, int k, double last_max, double *last_rowsum, double *rowsum,
    int *column_cache) {
    double alpha = 0.0;
    // read through the compressed column
    double estimate_squared = 0.0;
    double real_squared = 0.0;
    int entry = -1;
    int col_entry_pos = 0;
    if (verbose && k == interesting_col) {
        printf("X col %d contains %d entries\t", k, X.col_nz[k]);
        printf("last_rowsum: %d\n", last_rowsum);
    }
	for (int i = 0; i < X.col_nwords[k]; i++) {
		S8bWord word = X.compressed_indices[k][i];
		unsigned long values = word.values;
		for (int j = 0; j <= group_size[word.selector]; j++) {
			int diff = values & masks[word.selector];
			if (diff != 0) {
				entry += diff;
				column_cache[col_entry_pos] = entry;
				col_entry_pos++;

                // do whatever we need here with the index below:
                if (k == interesting_col) {
                    // printf("entry: %d\n", entry);
                }
                if (k == interesting_col && (entry == 0 || entry == 999)) {
                    // printf("ri: %f, l_ri: %f\n", rowsum[entry], last_rowsum[entry]);
                }
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
    if (verbose && k==interesting_col)
        printf("alpha: %f = %f/%f\n", alpha, estimate_squared, real_squared);

    //if (alpha == 1.0) {
    //    alpha = 0.99;
    //}

    double remainder = pessimistic_estimate(alpha, last_rowsum, rowsum, X, k, column_cache);

    double total_estimate = fabs(last_max*alpha) + remainder;
    if (verbose && k==interesting_col)
        printf("effect %d total estimate: %f = %f*%f + %f\n", k, total_estimate, last_max, alpha, remainder);
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
//TODO: should beta[k] be in here?
int wont_update_effect(XMatrixSparse X, double lambda, int k, double last_max, double *last_rowsum, double *rowsum, int *column_cache, double *beta) {
    double upper_bound = l2_combined_estimate(X, lambda, k, last_max, last_rowsum, rowsum, column_cache);
    if (verbose && k==interesting_col) {
        printf("beta[%d] = %f\n", k, beta[k]);
        printf("%d: upper bound: %f < lambda: %f?\n", k, upper_bound, lambda*(X.n/2));
    }
    return  upper_bound <= lambda*(X.n/2);
}