#include "liblasso.h"

double pessimistic_estimate(double alpha, double *last_rowsum, double *rowsum, XMatrixSparse X, int k) {

}

double exact_multiple() {
}

// the worst case effect is \leq last_max * alpha + pessimistic_estimate()
double l2_combined_estimate(XMatrixSparse X, double lambda, int k, double last_max, double *last_rowsum, double *rowsum) {
    double alpha = 0.0;
    // read through the compressed column
    double estimate_squared = 0.0;
    double real_squared = 0.0;
    int entry = 0;
    printf("X col %d contains %d entries\n", k, X.col_nz[k]);
	for (int i = 0; i < X.col_nwords[k]; i++) {
		S8bWord word = X.compressed_indices[k][i];
		unsigned long values = word.values;
		for (int j = 0; j < group_size[word.selector]; j++) {
			int diff = values & masks[word.selector];
			if (diff != 0) {
				entry += diff;
                printf("entry\n");
				// column_entries[col_entry_pos] = entry; //TODO: do we need this here
				// col_entry_pos++;

                // do whatever we need here with the index below:
                estimate_squared += rowsum[entry]*last_rowsum[entry];
                real_squared     += last_rowsum[entry]*last_rowsum[entry];
			}
			values >>= item_width[word.selector];
		}
	}
    alpha = estimate_squared/real_squared;
    //TODO: remove this
    printf("alpha: %f = %f/%f\n", alpha, estimate_squared, real_squared);

    return 0.0;
}

/// Branch pruning condition from Morvin & Vert
/// returns True if the branch k should be pruned
/// w is the maximum sum over i of y[i] - rowsum[i]
///      for any interaction with effect k at the last check.
/// last_rowsums is R^n, containing the rowsums for which last_max was checked.
/// rowsums is R^n, the current rowsums
int will_update_effect(XMatrixSparse X, double lambda, int k, double last_max, double *last_rowsum, double *rowsum) {
    return l2_combined_estimate(X, lambda, k, last_max, last_rowsum, rowsum) > lambda;
}