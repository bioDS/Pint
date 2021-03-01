#include "liblasso.h"
#include <omp.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>
#include <errno.h>
#include <limits.h>


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
double *col_ysum;
int max_size_given_entries[61];

double max_rowsums[NUM_MAX_ROWSUMS];
double max_cumulative_rowsums[NUM_MAX_ROWSUMS];

gsl_permutation *global_permutation;
gsl_permutation *global_permutation_inverse;
int_pair *cached_nums;

int min(int a, int b) {
	if (a < b)
		return a;
	return b;
}



static gsl_rng **thread_r;

//TODO: the compiler should really do this
void initialise_static_resources() {
	const gsl_rng_type *T = gsl_rng_default;
	NumCores = omp_get_num_procs();
	printf("using %d cores\n", NumCores);
	thread_r = malloc(NumCores*sizeof(gsl_rng*));
	for (int i = 0; i < NumCores; i++)
		thread_r[i] = gsl_rng_alloc(T);

	for (int i = 0; i < 60; i++) {
		max_size_given_entries[i] = 60/(i+1);
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
 * Assumes rng is capable of 2^32-1 values, it should be one of gsl_rng_{taus,mt19937,ran1xd1}
*/
long rand_long(gsl_rng *thread_rng, long max) {
	long r = -UINT_MAX;
	long rng_max = gsl_rng_max(thread_rng);
	if (rng_max != UINT_MAX) {
		fprintf(stderr, "Chosen psrng cannot generale all ints. This is most likely a mistake.\n");
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


void parallel_shuffle(gsl_permutation* permutation, long split_size, long final_split_size, long splits) {
	#pragma omp parallel for
	for (int i = 0; i < splits; i++) {
		//printf("%p, %p, %d, %d\n", permutation->data, &permutation->data[i*split_size], i, split_size);
		//printf("range %p-%p\n", &permutation->data[i*split_size], &permutation->data[i*split_size] + split_size);
		fisher_yates(&permutation->data[i*split_size], split_size, thread_r[omp_get_thread_num()]);
	}
	if (final_split_size > 0) {
		fisher_yates(&permutation->data[permutation->size - 1 - final_split_size], final_split_size, thread_r[omp_get_thread_num()]);
	}
}

long get_p_int(long p, long max_interaction_distance) {
	long p_int = 0;
	if (max_interaction_distance <= 0 || max_interaction_distance >= p/2)
		p_int = (p*(p+1))/2;
	else
		p_int = (p-max_interaction_distance)*(2*max_interaction_distance+1)
				+ max_interaction_distance*(max_interaction_distance-1);
	return p_int;
}


int max(int a, int b) {
	if (a > b)
		return a;
	return b;
}

XMatrix read_x_csv(char *fn, int n, int p) {
	char *buf = NULL;
	size_t line_size = 0;
	int **X = malloc(p*sizeof(int*));

	for (int i = 0; i < p; i++)
		X[i] = malloc(n*sizeof(int));

	FILE *fp = fopen(fn, "r");
	if (fp == NULL) {
		perror("opening failed");
	}

	int col = 0, row = 0, actual_cols = p;
	int readline_result = 0;
	while((readline_result = getline(&buf, &line_size, fp)) > 0) {
		// remove name from beginning (for the moment)
		int i = 1;
		while (buf[i] != '"')
			i++;
		i++;
		// read to the end of the line
		while (buf[i] != '\n' && i < line_size) {
			if (buf[i] == ',')
				{i++; continue;}
			if (buf[i] == '0') {
				X[col][row] = 0;
			}
			else if (buf[i] == '1') {
				X[col][row] = 1;
			}
			else {
				fprintf(stderr, "format error reading X from %s at row: %d, col: %d\n", fn, row, col);
				exit(0);
			}
			i++;
			if (++col >= p)
				break;
		}
		if (buf[i] != '\n')
			fprintf(stderr, "reached end of file without a newline\n");
		if (col < actual_cols)
			actual_cols = col;
		col = 0;
		if (++row >= n)
			break;
	}
	if (readline_result == -1)
		fprintf(stderr, "failed to read line, errno %d\n", errno);

	if (actual_cols < p) {
		printf("number of columns < p, should p have been %d?\n", actual_cols);
		p = actual_cols;
	}
	free(buf);
	XMatrix xmatrix;
	xmatrix.X = X;
	xmatrix.actual_cols = actual_cols;
	return xmatrix;
}

double *read_y_csv(char *fn, int n) {
	char *buf = malloc(BUF_SIZE);
	char *temp = malloc(BUF_SIZE);
	memset(buf, 0, BUF_SIZE);
	double *Y = malloc(n*sizeof(double));

	FILE *fp = fopen(fn, "r");
	if (fp == NULL) {
		perror("opening failed");
	}

	int col = 0, i = 0;
	while(fgets(buf, BUF_SIZE, fp) != NULL) {
		i = 1;
		// skip the name
		while(buf[i] != '"')
			i++;
		i++;
		if (buf[i] == ',')
			i++;
		// read the rest of the line as a float
		memset(temp, 0, BUF_SIZE);
		int j = 0;
		while(buf[i] != '\n')
			temp[j++] = buf[i++];
		Y[col] = atof(temp);
		col++;
	}

	// for comparison with implementations that normalise rather than
	// finding the intercept.
	if (NORMALISE_Y == 1) {
		printf("%d, normalising y values\n", NORMALISE_Y);
		double mean = 0.0;
		for (int i = 0; i < n; i++) {
			mean += Y[i];
		}
		mean /= n;
		for (int i = 0; i < n; i++) {
			Y[i] -= mean;
		}
	}

	free(buf);
	free(temp);
	return Y;
}

// n.b.: for glmnet gamma should be lambda * [alpha=1] = lambda
double soft_threshold(double z, double gamma) {
	double abs = fabs(z);
	if (abs < gamma)
		return 0.0;
	double val = abs - gamma;
	if (signbit(z))
		return -val;
	else
		return val;
}

double get_sump(int p, int k, int i, double *beta, int **X) {
	double sump = 0;
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
		max_interaction_distance = p_int/2+1;
	int_pair *nums = malloc(p_int*sizeof(int_pair));
	long offset = 0;
	for (int i = 0; i < p; i++) {
		for (int j = i; j < min(p, i+max_interaction_distance ); j++) {
			int_pair ip;
			ip.i = i;
			ip.j = j;
			nums[offset] = ip;
			offset++;
		}
	}
	return nums;
}

double update_beta_cyclic(XMatrix xmatrix, XMatrix_sparse xmatrix_sparse, double *Y, double *rowsum, int n, int p, double lambda, double *beta, long k, double intercept, int_pair *precalc_get_num, int *column_entry_cache) {
	double sumk = xmatrix_sparse.col_nz[k];
	double sumn = xmatrix_sparse.col_nz[k]*beta[k];
	int *column_entries = column_entry_cache;

	long col_entry_pos = 0;
	long entry = -1;
	for (int i = 0; i < xmatrix_sparse.col_nwords[k]; i++) {
		S8bWord word = xmatrix_sparse.compressed_indices[k][i];
		unsigned long values = word.values;
		for (int j = 0; j < group_size[word.selector]; j++) {
			int diff = values & masks[word.selector];
			if (diff != 0) {
				entry += diff;
				column_entries[col_entry_pos] = entry;
				sumn += Y[entry] - intercept - rowsum[entry];
				col_entry_pos++;
			}
			values >>= item_width[word.selector];
		}
	}

	// TODO: This is probably slower than necessary.
	double Bk_diff = beta[k];
	if (sumk == 0.0) {
		beta[k] = 0.0;
	} else {
		beta[k] = soft_threshold(sumn, lambda*n/2)/sumk;
	}
	Bk_diff = beta[k] - Bk_diff;
	// update every rowsum[i] w/ effects of beta change.
	if (Bk_diff != 0) {
		for (int e = 0; e < xmatrix_sparse.col_nz[k]; e++) {
			int i = column_entries[e];
			#pragma omp atomic
			rowsum[i] += Bk_diff;
		}
	} else {
		zero_updates++;
		zero_updates_entries += xmatrix_sparse.col_nz[k];
	}


	return Bk_diff;
}

double update_intercept_cyclic(double intercept, int **X, double *Y, double *beta, int n, int p) {
	double new_intercept = 0.0;
	double sumn = 0.0, sumx = 0.0;

	for (int i = 0; i < n; i++) {
		sumx = 0.0;
		for (int j = 0; j < p; j++) {
			sumx += X[i][j] * beta[j];
		}
		sumn += Y[i] - sumx;
	}
	new_intercept = sumn / n;
	return new_intercept;
}

static long log_file_offset;

// print to log: metadata required to resume from the log
FILE *init_log(char *filename, int n, int p, int num_betas, char **job_args, int job_args_num) {
	FILE *log_file = fopen(filename, "w+");
	fprintf(log_file, "still running\n");
	for (int i = 0; i < job_args_num; i++) {
		fprintf(log_file, "%s ", job_args[i]);
	}
	fprintf(log_file, "\n");
	fprintf(log_file, "lasso log file. metadata follows on the next few lines.\n \
The remaining log is {[ ]done/[w]ip} $current_iter, $current_lambda\\n $beta_1, $beta_2 ... $beta_k");
	fprintf(log_file, "num_rows, num_cols, num_betas\n");
	fprintf(log_file, "%d, %d, %d\n", n, p, num_betas);
	log_file_offset = ftell(log_file);
	return log_file;
}

static int log_pos = 0;
// save the current beta values to a log, so the program can be resumed if it is interrupted
void save_log(int iter, double lambda_value, int lambda_count, double *beta, int n_betas, FILE *log_file) {
	// Rather than filling the log with beta values, we want to only keep two copies.
	// The current one, and a backup in case we stop while writing the current one.
	if (log_pos % 2 == 0) {
		fseek(log_file, log_file_offset, SEEK_SET);
	}
	log_pos = (log_pos + 1) % 2;

	// n.b. Each log line will be the same number of bytes regardless of the actual values here.
	long line_start_pos = ftell(log_file);
	fprintf(log_file, "w");
	fprintf(log_file, "%.5d, %.5d, %+.6e\n", iter, lambda_count, lambda_value);
	for (int i = 0; i < n_betas; i++) {
		fprintf(log_file, "%+.6e, ", beta[i]);
	}
	fprintf(log_file, "\n");
	long line_end_pos = ftell(log_file);
	fseek(log_file, line_start_pos, SEEK_SET);
	fprintf(log_file, " ");
	fseek(log_file, line_end_pos, SEEK_SET);
}

void close_log(FILE *log_file) {
	fseek(log_file, 0, SEEK_SET);
	fprintf(log_file, "finished     \n");
	fclose(log_file);
}

int check_can_restore_from_log(char *filename, int n, int p, int num_betas, char **job_args, int job_args_num) {
	int buf_size = num_betas*16 + 500;
	int can_use = FALSE;
	FILE *log_file = fopen(filename, "r");
	if (log_file == NULL) {
		return FALSE;
	}
	char *our_args = malloc(500);
	char *buffer = malloc(buf_size);

	memset(our_args, 0, sizeof(our_args));
	for (int i = 0; i < job_args_num; i++) {
		sprintf(our_args + strlen(our_args), "%s ", job_args[i]);
	}
	sprintf(our_args + strlen(our_args), "\n");

	//printf("checking log\n");
	fgets(buffer, buf_size, log_file);
	//printf("comparing '%s', '%s'n", buffer, "still running");
	if (strcmp(buffer, "still running\n") == 0) {
		// there was an interrupted run, we should check if it was this one.
		fgets(buffer, buf_size, log_file);
		//printf("comparing '%s', '%s'\n", buffer, our_args);
		if (strcmp(buffer, our_args) == 0) {
			// the files were the same!
			can_use = TRUE;
		}

	}

	free(buffer);
	free(our_args);
	fclose(log_file);
	return can_use;
}

// returns the opened log for future use.
FILE *restore_from_log(char *filename, int n, int p, int num_betas, char **job_args, int job_args_num,
		int *actual_iter, int *actual_lambda_count, double *actual_lambda_value, double *actual_beta) {

	FILE *log_file = fopen(filename, "r+");
	int buf_size = num_betas*16 + 500;
	char *buffer = malloc(buf_size);
	Rprintf("restoring from log\n");

	// (none of this actually changes, we just need to set the log_file_offset)
	fprintf(log_file, "still running\n");
	for (int i = 0; i < job_args_num; i++) {
		fprintf(log_file, "%s ", job_args[i]);
	}
	fprintf(log_file, "\n");
	fprintf(log_file, "lasso log file. metadata follows on the next few lines.\n \
The remaining log is {[ ]done/[w]ip} $current_iter, $current_lambda\\n $beta_1, $beta_2 ... $beta_k");
	fprintf(log_file, "num_rows, num_cols, num_betas\n");
	fprintf(log_file, "%d, %d, %d\n", n, p, num_betas);
	log_file_offset = ftell(log_file);

	// now we're at the first saved line, check whether it's a complete checkpoint.
	int can_restore = TRUE;
	long first_pos = ftell(log_file);
	fgets(buffer, buf_size, log_file);
	printf("first buf: '%s'\n", buffer);
	if (strncmp(buffer, "w", 1) == 0) {
		printf("first entry unusable\n");
		// line is incomplete, don't use it.
		fgets(buffer, buf_size, log_file);
		fgets(buffer, buf_size, log_file);
		printf("second buf: '%s'\n", buffer);
		if (strncmp(buffer, "w", 1) == 0) {
			printf("second entry unusable\n");
			// so is the other one, don't change anything.
			can_restore = FALSE;
			Rprintf("warning: failed to restore from log, all entries were invalid.\n");
		}
	} else {
		// the first one was fine, but the second one may be more recent.
		int first_iter, first_lambda_count;
		int second_iter, second_lambda_count;
		double first_lambda_value;
		double second_lambda_value;
		sscanf(buffer, " %d, %d, %le\n", &first_iter, &first_lambda_count, &first_lambda_value);
		fgets(buffer, buf_size, log_file);
		fgets(buffer, buf_size, log_file);
		long second_pos = ftell(log_file);
		sscanf(buffer, " %d, %d, %le\n", &second_iter, &second_lambda_count, &second_lambda_value);
		printf("first_lambda_count: %d\n", first_lambda_count);
		printf("second_lambda_count: %d\n", second_lambda_count);
		if (strncmp(buffer, "w", 1) == 0 
		|| first_lambda_count > second_lambda_count
		|| (first_lambda_count == second_lambda_count && first_iter > second_iter)) {
			printf("first entry > second_entry\n");
			// but we can't/shouldn't use this one, go back to the first
			fseek(log_file, first_pos, SEEK_SET);
			fgets(buffer, buf_size, log_file);
		}
	}
	if (can_restore) {
		// buffer contains the current lambda and iter values.
		int first_iter = -1, first_lambda_count = -1;
		double first_lambda_value = -1; 
		printf("final buf: '%s'\n", buffer);
		sscanf(buffer, " %d, %d, %le", &first_iter, &first_lambda_count, &first_lambda_value);
		printf("%d, %d, %f\n", first_iter, first_lambda_count, first_lambda_value);
		sscanf(buffer, " %d, %d, %le\n", actual_iter, actual_lambda_count, actual_lambda_value);
		printf("lambda_count is now %d, lambda is now %f, iter is now %d\n", *actual_lambda_count, *actual_lambda_value, *actual_iter);
		// we actually only need the beta values, which are on the current line.
		fgets(buffer, buf_size, log_file);

		//printf("values_buf: '%s'\n", buffer);
		long offset = 0;
		for (int i = 0; i < num_betas; i++) {
			//printf("values_buf: '%s'\n", buffer + offset);
			//printf("reading beta %d\n", i);
			actual_beta[i] = 0.0;
			int ret = sscanf(buffer + offset, "%le, ", &actual_beta[i]);
			if (ret != 1) {
				printf("failed to match value in log, bad things will now happen\n");
			}
			//printf("value %lf\n", actual_beta[i]);
			offset += 15;
		}
	}

	free(buffer);
	Rprintf("done restoring from log\n");
	return log_file;
}

// check a particular pair of betas in the adaptive calibration scheme
int adaptive_calibration_check_beta(double c_bar, double lambda_1, Sparse_Betas beta_1, double lambda_2, Sparse_Betas beta_2, int beta_length) {
	double max_diff = 0.0;
	double adjusted_max_diff = 0.0;

	int b1_ind = 0;
	int b2_ind = 0;

	while (b1_ind < beta_1.count && b2_ind < beta_2.count) {
		while (beta_1.indices[b1_ind] < beta_2.indices[b2_ind] && b1_ind < beta_1.count)
			b1_ind++;
		while (beta_2.indices[b2_ind] < beta_1.indices[b1_ind] && b2_ind < beta_2.count)
			b2_ind++;
		if (b1_ind < beta_1.count && b2_ind < beta_2.count &&
			beta_1.indices[b1_ind] == beta_2.indices[b2_ind]) {
			double diff = fabs(beta_1.betas[b1_ind] - beta_2.betas[b2_ind]);
			if (diff > max_diff)
				max_diff = diff;
			b1_ind++;
		}
	}

	adjusted_max_diff = max_diff / (lambda_1 + lambda_2);

	if (adjusted_max_diff <= c_bar) {
		return 1;
	}
	return 0;
}

// checks whether the last element in the beta_sequence is the one we should stop at, according to
// Chichignoud et als 'Adaptive Calibration Scheme'
// returns TRUE if we are finished, FALSE if we should continue.
int check_adaptive_calibration(double c_bar, Beta_Sequence beta_sequence) {
	// printf("\nchecking %d betas\n", beta_sequence.count);
	for (int i = 0; i < beta_sequence.count; i++) {
		int this_result = adaptive_calibration_check_beta(c_bar, beta_sequence.lambdas[beta_sequence.count-1], beta_sequence.betas[beta_sequence.count-1],
															beta_sequence.lambdas[i], beta_sequence.betas[i],
													beta_sequence.vec_length);
		// printf("result: %d\n", this_result);
		if (this_result == 0) {
			return TRUE;
		}
	}
	return FALSE;
}

/* Edgeworths's algorithm:
 * \mu is zero for the moment, since the intercept (where no effects occurred)
 * would have no effect on fitness, so 1x survived. log(1) = 0.
 * This is probably assuming that the population doesn't grow, which we may
 * not want.
 * TODO: add an intercept
 * TODO: haschanged can only have an effect if an entire iteration does nothing. This should never happen.
 */

int **X2_from_X(int **X, int n, int p) {
	int **X2 = malloc(n*sizeof(int*));
	for (int row = 0; row < n; row++) {
		X2[row] = malloc(((p*(p+1))/2)*sizeof(int));
		int offset = 0;
		for (int i = 0; i < p; i++) {
			for (int j = i; j < p; j++) {
				X2[row][offset++] = X[row][i] * X[row][j];
			}
		}
	}
	return X2;
}

