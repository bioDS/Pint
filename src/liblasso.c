#include "liblasso.h"
#include "config.h"
#include <omp.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>
#include <errno.h>
#include <time.h>
#ifdef NOT_R
	#define Rprintf(args...) printf (args);
#else
	#include <R.h>
#endif

static int NumCores = 1;

const static int NORMALISE_Y = 0;
int skipped_updates = 0;
int total_updates = 0;
int skipped_updates_entries = 0;
int total_updates_entries = 0;
static int zero_updates = 0;
static int zero_updates_entries = 0;

static int VERBOSE = 1;
static int *colsum;
static double *col_ysum;
static int max_size_given_entries[61];
//static double max_rowsum = 0;

#define NUM_MAX_ROWSUMS 1
static double max_rowsums[NUM_MAX_ROWSUMS];
static double max_cumulative_rowsums[NUM_MAX_ROWSUMS];

static gsl_permutation *global_permutation;
static gsl_permutation *global_permutation_inverse;
static int_pair *cached_nums;

int min(int a, int b) {
	if (a < b)
		return a;
	return b;
}

typedef struct Queue_Item {
	void *contents;
	void *next;
} Queue_Item;

typedef struct Queue {
	Queue_Item *first_item;
	Queue_Item *last_item;
	int length;
} Queue;

Queue *queue_new() {
	Queue *new_queue = malloc(sizeof(Queue));
	new_queue->length = 0;
	new_queue->first_item = NULL;
	new_queue->last_item = NULL;

	return new_queue;
}

int queue_is_empty(Queue *q) {
	if (q->length == 0)
		return TRUE;
	return FALSE;
}

void queue_push_tail(Queue *q, void *item) {
	struct Queue_Item *new_queue_item = malloc(sizeof(Queue_Item));
	new_queue_item->contents = item;
	new_queue_item->next = NULL;
	// if the queue is currently empty we set both first and last
	// item, rather than  last_item->next
	if (queue_is_empty(q)) {
		q->first_item = new_queue_item;
		q->last_item = new_queue_item;
	} else {
		q->last_item->next = new_queue_item;
		q->last_item = new_queue_item;
	}
	q->length++;
}

int queue_get_length(Queue *q) {
	return q->length;
}

void *queue_pop_head(Queue *q) {
	Queue_Item *first_item = q->first_item;
	if (first_item == NULL) {
		return NULL;
	}

	q->first_item = first_item->next;
	// if we pop'd the only item, don't keep it as last.
	if (NULL == q->first_item) {
		q->last_item = NULL;
	}
	q->length--;

	void *contents = first_item->contents;
	free(first_item);

	return contents;
}

/// Currently assumes that we want to free the contents
/// of everything in the queue as well
void queue_free(Queue *q) {
	Queue_Item *current_item = q->first_item;
	Queue_Item *next_item = current_item->next;

	// free the queue contents
	while (current_item != NULL) {
		free(current_item->contents);
		next_item = current_item->next;
		free(current_item);
		current_item = next_item;
	}
	// and the queue itself
	free(q);
}

static int N;

//TODO: the compiler should really do this
void initialise_static_resources() {
	for (int i = 0; i < 60; i++) {
		max_size_given_entries[i] = 60/(i+1);
	}
	max_size_given_entries[60] = 0;
}

void free_static_resources() {
	if (global_permutation != NULL)
		free(global_permutation);
	if (global_permutation_inverse != NULL)
		free(global_permutation_inverse);
	if (cached_nums != NULL)
		free(cached_nums);
}

int get_p_int(int p, int max_interaction_distance) {
	int p_int = 0;
	//if (max_interaction_distance <= 0 || max_interaction_distance >= p/2)
		p_int = (p*(p+1))/2;
	//else
	//	p_int = p*(2*max_interaction_distance+1);
	return p_int;
}

S8bWord to_s8b(int count, int *vals) {
    S8bWord word;
	word.values = 0;
    word.selector = 0;
	int t = 0;
	//TODO: improve on this
	while(group_size[t] >= count && t < 16)
		t++;
	word.selector = t-1;
	for (int i = 0; i < count; i++) {
		word.values |= vals[count-i-1];
		if (i < count - 1)
			word.values <<= item_width[word.selector];
	}
    return word;
}

int max(int a, int b) {
	if (a > b)
		return a;
	return b;
}

//void update_max_rowsums(double new_value) {
//       if (new_value < max_rowsums[NUM_MAX_ROWSUMS - 1])
//               return;
//
//       #pragma omp critical
//       {
//               //TODO: reasonable search algorithm.
//               int i = NUM_MAX_ROWSUMS - 1;
//               for (; i > 0; i--) {
//                       if (new_value < max_rowsums[i])
//                               break;
//               }
//
//               // i is the index of the smallest value greater than our new one.
//               // shift everything else down
//               for (int j = i; j > NUM_MAX_ROWSUMS; j++) {
//                       max_rowsums[j+1] = max_rowsums[j];
//               }
//               max_rowsums[i] = new_value;
//
//               max_cumulative_rowsums[0] = max_rowsums[0];
//               for (int i = 1; i < NUM_MAX_ROWSUMS; i++) {
//                       max_cumulative_rowsums[i] = max_cumulative_rowsums[i-1] + max_rowsums[i];
//               }
//       }
//}


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
	printf("X: read %d columns from file\n", actual_cols);
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


int_pair get_num(int num, int p) {
	int num_post_permutation = gsl_permutation_get(global_permutation, num);
	return cached_nums[num_post_permutation];
}

int_pair *get_all_nums(int p, int max_interaction_distance) {
	int p_int = get_p_int(p, max_interaction_distance);
	if (max_interaction_distance == -1)
		max_interaction_distance = p_int/2+1;
	int_pair *nums = malloc(p_int*sizeof(int_pair));
	int offset = 0;
	for (int i = 0; i < p; i++) {
		for (int j = i; j < min(p, i+max_interaction_distance + 1); j++) {
			int_pair ip;
			ip.i = i;
			ip.j = j;
			nums[offset] = ip;
			offset++;
		}
	}
	return nums;
}

double update_beta_partition(XMatrix xmatrix, XMatrix_sparse X2, double *Y, double *rowsum, int n, int p, 
						  double lambda, double *beta, double dBMax, double intercept,
						  int_pair *precalc_get_num, int *column_entry_cache, Column_Partition column_partition) {
	int *column_entries = column_entry_cache;

	//TODO: not this.
	double *delta_beta = malloc(X2.p * sizeof(double));
	double *delta_beta_hat = malloc(X2.p * sizeof(double));

	// for every block b
	for (int b = 0; b < column_partition.count; b++) {
		// for every column k in block b at position ki in the block
		for (int ki = 0; ki < column_partition.sets[b].size; ki++) {
			int k = column_partition.sets[b].cols[ki];
			delta_beta[ki] = 0.0;
			double sumk = X2.col_nz[k];
			//double sumn = X2.col_nz[k]*beta[k];
			double sumn = 0.0;

			// find the delta_beta for column j

			//if (X2.col_nz[k] > 0) {
			//	for (int i = 0; i < n; i++) {
			//		delta_beta[ki] += testX2Tiny[i][k] * (Y[i] - rowsum[i]);
			//	}
			//	delta_beta[ki] /= X2.col_nz[k];
			//}

			int col_entry_pos = 0;
			int entry = -1;
			for (int i = 0; i < X2.col_nwords[k]; i++) {
				S8bWord word = X2.compressed_indices[k][i];
				for (int j = 0; j < group_size[word.selector]; j++) {
					int diff = word.values & masks[word.selector];
					if (diff != 0) {
						entry += diff;
						sumn += Y[entry] - intercept - rowsum[entry];
						col_entry_pos++;
					}
					word.values >>= item_width[word.selector];
				}
			}

			// TODO: This is probably slower than necessary.
			double Bk_diff = beta[k];
			if (sumk == 0.0) {
				delta_beta[ki] = 0.0;
				beta[k] = 0.0;
			} else {
				delta_beta[ki] = soft_threshold(sumn, lambda*n/2)/sumk;
			}

		}
		// then correct for simultaneous updates
		//TODO: we decompress the column a second time here, should we cache the entire block instead?
		double Bk_diff = correct_beta_updates(column_partition.sets[b], beta, delta_beta, p, delta_beta_hat, rowsum, X2);
		Bk_diff *= Bk_diff;
		if (Bk_diff > dBMax)
			dBMax = Bk_diff;
		}
	free(delta_beta);
	free(delta_beta_hat);

	return dBMax;
}

double update_beta_cyclic(XMatrix xmatrix, XMatrix_sparse X2, double *Y, double *rowsum, int n, int p, 
						  double lambda, double *beta, int k, double dBMax, double intercept,
						  int_pair *precalc_get_num, int *column_entry_cache) {
	double sumk = X2.col_nz[k];
	double sumn = X2.col_nz[k]*beta[k];
	int *column_entries = column_entry_cache;

	int col_entry_pos = 0;
	int entry = -1;
	for (int i = 0; i < X2.col_nwords[k]; i++) {
		S8bWord word = X2.compressed_indices[k][i];
		for (int j = 0; j < group_size[word.selector]; j++) {
			int diff = word.values & masks[word.selector];
			if (diff != 0) {
				entry += diff;
				column_entries[col_entry_pos] = entry;
				sumn += Y[entry] - intercept - rowsum[entry];
				col_entry_pos++;
			}
			word.values >>= item_width[word.selector];
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
		for (int e = 0; e < X2.col_nz[k]; e++) {
			int i = column_entries[e];
			#pragma omp atomic
			rowsum[i] += Bk_diff;
		}
	} else {
		zero_updates++;
		zero_updates_entries += X2.col_nz[k];
	}


	Bk_diff *= Bk_diff;
	if (Bk_diff > dBMax)
		dBMax = Bk_diff;
	return dBMax;
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

//void save_log_full(int iter, double lambda, double *beta, int n_betas, FILE *log_file) {
//	fprintf(log_file, "%d, %f\n", iter, lambda);
//	for (int i = 0; i < n_betas; i++) {
//		fprintf(log_file, "%f, ", beta[i]);
//	}
//	fprintf(log_file, "\n");
//}

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

//TODO: think about +ve and -ve Y/rowsums here, this doesn't seem right
//int worth_updating(double *col_ysum, XMatrix_sparse X2, int k, int n, int lambda) {
//	if (fabs(col_ysum[k] - max_cumulative_rowsums[min(X2.col_nz[k], NUM_MAX_ROWSUMS - 1)]) > n*lambda/2) {
//		return TRUE;
//	}
//	return FALSE;
//}

double calculate_error(int n, int p_int, XMatrix_sparse X2, double *Y, int **X, double *beta, double p, double intercept, double *rowsum) {
	double error = 0.0;
	for (int row = 0; row < n; row++) {
		double row_err = Y[row] - intercept - rowsum[row];
		error += row_err*row_err;
	}
	return error;
}

/* decompresses column k from X2 into full_column
* returns the size of the column
*/
int decompress_column(XMatrix_sparse X2, int *full_column, int max_column_size, int k) {
	int col_entry_pos = 0;
	int entry = -1;
	for (int i = 0; i < X2.col_nwords[k]; i++) {
		S8bWord word = X2.compressed_indices[k][i];
		for (int j = 0; j < group_size[word.selector]; j++) {
			int diff = word.values & masks[word.selector];
			if (diff != 0) {
				entry += diff;
				full_column[col_entry_pos] = entry;
				col_entry_pos++;
			}
			word.values >>= item_width[word.selector];
		}
	}

	return col_entry_pos;
}

/* Finds the overlap between two fully decompressed (but still sparse) columns.
 */
int find_overlap(int *col1, int *col2, int col1_size, int col2_size) {
	int current_col1_index = 0;
	int current_col2_index = 0;
	int overlap = 0;
	while(current_col1_index < col1_size && current_col2_index < col2_size) {
		while (col1[current_col1_index] < col2[current_col2_index] && current_col1_index < col1_size) {
			current_col1_index++;
		}
		while (col2[current_col2_index] < col1[current_col1_index] && current_col2_index < col2_size) {
			current_col2_index++;
		}

		if (col1[current_col1_index] == col2[current_col2_index]
		 && current_col1_index < col1_size
		 && current_col2_index < col2_size) {
			overlap++;
			current_col2_index++;
		} 
		//else {
		//	if (col1[current_col1_index] < col2[current_col2_index])
		//		current_col1_index++;
		//	if (col1[current_col2_index] < col1[current_col1_index])
		//		current_col2_index++;
		//}
	}

	return overlap;
}

/* TODO: we should sort the matrix so the blocks end up containing similar sized columns.
 * TODO: This function/struct is overkill if we're doing sequential chunks of the matrix.
 * TODO: we only need the diagonal of the overlap matrix
 * Divides the X2 matrix into sets of columns, with determined
 * error overlap between every pair of columns in the set.
 */
Column_Partition divide_into_blocks_of_size(XMatrix_sparse X2, int block_size, int total_columns) {
	int number_of_column_sets = 0;
	Queue *column_set_queue = queue_new();
	int *first_column = malloc(X2.n*sizeof(int));
	int *second_column = malloc(X2.n*sizeof(int));

	Rprintf("dividing matrix into blocks of size %d\n", block_size);

	// iterate through the matrix, assigning each sequential group of `block_size` columns to a new set.
	for (int block_start_column = 0; block_start_column < total_columns; block_start_column += block_size) {
		int size = min(total_columns - block_start_column, block_size);
		int *cols = malloc(size*sizeof(int));
		int **overlap_matrix = malloc(block_size * sizeof(int *));
		for (int i = 0; i < block_size; i++) {
			overlap_matrix[i] = malloc(block_size * sizeof(int));
		}

		// assign everything from block_start to block_start + size to this block.
		for (int column_in_block = 0; column_in_block < size; column_in_block++) {
			cols[column_in_block] = block_start_column + column_in_block;
		}

		// find \forall {i, j} \sum_i^n ( xik . xij )
		// i.e. for each pair of columns in the matrix, how much they overlap.
		for (int first_col = 0; first_col < size; first_col++) {
			for (int second_col = 0; second_col < first_col; second_col++) {
				// find the overlap between first_col and second_col
				int actual_first_col = cols[first_col];
				int actual_second_col = cols[second_col];

				//printf("%d: %d, %d: %d\n", first_col, actual_first_col, second_col, actual_second_col);

				int col1_size = decompress_column(X2, first_column, X2.n, actual_first_col);
				int col2_size = decompress_column(X2, second_column, X2.n, actual_second_col);

				int overlap = find_overlap(first_column, second_column, col1_size, col2_size);

				overlap_matrix[first_col][second_col] = overlap;
				overlap_matrix[second_col][first_col] = overlap;
			}
		}

		Column_Set *new_colset = malloc(sizeof(Column_Set));
		new_colset->size = size;
		new_colset->cols = cols;
		new_colset->overlap_matrix = overlap_matrix;
		number_of_column_sets++;
		queue_push_tail(column_set_queue, new_colset);
	}

	Column_Set *sets = malloc(number_of_column_sets * sizeof(Column_Set));
	for (int i = 0; !queue_is_empty(column_set_queue); i++) {
		Column_Set *current_colset = queue_pop_head(column_set_queue);
		memcpy(&sets[i], current_colset, sizeof(Column_Set));
		free(current_colset);
	}
	

	queue_free(column_set_queue);
	free(first_column);
	free(second_column);
	Column_Partition column_partition;
	column_partition.count = number_of_column_sets;
	column_partition.sets = sets;
	return column_partition;
}

/* Corrects the beta values updated in column set, assuming they were updated at the same time
 * and we want them to be effectively sequential.
 * TODO: currently sequentially uses delta_beta_hat, we would like to use delta_beta in parallel,
 * but there are some scalability questions to be answered there.
 * we pass delta_beta_hat to avoid excessive malloc'ing.
 * 
 * return value: the largest update performed.
 */
double correct_beta_updates(Column_Set column_set, double *beta, double *delta_beta, int num_beta, double *delta_beta_hat, double *rowsum, XMatrix_sparse X2) {
	double largest_delta_beta_hat = 0.0;
	for (int k = 0; k < column_set.size; k++) {
		int actual_k = column_set.cols[k];
		printf("beta[%d]: %f, delta_beta[%d] = %f\n", actual_k, beta[actual_k], k, delta_beta[k]);
		double diff = 0.0;
		if (X2.col_nz[actual_k] != 0) {
			// sum over j < k
			for (int j = 0; j < k; j++) {
				diff += delta_beta_hat[j] * (double)column_set.overlap_matrix[k][j];
				printf("diff += %f * %d (overlap between %d,%d)\n", delta_beta_hat[j], column_set.overlap_matrix[k][j], k, j);
			}

			// divide by 1/{s_k}
			diff /= (double)X2.col_nz[actual_k];
			printf("diff: %f\n", diff);
			//finally, add \delta \beta_k
			delta_beta_hat[k] = delta_beta[k] - diff;
			beta[actual_k] += delta_beta_hat[k];
			if (delta_beta_hat[k] > largest_delta_beta_hat)
				largest_delta_beta_hat = delta_beta_hat[k];
			// update rowsums
			int entry = -1;
			for (int i = 0; i < X2.col_nwords[actual_k]; i++) {
				S8bWord word = X2.compressed_indices[actual_k][i];
				for (int j = 0; j < group_size[word.selector]; j++) {
					int col_diff = word.values & masks[word.selector];
					if (col_diff != 0) {
						entry += col_diff;
						rowsum[entry] += delta_beta_hat[k];
						printf("rowsum[%d] += delta_beta_hat[%d] : %f\n", entry, k, delta_beta_hat[k]);
					}
					word.values >>= item_width[word.selector];
				}
			}
		}
		printf("beta[%d]: %f, delta_beta_hat[%d] = %f\n", actual_k, beta[actual_k], k, delta_beta_hat[k]);
	}
	return largest_delta_beta_hat;
}

/* Edgeworths's algorithm:
 * \mu is zero for the moment, since the intercept (where no effects occurred)
 * would have no effect on fitness, so 1x survived. log(1) = 0.
 * This is probably assuming that the population doesn't grow, which we may
 * not want.
 * TODO: add an intercept
 * TODO: haschanged can only have an effect if an entire iteration does nothing. This should never happen.
 */
double *simple_coordinate_descent_lasso(XMatrix xmatrix, double *Y, int n, int p, 
		int max_interaction_distance, double lambda_min, double lambda_max, char *method, 
		int max_iter, int verbose, double frac_overlap_allowed, double halt_beta_diff, enum LOG_LEVEL log_level,
		char **job_args, int job_args_num) {
	double lambda = lambda_max;
	VERBOSE = verbose;
	int_pair *precalc_get_num;
	int **X = xmatrix.X;
	N = n;

	NumCores = omp_get_num_procs();
	Rprintf("using %d threads\n", NumCores);

	XMatrix_sparse X2 = sparse_X2_from_X(X, n, p, max_interaction_distance, TRUE);
	int p_int = get_p_int(p, max_interaction_distance);

	long total_column_size = 0;
	for (int i = 0; i < p_int; i++) {
		total_column_size += X2.col_nz[i];
	}
	Rprintf("mean column size: %f\n", (double)total_column_size/(double)p_int);

	Rprintf("dividing matrix into updateable sets\n");

	for (int i = 0; i < NUM_MAX_ROWSUMS; i++) {
		max_rowsums[i] = 0;
		max_cumulative_rowsums[i] = 0;
	}

	if (max_interaction_distance == -1) {
		max_interaction_distance = p_int/2+1;
	}
	double *beta;
	beta = malloc(p_int*sizeof(double)); // probably too big in most cases.
	memset(beta, 0, p_int*sizeof(double));
	
	precalc_get_num = malloc(p_int*sizeof(int_pair));
	int offset = 0;
	for (int i = 0; i < p; i++) {
		for (int j = i; j < min(p, i+max_interaction_distance + 1); j++) {
			precalc_get_num[gsl_permutation_get(global_permutation_inverse,offset)].i = i;
			precalc_get_num[gsl_permutation_get(global_permutation_inverse,offset)].j = j;
			offset++;
		}
	}

	cached_nums = get_all_nums(p, max_interaction_distance);

	double error = 0.0, prev_error;
	for (int i = 0; i < n; i++) {
		error += Y[i]*Y[i];
	}
	double intercept = 0.0;

	// initially every value will be 0, since all betas are 0.
	double rowsum[n];
	memset(rowsum, 0, n*sizeof(double));

	colsum = malloc(p_int*sizeof(double));
	memset(colsum, 0, p_int*sizeof(double));

	col_ysum = malloc(p_int*sizeof(double));
	memset(col_ysum, 0, p_int*sizeof(double));
	for (int col = 0; col < p_int; col++) {
		int entry = -1;
		for (int i = 0; i < X2.col_nwords[col]; i++) {
			S8bWord word = X2.compressed_indices[col][i];
			for (int j = 0; j < group_size[word.selector]; j++) {
				int diff = word.values & masks[word.selector];
				if (diff != 0) {
					entry += diff;
					col_ysum[col] += Y[entry];
				}
				word.values >>= item_width[word.selector];
			}
		}
	}

	// find largest number of non-zeros in any column
	int largest_col = 0;
	long total_col = 0;
	for (int i = 0; i < p_int; i++) {
		if (X2.col_nz[i] > largest_col) {
			largest_col = X2.col_nz[i];
		}
		total_col += X2.col_nz[i];
	}
	int main_sum = 0;
	for (int i = 0; i < p; i++)
		for (int j = 0; j < n; j++)
			main_sum += X[i][j];

	struct timespec start, end;
	double cpu_time_used;

	int set_min_lambda = FALSE;
	gsl_permutation *iter_permutation = gsl_permutation_alloc(p_int);
	gsl_rng *iter_rng;
	gsl_permutation_init(iter_permutation);
	gsl_rng_env_setup();
	const gsl_rng_type *T = gsl_rng_default;
	iter_rng = gsl_rng_alloc(T);
	gsl_ran_shuffle(iter_rng, iter_permutation->data, p_int, sizeof(size_t));
	clock_gettime(CLOCK_REALTIME, &start);
	//TODO: make ratio an option
	//double final_lambda = lambda_min;
	//TODO: having this >0 broke the logs. ==0 might break something else.
	//TODO: hacky that doesn't work. figure out what this should be.
	double final_lambda = 0;
	final_lambda = (pow(0.9,50))*lambda;
	Rprintf("running from lambda %.2f to lambda %.2f\n", lambda, final_lambda);
	int lambda_count = 1;
	int iter_count = 0;

	int max_num_threads = omp_get_max_threads();
	int **thread_column_caches = malloc(max_num_threads*sizeof(int*));
	for (int i = 0; i <  max_num_threads; i++) {
		thread_column_caches[i] = malloc(largest_col*sizeof(int));
	}

	FILE *log_file;
	char *log_filename = "lasso_log.log";
	int iter = 0;
	if (check_can_restore_from_log(log_filename, n, p, p_int, job_args, job_args_num)) {
		Rprintf("We can restore from a partial log!\n");
		restore_from_log(log_filename, n, p, p_int, job_args, job_args_num, &iter, 
				&lambda_count, &lambda, beta);
		// we need to recalculate the rowsums
		for (int col = 0; col < p_int; col++) {
			int entry = -1;
			for (int i = 0; i < X2.col_nwords[col]; i++) {
				S8bWord word = X2.compressed_indices[col][i];
				for (int j = 0; j < group_size[word.selector]; j++) {
					int diff = word.values & masks[word.selector];
					if (diff != 0) {
						entry += diff;
						rowsum[entry] += beta[col];
					}
					word.values >>= item_width[word.selector];
				}
			}
		}
	} else {
		Rprintf("no partial log for current job found\n");
	}
	if (log_level != NONE)
		log_file = init_log(log_filename, n, p, p_int, job_args, job_args_num);

	//int set_step_size
	for (; lambda > final_lambda; iter++) {
		// save current beta values to log each iteration
		if (log_level == ITER)
			save_log(iter, lambda, lambda_count, beta, p_int, log_file);
		prev_error = error;
		double dBMax = 0.0; // largest beta diff this cycle

		// update intercept (don't for the moment, it should be 0 anyway)
		//intercept = update_intercept_cyclic(intercept, X, Y, beta, n, p);

		// update the predictor \Beta_k
		//TODO: shuffling is single-threaded and significantly-ish (40s -> 60s? on the workstation) slows things down.
		// it might be possible to do something better than this.
		if (set_min_lambda == TRUE)
			gsl_ran_shuffle(iter_rng, iter_permutation->data, p_int, sizeof(size_t));
		#pragma omp parallel for num_threads(NumCores) private(max_rowsums, max_cumulative_rowsums) shared(col_ysum, xmatrix, X2, Y, rowsum, beta, precalc_get_num) reduction(+:total_updates, skipped_updates, skipped_updates_entries, total_updates_entries, error) reduction(max: dBMax) //schedule(static, 1)
		for (int i = 0; i < p_int; i++) {
			int k = iter_permutation->data[i];

			dBMax = update_beta_cyclic(xmatrix, X2, Y, rowsum, n, p, lambda, beta, k, dBMax, intercept, precalc_get_num, thread_column_caches[omp_get_thread_num()]);
			total_updates++;
			total_updates_entries += X2.col_nz[k];
		}

		if (!set_min_lambda) {
			if (fabs(dBMax) > 0) {
				set_min_lambda = TRUE;
				//final_lambda = (pow(0.9,50))*lambda;
				Rprintf("first change at lambda %f, stopping at lambda %f\n", lambda, final_lambda);
			} else {
				Rprintf("done lambda %d after %d iterations (dbmax: %f), final error %.1f\n", lambda_count, iter + 1, dBMax, error);
				Rprintf(" %d(%f)", lambda_count, lambda);
				lambda_count++;
				lambda *= 0.9;
				iter_count += iter;
				iter = 0;
			}
		}
		error = calculate_error(n, p_int, X2, Y, X, beta, p, intercept, rowsum);


		// Be sure to clean up anything extra we allocate
		// TODO: don't actually do this, see glmnet convergence conditions for a more detailed approach.
		if (prev_error/error < halt_beta_diff) {
			Rprintf("largest change (%f) was less than %f, halting after %d iterations\n", prev_error/error, halt_beta_diff, iter + 1);
			Rprintf("done lambda %d after %d iterations (dbmax: %f), final error %.1f\n", lambda_count, iter + 1, dBMax, error);
			Rprintf(" %d(%f)", lambda_count, lambda);

			if (log_level == LAMBDA)
				save_log(iter, lambda, lambda_count, beta, p_int, log_file);

			lambda_count++;
			lambda *= 0.9;
			iter_count += iter;
			iter = 0;
		} else if (iter == max_iter) {
			Rprintf("stopping after iter (%d) = max_iter (%d) iterations\n", iter + 1, max_iter);

			if (log_level == LAMBDA)
				save_log(iter, lambda, lambda_count, beta, p_int, log_file);

			lambda_count++;
			lambda *= 0.9;
			iter_count += iter;
			iter = 0;
		}
	}
	if (log_level != NONE)
		close_log(log_file);
	Rprintf("\nfinished at lambda = %f\n", lambda);
	Rprintf("after %d total iters\n", iter_count);

	clock_gettime(CLOCK_REALTIME, &end);
	cpu_time_used = ((double)(end.tv_nsec-start.tv_nsec))/1e9 + (end.tv_sec - start.tv_sec);

	Rprintf("lasso done in %.4f seconds, columns skipped %ld out of %ld a.k.a (%f%%)\n", cpu_time_used, skipped_updates, total_updates, (skipped_updates*100.0)/((long)total_updates));
	Rprintf("cols: performed %d zero updates (%f%%)\n", zero_updates, ((float)zero_updates/(total_updates)) * 100);
	Rprintf("skipped entries %ld out of %ld a.k.a (%f%%)\n", skipped_updates_entries, total_updates_entries, (skipped_updates_entries*100.0)/((long)total_updates_entries));
	free(precalc_get_num);
	Rprintf("entries: performed %d zero updates (%f%%)\n", zero_updates_entries, ((float)zero_updates_entries/(total_updates_entries)) * 100);

	//TODO: this really should be 0. Fix things until it is.
	Rprintf("checking how much rowsums have diverged:\n");
	double *temp_rowsum = malloc(n*sizeof(double));
	memset(temp_rowsum, 0, n*sizeof(double));
	for (int col = 0; col < p_int; col++) {
		int entry = -1;
		for (int i = 0; i < X2.col_nwords[col]; i++) {
			S8bWord word = X2.compressed_indices[col][i];
			for (int j = 0; j < group_size[word.selector]; j++) {
				int diff = word.values & masks[word.selector];
				if (diff != 0) {
					entry += diff;
					temp_rowsum[entry] += beta[col];
				}
				word.values >>= item_width[word.selector];
			}
		}
	}
	double total_rowsum_diff = 0;
	double frac_rowsum_diff = 0;
	for (int i = 0; i < n; i++) {
		total_rowsum_diff += fabs((temp_rowsum[i] - rowsum[i]));
		if (fabs(rowsum[i]) > 1)
			frac_rowsum_diff += fabs((temp_rowsum[i] - rowsum[i])/rowsum[i]);
	}
	Rprintf("mean diff: %.2f (%.2f%%)\n", total_rowsum_diff/n, (frac_rowsum_diff*100));
	free(temp_rowsum);

	Rprintf("freeing stuff\n");
	// free beta sets
	// free X2
	for (int i = 0; i < p_int; i++) {
		if (X2.col_nwords[i] > 0)
			free(X2.compressed_indices[i]);
	}
	free(X2.compressed_indices);
	free(X2.col_nz);
	free(X2.col_nwords);


	return beta;
}

int **X2_from_X(int **X, int n, int p) {
	int **X2 = malloc(n*sizeof(int*));
	for (int row = 0; row < n; row++) {
		X2[row] = malloc(((p*(p+1))/2)*sizeof(int));
		int offset = 0;
		for (int i = 0; i < p; i++) {
			for (int j = i; j < p; j++) {
				X2[row][offset++] = X[i][row] * X[j][row];
			}
		}
	}
	return X2;
}


// TODO: write a test comparing this to non-sparse X2
XMatrix_sparse sparse_X2_from_X(int **X, int n, int p, int max_interaction_distance, int shuffle) {
	XMatrix_sparse X2;
	int colno, val, length;
	
	int iter_done = 0;
	int actual_p_int = 0;
	int p_int = p*(p+1)/2;
	//TODO: for the moment we use the maximum possible p_int for allocation, because things assume it.
	p_int = get_p_int(p, max_interaction_distance);
	if (max_interaction_distance < 0)
		max_interaction_distance = p;

	//TODO: granted all these pointers are the same size, but it's messy
	X2.compressed_indices = malloc(p_int*sizeof(int *));
	//X2.col_nz_indices = malloc(p_int*sizeof(int *));
	X2.col_nz = malloc(p_int*sizeof(int));
	memset(X2.col_nz, 0, p_int*sizeof(int));
	X2.col_nwords = malloc(p_int*sizeof(int));
	memset(X2.col_nwords, 0, p_int*sizeof(int));
	actual_p_int = p_int;

	long total_count = 0;
	long total_sum = 0;
	//TODO: iter_done isn't exactly being updated safely
	#pragma omp parallel for shared(X2, X, iter_done) private(length, val, colno) num_threads(NumCores) reduction(+:total_count, total_sum)
	for (int i = 0; i < p; i++) {
		for (int j = i; j < min(i+max_interaction_distance,p); j++) {
			//GQueue *current_col = g_queue_new();
			//GQueue *current_col_actual = g_queue_new();
			Queue *current_col = queue_new();
			Queue *current_col_actual = queue_new();
			// worked out by hand as being equivalent to the offset we would have reached.
			//TODO: include max_interaction_distance
			colno = (2*(p-1) + 2*(p-1)*(i-1) - (i-1)*(i-1) - (i-1))/2 + j;

			// Read through the the current column entries, and append them to X2 as an s8b-encoded list of offsets
			int *col_entries = malloc(60*sizeof(int));
			int count = 0;
			int largest_entry = 0;
			int max_bits = max_size_given_entries[0];
			int diff = 0;
			int prev_row = -1;
			int total_nz_entries = 0;
			for (int row = 0; row < n; row++) {
				val = X[i][row] * X[j][row];
				if (val == 1) {
					total_nz_entries++;
					diff = row - prev_row;
					total_sum += diff;
					int used = 0;
					int tdiff = diff;
					while (tdiff > 0) {
						used++;
						tdiff >>= 1;
					}
					max_bits = max_size_given_entries[count+1];
					// if the current diff won't fit in the s8b word, push the word and start a new one
					if (max(used,largest_entry) > max_size_given_entries[count+1]) {
						S8bWord *word = malloc(sizeof(S8bWord)); // we (maybe?) can't rely on this being the size of a pointer, so we'll add by reference
						S8bWord tempword = to_s8b(count, col_entries);
						total_count += count;
						memcpy(word, &tempword, sizeof(S8bWord));
						queue_push_tail(current_col, word);
						count = 0;
						largest_entry = 0;
						max_bits = max_size_given_entries[1];
					}
					// things for the next iter
					col_entries[count] = diff;
					count++;
					if (used > largest_entry)
						largest_entry = used;
					prev_row = row;
				}
				else if (val != 0)
					fprintf(stderr, "Attempted to convert a non-binary matrix, values will be missing!\n");
			}
			//push the last (non-full) word
			S8bWord *word = malloc(sizeof(S8bWord));
			S8bWord tempword = to_s8b(count, col_entries);
			memcpy(word, &tempword, sizeof(S8bWord));
			queue_push_tail(current_col, word);
			free(col_entries);
			length = queue_get_length(current_col);

			// push all our words to an array in X2
			X2.compressed_indices[colno] = malloc(length*sizeof(S8bWord));
			X2.col_nz[colno] = total_nz_entries;
			X2.col_nwords[colno] = length;
			count = 0;
			while (!queue_is_empty(current_col)) {
				S8bWord *current_word = queue_pop_head(current_col);
				X2.compressed_indices[colno][count] = *current_word;
				free(current_word);
				count++;
			}

			//TODO: remove thise
			// push actual columns for testing purposes
			queue_free(current_col_actual);
			queue_free(current_col);
			current_col = NULL;
			current_col_actual = NULL;
		}
		iter_done++;
	}

	long total_words = 0;
	long total_entries = 0;
	for (int i = 0; i < actual_p_int; i++) {
		total_words += X2.col_nwords[i];
		total_entries += X2.col_nz[i];
	}
	printf("mean count: %f\n", (double)total_count/(double)total_words);
	printf("mean size: %f\n", (double)total_sum/(double)total_entries);

	gsl_rng *r;
	gsl_permutation *permutation = gsl_permutation_alloc(actual_p_int);
	gsl_permutation_init(permutation);
	gsl_rng_env_setup();
	const gsl_rng_type *T = gsl_rng_default;
	r = gsl_rng_alloc(T);
	if (shuffle == TRUE)
		gsl_ran_shuffle(r, permutation->data, actual_p_int, sizeof(size_t));
	//TODO: remove
	S8bWord **permuted_indices = malloc(actual_p_int * sizeof(S8bWord*));
	int *permuted_nz = malloc(actual_p_int * sizeof(int));
	int *permuted_nwords = malloc(actual_p_int *sizeof(int));
	for (int i = 0; i < actual_p_int; i++) {
		permuted_indices[i] = X2.compressed_indices[permutation->data[i]];
		permuted_nz[i] = X2.col_nz[permutation->data[i]];
		permuted_nwords[i] = X2.col_nwords[permutation->data[i]];
	}
	free(X2.compressed_indices);
	free(X2.col_nz);
	free(X2.col_nwords);
	free(r);
	X2.n = n;
	X2.p = p_int;
	X2.compressed_indices = permuted_indices;
	X2.col_nz = permuted_nz;
	X2.col_nwords = permuted_nwords;
	X2.permutation = permutation;
	global_permutation = permutation;
	global_permutation_inverse = gsl_permutation_alloc(permutation->size);
	gsl_permutation_inverse(global_permutation_inverse, permutation);

	return X2;
}
