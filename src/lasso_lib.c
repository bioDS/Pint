#include "lasso_lib.h"
#include <omp.h>
#include <glib-2.0/glib.h>
#include <ncurses.h>

#define NumCores 4

const static int NORMALISE_Y = 0;
int skipped_updates = 0;
int total_updates = 0;

static int VERBOSE = 0;
static int zero_updates = 0;
static int haschanged = 1;
static int *colsum;
static double *col_ysum;
static double max_rowsum = 0;


//TODO: try using dancing links
//TODO: stop after |set| = numCores?
Beta_Sets find_beta_sets(XMatrix_sparse x2col, XMatrix_sparse_row x2row, int actual_p_int, int n) {
	Beta_Sets beta_sets;

	int remaining_columns = actual_p_int;
	int iteration_count = 0;
	int *allowable_columns = malloc(actual_p_int*sizeof(int));
	int *todo_columns = malloc(actual_p_int*sizeof(int));
	for (int i = 0; i < actual_p_int; i++) {
		allowable_columns[i] = 1;
		todo_columns[i] = 1;
	}

	GSList *all_sets = NULL;

	// do one iteration only
	while (remaining_columns > 0 && iteration_count < 10) {
		GSList *current_set = NULL;
		//printf("beginning iteration %d, remaining_columns %d\n", iteration_count++, remaining_columns);
		for (int row = 0; row < n; row++) {
			//printf("\nchecking row %d\n", row);
			int removed_one = 0;
			for (int col_ind = 0; col_ind < x2row.row_nz[row]; col_ind++) {
				int col = x2row.row_nz_indices[row][col_ind];
				//printf("\t checking column %d\n", col);
				// remove this column from those allowed to be updated at the same time as the set so far
				// (if it is currently allowed)
				if (allowable_columns[col] == 1) {
					if (removed_one == 0) {
						//printf("\t keeping column %d\n", col);
						removed_one++;
					}
					else {
						//printf("\t removing column %d\n", col);
						allowable_columns[col] = 0;
					}
				} else {
					//printf("\t \t column %d not allowed\n", col);
				}
			}

			//printf("  allowed columns: ");
			//for (int i = 0; i < actual_p_int; i++) {
			//	if (allowable_columns[i] == 1)
			//		printf("%d ", i);
			//}
			//printf("\n");
		}

		//printf("allowed at the same time: \n");
		for (int i = 0; i < actual_p_int; i++) {
			if (allowable_columns[i] == 1) {
				todo_columns[i] = 0;
				remaining_columns--;
				//printf("%d ", i);
				current_set = g_slist_prepend(current_set, (void *)(long)i);
			}
		}
		//printf("\n");
		current_set = g_slist_reverse(current_set);
		all_sets = g_slist_prepend(all_sets, current_set);

		for (int i = 0; i < actual_p_int; i++) {
			allowable_columns[i] = todo_columns[i];
		}
	}

	all_sets = g_slist_reverse(all_sets);

	beta_sets.number_of_sets = g_slist_length(all_sets);
	//printf("printing values from list (length %d):\n", beta_sets.number_of_sets);
	beta_sets.sets = malloc(beta_sets.number_of_sets*sizeof(struct Beta_Set));
	GSList *temp_set_pointer = all_sets;
	int counter = 0;
	while (temp_set_pointer != NULL) {
		//printf("\n");
		GSList *temp_val_pointer = temp_set_pointer->data;
		//printf("reading list %d (length %d)\n", counter, g_slist_length(temp_set_pointer));
		//struct Beta_Set temp_beta_set = malloc(sizeof(struct Beta_Set));
		beta_sets.sets[counter].set = temp_set_pointer->data;
		beta_sets.sets[counter].set_size = g_slist_length(temp_set_pointer->data);
		//while (temp_val_pointer != NULL) {
		//	printf("%ld ", (long)temp_val_pointer->data);

		//	temp_val_pointer = temp_val_pointer->next;
		//}
		temp_set_pointer = temp_set_pointer->next;
		counter++;
	}

	//int current_col = 0;
	//for (int row_ind = 0; row_ind < x2col.col_nz[current_col]; row_ind++) {
	//	int row = x2col.col_nz_indices[current_col][row_ind];
	//	for (int compare_col_ind = 0; compare_col_ind < x2row.row_nz[row]; compare_col_ind++) {
	//		int compare_col = x2row.row_nz_indices[row][compare_col_ind];
	//		if (compare_col == current_col)
	//			continue;
	//		allowable_columns[compare_col] = 0;
	//	}
	//}

	free(allowable_columns);
	return beta_sets;
}

XMatrix read_x_csv(char *fn, int n, int p) {
	char *buf = NULL;
	size_t line_size = 0;
	int **X = malloc(p*sizeof(int*));
	gsl_spmatrix *X_sparse = gsl_spmatrix_alloc(n, p);

	// forces X[...] to be sequential. (and adds some segfaults).
	//int *Xq = malloc(n*p*sizeof(int));
	//for (int i = 0; i < n; i++)
	//	X[i] = &Xq[p*i];

	for (int i = 0; i < p; i++)
		X[i] = malloc(n*sizeof(int));

	move(1,0);
	printw("reading X from: \"%s\"\n", fn);
	refresh();

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
			//printf("setting X[%d][%d] to %c\n", row, col, buf[i]);
			if (buf[i] == '0') {
				X[col][row] = 0;
			}
			else if (buf[i] == '1') {
				X[col][row] = 1;
				gsl_spmatrix_set(X_sparse, row, col, 1);
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
	move(2,0);
	printw("read %dx%d, freeing stuff\n", row, actual_cols);
	refresh();
	free(buf);
	XMatrix xmatrix;
	xmatrix.X = X;
	xmatrix.X_sparse = gsl_spmatrix_ccs(X_sparse);
	xmatrix.actual_cols = actual_cols;
	return xmatrix;
}

double *read_y_csv(char *fn, int n) {
	char *buf = malloc(BUF_SIZE);
	char *temp = malloc(BUF_SIZE);
	memset(buf, 0, BUF_SIZE);
	double *Y = malloc(n*sizeof(double));

	move(3,0);
	printw("reading Y from: \"%s\"\n", fn);
	refresh();
	FILE *fp = fopen(fn, "r");
	if (fp == NULL) {
		perror("opening failed");
	}

	int col = 0, i = 0;
	// drop the first line
	//if (fgets(buf, BUF_SIZE, fp) == NULL)
	//	fprintf(stderr, "failed to read first line of Y from \"%s\"\n", fn);
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
		//printf("temp '%s' set %d: %f\n", temp, col, Y[col]);
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

	move(4,0);
	printw("read %d lines, freeing stuff\n", col);
	refresh();
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

//TODO: applies when the x variables are standardized to have unit variance, is this the case?
//TODO: glmnet also standardizes Y before computing its lambda sequence.
double update_beta_glmnet(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax, double intercept) {
	double derivative = 0.0;
	double sumk = 0.0;
	double sumn = 0.0;
	double sump;
	double new_beta;

	for (int i = 0; i < n; i++) {
		sump = 0.0;
		for (int j = 0; j < p; j++) {
			if (j != k)
				sump += X[i][j]?beta[j]:0.0;
		}
		//sumn += (Y[i] - sump)*(double)X[i][k];
		sumn += X[i][k]?(Y[i] - intercept - sump):0.0;
		sumk += X[i][k] * X[i][k];
	}

	new_beta = soft_threshold(sumn/n, lambda);
	// soft thresholding of n, lambda*[alpha=1]

	if (fabs(beta[k] - new_beta) > dBMax)
		dBMax = fabs(beta[k] - new_beta);
	beta[k] = new_beta;
	if (VERBOSE)
		printf("beta_%d is now %f\n", k, beta[k]);
	return dBMax;
}

// separated to make profiling easier.
// TODO: this is taking most of the time, worth avoiding.
//		- has not been adjusted for on the fly X2.
double get_sump(int p, int k, int i, double *beta, int **X) {
	double sump = 0;
	for (int j = 0; j < p; j++) {
		if (j != k)
			//sump += X[i][j]?beta[j]:0.0;
			sump += X[i][j] * beta[j];
	}
	return sump;
}


//TODO: this takes far too long.
//	-could we store one row (of essentially these) instead?
int_pair get_num(int num, int p) {
	int offset = 0;
	int_pair ip;
	ip.i = -1;
	ip.j = -1;
	for (int i = 0; i < p; i++) {
		for (int j = i; j < p; j++) {
			if (offset == num) {
				ip.i = i;
				ip.j = j;
				return ip;
			}
			offset++;
		}
	}
	return ip;
}

// N.B. main effects are not first in the matrix, X[x][1] is the interaction between genes 0 and 1. (the main effect for gene 1 is at X[1][p])
// That is to say that k<p is not a good indication of whether we are looking at an interaction or not.
double update_beta_cyclic(XMatrix xmatrix, XMatrix_sparse xmatrix_sparse, double *Y, double *rowsum, int n, int p, double lambda, double *beta, int k, double dBMax, double intercept, int USE_INT, int_pair *precalc_get_num) {
	double derivative = 0.0;
	double sumk = xmatrix_sparse.col_nz[k];
	double sumn = xmatrix_sparse.col_nz[k]*beta[k];
	double sump;
	int **X = xmatrix.X;
	gsl_spmatrix *X_sparse = xmatrix.X_sparse;
	int pairwise_product = 0;
	int_pair ip;
	if (USE_INT) {
		//ip = get_num(k, p);
		ip = precalc_get_num[k];
		if (VERBOSE)
			printf("using interaction %d,%d (k: %d)\n", ip.i, ip.j, k);
	} else {
		if (VERBOSE)
			printf("using main effect %d\n", k);
		ip.i = k;
		ip.j = k;
	}
	// From here on things should behave the same (this is set mostly for testing)
	USE_INT = 1;

	int i, j, row;
	//#pragma omp parallel for num_threads(1) private(i) shared(X, Y, xmatrix_sparse, rowsum) reduction (+:sumn, sumk, total_updates)
	if (haschanged == 1) {
		for (int e = 0; e < xmatrix_sparse.col_nz[k]; e++) {
			// TODO: avoid unnecessary calculations for large lambda.
			i = xmatrix_sparse.col_nz_indices[k][e];
			// TODO: assumes X is binary
			sumn += Y[i] - intercept - rowsum[i];
		}
	} else {
		sumn = colsum[k];
	}
	total_updates += xmatrix_sparse.col_nz[k];

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
		haschanged = 1;
		for (int e = 0; e < xmatrix_sparse.col_nz[k]; e++) {
			i = xmatrix_sparse.col_nz_indices[k][e];
			rowsum[i] += Bk_diff;
			if (rowsum[i] < max_rowsum)
				max_rowsum = rowsum[i];
		}
	} else {
		zero_updates += xmatrix_sparse.col_nz[k];
	}


	Bk_diff *= Bk_diff;
	if (Bk_diff > dBMax)
		dBMax = Bk_diff;
	if (VERBOSE)
		printf("beta_%d is now %f\n", k, beta[k]);
	return dBMax;
}

void update_beta_shoot() {
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

double update_beta_greedy_l1(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax) {
	double derivative = 0.0;
	double sumk = 0.0;
	double sumn = 0.0;
	double sump;

	for (int i = 0; i < n; i++) {
		sump = 0.0;
		for (int j = 0; j < p; j++) {
			// if j != k ?
				//sump += X[i][j] * beta[j];
				sump += X[i][j]?beta[j]:0.0;
		}
		//sumn += (Y[i] - sump)*(double)X[i][k];
		sumn += X[i][k]?(Y[i] - sump):0.0;
		sumk += X[i][k] * X[i][k];
	}
	derivative = -sumn;

	// TODO: This is probably slower than necessary.
	double Bkn = fmin(0.0, beta[k] - (derivative - lambda)/(sumk));
	double Bkp = fmax(0.0, beta[k] - (derivative + lambda)/(sumk));
	double Bk_diff = beta[k];
	if (Bkn < 0.0)
		beta[k] = Bkn;
	else if (Bkp > 0.0)
		beta[k] = Bkp;
	else {
		beta[k] = 0.0;
		if (VERBOSE)
			fprintf(stderr, "both \\Beta_k- (%f) and \\Beta_k+ (%f) were invalid\n", Bkn, Bkp);
	}
	Bk_diff = fabs(beta[k] - Bk_diff);
	Bk_diff *= Bk_diff;
	if (Bk_diff > dBMax)
		dBMax = Bk_diff;
	if (VERBOSE)
		printf("beta_%d is now %f\n", k, beta[k]);
	return dBMax;
}

/* Edgeworths's algorithm:
 * \mu is zero for the moment, since the intercept (where no effects occurred)
 * would have no effect on fitness, so 1x survived. log(1) = 0.
 * This is probably assuming that the population doesn't grow, which we may
 * not want.
 * TODO: add an intercept
 */
double *simple_coordinate_descent_lasso(XMatrix xmatrix, double *Y, int n, int p, double lambda, char *method, int max_iter, int USE_INT, int verbose) {
	// TODO: until converged
		// TODO: for each main effect x_i or interaction x_ij
			// TODO: choose index i to update uniformly at random
			// TODO: update x_i in the direction -(dF(x)/de_i / B)
	//TODO: free
	VERBOSE = verbose;
	int_pair *precalc_get_num;
	int **X = xmatrix.X;
	gsl_spmatrix *X_sparse = xmatrix.X_sparse;

	move(7,0);
	printw("calculating sparse interaction matrix\n");
	refresh();
	XMatrix_sparse X2 = sparse_X2_from_X(X, n, p, USE_INT);
	XMatrix_sparse_row X2row = sparse_horizontal_X2_from_X(X, n, p, USE_INT);

	int p_int = p*(p+1)/2;
	double *beta;
	if (USE_INT) {
		beta = malloc(p_int*sizeof(double)); // probably too big in most cases.
		memset(beta, 0, p_int*sizeof(double));
	}
	else {
		beta = malloc(p*sizeof(double)); // probably too big in most cases.
		memset(beta, 0, p*sizeof(double));
	}
	if (USE_INT) {
		precalc_get_num = malloc(p_int*sizeof(int_pair));
		int offset = 0;
		for (int i = 0; i < p; i++) {
			for (int j = i; j < p; j++) {
				precalc_get_num[offset].i = i;
				precalc_get_num[offset].j = j;
				offset++;
			}
		}
	} else {
		p_int = p;
	}

	//int skip_count = 0;
	//int skip_entire_column[p_int];
	//for (int i = 0; i < p; i++)
	//	for (int j = 0; j < i; j++) {
	//		int skip_this = 1;
	//		for (int k = 0; k < n; k++) {
	//			if (X[k][i] != 0 && X[k][j] != 0) {
	//				skip_this = 0;
	//			}
	//		}
	//		if (skip_this == 1) {
	//			skip_count++;
	//			skip_entire_column[i*p + j] = 1;
	//		}
	//	}
	//printf("should skip %d columns\n", skip_count);

	double error = 0, prev_error;
	double intercept = 0.0;
	double iter_lambda;
	int use_cyclic = 0, use_greedy = 0;

	//printw("original lambda: %f n: %d ", lambda, n);
	//lambda = lambda;
	//printw("effective lambda is %f\n", lambda);

	move(8,0);
	if (strcmp(method,"cyclic") == 0) {
		printw("using cyclic descent\n");
		use_cyclic = 1;
	} else if (strcmp(method, "greedy") == 0) {
		printw("using greedy descent\n");
		use_greedy = 1;
	}
	refresh();

	if (use_greedy == 0 && use_cyclic == 0) {
		fprintf(stderr, "exactly one of cyclic/greedy must be specified\n");
		return NULL;
	}

	// initially every value will be 0, since all betas are 0.
	double rowsum[n];
	memset(rowsum, 0, n*sizeof(double));

	colsum = malloc(p_int*sizeof(double));
	memset(colsum, 0, p_int*sizeof(double));

	col_ysum = malloc(p_int*sizeof(double));
	for (int col = 0; col < p_int; col++) {
		for (int row_ind = 0; row_ind < X2.col_nz[col]; row_ind++) {
			col_ysum[col] += Y[X2.col_nz_indices[col][row_ind]];
		}
	}

	// find largest number of non-zeros in any column
	int largest_col = 0;
	long total_col = 0;
	for (int i = 0; i < p_int; i++) {
		if (X2.col_nz[i] > largest_col) {
			largest_col = X2.col_nz[i];
			total_col += X2.col_nz[i];
		}
	}
	move(9,0);
	printw("largest column has %d non-zero entries (out of %d)\n", largest_col, n);
	move(10,0);
	printw("mean column has %f non-zero entries (out of %d)\n", (float)total_col/n, n);
	refresh();

	Beta_Sets beta_sets;
	if (USE_INT == 1)
		beta_sets = find_beta_sets(X2, X2row, p_int, n);
	else
		beta_sets = find_beta_sets(X2, X2row, p, n);

	for (int iter = 0; iter < max_iter; iter++) {
		refresh();
		prev_error = error;
		error = 0;
		double dBMax = 0.0; // largest beta diff this cycle

		// update intercept (don't for the moment, it should be 0 anyway)
		//intercept = update_intercept_cyclic(intercept, X, Y, beta, n, p);
		//iter_lambda = lambda*(max_iter-iter)/max_iter;
		//printf("using lambda = %f\n", iter_lambda);

		haschanged = 1;
		int count=5;
		//#pragma omp parallel for num_threads(1) reduction(+:count) // >1 threads will (unsurprisingly) lead to inconsistent (& not reproducable) results
		//for (int k = 0; k < p_int; k++) {
		//	if (k % (p_int/100) == 0) {
		//		move(12,0);
		//		printw("iteration %d: ", iter);
		//		refresh();
		//		move(12,15);
		//		printw("%d%%", count++);
		//		refresh();
		//	}

			// update the predictor \Beta_k
			// TODO: this check is currently slower than just calculating every non-zero column (for non-huge lambda)
			// TODO: is there any way to keep a running total of the sum of the n largest lambda?
			int *cols_to_update = malloc(p_int*sizeof(int));
			for (int i = 0; i < p_int; i++)
				cols_to_update[i] = -1;
			for (int i = 0; i <  beta_sets.number_of_sets; i++) {
				GSList *temp_list = beta_sets.sets[i].set;
				int counter = 0;
				while (temp_list->next != NULL) {
					cols_to_update[counter++] = (int)(long)temp_list->data;

					temp_list = temp_list->next;
				}
				for (int j = 0; j < beta_sets.sets[i].set_size; j++) {
					int k = cols_to_update[j];
					if (fabs(col_ysum[k] - X2.col_nz[k]*max_rowsum) > n*lambda/2) {
						dBMax = update_beta_cyclic(xmatrix, X2, Y, rowsum, n, p, lambda, beta, k, dBMax, intercept, USE_INT, precalc_get_num);
					}
					else {
						skipped_updates += X2.col_nz[k];
						total_updates += X2.col_nz[k];
					}
				}
			}
		//}
		haschanged = 0;
		printw("\n\n");

		// caculate cumulative error after update
		if (USE_INT == 0)
			for (int row = 0; row < n; row++) {
				double sum = 0;
				for (int k = 0; k < p; k++) {
					sum += X[k][row]*beta[k];
				}
				double e_diff = Y[row] - intercept - sum;
				e_diff *= e_diff;
				error += e_diff;
			}
		else
			for (int row = 0; row < n; row++) {
				double sum = 0;
				for (int k = 0; k < p; k++) {
					sum += X[k][row]*beta[k];
				}
				double e_diff = Y[row] - intercept - sum;
				e_diff *= e_diff;
				error += e_diff;
			}
		error /= n;
		printw("mean squared error is now %f, w/ intercept %f\n", error, intercept);
		printw("indices significantly negative (-500):\n");
		for (int i = 0; i < p_int; i++) {
			if (beta[i] < -500) {
				int_pair ip = get_num(i, p);
				if (ip.i == ip.j)
					printw("main: %d (%d):\t\t\t %f\n", i, ip.i, beta[i]);
				else
					printw("int: %d  (%d, %d):\t\t %f\n", i, ip.i, ip.j, beta[i]);
			}
		}
		// Be sure to clean up anything extra we allocate
		// TODO: don't actually do this, see glmnet convergence conditions for a more detailed approach.
		if (dBMax < HALT_BETA_DIFF) {
			printw("largest change (%f) was less than %d, halting\n", dBMax, HALT_BETA_DIFF);
			return beta;
		}

		printw("done iteration %d\n", iter);
		clrtobot();
	}

	if (USE_INT)
		printw("lasso done, skipped_updates %ld out of %ld a.k.a (%f\%)\n", skipped_updates, total_updates, (skipped_updates*100.0)/((long)total_updates));
	else
		printw("lasso done, skipped_updates %ld out of %ld a.k.a (%f\%)\n", skipped_updates, total_updates, (skipped_updates*100.0)/((long)total_updates));
	free(precalc_get_num);
	printw("performed %d zero updates (%f\%)\n", zero_updates, ((float)zero_updates/(total_updates)) * 100);

	return beta;
}

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


// TODO: write a test comparing this to non-sparse X2
XMatrix_sparse sparse_X2_from_X(int **X, int n, int p, int USE_INT) {
	XMatrix_sparse X2;
	int colno, val, length;
	int p_int = (p*(p+1))/2;
	int iter_done = 0;

	if (!USE_INT) {
		X2.col_nz_indices = malloc(p*sizeof(int *));
		X2.col_nz = malloc(p*sizeof(int));
	} else {
		X2.col_nz_indices = malloc(p_int*sizeof(int *));
		X2.col_nz = malloc(p_int*sizeof(int));
	}

	//TODO: iter_done isn't exactly being updated safely
	#pragma omp parallel for shared(X2, X, iter_done) private(length, val, colno)
	for (int i = 0; i < p; i++) {
		for (int j = i; j < p; j++) {
			GSList *current_col = NULL;
			// only include main effects (where i==j) unless USE_INT is set.
			if (USE_INT || j == i) {
				if (USE_INT)
					// worked out by hand as being equivalent to the offset we would have reached.
					colno = (2*(p-1) + 2*(p-1)*(i-1) - (i-1)*(i-1) - (i-1))/2 + j;
				else
					colno = i;

				for (int row = 0; row < n; row++) {
					val = X[i][row] * X[j][row];
					if (val == 1) {
						current_col = g_slist_prepend(current_col, (void*)(long)row);
					}
					else if (val != 0)
						fprintf(stderr, "Attempted to convert a non-binary matrix, values will be missing!\n");
				}
				length = g_slist_length(current_col);
				current_col = g_slist_reverse(current_col);

				X2.col_nz_indices[colno] = malloc(length*sizeof(int));
				X2.col_nz[colno] = length;

				GSList *current_col_ind = current_col;
				int temp_counter = 0;
				while(current_col_ind != NULL) {
					X2.col_nz_indices[colno][temp_counter++] = (int)(long)current_col_ind->data;
					current_col_ind = current_col_ind->next;
				}

				g_slist_free(current_col);
				current_col = NULL;
			}
		}
		iter_done++;
		if (omp_get_thread_num() == 0) {
			move(7,40);
			printw("%.1f%%\n", (float)iter_done*100/p);
			refresh();
		}
	}
	return X2;
}

//TODO: sparse row matrix (for interaction counts)
XMatrix_sparse_row sparse_horizontal_X2_from_X(int **X, int n, int p, int USE_INT) {
	XMatrix_sparse_row X2;
	int rowno, val, length, colno;
	int p_int = (p*(p+1))/2;

	X2.row_nz_indices = malloc(n*sizeof(int *));
	X2.row_nz = malloc(n*sizeof(int));

	#pragma omp parallel for shared(X2, X) private(length, val, colno)
	for (int rowno = 0; rowno < n; rowno++) {
		GSList *current_row = NULL;
		// only include main effects (where i==j) unless USE_INT is set.
		for (int i = 0; i < p; i++) {
			for (int j = i; j < p; j++) {
				// only include main effects (where i==j) unless USE_INT is set.
				if (USE_INT || j == i) {
					if (USE_INT)
						// worked out by hand as being equivalent to the offset we would have reached.
						colno = (2*(p-1) + 2*(p-1)*(i-1) - (i-1)*(i-1) - (i-1))/2 + j;
					else
						colno = i;
					if (X[i][rowno] * X[j][rowno] == 1) {
						current_row = g_slist_prepend(current_row, (void*)(long)colno);
					}

				}
			}
		}
		length = g_slist_length(current_row);
		current_row = g_slist_reverse(current_row);

		X2.row_nz_indices[rowno] = malloc(length*sizeof(int));
		X2.row_nz[rowno] = length;

		GSList *current_row_ind = current_row;
		int temp_counter = 0;
		while(current_row_ind != NULL) {
			X2.row_nz_indices[rowno][temp_counter++] = (int)(long)current_row_ind->data;
			current_row_ind = current_row_ind->next;
		}

		g_slist_free(current_row);
		current_row = NULL;
	}
	return X2;
}
