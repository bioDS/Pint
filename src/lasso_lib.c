#include "lasso_lib.h"
#include <omp.h>
#include <glib-2.0/glib.h>

#define NumCores 4

const static int NORMALISE_Y = 0;
int skipped_updates = 0;
int total_updates = 0;

static int VERBOSE = 0;

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

	printf("reading X from: \"%s\"\n", fn);

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
	printf("read %dx%d, freeing stuff\n", row, actual_cols);
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

	printf("reading Y from: \"%s\"\n", fn);
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

	printf("read %d lines, freeing stuff\n", col);
	free(buf);
	free(temp);
	return Y;
}

// n.b.: for glmnet gamma should be lambda * [alpha=1] = lambda
double soft_threshold(double z, double gamma) {
	if (fabs(z) < gamma)
		return 0.0;
	double val = fabs(z) - gamma;
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
	double sumk = 0.0;
	double sumn = 0.0;
	double sump;
	int **X = xmatrix.X;
	gsl_spmatrix *X_sparse = xmatrix.X_sparse;
	int pairwise_product = 0;
	int_pair ip;
	USE_INT = 1;
	if (USE_INT) {
		//ip = get_num(k, p);
		ip = precalc_get_num[k];
		if (VERBOSE)
			printf("using interaction %d,%d (k: %d)\n", ip.i, ip.j, k);
	} else {
		if (VERBOSE)
			printf("using main effect %d\n", k);
	}

	int i, j, row;
	#pragma omp parallel for num_threads(1) private(sump) shared(X) reduction (+:sumn, sumk)
	for (int e = 0; e < xmatrix_sparse.col_nz[k]; e++) {
		sump = 0.0;
		// TODO: avoid unnecessary calculations for large lambda.
		// e.g. store current_row_count[1..n], sum_largest_betas[1..largest_row_count].
		// - would prevent updating sufficiently small rows only, since we don't know which betas matter.
		//	surely the required row size could be calculated instead.
		// e.g.2. store each row's sum(beta_i*x[row][i]), sump = total - (current k).
		// TODO; linked list to next non-zero row?
		//		- or a column-major sparse format
		if (!USE_INT)
			pairwise_product = X[k][i];
		else {
			row = xmatrix_sparse.col_nz_indices[k][e];
			i = row;
			pairwise_product = 1;
			//pairwise_product = X[ip.i][i] * X[ip.j][i];
		}
		if (pairwise_product != 0) {
			//sump = get_sump(p, k, i, beta, X);
			if (!USE_INT)
				sump = rowsum[i] - X[k][i]*beta[k];
			else {
				// TODO: what if X is not binary?
				if (X[ip.i][i] != 0 && X[ip.j][i] != 0) {
					sump = rowsum[i] - beta[k];
				}
			}
			if (VERBOSE)
				printf("rowsum[%d]: %f\n", i, rowsum[i]);
			if (VERBOSE)
				printf("sump: %f, Y[%d]: %f, intercept: %f\n", sump, i, Y[i], intercept);
			//sumn += X[i][k]?(Y[i] - intercept - sump):0.0;
			if (!USE_INT)
				sumn += (Y[i] - intercept - sump)*(double)X[k][i];
			else
				//TODO: assumes X is binary
				if (pairwise_product != 0)
					sumn += Y[i] - intercept - sump;
			if (VERBOSE)
				printf("adding %f\n", X[k][i]?(Y[i] - intercept - sump):0.0);
			//X_col_totals[k] = sump + X[i][k]?beta[k]:0.0;
		} else {
			skipped_updates++;
		}
		if (!USE_INT)
			sumk += X[k][i] * X[k][i];
		else
			if (pairwise_product != 0)
				sumk++;
		total_updates++;
	}
	if (VERBOSE)
		printf("sumn: %f\n", sumn);
	if (VERBOSE)
		printf("sumk: %f\n", sumk);
	derivative = -sumn;

	// TODO: This is probably slower than necessary.
	double Bk_diff = beta[k];
	if (sumk == 0.0) {
		beta[k] = 0.0;
	} else {
		beta[k] = soft_threshold(sumn, lambda*n/2)/sumk;
		//double Bkn = fmin(0.0, -(derivative + lambda)/(sumk));
		//double Bkp = fmax(0.0, -(derivative - lambda)/(sumk));
		//if (Bkn < 0.0)
		//	beta[k] = Bkn;
		//else if (Bkp > 0.0)
		//	beta[k] = Bkp;
		//else {
		//	beta[k] = 0.0;
		//	//if (VERBOSE)
		//	//	fprintf(stderr, "both \\Beta_k- (%f) and \\Beta_k+ (%f) were invalid\n", Bkn, Bkp);
		//}
	}
	Bk_diff = beta[k] - Bk_diff;
	// update every rowsum[i] w/ effects of beta change.
	if (Bk_diff != 0)
		for (int i = 0; i < n; i++) {
			if (!USE_INT)
				rowsum[i] += Bk_diff * X[k][i];
			else {
				//TODO: again, non-binary?
				if (X[ip.i][i] != 0 && X[ip.j][i] != 0)
					rowsum[i] += Bk_diff;
			}
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

	printf("calculating sparse interaction matrix\n");
	XMatrix_sparse X2 = sparse_X2_from_X(X, n, p, 1);

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

	printf("original lambda: %f n: %d ", lambda, n);
	lambda = lambda;
	printf("effective lambda is %f\n", lambda);

	if (strcmp(method,"cyclic") == 0) {
		printf("using cyclic descent\n");
		use_cyclic = 1;
	} else if (strcmp(method, "greedy") == 0) {
		printf("using greedy descent\n");
		use_greedy = 1;
	}

	if (use_greedy == 0 && use_cyclic == 0) {
		fprintf(stderr, "exactly one of cyclic/greedy must be specified\n");
		return NULL;
	}

	// initially every value will be 0, since all betas are 0.
	double rowsum[n];
	memset(rowsum, 0, n*sizeof(double));

	for (int iter = 0; iter < max_iter; iter++) {
		prev_error = error;
		error = 0;
		double dBMax = 0.0; // largest beta diff this cycle

		// update intercept (don't for the moment, it should be 0 anyway)
		//intercept = update_intercept_cyclic(intercept, X, Y, beta, n, p);
		//iter_lambda = lambda*(max_iter-iter)/max_iter;
		//printf("using lambda = %f\n", iter_lambda);

		if (USE_INT == 0)
			for (int k = 0; k < p; k++) {
				// update the predictor \Beta_k
				//TODO: NULL here seems somewhat unsafe.
				XMatrix_sparse empty_sparse_x2;
				dBMax = update_beta_cyclic(xmatrix, X2, Y, rowsum, n, p, lambda, beta, k, dBMax, intercept, USE_INT, NULL);
			}
		else
			for (int k = 0; k < p_int; k++) {
				if (k % (p_int/100) == 0) {
					printf("*");
					fflush(stdout);
				}

				// update the predictor \Beta_k
				dBMax = update_beta_cyclic(xmatrix, X2, Y, rowsum, n, p, lambda, beta, k, dBMax, intercept, USE_INT, precalc_get_num);
			}

		// caculate cumulative error after update
		printf("calculating error\n");
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
		printf("mean squared error is now %f, w/ intercept %f\n", error, intercept);
		// Be sure to clean up anything extra we allocate
		// TODO: don't actually do this, see glmnet convergence conditions for a more detailed approach.
		if (dBMax < HALT_BETA_DIFF) {
			printf("largest change (%f) was less than %d, halting\n", dBMax, HALT_BETA_DIFF);
			return beta;
		}

		printf("done iteration %d\n", iter);
	}

	if (USE_INT)
		printf("lasso done, skipped_updates %d out of %d (which should be %d) a.k.a (%f\%)\n", skipped_updates, p_int*n*max_iter, total_updates, (skipped_updates*100.0)/(p_int*n*max_iter));
	else
		printf("lasso done, skipped_updates %d out of %d (which should be %d) a.k.a (%f\%)\n", skipped_updates, p*n*max_iter, total_updates, (skipped_updates*100.0)/(p*n*max_iter));
	free(precalc_get_num);
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
	GSList *current_col = NULL;
	int colno, val, length;
	int p_int = (p*(p+1))/2;

	if (!USE_INT) {
		X2.col_nz_indices = malloc(p*sizeof(int *));
		X2.col_nz = malloc(p*sizeof(int));
	} else {
		X2.col_nz_indices = malloc(p_int*sizeof(int *));
		X2.col_nz = malloc(p_int*sizeof(int));
	}

	int offset = 0;
	for (int i = 0; i < p; i++) {
		for (int j = i; j < p; j++) {
			if (!USE_INT && j != i)
				continue;
			colno = offset++;

			for (int row = 0; row < n; row++) {
				val = X[i][row] * X[j][row];
				if (val == 1) {
					current_col = g_slist_prepend(current_col, (void*)(long)row);
				}
				else if (val != 0)
					fprintf(stderr, "Attempted to convert a non-binary matrix, values will be missing!\n");
			}
			length = g_slist_length(current_col);

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
	return X2;
}
