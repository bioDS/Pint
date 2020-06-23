#include <glib-2.0/glib.h>
#include "../src/liblasso.h"
#include <locale.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_permutation.h>
#include <glib-2.0/glib.h>
#include <omp.h>

/* int **X2_from_X(int **X, int n, int p); */
/* double *simple_coordinate_descent_lasso(int **X, double *Y, int n, int p, double lambda, char *method); */
/* double update_beta_greedy_l1(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax); */
/* double update_intercept_cyclic(double intercept, int **X, double *Y, double *beta, int n, int p); */
/* double update_beta_cyclic(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax, double intercept); */
/* double update_beta_glmnet(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax, double intercept); */
/* double soft_threshold(double z, double gamma); */
/* double *read_y_csv(char *fn, int n); */
/* XMatrix read_x_csv(char *fn, int n, int p); */

#define NumCores 4

typedef struct {
	int n;
	int p;
	XMatrix xmatrix;
	int **X;
	double *Y;
	double *rowsum;
	double lambda;
	double *beta;
	int k;
	double dBMax;
	double intercept;
	XMatrix_sparse xmatrix_sparse;
	int_pair *precalc_get_num;
	int *column_cache;
} UpdateFixture;

const static double small_X2_correct_beta[630] = {-83.112248,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-39.419762,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-431.597831,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-56.125867,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-54.818886,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-144.076649,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-64.023489,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-33.646329,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-62.705188,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-334.676519,0.000000,0.000000,-215.196793,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-165.866118,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-112.678381,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-1.284220,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-58.031513,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,3.916624,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-73.009253,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,6.958046,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-120.529141,0.000000,0.000000,0.000000,0.000000,-80.263024};

static void test_update_beta_greedy_l1() {
	printf("not implemented yet\n");
}

static void test_update_intercept_cyclic() {
	printf("not implemented yet\n");
}

static void update_beta_fixture_set_up(UpdateFixture *fixture, gconstpointer user_data) {
	fixture->n = 1000;
	fixture->p = 35;
	fixture->xmatrix = read_x_csv("/home/kieran/work/lasso_testing/testXSmall.csv", fixture->n, fixture->p);
	fixture->X = fixture->xmatrix.X;
	fixture->Y = read_y_csv("/home/kieran/work/lasso_testing/testYSmall.csv", fixture->n);
	fixture->rowsum = malloc(fixture->n*sizeof(double));
	fixture->lambda = 6.46;
	fixture->beta = malloc(fixture->p*sizeof(double));
	memset(fixture->beta, 0, fixture->p*sizeof(double));
	fixture->k = 27;
	fixture->dBMax = 0;
	fixture->intercept = 0;
	printf("%d\n", fixture->X[1][0]);
	int p_int = (fixture->p*(fixture->p+1))/2;
	int_pair *precalc_get_num = malloc(p_int*sizeof(int_pair));
	int offset = 0;
	for (int i = 0; i < fixture->p; i++) {
		for (int j = i; j < fixture->p; j++) {
			precalc_get_num[offset].i = i;
			precalc_get_num[offset].j = j;
			offset++;
		}
	}
	fixture->precalc_get_num = precalc_get_num;

	for (int i = 0; i < fixture->n; i++)
		fixture->rowsum[i] = 0;
	fixture->column_cache = malloc(fixture->n*sizeof(int));
}

static void update_beta_fixture_tear_down(UpdateFixture *fixture, gconstpointer user_data) {
	for (int i = 0; i < fixture->p; i++) {
		free(fixture->xmatrix.X[i]);
	}
	free(fixture->Y);
	free(fixture->rowsum);
	free(fixture->beta);
	free(fixture->precalc_get_num);
	free(fixture->column_cache);
}

static void test_update_beta_cyclic(UpdateFixture *fixture, gconstpointer user_data) {
	printf("beta[27]: %f\n", fixture->beta[27]);
	fixture->xmatrix_sparse = sparse_X2_from_X(fixture->X, fixture->n, fixture->p, -1, FALSE, FALSE);
	update_beta_cyclic(fixture->xmatrix, fixture->xmatrix_sparse, fixture->Y, fixture->rowsum, fixture->n, fixture->p, fixture->lambda, fixture->beta, fixture->k, fixture->dBMax, fixture->intercept, fixture->precalc_get_num, fixture->column_cache);
	printf("beta[27]: %f\n", fixture->beta[27]);
	g_assert_true(fixture->beta[27] != 0.0);
	g_assert_true(fixture->beta[27] < -263.94);
	g_assert_true(fixture->beta[27] > -263.941);
}


static void test_soft_threshold() {
	printf("not implemented yet\n");
}

static void test_read_x_csv() {
	int n = 1000;
	int p = 100;
	XMatrix xmatrix = read_x_csv("/home/kieran/work/lasso_testing/testX.csv", n, p);
	g_assert_true(xmatrix.actual_cols == 100);
	g_assert_true(xmatrix.X[0][0] == 0);
	g_assert_true(xmatrix.X[99][999] == 0);
	g_assert_true(xmatrix.X[16][575] == 1);

	int sum = 0;
	for (int i = 0; i < p; i++) {
		sum += xmatrix.X[i][321];
	}
	g_assert_true(sum == 8);
}

static void test_X2_from_X() {
	printf("not implemented yet\n");
}

static void test_simple_coordinate_descent_set_up(UpdateFixture *fixture, gconstpointer user_data) {
	fixture->n = 1000;
	fixture->p = 35;
	fixture->xmatrix = read_x_csv("/home/kieran/work/lasso_testing/testXSmall.csv", fixture->n, fixture->p);
	fixture->X = fixture->xmatrix.X;
	fixture->Y = read_y_csv("/home/kieran/work/lasso_testing/testYSmall.csv", fixture->n);
	fixture->rowsum = malloc(fixture->n*sizeof(double));
	fixture->lambda = 20;
	int p_int = fixture->p*(fixture->p+1)/2;
	fixture->beta = malloc(p_int*sizeof(double));
	memset(fixture->beta, 0, p_int*sizeof(double));
	fixture->k = 27;
	fixture->dBMax = 0;
	fixture->intercept = 0;
	int_pair *precalc_get_num = malloc(p_int*sizeof(int_pair));
	int offset = 0;
	for (int i = 0; i < fixture->p; i++) {
		for (int j = i; j < fixture->p; j++) {
			precalc_get_num[offset].i = i;
			precalc_get_num[offset].j = j;
			offset++;
		}
	}
	fixture->precalc_get_num = precalc_get_num;

	for (int i = 0; i < fixture->n; i++)
		fixture->rowsum[i] = 0;
}

static void test_simple_coordinate_descent_tear_down(UpdateFixture *fixture, gconstpointer user_data) {
	for (int i = 0; i < fixture->p; i++) {
		free(fixture->xmatrix.X[i]);
	}
	free(fixture->xmatrix.X);
	free(fixture->Y);
	free(fixture->rowsum);
	free(fixture->beta);
	free(fixture->precalc_get_num);
	for (int i = 0; i < fixture->p*(fixture->p+1)/2; i++) {
		#ifdef DENSE_X2
		#ifndef LIMIT_OVERLAP
			free(fixture->xmatrix_sparse.col_nz_indices[i]);
		#endif
		#endif
		free(fixture->xmatrix_sparse.compressed_indices[i]);
	}
	#ifdef DENSE_X2
	free(fixture->xmatrix_sparse.col_nz_indices);
	#endif
	free(fixture->xmatrix_sparse.compressed_indices);
	free(fixture->xmatrix_sparse.col_nz);
	free(fixture->xmatrix_sparse.col_nwords);
}

static void test_simple_coordinate_descent_int(UpdateFixture *fixture, gconstpointer user_data) {
	// are we running the shuffle test, or sequential?
	double acceptable_diff = 0.1;
	int shuffle = FALSE;
	if (user_data == TRUE) {
		printf("\nrunning shuffle test!\n");
		acceptable_diff = 10;
		shuffle = TRUE;
	}
	double *glmnet_beta = read_y_csv("/home/kieran/work/lasso_testing/glmnet_small_output.csv", 630);
	printf("starting interaction test\n");
	fixture->xmatrix = read_x_csv("/home/kieran/work/lasso_testing/testXSmall.csv", fixture->n, fixture->p);
	fixture->X = fixture->xmatrix.X;
	fixture->xmatrix_sparse = sparse_X2_from_X(fixture->X, fixture->n, fixture->p, -1, shuffle, FALSE);
	int p_int = fixture->p*(fixture->p+1)/2;
	double *beta = fixture->beta;

	double dBMax;
	for (int j = 0; j < 10; j++)
		for (int i = 0; i < p_int; i++) {
			//int k = gsl_permutation_get(fixture->xmatrix_sparse.permutation, i);
			int k = fixture->xmatrix_sparse.permutation->data[i];
			//int k = i;
			dBMax = update_beta_cyclic(fixture->xmatrix, fixture->xmatrix_sparse, fixture->Y, fixture->rowsum, fixture->n, fixture->p, fixture->lambda, beta, k, dBMax, 0, fixture->precalc_get_num, fixture->column_cache);
		}

	int no_agreeing = 0;
	for (int i = 0; i < p_int; i++) {
		int k = gsl_permutation_get(fixture->xmatrix_sparse.permutation, i);
		//int k = i;
		printf("testing beta[%d] (%f) ~ %f [", i, beta[i], small_X2_correct_beta[k]);

		if (	(beta[i] < small_X2_correct_beta[k] + acceptable_diff)
			&& 	(beta[i] > small_X2_correct_beta[k] - acceptable_diff)) {
				no_agreeing++;
				printf("x]\n");
			} else {
				printf(" ]\n");
			}
	}
	printf("frac agreement: %f\n", (double)no_agreeing/p_int);
	g_assert_true(no_agreeing == p_int);
}

static void test_simple_coordinate_descent_vs_glmnet(UpdateFixture *fixture, gconstpointer user_data) {
	double *glmnet_beta = read_y_csv("/home/kieran/work/lasso_testing/glmnet_small_output.csv", 630);
	printf("starting interaction test\n");
	fixture->p = 35;
	fixture->xmatrix = read_x_csv("/home/kieran/work/lasso_testing/testXSmall.csv", fixture->n, fixture->p);
	fixture->X = fixture->xmatrix.X;
	fixture->xmatrix_sparse = sparse_X2_from_X(fixture->X, fixture->n, fixture->p, -1, FALSE, FALSE);
	int p_int = fixture->p*(fixture->p+1)/2;
	double *beta = fixture->beta;

	beta = simple_coordinate_descent_lasso(fixture->xmatrix, fixture->Y, fixture->n, fixture->p, -1, 0.05, 1000, "cyclic", 100, 0, 0.01, 1.0001, FALSE, 1, "test");

	double acceptable_diff = 10;
	int no_agreeing = 0;
	for (int i = 0; i < p_int; i++) {
		int k = gsl_permutation_get(fixture->xmatrix_sparse.permutation, i);
		printf("testing beta[%d] (%f) ~ %f [", i, beta[k], glmnet_beta[i]);

		if (	(beta[k] < glmnet_beta[i] + acceptable_diff)
			&& 	(beta[k] > glmnet_beta[i] - acceptable_diff)) {
				no_agreeing++;
				printf("x]\n");
			} else {
				printf(" ]\n");
			}
	}
	printf("frac agreement: %f\n", (double)no_agreeing/p_int);
	g_assert_true(no_agreeing >= 0.8*p_int);
}

// will fail if Y has been normalised
static void test_read_y_csv() {
	int n = 1000;
	double *Y = read_y_csv("/home/kieran/work/lasso_testing/testYSmall.csv", n);
	g_assert_true(Y[0] == -133.351709197933);
	g_assert_true(Y[999] == -352.293608898344);
}

//assumes little endian
void printBits(size_t const size, void const * const ptr) {
    unsigned char *b = (unsigned char*) ptr;
    unsigned char byte;
    int i, j;

    for (i=size-1;i>=0;i--)
    {
        for (j=7;j>=0;j--)
        {
            byte = (b[i] >> j) & 1;
            printf("%u", byte);
        }
    }
    puts("");
}

static void check_X2_encoding() {
	int n = 1000;
	int p = 35;
	int p_int = p*(p+1)/2;
	XMatrix xmatrix = read_x_csv("/home/kieran/work/lasso_testing/testXSmall.csv", n, p);
	XMatrix_sparse xmatrix_sparse = sparse_X2_from_X(xmatrix.X, n, p, -1, FALSE, FALSE);

	// mean entry size
	long total = 0;
	int no_entries = 0;
	for (int i = 0; i < p_int; i++) {
		no_entries += xmatrix_sparse.col_nz[i];
		for (int j = 0; j < xmatrix_sparse.col_nz[i]; j++) {
			total += xmatrix_sparse.col_nz_indices[i][j];
		}
	}
	printf("\nmean entry size: %f\n", (double)total/(double)no_entries);

	// mean diff size
	total = 0;
	int prev_entry = 0;
	for (int i = 0; i < p_int; i++) {
		prev_entry = 0;
		for (int j = 0; j < xmatrix_sparse.col_nz[i]; j++) {
			total += xmatrix_sparse.col_nz_indices[i][j] - prev_entry;
			prev_entry = xmatrix_sparse.col_nz_indices[i][j];
		}
	}
	printf("mean diff size: %f\n", (double)total/(double)no_entries);

	printf("size of s8bword struct: %d (int is %ld)\n", sizeof(S8bWord), sizeof(int));

	int item_width[16] = {0,   0,   1,  2,  3,  4,  5,  6,  7, 8, 10, 12, 15, 20, 30, 60};
	int group_size[16] = {240, 120, 60, 30, 20, 15, 12, 10, 8, 7, 6,  5,  4,  3,  2,  1};
	int masks[16];
	for (int i = 0; i < 16; i++)
		masks[i] = (1<<item_width[i]) - 1;

	S8bWord test_word;
	test_word.selector = 7;
	test_word.values = 0;
	unsigned int numbers[10] = {3,2,4,20,1,14,30,52,10,63};
	for (int i = 0; i < 10; i++) {
		test_word.values |= numbers[9-i];
		if (i < 9)
			test_word.values <<= item_width[test_word.selector];
	}

	S8bWord w2 =  to_s8b(10, numbers);

	g_assert_true(sizeof(S8bWord) == 8);
	g_assert_true(test_word.selector == w2.selector);
	g_assert_true(test_word.values == w2.values);

	int max_size_given_entries[61];
	for (int i = 0; i < 60; i++) {
		max_size_given_entries[i] = 60/(i+1);
	}
	max_size_given_entries[60] = 0;

	printf("num entries in col 0: %d\n", xmatrix_sparse.col_nz[0]);
	int *col_entries = malloc(60*sizeof(int));
	int count = 0;
	//GList *s8b_col = NULL;
	GQueue *s8b_col = g_queue_new();
	// work out s8b compressed equivalent of col 0
	int largest_entry = 0;
	int max_bits = max_size_given_entries[0];
	int diff = xmatrix_sparse.col_nz_indices[0][0] + 1;
	for (int i = 0; i < xmatrix_sparse.col_nz[0]; i++) {
		if (i != 0)
			diff = xmatrix_sparse.col_nz_indices[0][i] - xmatrix_sparse.col_nz_indices[0][i-1];
		printf("current no. %d, diff %d. available bits %d\n", xmatrix_sparse.col_nz_indices[0][i], diff, max_bits);
		// update max bits.
		int used = 0;
		int tdiff = diff;
		while (tdiff > 0) {
			used++;
			tdiff >>= 1;
		}
		max_bits = max_size_given_entries[count+1];
		// if the current diff won't fit in the s8b word, push the word and start a new one
		if (diff > 1<<max_bits || largest_entry > max_size_given_entries[count+1]) {
			if (diff > 1<<max_bits)
				printf(" b ");
			if (largest_entry > max_size_given_entries[count+1])
				printf(" c ");
			printf("pushing word with %d entries: ", count);
			for (int j = 0; j < count; j++)
				printf("%d ", col_entries[j]);
			printf("\n");
			S8bWord *word = malloc(sizeof(S8bWord));
			S8bWord tempword = to_s8b(count, col_entries);
			memcpy(word, &tempword, sizeof(S8bWord));
			g_queue_push_tail(s8b_col, word);
			count = 0;
			largest_entry = 0;
			max_bits = max_size_given_entries[1];
		}
		col_entries[count] = diff;
		count++;
		if (used > largest_entry)
			largest_entry = used;
	}
	//push the last (non-full) word
	S8bWord *word = malloc(sizeof(S8bWord));
	S8bWord tempword = to_s8b(count, col_entries);
	memcpy(word, &tempword, sizeof(S8bWord));
	g_queue_push_tail(s8b_col, word);

	free(col_entries);
	int length = g_queue_get_length(s8b_col);

	S8bWord *actual_col = malloc(length*sizeof(S8bWord));
	count = 0;
	while (!g_queue_is_empty(s8b_col)) {
		S8bWord *current_word = g_queue_pop_head(s8b_col);
		memcpy(&actual_col[count], current_word, sizeof(S8bWord));
		count++;
	}

	printf("checking [s8b] == [short]\n");
	int col_entry_pos = 0;
	int entry = -1;
	for (int i = 0; i < length; i++) {
		S8bWord word = actual_col[i];
		for (int j = 0; j < group_size[word.selector]; j++) {
			int diff = word.values & masks[word.selector];
			entry += diff;
			//printf("b: ");
			//printBits(8, &word);
			//printf("\n");
			if (diff != 0) {
				printf("%d == %d\n", entry, xmatrix_sparse.col_nz_indices[0][col_entry_pos]);
				g_assert_true(entry == xmatrix_sparse.col_nz_indices[0][col_entry_pos]);
				col_entry_pos++;
			}
			word.values >>= item_width[word.selector];
		}
	}

	int bytes = length*sizeof(S8bWord);
	printf("col[0] contains %d words, for a toal of %d bytes, instead of %d shorts (%d bytes). Effective reduction %f\n",
		length, bytes, xmatrix_sparse.col_nz[0], xmatrix_sparse.col_nz[0]*sizeof(short), (double)bytes/(xmatrix_sparse.col_nz[0]*sizeof(short)));

	printf("liblasso vs test compressed first col:\n");
	for (int i = 0; i < xmatrix_sparse.col_nwords[0]; i++) {
		printf("%d == %d\n", xmatrix_sparse.compressed_indices[0][i].selector, actual_col[i].selector);
		g_assert_true(xmatrix_sparse.compressed_indices[0][i].selector == actual_col[i].selector);
		printf("%d == %d\n", xmatrix_sparse.compressed_indices[0][i].values, actual_col[i].values);
		g_assert_true(xmatrix_sparse.compressed_indices[0][i].values == actual_col[i].values);
	}
	g_assert_true(xmatrix_sparse.col_nwords[0] == length);
	printf("correct number of words\n");
}

void test_find_overlap() {
	//col1: {1,0,0,1,0,0,1,0,0,1};
	//col2: {0,0,1,0,1,1,1,0,0,1};

	// extra overlap at end to see if we end up hitting it somehow,
	// last value is not actually part of the column.
	{
		int col1[5] = {0,3,6,7,10};
		int col2[6] = {2,4,5,6,9,10};

		int overlap = find_overlap(col1, col2, 4, 5);
		printf("overlap: %d == 1?\n", overlap);
		g_assert_true(overlap == 1);

		overlap = find_overlap(col2, col1, 5, 4);
		printf("overlap: %d == 1?\n", overlap);
		g_assert_true(overlap == 1);
	}

	{
		int col1[5] = {0,3,4,7,10};
		int col2[6] = {4,5,6,7,9,10};

		int overlap = find_overlap(col1, col2, 4, 5);
		printf("overlap: %d == 2?\n", overlap);
		g_assert_true(overlap == 2);

		overlap = find_overlap(col2, col1, 5, 4);
		printf("overlap: %d == 2?\n", overlap);
		g_assert_true(overlap == 2);
	}
}

void test_block_division() {
	XMatrix X = read_x_csv("/home/kieran/work/lasso_testing/testXTiny.csv", 11, 4);
	int **testX2Tiny = X2_from_X(X.X, 11, 4);
	for (int i = 0; i < 11; i++) {
		for (int j = 0; j < 10; j++) {
			printf("%d,", testX2Tiny[i][j]);
		}
		printf("\n");
	}
	XMatrix_sparse X2 = sparse_X2_from_X(X.X, 11, 4, -1, FALSE, FALSE);
	int block_size = 4;
	Column_Partition column_partition = divide_into_blocks_of_size(X2, block_size, 10);

	g_assert_true(column_partition.count == 3);

	g_assert_true(column_partition.sets[0].size == 4);
	{
		int correct_values[4] = {0,1,2,3};
		for (int i = 0; i < 4; i++)
			g_assert_true(column_partition.sets[0].cols[i] == correct_values[i]);
		g_assert_true(column_partition.sets[0].overlap_matrix[0][1] == 0);
		g_assert_true(column_partition.sets[0].overlap_matrix[1][2] == 0);
		g_assert_true(column_partition.sets[0].overlap_matrix[0][3] == 2);
	}

	g_assert_true(column_partition.sets[1].size == 4);
	{
		int correct_values[4] = {4,5,6,7};
		for (int i = 0; i < 4; i++)
			g_assert_true(column_partition.sets[1].cols[i] == correct_values[i]);
		g_assert_true(column_partition.sets[1].overlap_matrix[0][1] == 1);
		g_assert_true(column_partition.sets[1].overlap_matrix[0][3] == 1);
		g_assert_true(column_partition.sets[1].overlap_matrix[0][2] == 1);
		g_assert_true(column_partition.sets[1].overlap_matrix[1][2] == 0);
		g_assert_true(column_partition.sets[1].overlap_matrix[1][3] == 1);
		g_assert_true(column_partition.sets[1].overlap_matrix[2][3] == 0);
		g_assert_true(column_partition.sets[1].overlap_matrix[2][3] == 0);
	}

	g_assert_true(column_partition.sets[2].size == 2);
	{
		int correct_values[4] = {8,9,-1,-1};
		for (int i = 0; i < 2; i++)
			g_assert_true(column_partition.sets[2].cols[i] == correct_values[i]);
		g_assert_true(column_partition.sets[2].overlap_matrix[0][1] == 2);
	}

}

void test_block_division_large_time() {
	int n = 10000;
	int p = 1000;
	XMatrix X = read_x_csv("/home/kieran/work/lasso_testing/X_nlethals50_v15803.csv", n, p);
	XMatrix_sparse X2 = sparse_X2_from_X(X.X, n, p, -1, TRUE, FALSE);
	int block_size = 400;

	double start_time = omp_get_wtime();
	Column_Partition column_partition = divide_into_blocks_of_size(X2, block_size, X2.p);
	double end_time = omp_get_wtime();

	printf("time taken: %f seconds\n", end_time - start_time);
}

void test_update_beta_block() {

}

void test_correct_beta_updates() {
	XMatrix X = read_x_csv("/home/kieran/work/lasso_testing/testXTiny.csv", 11, 4);
	int **testX2Tiny = X2_from_X(X.X, 11, 4);
	printf("X2:\n");
	for (int i = 0; i < 11; i++) {
		for (int j = 0; j < 10; j++) {
			printf("%d,", testX2Tiny[i][j]);
		}
		printf("\n");
	}
	XMatrix_sparse X2 = sparse_X2_from_X(X.X, 11, 4, -1, FALSE, FALSE);
	int block_size = 4;

	int **column_entry_caches = malloc(block_size*sizeof(int*));
	for (int i = 0; i < block_size; i++) {
		column_entry_caches[i] = malloc(X2.n*sizeof(int));
	}

	Column_Partition column_partition = divide_into_blocks_of_size(X2, block_size, 10);

	// arbitrary beta changes for each column, check to see if they're accumulated correctly.
	double Y[11] = {17.1, 79.10, 29.10, 95.5, 27.3, 36.3, 37.1, 49.8, 89.2, 89.8, 42.3, 90.6};
	double beta[10] = {3.4, 5.7, 2.3, 55.0, 34.2, 23.1, 56.2, 17.2, 19.2, 0.2, 10.9};
	double check_beta[10] = {3.4, 5.7, 2.3, 55.0, 34.2, 23.1, 56.2, 17.2, 19.2, 0.2, 10.9};
	double delta_beta[10];
	double check_delta_beta[10];
	double delta_beta_hat[10];
	//double error[11];
	double rowsum[11];
	double check_rowsum[11];

	int n = 11;
	int p = 10;

	// Initialise rowsums
	for (int i = 0; i < n; i++) {
		rowsum[i] = 0.0;
		check_rowsum[i] = 0.0;
		for (int j = 0; j < p; j++) {
			rowsum[i] += beta[j] * (double)testX2Tiny[i][j];
			check_rowsum[i] += beta[j] * (double)testX2Tiny[i][j];
		}
		//error[i] = Y[i] - rowsum[i];
		printf("r%d: %f\n", i, rowsum[i]);
	}

	//update_beta_cyclic(X, X2, Y, rowsum, n, p, 0.0, beta, j, 0)
	// Calculate delta betas
	for (int b = 0; b < column_partition.count; b++) {
		for (int ji = 0; ji < column_partition.sets[b].size; ji++) {
			int j = column_partition.sets[b].cols[ji];
			delta_beta[ji] = 0.0;
			if (X2.col_nz[j] > 0) {
				for (int i = 0; i < n; i++) {
					delta_beta[ji] += testX2Tiny[i][j] * (Y[i] - rowsum[i]);
				}
				// delta_beta[ji] /= X2.col_nz[j];
			}
			printf("setting column_entry_caches[%d]\n", ji);
			decompress_column(X2, column_entry_caches[ji], X2.n, j);
		}
		correct_beta_updates(column_partition.sets[b], beta, delta_beta, p, delta_beta_hat, rowsum, X2, 0.0, column_entry_caches);
	}
	for (int j = 0; j < p; j++) {
		check_delta_beta[j] = 0.0;
		if (X2.col_nz[j] > 0) {
			for (int i = 0; i < n; i++) {
				check_delta_beta[j] += testX2Tiny[i][j] * (Y[i] - check_rowsum[i]);
			}
			check_delta_beta[j] /= X2.col_nz[j];
			for (int i = 0; i < n; i++) {
				check_rowsum[i] += check_delta_beta[j] * (double)testX2Tiny[i][j];
			}
			check_beta[j] += check_delta_beta[j];
		}
	}

	for (int j = 0; j < p; j++) {
		printf("actual delta_hat_beta[%d]: %f\n", j, check_delta_beta[j]);
	}

	for (int j = 0; j < 10; j++) {
		printf("checking beta[%d] (%f) == %f\n", j, beta[j], check_beta[j]);
		g_assert_true(fabs(beta[j] - check_beta[j]) < 0.001);
	}
	for (int i = 0; i < 11; i++) {
		printf("checking rowsum[%d] (%f) == %f\n", i, rowsum[i], check_rowsum[i]);
		g_assert_true(fabs(rowsum[i] - check_rowsum[i]) < 0.001);
	}

	for (int i = 0; i < block_size; i++) {
		free(column_entry_caches[i]);
	}
	free(column_entry_caches);
}

void test_update_beta_partition(double lambda);

void test_update_beta_partition_lambda0() {
	test_update_beta_partition(0.0);
}
void test_update_beta_partition_lambda1() {
	test_update_beta_partition(6.4);
}

void test_update_beta_partition(double lambda) {
	XMatrix X = read_x_csv("/home/kieran/work/lasso_testing/testXTiny.csv", 11, 4);
	int **testX2Tiny = X2_from_X(X.X, 11, 4);
	printf("X2:\n");
	for (int i = 0; i < 11; i++) {
		for (int j = 0; j < 10; j++) {
			printf("%d,", testX2Tiny[i][j]);
		}
		printf("\n");
	}
	int n = 11;
	int p = 10;
	XMatrix_sparse X2 = sparse_X2_from_X(X.X, 11, 4, -1, FALSE, FALSE);
	int block_size = 4;

	Column_Partition column_partition = divide_into_blocks_of_size(X2, block_size, p);

	// arbitrary beta changes for each column, check to see if they're accumulated correctly.
	double Y[11] = {17.1, 79.10, 29.10, 95.5, 27.3, 36.3, 37.1, 49.8, 89.2, 89.8, 42.3, 90.6};
	double beta[10] = {3.4, 0.0, 2.3, 55.0, 34.2, 23.1, 56.2, 17.2, 19.2, 0.2, 10.9};
	double check_beta[10] = {3.4, 0.0, 2.3, 55.0, 34.2, 23.1, 56.2, 17.2, 19.2, 0.2, 10.9};
	double check_delta_beta[10];
	//double error[11];
	double rowsum[11];
	double check_rowsum[11];


	// Initialise rowsums
	for (int i = 0; i < n; i++) {
		rowsum[i] = 0.0;
		check_rowsum[i] = 0.0;
		for (int j = 0; j < p; j++) {
			rowsum[i] 		+= beta[j] * (double)testX2Tiny[i][j];
			check_rowsum[i]	+= beta[j] * (double)testX2Tiny[i][j];
		}
		//error[i] = Y[i] - rowsum[i];
	}
	int_pair *precalc_get_num = malloc(p*sizeof(int_pair));
	int offset = 0;
	for (int i = 0; i < p; i++) {
		for (int j = i; j < 4; j++) {
			precalc_get_num[offset].i = i;
			precalc_get_num[offset].j = j;
			offset++;
		}
	}

	int max_num_threads = 4;
	int largest_col = n;
	int **thread_column_caches = malloc(max_num_threads*sizeof(int*));
	for (int i = 0; i <  max_num_threads; i++) {
		thread_column_caches[i] = malloc(largest_col*sizeof(int));
	}
	for (int k = 0; k < p; k++)
		update_beta_cyclic(X, X2, Y, check_rowsum, n, p, lambda, check_beta, k, 0.0, 0.0, precalc_get_num, thread_column_caches[0]);

	double *delta_beta = malloc(block_size*sizeof(double));
	double *delta_beta_hat = malloc(block_size*sizeof(double));
	update_beta_partition(X, X2, Y, rowsum, n, p, lambda, beta, 0.0, 0.0, precalc_get_num, thread_column_caches, column_partition,
						delta_beta, delta_beta_hat, MULTIPLIER);

	for (int j = 0; j < p; j++) {
		printf("checking beta[%d] (%f) == %f\n", j, beta[j], check_beta[j]);
		g_assert_true(fabs(beta[j] - check_beta[j]) < 0.001 );
	}
	for (int i = 0; i < n; i++) {
		printf("checking rowsum[%d] (%f) == %f\n", i, rowsum[i], check_rowsum[i]);
		g_assert_true(fabs(rowsum[i] - check_rowsum[i]) < 0.001 );
	}

	for (int i = 0; i <  max_num_threads; i++) {
		free(thread_column_caches[i]);
	}
	free(precalc_get_num);
	free(delta_beta);
	free(delta_beta_hat);
}
double test_update_beta_partition_repeat_multiplier(int num_updates, int multiplier, XMatrix X, XMatrix_sparse X2, double *Y) {
	int n = 10000;
	int p = 1000;
	int block_size = omp_get_max_threads()*multiplier;
	//XMatrix X = read_x_csv("/home/kieran/work/lasso_testing/X_nlethals50_v15803.csv", n, p);
	//printf("reading Y\n");
	//double *Y = read_y_csv("/home/kieran/work/lasso_testing/Y_nlethals50_v15803.csv", n);
	int p_int = (p*(p+1))/2;
	//XMatrix_sparse X2 = sparse_X2_from_X(X.X, n, p, -1, FALSE, FALSE);
	double lambda = 6.46;

	double *beta = malloc(p_int * sizeof(double));
	memset(beta, 0, p_int*sizeof(double));
	double *rowsum = malloc(n * sizeof(double));
	memset(rowsum, 0, n*sizeof(double));

	printf("dividing into blocks\n");
	Column_Partition column_partition = divide_into_blocks_of_size(X2, block_size, X2.p);

	printf("initialising variables\n");
	// omp_set_num_threads(8);
	// Initialise rowsums
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
	int_pair *precalc_get_num = malloc(p*sizeof(int_pair));
	int offset = 0;
	for (int i = 0; i < p; i++) {
		for (int j = i; j < 4; j++) {
			precalc_get_num[offset].i = i;
			precalc_get_num[offset].j = j;
			offset++;
		}
	}

	// int max_num_threads = 4;
	int max_num_threads = block_size;
	int largest_col = X2.n;
	int **thread_column_caches = malloc(max_num_threads*sizeof(int*));
	for (int i = 0; i <  max_num_threads; i++) {
		thread_column_caches[i] = malloc(largest_col*sizeof(int));
	}

	double *delta_beta = malloc(100*block_size*sizeof(double));
	double *delta_beta_hat = malloc(block_size*sizeof(double));

	printf("updating beta %d times\n", num_updates);
	double time1 = omp_get_wtime();
	for (int i = 0; i < num_updates; i++)
		update_beta_partition(X, X2, Y, rowsum, n, p, lambda, beta, 0.0, 0.0, precalc_get_num, thread_column_caches, column_partition,
						delta_beta, delta_beta_hat, MULTIPLIER);
	double time2 = omp_get_wtime();
	double total_time = time2 - time1;
	printf("total time: %f\n", total_time);
	printf("iters/second: %f\n", (double)num_updates/total_time);

	//for (int j = 0; j < p; j++) {
	//	printf("checking beta[%d] (%f) == %f\n", j, beta[j], check_beta[j]);
	//	g_assert_true(fabs(beta[j] - check_beta[j]) < 0.001 );
	//}
	//for (int i = 0; i < n; i++) {
	//	printf("checking rowsum[%d] (%f) == %f\n", i, rowsum[i], check_rowsum[i]);
	//	g_assert_true(fabs(rowsum[i] - check_rowsum[i]) < 0.001 );
	//}

	double sum = 0.0;
	for (int i = 0; i < p_int; i++)
		sum += beta[i];
	printf("final beta sum (to avoid optimising out updates) %f\n", sum);
	printf("freeing things\n");
	for (int i = 0; i <  max_num_threads; i++) {
		free(thread_column_caches[i]);
	}
	free(thread_column_caches);
	free(precalc_get_num);
	free(delta_beta);
	free(delta_beta_hat);
	free(beta);
	return total_time;
}
void test_update_beta_partition_repeat() {
	int n = 10000;
	int p = 1000;
	XMatrix X = read_x_csv("/home/kieran/work/lasso_testing/X_nlethals50_v15803.csv", n, p);
	double *Y = read_y_csv("/home/kieran/work/lasso_testing/Y_nlethals50_v15803.csv", n);
	int p_int = (p*(p+1))/2;
	XMatrix_sparse X2 = sparse_X2_from_X(X.X, n, p, -1, FALSE, FALSE);
	double lambda = 6.46;
	int num_updates = 5;
	int max_threads = omp_get_max_threads();
	for (int multiplier = 10; multiplier < 1025; multiplier *= 2) {
		printf("using multiplier: %d\n", multiplier);
		printf("1 thread:\n");
		omp_set_num_threads(1);
		double time1 = test_update_beta_partition_repeat_multiplier(num_updates, multiplier, X, X2, Y);
		printf("%d threads:\n", max_threads);
		omp_set_num_threads(max_threads);
		double time4 = test_update_beta_partition_repeat_multiplier(num_updates, multiplier, X, X2, Y);
		printf("relative speedup: %f\n", time1/time4);
	}
}
void test_update_beta_cyclic_repeat() {
	int n = 10000;
	int p = 1000;
	XMatrix X = read_x_csv("/home/kieran/work/lasso_testing/X_nlethals50_v15803.csv", n, p);
	printf("reading Y\n");
	double *Y = read_y_csv("/home/kieran/work/lasso_testing/Y_nlethals50_v15803.csv", n);
	int p_int = (p*(p+1))/2;
	XMatrix_sparse X2 = sparse_X2_from_X(X.X, n, p, 1, FALSE, FALSE);
	double lambda = 6.46;

	double *beta = malloc(p_int * sizeof(double));
	memset(beta, 0, p_int*sizeof(double));
	double *rowsum = malloc(n * sizeof(double));
	memset(rowsum, 0, n*sizeof(double));

	printf("initialising variables\n");
	// omp_set_num_threads(8);
	// Initialise rowsums
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
	int_pair *precalc_get_num = malloc(p*sizeof(int_pair));
	int offset = 0;
	for (int i = 0; i < p; i++) {
		for (int j = i; j < 4; j++) {
			precalc_get_num[offset].i = i;
			precalc_get_num[offset].j = j;
			offset++;
		}
	}

	int max_num_threads = 1;
	int largest_col = n;
	int **thread_column_caches = malloc(max_num_threads*sizeof(int*));
	for (int i = 0; i <  max_num_threads; i++) {
		thread_column_caches[i] = malloc(largest_col*sizeof(int));
	}

	int num_updates = 1000;
	printf("updating beta %d times\n", num_updates);
	double time1 = omp_get_wtime();
	for (int i = 0; i < num_updates; i++)
		for (int k = 0; k < X2.p; k++)
			update_beta_cyclic(X, X2, Y, rowsum, n, p, lambda, beta, k, 0.0, 0.0, precalc_get_num, thread_column_caches[omp_get_thread_num()]);
	double time2 = omp_get_wtime();
	printf("total time: %f\n", time2 - time1);

	//for (int j = 0; j < p; j++) {
	//	printf("checking beta[%d] (%f) == %f\n", j, beta[j], check_beta[j]);
	//	g_assert_true(fabs(beta[j] - check_beta[j]) < 0.001 );
	//}
	//for (int i = 0; i < n; i++) {
	//	printf("checking rowsum[%d] (%f) == %f\n", i, rowsum[i], check_rowsum[i]);
	//	g_assert_true(fabs(rowsum[i] - check_rowsum[i]) < 0.001 );
	//}

	double sum = 0.0;
	for (int i = 0; i < p_int; i++)
		sum += beta[i];
	printf("final beta sum (to avoid optimising out updates) %f\n", sum);
	printf("freeing things\n");
	for (int i = 0; i <  max_num_threads; i++) {
		free(thread_column_caches[i]);
	}
	free(thread_column_caches);
	free(precalc_get_num);
	free(beta);
}

int main (int argc, char *argv[]) {
	initialise_static_resources();
	setlocale (LC_ALL, "");
	g_test_init (&argc, &argv, NULL);

	g_test_add_func("/func/test-read-y-csv", test_read_y_csv);
	g_test_add_func("/func/test-read-x-csv", test_read_x_csv);
	g_test_add_func("/func/test-soft-threshol", test_soft_threshold);
	g_test_add("/func/test-update-beta-cyclic", UpdateFixture, NULL, update_beta_fixture_set_up, test_update_beta_cyclic, update_beta_fixture_tear_down);
	g_test_add_func("/func/test-update-intercept-cyclic", test_update_intercept_cyclic);
	g_test_add_func("/func/test-X2_from_X", test_X2_from_X);
	g_test_add("/func/test-simple-coordinate-descent-int", UpdateFixture, FALSE, test_simple_coordinate_descent_set_up, test_simple_coordinate_descent_int, test_simple_coordinate_descent_tear_down);
	g_test_add("/func/test-simple-coordinate-descent-int-shuffle", UpdateFixture, TRUE, test_simple_coordinate_descent_set_up, test_simple_coordinate_descent_int, test_simple_coordinate_descent_tear_down);
	g_test_add("/func/test-simple-coordinate-descent-vs-glmnet", UpdateFixture, TRUE, test_simple_coordinate_descent_set_up, test_simple_coordinate_descent_vs_glmnet, test_simple_coordinate_descent_tear_down);
	g_test_add_func("/func/test-block-division", test_block_division);
	g_test_add_func("/func/tests-block-division-large-time", test_block_division_large_time);
	g_test_add_func("/func/test-X2-encoding", check_X2_encoding);
	g_test_add_func("/func/test-find-overlap", test_find_overlap);
	g_test_add_func("/func/test-correct-beta-updates", test_correct_beta_updates);
	g_test_add_func("/func/test-update-beta-partition", test_update_beta_partition_lambda0);
	g_test_add_func("/func/test-update-beta-partition-lambda", test_update_beta_partition_lambda1);
	g_test_add_func("/func/test-update-beta-partition-repeat", test_update_beta_partition_repeat);
	g_test_add_func("/func/test-update-beta-cyclic-repeat", test_update_beta_cyclic_repeat);

	return g_test_run();
}
