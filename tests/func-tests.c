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
	fixture->xmatrix_sparse = sparse_X2_from_X(fixture->X, fixture->n, fixture->p, 0, -1);
	update_beta_cyclic(fixture->xmatrix, fixture->xmatrix_sparse, fixture->Y, fixture->rowsum, fixture->n, fixture->p, fixture->lambda, fixture->beta, fixture->k, fixture->intercept, fixture->precalc_get_num, fixture->column_cache);
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
	fixture->xmatrix_sparse = sparse_X2_from_X(fixture->X, fixture->n, fixture->p, -1, shuffle);
	int p_int = fixture->p*(fixture->p+1)/2;
	double *beta = fixture->beta;

	double dBMax;
	for (int j = 0; j < 10; j++)
		for (int i = 0; i < p_int; i++) {
			//int k = gsl_permutation_get(fixture->xmatrix_sparse.permutation, i);
			int k = fixture->xmatrix_sparse.permutation->data[i];
			//int k = i;
			dBMax = update_beta_cyclic(fixture->xmatrix, fixture->xmatrix_sparse, fixture->Y, fixture->rowsum, fixture->n, fixture->p, fixture->lambda, beta, k, 0, fixture->precalc_get_num, fixture->column_cache);
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
	fixture->xmatrix_sparse = sparse_X2_from_X(fixture->X, fixture->n, fixture->p, -1, FALSE);
	int p_int = fixture->p*(fixture->p+1)/2;
	double *beta = fixture->beta;

	beta = simple_coordinate_descent_lasso(fixture->xmatrix, fixture->Y, fixture->n, fixture->p, -1, 0.05, 1000, 100, 0, 0.01, 1.0001, FALSE, 1, "test", FALSE, -1);

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
	XMatrix_sparse xmatrix_sparse = sparse_X2_from_X(xmatrix.X, n, p, 1, -1);

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

	int item_width[15] = {0,   0,   1,  2,  3,  4,  5,  6,  7, 8, 10, 12, 15, 20, 30, 60};
	int group_size[15] = {240, 120, 60, 30, 20, 15, 12, 10, 8, 7, 6,  5,  4,  3,  2,  1};
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

static void check_permutation() {
	int threads = omp_get_num_procs();
	gsl_rng **thread_r = malloc(threads*sizeof(gsl_rng*));
	for (int i = 0; i < threads; i++)
		thread_r[i] = gsl_rng_alloc(gsl_rng_taus2);

	long perm_size = 3235; //<< 12 + 67;
	printf("perm_size %ld\n", perm_size);
	gsl_permutation *perm = gsl_permutation_alloc(perm_size);
	gsl_permutation_init(perm);

	parallel_shuffle(perm, perm_size/threads, perm_size%threads, threads);

	int *found = malloc(perm_size*sizeof(int));
	memset(found, 0, perm_size*sizeof(int));
	for (int i = 0; i < perm_size; i++) {
		size_t val = perm->data[i];
		found[val] = 1;
		printf("found %d\n", val);
	}
	for (int i = 0; i < perm_size; i++) {
		printf("checking %d is present\n", i);
		printf("found[%d] = %d\n", i, found[i]);
		printf("found[%d+1] = %d\n", i, found[i+1]);
		g_assert_true(found[i] == 1);
	}
	free(found);
	gsl_permutation_free(perm);

	perm_size = 123123; //<< 12 + 67;
	printf("perm_size %ld\n", perm_size);
	perm = gsl_permutation_alloc(perm_size);
	gsl_permutation_init(perm);

	parallel_shuffle(perm, perm_size/threads, perm_size%threads, threads);

	found = malloc(perm_size*sizeof(int));
	memset(found, 0, perm_size);
	for (long i = 0; i < perm_size; i++) {
		long val = perm->data[i];
		found[val] = 1;
		printf("found %d\n", val);
	}
	for (long i = 0; i < perm_size; i++) {
		printf("checking %d is present\n", i);
		printf("found[%d] = %d\n", i, found[i]);
		printf("found[%d+1] = %d\n", i, found[i+1]);
		g_assert_true(found[i] == 1);
	}
	free(found);
	gsl_permutation_free(perm);
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
	g_test_add_func("/func/test-X2-encoding", check_X2_encoding);
	g_test_add_func("/func/test-permutation", check_permutation);

	return g_test_run();
}
