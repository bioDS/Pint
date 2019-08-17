#include <glib-2.0/glib.h>
#include "../src/liblasso.h"
#include <locale.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_permutation.h>

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
} UpdateFixture;

typedef struct {
	Mergeset *all_sets;
	int **set_bins_of_size;
	int *num_bins_of_size;
	int *valid_mergesets;
	int p_int;
} Merge_Fixture;


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
}

static void update_beta_fixture_tear_down(UpdateFixture *fixture, gconstpointer user_data) {
	for (int i = 0; i < fixture->p; i++) {
		free(fixture->xmatrix.X[i]);
	}
	free(fixture->Y);
	free(fixture->rowsum);
	free(fixture->beta);
	free(fixture->precalc_get_num);
}

static void test_update_beta_cyclic(UpdateFixture *fixture, gconstpointer user_data) {
	printf("beta[27]: %f\n", fixture->beta[27]);
	fixture->xmatrix_sparse = sparse_X2_from_X(fixture->X, fixture->n, fixture->p, 0, FALSE);
	update_beta_cyclic(fixture->xmatrix, fixture->xmatrix_sparse, fixture->Y, fixture->rowsum, fixture->n, fixture->p, fixture->lambda, fixture->beta, fixture->k, fixture->dBMax, fixture->intercept, 0, fixture->precalc_get_num);
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
	fixture->p = 100;
	fixture->Y = read_y_csv("/home/kieran/work/lasso_testing/testYSmall.csv", fixture->n);
	fixture->lambda = 20;
	fixture->k = 27;
	fixture->dBMax = 0;
	fixture->intercept = 0;
}

static void test_simple_coordinate_descent_main(UpdateFixture *fixture, gconstpointer user_data) {

	fixture->p = 630;
	fixture->xmatrix = read_x_csv("/home/kieran/work/lasso_testing/testX2Small.csv", fixture->n, fixture->p);
	fixture->X = fixture->xmatrix.X;
	fixture->xmatrix_sparse = sparse_X2_from_X(fixture->X, fixture->n, fixture->p, 0, TRUE);
	fixture->beta = simple_coordinate_descent_lasso(fixture->xmatrix, fixture->Y, fixture->n, fixture->xmatrix.actual_cols, 0.01, fixture->lambda, "cyclic", 10, 0, 0, 0.0);

	double acceptable_diff = 10;
	for (int i = 0; i < fixture->p; i++) {
		int k = gsl_permutation_get(fixture->xmatrix_sparse.permutation, i);
		printf("testing beta[%d] (%f) ~ %f\n", i, fixture->beta[i], small_X2_correct_beta[k]);
		g_assert_true(fixture->beta[i] < small_X2_correct_beta[k] + acceptable_diff);
		g_assert_true(fixture->beta[i] > small_X2_correct_beta[k] - acceptable_diff);
	}
}

static void test_simple_coordinate_descent_int(UpdateFixture *fixture, gconstpointer user_data) {
	printf("starting interaction test\n");
	fixture->p = 35;
	fixture->xmatrix = read_x_csv("/home/kieran/work/lasso_testing/testXSmall.csv", fixture->n, fixture->p);
	fixture->X = fixture->xmatrix.X;
	int p_int = fixture->p*(fixture->p+1)/2;
	fixture->xmatrix_sparse = sparse_X2_from_X(fixture->X, fixture->n, fixture->p, 1, TRUE);
	fixture->beta = simple_coordinate_descent_lasso(fixture->xmatrix, fixture->Y, fixture->n, fixture->xmatrix.actual_cols, 0.01, fixture->lambda, "cyclic", 10, 1, 0, 0.0);

	double acceptable_diff = 10;
	for (int i = 0; i < p_int; i++) {
		int k = gsl_permutation_get(fixture->xmatrix_sparse.permutation, i);
		printf("testing beta[%d] (%f) ~ %f\n", i, fixture->beta[i], small_X2_correct_beta[k]);
		g_assert_true(fixture->beta[i] < small_X2_correct_beta[k] + acceptable_diff);
		g_assert_true(fixture->beta[i] > small_X2_correct_beta[k] - acceptable_diff);
	}
}

// will fail if Y has been normalised
static void test_read_y_csv() {
	int n = 1000;
	double *Y = read_y_csv("/home/kieran/work/lasso_testing/testYSmall.csv", n);
	g_assert_true(Y[0] == -133.351709197933);
	g_assert_true(Y[999] == -352.293608898344);
}

static void test_find_beta_sets() {
	initialise_static_resources();
	int n = 6;
	int p = 7;
	int Xt[7][6] = { {1,1,0,0,0,0},
					{0,0,0,0,1,1},
					{0,0,0,1,1,0},
					{1,1,1,0,0,0},
					{0,0,1,1,0,0},
					{0,0,0,1,1,0},
					{1,0,1,0,1,1}};
	int **X = malloc(p*sizeof(int*));
	for (int i = 0; i < p; i++) {
		X[i] = malloc(n*sizeof(int));
		memcpy(X[i], Xt[i], n*sizeof(int));
	}
	XMatrix_sparse x2col = sparse_X2_from_X(X, n, p, 0, FALSE);
	//XMatrix_sparse_row x2row = sparse_horizontal_X2_from_X(X, n, p, 0);

	//printf("\nsparse row 0 (%d entries):\n", x2row.row_nz[0]);
	//for (int i = 0; i < x2row.row_nz[0]; i++)
	//	printf("'%d' ", x2row.row_nz_indices[0][i]);
	//printf("\n");



	//find_beta_sets(x2col, x2row, p*(p+1)/2, n);
	Beta_Sets beta_sets = find_beta_sets(x2col, p, n, 0.0);
	for (int i = 0; i < 7; i++)
		free(X[i]);

	int correct_set_sizes[5] = {2,2,1,1,1};
	int correct_cols_for_set[7] = {0, 1, 2, 3, 4, 5, 6};

	printf("\nchecking number of sets (%d) == 5\n", beta_sets.number_of_sets);
	g_assert_true(beta_sets.number_of_sets == 5);


	int val_counter = 0;
	for (int set = 0; set < beta_sets.number_of_sets; set++) {
		//GList *temp_set = beta_sets.sets[set].set;
		struct Beta_Set colset = beta_sets.sets[set];
		g_assert_true(colset.set_size == correct_set_sizes[set]);
		for (int entry = 0; entry < colset.set_size; entry++) {
			int val = colset.set[entry];
			printf("comparing set %d, entry %d: %d == %d\n", set, entry, val, correct_cols_for_set[val_counter]);
			g_assert_true(val == correct_cols_for_set[val_counter++]);
		}
	}


	printf("checking find_beta for testX\n");
	n = 1000;
	p = 35;
	int p_int = p*(p+1)/2;
	XMatrix xmatrix = read_x_csv("/home/kieran/work/lasso_testing/testXSmall.csv", n, p);
	x2col = sparse_X2_from_X(xmatrix.X, n, p, 1, FALSE);
	//x2row = sparse_horizontal_X2_from_X(xmatrix.X, n, p, 1);
	beta_sets = find_beta_sets(x2col, p_int, n, 0.0);

	printf("found %d sets\n", beta_sets.number_of_sets);
	printf("checking every element is present exactly once\n");

	int found[p_int];
		memset(found, 0, p_int*sizeof(int));
	for (int i = 0; i < beta_sets.number_of_sets; i++) {
		struct Beta_Set colset = beta_sets.sets[i];
		for (int entry = 0; entry < colset.set_size; entry++) {
			int k = colset.set[entry];
			g_assert_true(k >= 0);
			g_assert_true(k < p_int);
			g_assert_true(found[k] == 0);
			found[k] = 1;
		}
	}
	for (int i = 0; i < p_int; i++) {
		g_assert_true(found[i] == 1);
	}

	return;
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
	initialise_static_resources();
	int n = 1000;
	int p = 35;
	int p_int = p*(p+1)/2;
	XMatrix xmatrix = read_x_csv("./testXSmall.csv", n, p);
	XMatrix_sparse xmatrix_sparse = sparse_X2_from_X(xmatrix.X, n, p, 1, FALSE);

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

static void check_merge_n_set_up(Merge_Fixture *fixture, gconstpointer user_data) {
	int n = 1000;
	int p = 35;
	XMatrix xmatrix = read_x_csv("/home/kieran/work/lasso_testing/testXSmall.csv", n, p);
	int **X = xmatrix.X;
	double *Y = read_y_csv("/home/kieran/work/lasso_testing/testYSmall.csv", n);
	int *rowsum = malloc(n*sizeof(double));
	int lambda = 6.46;
	int *beta = malloc(p*sizeof(double));
	memset(beta, 0, p*sizeof(double));
	int k = 27;
	int dBMax = 0;
	int intercept = 0;
	int p_int = (p*(p+1))/2;
	int_pair *precalc_get_num = malloc(p_int*sizeof(int_pair));
	int offset = 0;
	for (int i = 0; i < p; i++) {
		for (int j = i; j < p; j++) {
			precalc_get_num[offset].i = i;
			precalc_get_num[offset].j = j;
			offset++;
		}
	}
	XMatrix_sparse xmatrix_sparse = sparse_X2_from_X(X, n, p, 1, FALSE);
	Mergeset *all_sets = malloc(p_int*sizeof(Mergeset));
	int *valid_mergesets = malloc(p_int*sizeof(int));

	for (int i = 0; i < p_int; i++) {
		all_sets[i].size = xmatrix_sparse.col_nz[i];
		all_sets[i].cols = malloc(sizeof(int));
		all_sets[i].cols[0] = i;
		all_sets[i].ncols = 1;
		all_sets[i].entries = malloc(xmatrix_sparse.col_nz[i]*sizeof(int));
		memcpy(all_sets[i].entries, xmatrix_sparse.col_nz_indices[i], all_sets[i].size*sizeof(int));
		valid_mergesets[i] = TRUE;
	}

	int new_mergeset_count = p_int;
	for (int i = 0; i < p_int - 2; i += 2) {
		if (valid_mergesets[i+1] && can_merge(all_sets, i, i+1, 0.0)) {
			merge_sets(all_sets, i, i+1);
			valid_mergesets[i+1] = FALSE;
			new_mergeset_count--;
		}
	}

	int **set_bins_of_size = malloc((NumCores+1)*sizeof(int*));
	for (int i = 0; i < NumCores+1; i++)
		set_bins_of_size[i] = malloc(p_int*sizeof(int));
	//int num_bins_of_size[NumCores+2];
	int *num_bins_of_size = malloc((NumCores+1)*sizeof(int));
	int valid_mergeset_indices[p_int];

	for (int i = 0; i <= NumCores+1; i++)
		num_bins_of_size[i] = 0;

	int count = 0;
	for (int i = 0; i < p_int; i++) {
		if (valid_mergesets[i]) {
			int set_size = all_sets[i].ncols;
			if (set_size > NumCores+1)
				set_size = NumCores+1;
			valid_mergeset_indices[count] = i;
			set_bins_of_size[set_size][num_bins_of_size[set_size]] = i;
			num_bins_of_size[set_size]++;
		}
	}

	fixture->all_sets = all_sets;
	fixture->valid_mergesets = valid_mergesets;
	fixture->num_bins_of_size = num_bins_of_size;
	fixture->set_bins_of_size = set_bins_of_size;
	fixture->p_int = p_int;
}

static void check_merge_n_tear_down(Merge_Fixture *fixture, gconstpointer user_data) {
	return;
	for (int i = 0; i < fixture->p_int; i++) {
		free(fixture->all_sets[i].entries);
		free(fixture->all_sets[i].cols);
	}
	free(fixture->all_sets);
	free(fixture->valid_mergesets);
	free(fixture->num_bins_of_size);
	for (int i = 0; i < NumCores+1; i++)
		free(fixture->set_bins_of_size[i]);
	free(fixture->set_bins_of_size);
}

static void test_check_n(Merge_Fixture *fx, gconstpointer user_data) {
	int *sets_to_merge = malloc(fx->p_int*sizeof(int));
	printf("bins of size 1: %d, size 2: %d\n", fx->num_bins_of_size[1], fx->num_bins_of_size[2]);
	int n = fx->num_bins_of_size[1];
	if (fx->num_bins_of_size[2] < n)
		n = fx->num_bins_of_size[2];
	int offset_small = 17, offset_large = 81;
	int small = 1;
	int large = 2;
	int no_sets_to_merge = compare_n(fx->all_sets, fx->valid_mergesets, fx->set_bins_of_size, fx->num_bins_of_size, sets_to_merge, small, large, n, offset_small, offset_large, 0.0);

	int successful_merge_count = 0;
	for (int i = 0; i < n; i++) {
		int small_no = (i + offset_small) % fx->num_bins_of_size[small];
		int large_no = (i + offset_large) % fx->num_bins_of_size[large];
		if (sets_to_merge[i] == 1) {
			printf("confirming [%d][%d] merges with [%d][%d]...", small, small_no, large, large_no);
			g_assert_true(can_merge(fx->all_sets, fx->set_bins_of_size[small][small_no], fx->set_bins_of_size[large][large_no], 0.0));
			successful_merge_count++;
			printf(" OK\n");
		}
		else
			g_assert_false(can_merge(fx->all_sets, fx->set_bins_of_size[small][small_no], fx->set_bins_of_size[large][large_no], 0.0));
	}
	g_assert_true(no_sets_to_merge == successful_merge_count);
	printf("compare_n succeeded on distinct sets\n");

	printf("running compare_n on sections of the same set\n");
	n = fx->num_bins_of_size[small];
	offset_small = 23;
	offset_large = n/2 + 23;
	small = large = 1;
	no_sets_to_merge = compare_n(fx->all_sets, fx->valid_mergesets, fx->set_bins_of_size, fx->num_bins_of_size, sets_to_merge, small, small, n/2, offset_small, offset_large, 0.0);

	successful_merge_count = 0;
	for (int i = 0; i < n/2; i++) {
		int small_no = (i + offset_small) % fx->num_bins_of_size[small];
		int large_no = (i + offset_large) % fx->num_bins_of_size[large];
		if (sets_to_merge[i] == 1) {
			printf("confirming [%d][%d] merges with [%d][%d]...", small, small_no, large, large_no);
			g_assert_true(can_merge(fx->all_sets, fx->set_bins_of_size[small][small_no], fx->set_bins_of_size[large][large_no], 0.0));
			successful_merge_count++;
			printf(" OK\n");
		}
		else
			g_assert_false(can_merge(fx->all_sets, fx->set_bins_of_size[small][small_no], fx->set_bins_of_size[large][large_no], 0.0));
	}
	g_assert_true(no_sets_to_merge = successful_merge_count);
	printf("compare_n succeeded on distinct sets\n");
}

void check_sets_occur_once(Merge_Fixture *fx) {
	int found_col[fx->p_int];
	memset(found_col, 0, fx->p_int*sizeof(int));
	printf("counting column occurances in result... ");
	for (int i = 0; i < NumCores + 1; i++) {
		for (int j = 0; j < fx->num_bins_of_size[i]; j++) {
			int set = fx->set_bins_of_size[i][j];
			g_assert_true(set >= 0 && set < fx->p_int);
			if (fx->valid_mergesets[set])
				for (int k = 0; k < fx->all_sets[set].ncols; k++) {
					int col = fx->all_sets[set].cols[k];
					found_col[col]++;
					if (col==20) {
						printf("\ncol20 found at [%d][%d]\n", i, j);
					}
				}
		}
	}
	printf("done\n");


	printf("checking every column occures exactly once... ");
	for (int i = 0; i < fx->p_int; i++) {
		g_assert_true(found_col[i] == 1);
	}
	printf("done\n");
}

static void test_merge_n(Merge_Fixture *fx, gconstpointer user_data) {
	int *sets_to_merge = malloc(fx->p_int*sizeof(int));
	printf("\nchecking distinct sets\n");

	printf("bins of size");
	for (int i = 0; i < NumCores+1; i++) {
		printf(" [%d]: %d,", i, fx->num_bins_of_size[i]);
	}
	printf("\n");

	int n = fx->num_bins_of_size[1];
	if (fx->num_bins_of_size[2] < n)
		n = fx->num_bins_of_size[2];
	int no_sets_to_merge = compare_n(fx->all_sets, fx->valid_mergesets, fx->set_bins_of_size, fx->num_bins_of_size, sets_to_merge, 1, 2, n, 0, 0, 0.0);
	merge_n(fx->all_sets, fx->set_bins_of_size, fx->num_bins_of_size, fx->valid_mergesets, sets_to_merge, 1, 2, n, 0, 0, no_sets_to_merge);

	printf("bins of size");
	for (int i = 0; i < NumCores+1; i++) {
		printf(" [%d]: %d,", i, fx->num_bins_of_size[i]);
	}
	printf("\n");

	printf("\nchecking the same set\n");

	check_sets_occur_once(fx);

	n = fx->num_bins_of_size[2];
	int offset = 23;
	no_sets_to_merge = compare_n(fx->all_sets, fx->valid_mergesets, fx->set_bins_of_size, fx->num_bins_of_size, sets_to_merge, 2, 2, n/2, offset, n/2+offset, 0.0);
	merge_n(fx->all_sets, fx->set_bins_of_size, fx->num_bins_of_size, fx->valid_mergesets, sets_to_merge, 2, 2, n/2, offset, n/2+offset, no_sets_to_merge);

	printf("bins of size");
	for (int i = 0; i < NumCores+1; i++) {
		printf(" [%d]: %d,", i, fx->num_bins_of_size[i]);
	}
	printf("\n");

	check_sets_occur_once(fx);
}

int main (int argc, char *argv[]) {
	setlocale (LC_ALL, "");
	g_test_init (&argc, &argv, NULL);

	g_test_add_func("/func/test-read-y-csv", test_read_y_csv);
	g_test_add_func("/func/test-read-x-csv", test_read_x_csv);
	g_test_add_func("/func/test-soft-threshol", test_soft_threshold);
	//g_test_add_func("/func/test-update-beta-cyclic", test_update_beta_cyclic);
	g_test_add("/func/test-update-beta-cyclic", UpdateFixture, NULL, update_beta_fixture_set_up, test_update_beta_cyclic, update_beta_fixture_tear_down);
	g_test_add_func("/func/test-update-intercept-cyclic", test_update_intercept_cyclic);
	g_test_add_func("/func/test-X2_from_X", test_X2_from_X);
	g_test_add("/func/test-simple-coordinate-descent-main", UpdateFixture, NULL, test_simple_coordinate_descent_set_up, test_simple_coordinate_descent_main, update_beta_fixture_tear_down);
	g_test_add("/func/test-simple-coordinate-descent-int", UpdateFixture, NULL, test_simple_coordinate_descent_set_up, test_simple_coordinate_descent_int, update_beta_fixture_tear_down);
	g_test_add_func("/func/test-find-beta-sets", test_find_beta_sets);
	g_test_add("/func/test-check-n", Merge_Fixture, NULL, check_merge_n_set_up, test_check_n, check_merge_n_tear_down);
	g_test_add("/func/test-merge-n", Merge_Fixture, NULL, check_merge_n_set_up, test_merge_n, check_merge_n_tear_down);
	g_test_add_func("/func/test-X2-encoding", check_X2_encoding);

	return g_test_run();
}
