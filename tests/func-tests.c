#include <glib.h>
#include "../src/lasso_lib.h"
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
	int *sets_to_merge;
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
	fixture->beta = simple_coordinate_descent_lasso(fixture->xmatrix, fixture->Y, fixture->n, fixture->xmatrix.actual_cols, fixture->lambda, "cyclic", 10, 0, 0);

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
	fixture->beta = simple_coordinate_descent_lasso(fixture->xmatrix, fixture->Y, fixture->n, fixture->xmatrix.actual_cols, fixture->lambda, "cyclic", 10, 1, 0);

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
	XMatrix_sparse_row x2row = sparse_horizontal_X2_from_X(X, n, p, 0);

	printf("\nsparse row 0 (%d entries):\n", x2row.row_nz[0]);
	for (int i = 0; i < x2row.row_nz[0]; i++)
		printf("'%d' ", x2row.row_nz_indices[0][i]);
	printf("\n");



	//find_beta_sets(x2col, x2row, p*(p+1)/2, n);
	Beta_Sets beta_sets = find_beta_sets(x2col, x2row, p, n);
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
	x2row = sparse_horizontal_X2_from_X(xmatrix.X, n, p, 1);
	beta_sets = find_beta_sets(x2col, x2row, p_int, n);

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

static void test_column_set_operations() {
	Column_Set test_set1;
	test_set1.size = 10;
	test_set1.cols = malloc(10*sizeof(ColEntry));

	for (int i = 0; i < 10; i++) {
		test_set1.cols[i].value = i;
		test_set1.cols[i].nextEntry = i+1;
	}

	Column_Set test_set2 = copy_column_set(test_set1);
	printf("checking set1.size (%d) == set2.size (%d)\n", test_set1.size, test_set2.size);
	g_assert_true(test_set2.size == test_set1.size);

	for (int i = 0; i < test_set1.size; i++) {
		printf("checking set1[%d] = (%d,%d) == set2[%d] (%d,%d)\n", i, test_set1.cols[i].value, test_set1.cols[i].nextEntry,
																	i, test_set2.cols[i].value, test_set2.cols[i].nextEntry);
		g_assert_true(test_set2.cols[i].value == test_set1.cols[i].value);
		g_assert_true(test_set2.cols[i].nextEntry == test_set1.cols[i].nextEntry);
	}

	for (int i = 0; i < test_set1.size; i++) {
		int found_ind = fancy_col_find_entry_value_or_next(test_set2, i);
		printf("checking found_ind: %d == actual location: %d\n", found_ind, i);
		g_assert_true(found_ind == i);
	}

	fancy_col_remove(test_set1, 4);
	for (int i = 0; i < 10; i++) {
		if (i == 4)
			g_assert_true(test_set1.cols[i].nextEntry == -5);
		else
			g_assert_true(test_set1.cols[i].nextEntry == i+1);
		g_assert_true(test_set1.cols[i].value == i);
	}

	fancy_col_remove(test_set1, 0);
	for (int i = 0; i < 10; i++) {
		if (i == 0)
			g_assert_true(test_set1.cols[i].nextEntry == -1);
		else if (i == 4)
			g_assert_true(test_set1.cols[i].nextEntry == -5);
		else
			g_assert_true(test_set1.cols[i].nextEntry == i+1);
		g_assert_true(test_set1.cols[i].value == i);
	}

	fancy_col_remove(test_set1, 9);
	for (int i = 0; i < 10; i++) {
		if (i == 0)
			g_assert_true(test_set1.cols[i].nextEntry == -1);
		else if (i == 4)
			g_assert_true(test_set1.cols[i].nextEntry == -5);
		else if (i == 9)
			g_assert_true(test_set1.cols[i].nextEntry == -10);
		else
			g_assert_true(test_set1.cols[i].nextEntry == i+1);
		g_assert_true(test_set1.cols[i].value == i);
	}
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
		if (valid_mergesets[i+1] && can_merge(all_sets, i, i+1)) {
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
	int no_sets_to_merge = compare_n(fx->all_sets, fx->valid_mergesets, fx->set_bins_of_size, fx->num_bins_of_size, sets_to_merge, 1, 2, n, 0, 0);

	int *nothing = malloc(10*sizeof(int));
	int successful_merge_count = 0;
	for (int i = 0; i < n; i++) {
		if (sets_to_merge[i] == 1) {
			g_assert_true(can_merge(fx->all_sets, fx->set_bins_of_size[1][i], fx->set_bins_of_size[2][i]));
			successful_merge_count++;
		}
		else
			g_assert_false(can_merge(fx->all_sets, fx->set_bins_of_size[1][i], fx->set_bins_of_size[2][i]));
	}
	g_assert_true(no_sets_to_merge = successful_merge_count);
	printf("compare_n succeeded on distinct sets\n");
}

static void test_merge_n(Merge_Fixture *fx, gconstpointer user_data) {
	int *sets_to_merge = malloc(fx->p_int*sizeof(int));
	printf("bins of size 1: %d, size 2: %d\n", fx->num_bins_of_size[1], fx->num_bins_of_size[2]);
	int n = fx->num_bins_of_size[1];
	if (fx->num_bins_of_size[2] < n)
		n = fx->num_bins_of_size[2];
	int no_sets_to_merge = compare_n(fx->all_sets, fx->valid_mergesets, fx->set_bins_of_size, fx->num_bins_of_size, sets_to_merge, 1, 2, n, 0, 0);
	merge_n(fx->all_sets, fx->set_bins_of_size, fx->num_bins_of_size, fx->valid_mergesets, fx->sets_to_merge, 1, 2, n, 0, 0, no_sets_to_merge);
	printf("bins of size 1: %d, size 2: %d\n", fx->num_bins_of_size[1], fx->num_bins_of_size[2]);
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
	g_test_add_func("/func/test-column-set-operations", test_column_set_operations);
	g_test_add("/func/test-check-n", Merge_Fixture, NULL, check_merge_n_set_up, test_check_n, check_merge_n_tear_down);
	g_test_add("/func/test-merge-n", Merge_Fixture, NULL, check_merge_n_set_up, test_merge_n, check_merge_n_tear_down);

	return g_test_run();
}
