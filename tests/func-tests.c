#include <glib.h>
#include "../src/lasso_lib.h"
#include <locale.h>

/* int **X2_from_X(int **X, int n, int p); */
/* double *simple_coordinate_descent_lasso(int **X, double *Y, int n, int p, double lambda, char *method); */
/* double update_beta_greedy_l1(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax); */
/* double update_intercept_cyclic(double intercept, int **X, double *Y, double *beta, int n, int p); */
/* double update_beta_cyclic(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax, double intercept); */
/* double update_beta_glmnet(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax, double intercept); */
/* double soft_threshold(double z, double gamma); */
/* double *read_y_csv(char *fn, int n); */
/* XMatrix read_x_csv(char *fn, int n, int p); */

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
	fixture->xmatrix_sparse = sparse_X2_from_X(fixture->X, fixture->n, fixture->p, 1);
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
	fixture->xmatrix_sparse = sparse_X2_from_X(fixture->X, fixture->n, fixture->p, 0);
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
	fixture->beta = simple_coordinate_descent_lasso(fixture->xmatrix, fixture->Y, fixture->n, fixture->xmatrix.actual_cols, fixture->lambda, "cyclic", 10, 0, 0);

	double acceptable_diff = 0.0001;
	for (int i = 0; i < fixture->p; i++) {
		printf("testing beta[%d] (%f) ~ %f\n", i, fixture->beta[i], small_X2_correct_beta[i]);
		g_assert_true(fixture->beta[i] < small_X2_correct_beta[i] + acceptable_diff);
		g_assert_true(fixture->beta[i] > small_X2_correct_beta[i] - acceptable_diff);
	}
}

static void test_simple_coordinate_descent_int(UpdateFixture *fixture, gconstpointer user_data) {

	fixture->p = 35;
	fixture->xmatrix = read_x_csv("/home/kieran/work/lasso_testing/testXSmall.csv", fixture->n, fixture->p);
	fixture->X = fixture->xmatrix.X;
	int p_int = fixture->p*(fixture->p+1)/2;
	fixture->beta = simple_coordinate_descent_lasso(fixture->xmatrix, fixture->Y, fixture->n, fixture->xmatrix.actual_cols, fixture->lambda, "cyclic", 10, 1, 0);

	double acceptable_diff = 0.0001;
	for (int i = 0; i < p_int; i++) {
		printf("testing beta[%d] (%f) ~ %f\n", i, fixture->beta[i], small_X2_correct_beta[i]);
		g_assert_true(fixture->beta[i] < small_X2_correct_beta[i] + acceptable_diff);
		g_assert_true(fixture->beta[i] > small_X2_correct_beta[i] - acceptable_diff);
	}
}

// will fail if Y has been normalised
static void test_read_y_csv() {
	int n = 1000;
	double *Y = read_y_csv("/home/kieran/work/lasso_testing/testYSmall.csv", n);
	g_assert_true(Y[0] == -133.351709197933);
	g_assert_true(Y[999] == -352.293608898344);
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

	return g_test_run();
}
