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
} UpdateFixture;

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

	for (int i = 0; i < fixture->p; i++)
		fixture->rowsum[i] = 0;
}

static void update_beta_fixture_tear_down(UpdateFixture *fixture, gconstpointer user_data) {
	for (int i = 0; i < fixture->n; i++) {
		free(fixture->xmatrix.X[i]);
	}
	free(fixture->Y);
	free(fixture->rowsum);
	free(fixture->beta);
}

static void test_update_beta_cyclic(UpdateFixture *fixture, gconstpointer user_data) {
	printf("beta[27]: %f\n", fixture->beta[27]);
	update_beta_cyclic(fixture->X, fixture->Y, fixture->rowsum, fixture->n, fixture->p, fixture->lambda, fixture->beta, fixture->k, fixture->dBMax, fixture->intercept, 0);
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
	g_assert_true(xmatrix.X[999][99] == 0);
	g_assert_true(xmatrix.X[575][16] == 1);

	int sum = 0;
	for (int i = 0; i < p; i++) {
		sum += xmatrix.X[321][i];
	}
	g_assert_true(sum == 8);
}

static void test_X2_from_X() {
	printf("not implemented yet\n");
}

static void test_simple_coordinate_descent() {
	double correct_beta[35] = {-94.48064159, -433.92736735, -71.25059672, -54.75560128, -163.96807642, -14.86901934, -41.69686004, -57.29189735, -75.18984525, 20.76153523, -76.05316284, -428.02834825, -340.97245868, 45.23866547, -168.97759338, -96.8576242, -129.47612499, 98.63679679, -0.98614496, -12.82189371, 0.0, 0.0, 0.0, -84.07942008, 20.70530852, -61.74223165, 0.0, 0.0, -32.28071432, 0.0, 88.06443817, 0.0, -125.35161412, 0.0, -95.00169196};
	int n = 1000;
	int p = 35;
	XMatrix xmatrix = read_x_csv("/home/kieran/work/lasso_testing/testXSmall.csv", n, p);
	int **X = xmatrix.X;
	double *Y = read_y_csv("/home/kieran/work/lasso_testing/testYSmall.csv", n);
	double lambda = 6.46;
	double *beta = malloc(p*sizeof(double));
	memset(beta, 0, p*sizeof(double));
	int k = 27;
	double dBMax = 0;
	double intercept = 0;

	beta = simple_coordinate_descent_lasso(X, Y, n, p, lambda, "cyclic", 10, 0);

	double acceptable_diff = 0.0001;
	for (int i = 0; i < p; i++) {
		printf("testing beta[%d] (%f) ~ %f\n", i, beta[i], correct_beta[i]);
		g_assert_true(beta[i] < correct_beta[i] + acceptable_diff);
		g_assert_true(beta[i] > correct_beta[i] - acceptable_diff);
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
	g_test_add_func("/func/test-simple-coordinate-descent-lasso", test_simple_coordinate_descent);

	return g_test_run();
}
