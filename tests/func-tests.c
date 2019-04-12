#include <glib.h>
#include "../src/lasso_lib.h"

/* int **X2_from_X(int **X, int n, int p); */
/* double *simple_coordinate_descent_lasso(int **X, double *Y, int n, int p, double lambda, char *method); */
/* double update_beta_greedy_l1(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax); */
/* double update_intercept_cyclic(double intercept, int **X, double *Y, double *beta, int n, int p); */
/* double update_beta_cyclic(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax, double intercept); */
/* double update_beta_glmnet(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax, double intercept); */
/* double soft_threshold(double z, double gamma); */
/* double *read_y_csv(char *fn, int n); */
/* XMatrix read_x_csv(char *fn, int n, int p); */

// will fail if Y has been normalised

static void test_update_beta_greedy_l1() {
}
static void test_update_intercept_cyclic() {
	int n = 1000;
	int p = 100;
	XMatrix xmatrix = read_x_csv("/home/kieran/work/lasso_testing/testXSmall.csv", n, p);
	int **X = xmatrix.X;
	double *Y = read_y_csv("/home/kieran/work/lasso_testing/testYSmall.csv", n);
	double lambda = 6.46;
	double *beta = malloc(p*sizeof(double));
	memset(beta, 0, p*sizeof(double));
	int k = 27;
	double dBMax = 0;
	double intercept = 0;

	printf("beta[27]: %f\n", beta[27]);
	update_beta_cyclic(X, Y, n, p, lambda, beta, k, dBMax, intercept);
	printf("beta[27]: %f\n", beta[27]);
	g_assert_true(beta[27] != 0.0);
	g_assert_true(beta[27] < -263.94);
	g_assert_true(beta[27] > -263.941);
}
static void test_soft_threshold() {
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
}
static void test_simple_coordinate_descent() {
}

static void test_read_y_csv() {
	double *Y = read_y_csv("/home/kieran/work/lasso_testing/testYSmall.csv", N);
	g_assert_true(Y[0] == -133.351709197933);
	g_assert_true(Y[999] == -352.293608898344);
}

int main (int argc, char *argv[]) {
	g_test_init (&argc, &argv, NULL);

	g_test_add_func("/func/test-read-y-csv", test_read_y_csv);
	g_test_add_func("/func/test-read-x-csv", test_read_x_csv);
	g_test_add_func("/func/test-soft-threshol", test_soft_threshold);
	g_test_add_func("/func/test-update-beta-cyclic", test_update_intercept_cyclic);
	g_test_add_func("/func/test-update-intercept-cyclic", test_update_intercept_cyclic);
	g_test_add_func("/func/test-X2_from_X", test_X2_from_X);
	g_test_add_func("/func/test-simple-coordinate-descent-lasso", test_simple_coordinate_descent);

	return g_test_run();
}
