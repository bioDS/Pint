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
}
static void test_soft_threshold() {
}
static void test_read_x_csv() {
}
static void test_X2_from_X() {
}
static void test_simple_coordinate_descent() {
}

static void test_read_y_csv() {
	double *Y = read_y_csv("/home/kieran/work/lasso_testing/testYSmall.csv", N);
	g_assert(Y[0] == -133.351709197933);
	g_assert(Y[999] == -352.293608898344);
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
