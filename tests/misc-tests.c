#include <glib.h>
#include "../src/lasso_lib.h"

static void test_always_succeeds() {
	g_assert (1 == 1);
}

static void test_always_fails() {
	g_assert (1 == 2);
}

// will fail if Y has been normalised
static void test_can_read_Y() {
	double *Y = read_y_csv("/home/kieran/work/lasso_testing/testYSmall.csv", N);
	printf("test stdout\n");
	printf("%f\n", Y[0]);
	g_assert(Y[0] == -133.351709197933);
}

int main (int argc, char *argv[]) {
	g_test_init (&argc, &argv, NULL);

	g_test_add_func("/misc/test-succeed", test_always_succeeds);
	//g_test_add_func("/misc/test-fail", test_always_fails);
	g_test_add_func("/misc/test-can-read-y", test_can_read_Y);

	return g_test_run();
}
