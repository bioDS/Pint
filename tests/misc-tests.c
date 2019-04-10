#include <glib.h>

static void test_always_succeeds() {
	g_assert (1 == 1);
}

static void test_always_fails() {
	g_assert (1 == 2);
}

int main (int argc, char *argv[]) {
	g_test_init (&argc, &argv, NULL);

	g_test_add_func("/misc/test-succeed", test_always_succeeds);
	g_test_add_func("/misc/test-fail", test_always_fails);

	return g_test_run();
}
