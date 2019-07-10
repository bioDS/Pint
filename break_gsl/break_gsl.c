#include <gsl/gsl_permutation.h>

int main() {
	// values larger than 2094134 break gsl (number diff. from non-toy example)
	//int test_p_int = 2095145;
	int test_p_int = 20000*(20001)/2;
//
	int shuffle_order[test_p_int];
//	gsl_rng *r;
	gsl_permutation *permutation = gsl_permutation_alloc(test_p_int);
	if (permutation == NULL) {
		fprintf(stderr, "inssufficient memory to create permutation of size %d\n", test_p_int);
		exit(1);
	}
	gsl_permutation_init(permutation);
	gsl_rng_env_setup();
}
