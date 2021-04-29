#include <gsl/gsl_vector.h>
#include "../../src/liblasso.h"

enum Output_Mode {quit, file, terminal};

int main(int argc, char** argv) {
	if (argc != 12) {
		fprintf(stderr, "usage: ./lasso_exe X.csv Y.csv [main/int] verbose=T/F [max lambda] N P [max interaction distance] [frac overlap allowed] [q/t/filename] [log_level [i]ter/[l]ambda/[n]one]\n");
		printf("actual args(%d): '", argc);
		for (int i = 0; i < argc; i++) {
			printf("%s ", argv[i]);
		}
		printf("\n");
		return 1;
	}


	char *scale = argv[3];
	char *verbose = argv[2];
	char *output_filename = argv[10];
	FILE *output_file = NULL;

	enum Output_Mode output_mode = terminal;
	if (strcmp(output_filename, "t") == 0);
	else if (strcmp(output_filename, "q") == 0)
		output_mode = quit;
	else {
		output_mode = file;
		output_file = fopen(output_filename, "w");
		if (output_file == NULL) {
			perror("opening output file failed");
		}
	}



	VERBOSE = 0;
	if (strcmp(verbose, "T") == 0)
		VERBOSE = 1;

	float lambda;

	if ((lambda = strtod(argv[5], NULL)) == 0)
		lambda = 3.604;
	printf("using lambda = %f\n", lambda);


	int N = atoi(argv[6]);
	int P = atoi(argv[7]);
	printf("using N = %d, P = %d\n", N, P);

	int max_interaction_distance = atoi(argv[8]);
	printf("using max interaction distance: %d\n", max_interaction_distance);

	float overlap = atof(argv[9]);
	printf("using frac: %.2f\n", overlap);

	enum LOG_LEVEL log_level = NONE;
	if (strcmp(argv[11], "i") == 0) {
		log_level = ITER;
	} else if (strcmp(argv[11], "l") == 0) {
		log_level = LAMBDA;
	} else if (strcmp(argv[11], "n") != 0) {
		printf("using 'log_level = NONE', no valid argument given");
	}

	initialise_static_resources();

	// testing: wip
	XMatrix xmatrix = read_x_csv(argv[1], N, P);
	float *Y = read_y_csv(argv[2], N);

	int **X2;
	int nbeta;
	nbeta = xmatrix.actual_cols;
	X2 = xmatrix.X;
	printf("using nbeta = %d\n", nbeta);

	if (xmatrix.X == NULL) {
		fprintf(stderr, "failed to read X\n");
		return 1;
	}
	if (Y == NULL) {
		fprintf(stderr, "failed to read Y\n");
		return 1;
	}

	printf("begginning coordinate descent\n");
	float *beta = simple_coordinate_descent_lasso(xmatrix, Y, N, nbeta, max_interaction_distance,
			0.04, lambda, 10000, VERBOSE, overlap, 1.0001, log_level, argv, argc, FALSE, 50);
	int nbeta_int = nbeta;
	nbeta_int = get_p_int(nbeta, max_interaction_distance);
	if (beta == NULL) {
		fprintf(stderr, "failed to estimate beta values\n");
		return 1;
	}
	//for (int i = 0; i < nbeta_int; i++) {
	//	printf("%f ", beta[i]);
	//}
	//printf("\n");

	printf("indices non-zero (|x| != 0):\n");
	int printed = 0;
	int sig_beta_count = 0;
	//TODO: remove hack to avoid printing too much for the terminal

	printf("\n\n");

	for (int i = 0; i < xmatrix.actual_cols; i++)
		free(xmatrix.X[i]);
	free(xmatrix.X);
	free(Y);
	printf("freeing X/Y\n");
	switch(output_mode){
		case terminal:
			for (int i = 0; i < nbeta_int && printed < 100; i++) {
				if (fabs(beta[i]) > 0) {
					printed++;
					sig_beta_count++;
					int_pair ip = get_num(i, nbeta);
					if (ip.i == ip.j)
						printf("main: %d (%d):     %f\n", i, ip.i + 1, beta[i]);
					else
						printf("int: %d  (%d, %d): %f\n", i, ip.i + 1, ip.j + 1, beta[i]);
				}
			}
		break;
		case file:
			for (int i = 0; i < nbeta_int; i++) {
				if (beta[i] != 0.0) {
					printed++;
					sig_beta_count++;
					int_pair ip = get_num(i, nbeta);
					if (ip.i == ip.j)
						fprintf(output_file, "main: %d (%d):     %f\n", i, ip.i + 1, beta[i]);
					else
						fprintf(output_file, "int: %d  (%d, %d): %f\n", i, ip.i + 1, ip.j + 1, beta[i]);
				}
			}
			fclose(output_file);
		break;
		case quit:
		break;
	}
	if (output_mode == terminal) {
	}
	//endwin();
	free(beta);
	free_static_resources();
	return 0;
}
