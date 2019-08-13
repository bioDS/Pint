#include <gsl/gsl_vector.h>
#include "../../src/liblasso.h"
#include <ncurses.h>

enum Output_Mode {quit, file, terminal};

int main(int argc, char** argv) {
	if (argc != 10) {
		fprintf(stderr, "usage: ./lasso-testing X.csv Y.csv [main/int] verbose=T/F [max lambda] N P [frac overlap allowed] [q/t/filename]\n");
		printf("actual args(%d): '", argc);
		for (int i = 0; i < argc; i++) {
			printf("%s ", argv[i]);
		}
		printf("\n");
		return 1;
	}


	initscr();
	refresh();

	char *scale = argv[3];
	char *verbose = argv[2];
	char *output_filename = argv[9];
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




	int USE_INT=0; // main effects only by default
	if (strcmp(scale, "int") == 0)
		USE_INT=1;

	VERBOSE = 0;
	if (strcmp(verbose, "T") == 0)
		VERBOSE = 1;

	double lambda;

	if ((lambda = strtod(argv[5], NULL)) == 0)
		lambda = 3.604;
	move(1,0);
	printw("using lambda = %f\n", lambda);


	int N = atoi(argv[6]);
	int P = atoi(argv[7]);
	move(1,0);
	printw("using N = %d, P = %d\n", N, P);

	double overlap = atof(argv[8]);
	printw("using frac: %.2f\n", overlap);


	// testing: wip
	XMatrix xmatrix = read_x_csv(argv[1], N, P);
	double *Y = read_y_csv(argv[2], N);

	int **X2;
	int nbeta;
	//if (USE_INT) {
	//	printf("converting to X2\n");
	//	X2 = X2_from_X(xmatrix.X, N, xmatrix.actual_cols);
	//	nbeta = (xmatrix.actual_cols*(xmatrix.actual_cols+1))/2;
	//} else {
		nbeta = xmatrix.actual_cols;
		X2 = xmatrix.X;
	//}
	move(5,0);
	printw("using nbeta = %d\n", nbeta);
	refresh();

	if (xmatrix.X == NULL) {
		fprintf(stderr, "failed to read X\n");
		return 1;
	}
	if (Y == NULL) {
		fprintf(stderr, "failed to read Y\n");
		return 1;
	}

	move(6,0);
	printw("begginning coordinate descent\n");
	refresh();
	double *beta = simple_coordinate_descent_lasso(xmatrix, Y, N, nbeta, lambda, "cyclic", 30, USE_INT, VERBOSE, overlap);
	int nbeta_int = nbeta;
	if (USE_INT) {
		nbeta_int = nbeta*(nbeta+1)/2;
	}
	//printw("done coordinate descent lasso, printing (%d) beta values:\n", nbeta_int);
	if (beta == NULL) {
		fprintf(stderr, "failed to estimate beta values\n");
		return 1;
	}
	//for (int i = 0; i < nbeta; i++) {
	//	printf("%f ", beta[i]);
	//}
	//printf("\n");

	printw("\n");
	//move(12,0);
	printw("indices significantly negative (-500):\n");
	int printed = 0;
	int sig_beta_count = 0;
	//TODO: remove hack to avoid printing too much for the terminal

	printw("\n\n");

	//printw("\n");
	free(xmatrix.X);
	free(Y);
	//move(22 + sig_beta_count,0);
	printw("freeing X/Y\n");
	switch(output_mode){
		case terminal:
			for (int i = 0; i < nbeta_int && printed < 10; i++) {
				if (beta[i] < -500) {
					printed++;
					sig_beta_count++;
					int_pair ip = get_num(i, nbeta);
					if (ip.i == ip.j)
						printw("main: %d (%d):     %f\n", i, ip.i + 1, beta[i]);
					else
						printw("int: %d  (%d, %d): %f\n", i, ip.i + 1, ip.j + 1, beta[i]);
				}
			}
			printw("finished! press q to exit");
			while(getch() != 'q');
			getch();
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
	endwin();
	free_static_resources();
	return 0;
}
