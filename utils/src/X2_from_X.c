#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include "../../src/liblasso.h"


void write_x_csv(char *filename, int **X, int n, int p) {
	FILE *fp = fopen(filename, "w");
	printf("writing %dx%d elements\n", n, p);

	for (int i = 0; i < n; i++) {
		fprintf(fp, "\"%d\",", i);
		for (int j = 0; j < p - 1; j++) {
			if (X[i][j] > 1) {
				printf("writing problem at %dx%d\n", i, j);
			}
			fprintf(fp, "%d,", X[i][j]);
		}
		fprintf(fp, "%d\n", X[i][p - 1]);
	}
}

int main(int argc, char **argv) {
	if (argc != 5) {
		printf("usage ./X2_from_X X.csv n p X2.csv\n");
		return 1;
	}

	char *Xf = argv[1];
	char *X2f = argv[4];
	int n = atoi(argv[2]);
	int p = atoi(argv[3]);

	printf("reading matrix\n");
	XMatrix xmatrix = read_x_csv(Xf, n, p);
	printf("translating to X2\n");
	int **X2 = X2_from_X(xmatrix.X, n, xmatrix.actual_cols);
	printf("writing X2\n");
	write_x_csv(X2f, X2, n, (p*(p+1))/2);

	return 0;
}
