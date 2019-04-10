#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include "test_lasso.h"

typedef struct XMatrix {
	int **X;
	int actual_cols;
} XMatrix;

XMatrix read_x_csv(char *fn, int n, int p) {
	char *buf = NULL;
	size_t line_size = 0;
	int **X = malloc(n*sizeof(int*));
	for (int i = 0; i < n; i++)
		X[i] = malloc(p*sizeof(int));

	printf("reading X from: \"%s\"\n", fn);

	FILE *fp = fopen(fn, "r");
	if (fp == NULL) {
		perror("opening failed");
	}

	int col = 0, row = 0, actual_cols = p;
	int readline_result = 0;
	while((readline_result = getline(&buf, &line_size, fp)) > 0) {
		// remove name from beginning (for the moment)
		int i = 1;
		while (buf[i] != '"')
			i++;
		i++;
		// read to the end of the line
		while (buf[i] != '\n' && i < line_size) {
			if (buf[i] == ',')
				{i++; continue;}
			if (buf[i] == '0')
				X[row][col] = 0;
			else if (buf[i] == '1')
				X[row][col] = 1;
			else {
				fprintf(stderr, "format error reading X from %s at row: %d, col: %d\n", fn, row, col);
				exit(0);
			}
			i++;
			if (++col >= p)
				break;
		}
		if (buf[i] != '\n')
			fprintf(stderr, "reached end of file without a newline\n");
		if (col < actual_cols)
			actual_cols = col;
		col = 0;
		if (++row >= n)
			break;
	}
	if (readline_result == -1)
		fprintf(stderr, "failed to read line, errno %d\n", errno);

	if (actual_cols < p)
		printf("number of columns < p, should p have been %d?\n", actual_cols);
	printf("read %dx%d, freeing stuff\n", row, actual_cols);
	free(buf);
	XMatrix xmatrix;
	xmatrix.X = X;
	xmatrix.actual_cols = actual_cols;
	return xmatrix;
}

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
