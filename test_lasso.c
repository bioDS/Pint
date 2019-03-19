#include<stdio.h>
#include <gsl/gsl_vector.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>

#define VECTOR_SIZE 3
// are all ids the same size?
#define ID_LEN 20
#define N 30856
//#define N 30
#define P 21110

void read_x_csv(char *fn, int n, int p) {
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
		//printf("new buffer, row %d, col: %d\n", row, col);
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
				//{printf("test3\n"); X[row*p + col] = 0.0; printf("test5\n");}
				X[row][col] = 0;
			else if (buf[i] == '1')
				//{printf("test4\n"); X[row*p + col] = 1.0;}
				//X[row*p + col] = 1.0;
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
		if (col < p)
			printf("number of columns < p, should p have been %d?\n", col);
		if (col < actual_cols)
			actual_cols = col;
		col = 0;
		if (++row >= n)
			break;
	}
	if (readline_result == -1)
		fprintf(stderr, "failed to read line, errno %d\n", errno);

	printf("read %dx%d, freeing stuff\n", row, actual_cols);
	free(buf);

}

void read_y_csv(char *fn, int n) {
}

int main(int argc, char** argv) {
	gsl_vector *v = gsl_vector_alloc(3);
	gsl_vector *w = gsl_vector_alloc(3);
	gsl_vector_set_zero(v);
	gsl_vector_set(v, 1, 2);
	gsl_vector_set(v, 2, 3);
	gsl_vector_memcpy(w, v);
	gsl_vector_set(w,2,1);

	int result = gsl_vector_mul(v,w);
	printf("result: %d\n", result);
	printf("v: ");
	for (int i = 0; i < VECTOR_SIZE; i++) {
		printf("%f ", gsl_vector_get(v, i));
	}
	printf("\n");


	// testing: wip
	read_x_csv("X.csv", N, P);

	return 0;
}
