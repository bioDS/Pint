#include "../../src/liblasso.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void write_x_csv(char* filename, long** X, long n, long p)
{
    FILE* fp = fopen(filename, "w");
    printf("writing %ldx%ld elements\n", n, p);

    for (long i = 0; i < n; i++) {
        fprintf(fp, "\"%ld\",", i);
        for (long j = 0; j < p - 1; j++) {
            if (X[i][j] > 1) {
                printf("writing problem at %ldx%ld\n", i, j);
            }
            fprintf(fp, "%ld,", X[i][j]);
        }
        fprintf(fp, "%ld\n", X[i][p - 1]);
    }
}

int main(long argc, char** argv)
{
    if (argc != 5) {
        printf("usage ./X2_from_X X.csv n p X2.csv\n");
        return 1;
    }

    char* Xf = argv[1];
    char* X2f = argv[4];
    long n = atoi(argv[2]);
    long p = atoi(argv[3]);

    printf("reading matrix\n");
    XMatrix xmatrix = read_x_csv(Xf, n, p);
    printf("translating to X2\n");
    long** X2 = X2_from_X(xmatrix.X, n, xmatrix.actual_cols);
    printf("writing X2\n");
    write_x_csv(X2f, X2, n, (p * (p + 1)) / 2);

    return 0;
}
