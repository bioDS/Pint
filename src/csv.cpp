#include "liblasso.h"
#include <fstream>

XMatrix read_x_csv(const char* fn, int_fast64_t n, int_fast64_t p)
{
    int_fast64_t** X = (int_fast64_t**)malloc(p * sizeof *X);

    for (int_fast64_t i = 0; i < p; i++)
        X[i] = (int_fast64_t*)malloc(n * sizeof *X[i]);

    std::ifstream in_file(fn, std::ifstream::in);

    bool printed_eof_err = false;
    int_fast64_t col = 0, row = 0, actual_cols = p;
    int_fast64_t readline_result = 0;
    for (std::string buf; std::getline(in_file, buf); ) {
        // remove name from beginning (for the moment)
        long unsigned int i = 1;
        while (buf[i] != '"')
            i++;
        i++;
        // read to the end of the line
        while (buf[i] != '\n' && i < buf.length()) {
            if (buf[i] == ',') {
                i++;
                continue;
            }
            if (buf[i] == '0') {
                X[col][row] = 0;
            } else if (buf[i] == '1') {
                X[col][row] = 1;
            } else {
                fprintf(stderr, "format error reading X from %s at row: %ld, col: %ld\n",
                    fn, row, col);
                exit(0);
            }
            i++;
            if (++col >= p)
                break;
        }
        if (buf[i] != '\n' && !printed_eof_err) {
            fprintf(stderr, "reached end of file without a newline\n");
            printed_eof_err = true;
        }
        if (col < actual_cols)
            actual_cols = col;
        col = 0;
        if (++row >= n)
            break;
    }
    if (readline_result == -1)
        fprintf(stderr, "failed to read line, errno %d\n", errno);

    if (actual_cols < p) {
        printf("number of columns < p, should p have been %ld?\n", actual_cols);
        p = actual_cols;
    }
    in_file.close();
    XMatrix xmatrix;
    xmatrix.X = X;
    xmatrix.actual_cols = actual_cols;
    return xmatrix;
}

float* read_y_csv(const char* fn, int_fast64_t n)
{
    char* buf = (char*)malloc(BUF_SIZE);
    char* temp = (char*)malloc(BUF_SIZE);
    memset(buf, 0, BUF_SIZE);
    float* Y = (float*)malloc(n * sizeof(float));

    FILE* fp = fopen(fn, "r");
    if (fp == NULL) {
        perror("opening failed");
    }

    int_fast64_t col = 0, i = 0;
    while (fgets(buf, BUF_SIZE, fp) != NULL) {
        i = 1;
        // skip the name
        while (buf[i] != '"')
            i++;
        i++;
        if (buf[i] == ',')
            i++;
        // read the rest of the line as a float
        memset(temp, 0, BUF_SIZE);
        int_fast64_t j = 0;
        while (buf[i] != '\n')
            temp[j++] = buf[i++];
        Y[col] = atof(temp);
        col++;
    }

    // for comparison with implementations that normalise rather than
    // finding the intercept.
    if (NORMALISE_Y == 1) {
        printf("%ld, normalising y values\n", NORMALISE_Y);
        float mean = 0.0;
        for (int_fast64_t i = 0; i < n; i++) {
            mean += Y[i];
        }
        mean /= n;
        for (int_fast64_t i = 0; i < n; i++) {
            Y[i] -= mean;
        }
    }

    free(buf);
    free(temp);
    return Y;
}