#include "liblasso.h"

XMatrix read_x_csv(char *fn, int n, int p) {
  char *buf = NULL;
  size_t line_size = 0;
  int **X = malloc(p * sizeof(int *));

  for (int i = 0; i < p; i++)
    X[i] = malloc(n * sizeof(int));

  FILE *fp = fopen(fn, "r");
  if (fp == NULL) {
    perror("opening failed");
  }

  int col = 0, row = 0, actual_cols = p;
  int readline_result = 0;
  while ((readline_result = getline(&buf, &line_size, fp)) > 0) {
    // remove name from beginning (for the moment)
    int i = 1;
    while (buf[i] != '"')
      i++;
    i++;
    // read to the end of the line
    while (buf[i] != '\n' && i < line_size) {
      if (buf[i] == ',') {
        i++;
        continue;
      }
      if (buf[i] == '0') {
        X[col][row] = 0;
      } else if (buf[i] == '1') {
        X[col][row] = 1;
      } else {
        fprintf(stderr, "format error reading X from %s at row: %d, col: %d\n",
                fn, row, col);
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

  if (actual_cols < p) {
    printf("number of columns < p, should p have been %d?\n", actual_cols);
    p = actual_cols;
  }
  free(buf);
  XMatrix xmatrix;
  xmatrix.X = X;
  xmatrix.actual_cols = actual_cols;
  return xmatrix;
}

float *read_y_csv(char *fn, int n) {
  char *buf = malloc(BUF_SIZE);
  char *temp = malloc(BUF_SIZE);
  memset(buf, 0, BUF_SIZE);
  float *Y = malloc(n * sizeof(float));

  FILE *fp = fopen(fn, "r");
  if (fp == NULL) {
    perror("opening failed");
  }

  int col = 0, i = 0;
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
    int j = 0;
    while (buf[i] != '\n')
      temp[j++] = buf[i++];
    Y[col] = atof(temp);
    col++;
  }

  // for comparison with implementations that normalise rather than
  // finding the intercept.
  if (NORMALISE_Y == 1) {
    printf("%d, normalising y values\n", NORMALISE_Y);
    float mean = 0.0;
    for (int i = 0; i < n; i++) {
      mean += Y[i];
    }
    mean /= n;
    for (int i = 0; i < n; i++) {
      Y[i] -= mean;
    }
  }

  free(buf);
  free(temp);
  return Y;
}