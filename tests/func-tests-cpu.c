#include "../src/liblasso.h"
#include <glib-2.0/glib.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_rng.h>
#include <locale.h>
#include <omp.h>

/* long **X2_from_X(long **X, long n, long p); */
/* double *simple_coordinate_descent_lasso(long **X, double *Y, long n, long p,
 * double lambda, char *method); */
/* double update_beta_greedy_l1(long **X, double *Y, long n, long p, double lambda,
 * double *beta, long k, double dBMax); */
/* double update_intercept_cyclic(double intercept, long **X, double *Y, double
 * *beta, long n, long p); */
/* double update_beta_cyclic(long **X, double *Y, long n, long p, double lambda,
 * double *beta, long k, double dBMax, double intercept); */
/* double update_beta_glmnet(long **X, double *Y, long n, long p, double lambda,
 * double *beta, long k, double dBMax, double intercept); */
/* double soft_threshold(double z, double gamma); */
/* double *read_y_csv(char *fn, long n); */
/* XMatrix read_x_csv(char *fn, long n, long p); */

// #define NumCores 4
#define HALT_ERROR_DIFF 1.01

struct timespec start, end;
static double x2_conversion_time = 0.0;

typedef struct {
  long n;
  long p;
  XMatrix xmatrix;
  long **X;
  double *Y;
  double *rowsum;
  double lambda;
  double *beta;
  long k;
  double dBMax;
  double intercept;
  XMatrixSparse xmatrix_sparse;
  int_pair *precalc_get_num;
  long **column_caches;
  XMatrixSparse Xc;
  XMatrixSparse X2c;
} UpdateFixture;

typedef struct {
  long *col_i;
  long *col_j;
} Thread_Cache;

const static double small_X2_correct_beta[630] = {
    -83.112248,  0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    -39.419762, 0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, -431.597831,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    -56.125867,  0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    -54.818886,  0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   -144.076649, 0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    -64.023489,  0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   -33.646329,  0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, -62.705188,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    -334.676519, 0.000000,   0.000000,    -215.196793, 0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    -165.866118, 0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    -112.678381, 0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   -1.284220,   0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    -58.031513,  0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    3.916624,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, -73.009253,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    6.958046,    0.000000, 0.000000,
    0.000000,    0.000000,   0.000000,    0.000000,    0.000000, 0.000000,
    -120.529141, 0.000000,   0.000000,    0.000000,    0.000000, -80.263024};

static void test_update_beta_greedy_l1() { printf("not implemented yet\n"); }

static void test_update_intercept_cyclic() { printf("not implemented yet\n"); }

static void update_beta_fixture_set_up(UpdateFixture *fixture,
                                       gconstpointer user_data) {
  fixture->n = 1000;
  fixture->p = 35;
  fixture->xmatrix = read_x_csv("../testXSmall.csv", fixture->n, fixture->p);
  fixture->X = fixture->xmatrix.X;
  fixture->Y = read_y_csv("../testYSmall.csv", fixture->n);
  fixture->rowsum = malloc(fixture->n * sizeof(double));
  fixture->lambda = 6.46;
  fixture->beta = malloc(fixture->p * sizeof(double));
  memset(fixture->beta, 0, fixture->p * sizeof(double));
  fixture->k = 27;
  fixture->dBMax = 0;
  fixture->intercept = 0;
  printf("%d\n", fixture->X[1][0]);
  long p_long = (fixture->p * (fixture->p + 1)) / 2;
  int_pair *precalc_get_num = malloc(p_long * sizeof(int_pair));
  long offset = 0;
  for (long i = 0; i < fixture->p; i++) {
    for (long j = i; j < fixture->p; j++) {
      precalc_get_num[offset].i = i;
      precalc_get_num[offset].j = j;
      offset++;
    }
  }
  fixture->precalc_get_num = precalc_get_num;

  long **thread_column_caches = malloc(NumCores * sizeof(long *));
  for (long i = 0; i < NumCores; i++) {
    thread_column_caches[i] = malloc(fixture->n * sizeof(int));
  }
  fixture->column_caches = thread_column_caches;
}

static void update_beta_fixture_tear_down(UpdateFixture *fixture,
                                          gconstpointer user_data) {
  for (long i = 0; i < fixture->p; i++) {
    free(fixture->xmatrix.X[i]);
  }
  free(fixture->Y);
  free(fixture->rowsum);
  free(fixture->beta);
  free(fixture->precalc_get_num);
  for (long i = 0; i < NumCores; i++) {
    free(fixture->column_caches[i]);
  }
  free(fixture->column_caches);
}

static void test_update_beta_cyclic(UpdateFixture *fixture,
                                    gconstpointer user_data) {
  printf("beta[27]: %f\n", fixture->beta[27]);
  fixture->xmatrix_sparse =
      sparse_X2_from_X(fixture->X, fixture->n, fixture->p, 0, -1);
  update_beta_cyclic_old(fixture->xmatrix_sparse, fixture->Y, fixture->rowsum,
                         fixture->n, fixture->p, fixture->lambda, fixture->beta,
                         fixture->k, fixture->intercept,
                         fixture->precalc_get_num, fixture->column_caches);
  printf("beta[27]: %f\n", fixture->beta[27]);
  g_assert_true(fixture->beta[27] != 0.0);
  g_assert_true(fixture->beta[27] < -263.94);
  g_assert_true(fixture->beta[27] > -263.941);
}

static void test_soft_threshold() { printf("not implemented yet\n"); }

static void test_read_x_csv() {
  long n = 1000;
  long p = 100;
  XMatrix xmatrix = read_x_csv("../testX.csv", n, p);
  g_assert_true(xmatrix.actual_cols == 100);
  g_assert_true(xmatrix.X[0][0] == 0);
  g_assert_true(xmatrix.X[99][999] == 0);
  g_assert_true(xmatrix.X[16][575] == 1);

  long sum = 0;
  for (long i = 0; i < p; i++) {
    sum += xmatrix.X[i][321];
  }
  g_assert_true(sum == 8);
}

static void test_compressed_main_X() {
  long n = 1000;
  long p = 100;
  XMatrix xm = read_x_csv("../testX.csv", n, p);

  XMatrixSparse Xs = sparsify_X(xm.X, n, p);

  g_assert_true(Xs.n == n);
  g_assert_true(Xs.p == p);

  long *column_entries[n];

  long agreed_on = 0;
  for (long k = 0; k < p; k++) {
    long col_entry_pos = 0;
    long entry = -1;
    memset(column_entries, 0, sizeof *column_entries * n);
    for (long i = 0; i < Xs.cols[k].nwords; i++) {
      S8bWord word = Xs.cols[k].compressed_indices[i];
      unsigned long values = word.values;
      for (long j = 0; j <= group_size[word.selector]; j++) {
        long diff = values & masks[word.selector];
        if (diff != 0) {
          entry += diff;
          column_entries[col_entry_pos] = entry;
          col_entry_pos++;
        }
        values >>= item_width[word.selector];
      }
    }
    // check the read column agrees with k of testX2.csv
    // n.b. XMatrix.X is column-major
    col_entry_pos = 0;
    for (long i = 0; i < n; i++) {
      // printf("\ncolumn %d contains %d entries", k, X2s.nz[k]);
      if (col_entry_pos > Xs.cols[k].nz || column_entries[col_entry_pos] < i) {
        if (xm.X[k][i] != 0) {
          printf("\n[%d][%d] is not in the index but should be", k, i);
          g_assert_true(FALSE);
        }
      } else if (Xs.cols[k].nz > 0 && column_entries[col_entry_pos] == i) {
        if (xm.X[k][i] != 1) {
          printf("\n[%d][%d] missing from \n", k, i);
          g_assert_true(FALSE);
        } else {
          agreed_on++;
        }
        col_entry_pos++;
      }
    }
    // printf("\nfinished column %d", k);
  }
  printf("agreed on %d\n", agreed_on);
}

static void test_X2_from_X() {
  long n = 1000;
  long p = 100;
  long p_long = p * (p + 1) / 2;
  XMatrix xm = read_x_csv("../testX.csv", n, p);
  XMatrix xm2 = read_x_csv("../testX2.csv", n, p_int);

  XMatrixSparse X2s = sparse_X2_from_X(xm.X, n, p, -1, FALSE);

  g_assert_true(X2s.n == n);
  g_assert_true(X2s.p == p_int);

  // prlong X2s
  // printf("X2s:\n");
  // for (long k = 0; k < p_int; k++) {
  //  long entry = -1;
  //  printf("%d: ", k);
  //  for (long i = 0; i < X2s.cols[k].nwords; i++) {
  //    S8bWord word = X2s.cols[k].compressed_indices[i];
  //    unsigned long values = word.values;
  //    for (long j = 0; j <= group_size[word.selector]; j++) {
  //      long diff = values & masks[word.selector];
  //      if (diff != 0) {
  //        entry += diff;
  //        printf(" %d", entry);
  //      }
  //      values >>= item_width[word.selector];
  //    }
  //  }
  //  printf("\n");
  //}

  // printf("X2 (printed rows are file columns)\n");
  // for (long j = 0; j < p_int; j++) {
  //  for (long i = 0; i < n; i++) {
  //    printf(" %d", xm2.X[j][i]);
  //  }
  //  printf("\n");
  //}

  long *column_entries[n];

  for (long k = 0; k < p_int; k++) {
    if (k == 2905) {
      printf("xm2.X[%d][0] == %d\n", k, xm2.X[k][0]);
    }
    long col_entry_pos = 0;
    long entry = -1;
    memset(column_entries, 0, sizeof *column_entries * n);
    for (long i = 0; i < X2s.cols[k].nwords; i++) {
      S8bWord word = X2s.cols[k].compressed_indices[i];
      unsigned long values = word.values;
      for (long j = 0; j <= group_size[word.selector]; j++) {
        long diff = values & masks[word.selector];
        if (diff != 0) {
          entry += diff;
          column_entries[col_entry_pos] = entry;
          col_entry_pos++;
        }
        values >>= item_width[word.selector];
      }
    }
    // check the read column agrees with k of testX2.csv
    // n.b. XMatrix.X is column-major
    col_entry_pos = 0;
    for (long i = 0; i < n; i++) {
      // printf("\ncolumn %d contains %d entries", k, X2s.nz[k]);
      if (col_entry_pos > X2s.cols[k].nz || column_entries[col_entry_pos] < i) {
        if (xm2.X[k][i] != 0) {
          printf("\n[%d][%d] is not in the index but should be", k, i);
          g_assert_true(FALSE);
        }
      } else if (X2s.cols[k].nz > 0 && column_entries[col_entry_pos] == i) {
        if (xm2.X[k][i] != 1) {
          printf("\n[%d][%d] missing from \n", k, i);
          g_assert_true(FALSE);
        }
        col_entry_pos++;
      }
    }
    // printf("\nfinished column %d", k);
  }
}

static void test_simple_coordinate_descent_set_up(UpdateFixture *fixture,
                                                  gconstpointer use_big) {
  char *xfile, *yfile;
  if (use_big == 2) {
    printf("\nusing huge test case\n");
    fixture->n = 10000;
    fixture->p = 5000;
    xfile = "../n10000_p5000_SNR5_nbi50_nbij1000_nlethals250_viol0_40452/X.csv";
    yfile = "../n10000_p5000_SNR5_nbi50_nbij1000_nlethals250_viol0_40452/Y.csv";
  } else if (use_big == 1) {
    printf("\nusing large test case\n");
    fixture->n = 10000;
    fixture->p = 1000;
    xfile = "../X_nlethals50_v15803.csv";
    yfile = "../Y_nlethals50_v15803.csv";
  } else if (use_big == 0) {
    printf("\nusing small test case\n");
    fixture->n = 1000;
    fixture->p = 100;
    xfile = "../testX.csv";
    yfile = "../testY.csv";
  }
  printf("reading X from %s\n", xfile);
  fixture->xmatrix = read_x_csv(xfile, fixture->n, fixture->p);
  fixture->X = fixture->xmatrix.X;
  printf("reading Y from %s\n", yfile);
  fixture->Y = read_y_csv(yfile, fixture->n);
  fixture->rowsum = malloc(fixture->n * sizeof(double));
  fixture->lambda = 20;
  long p_long = fixture->p * (fixture->p + 1) / 2;
  fixture->beta = malloc(p_long * sizeof(double));
  memset(fixture->beta, 0, p_long * sizeof(double));
  fixture->k = 27;
  fixture->dBMax = 0;
  fixture->intercept = 0;
  int_pair *precalc_get_num = malloc(p_long * sizeof(int_pair));
  long offset = 0;
  for (long i = 0; i < fixture->p; i++) {
    for (long j = i; j < fixture->p; j++) {
      precalc_get_num[offset].i = i;
      precalc_get_num[offset].j = j;
      offset++;
    }
  }
  fixture->precalc_get_num = precalc_get_num;

  cached_nums = get_all_nums(fixture->p, p_int);

  for (long i = 0; i < fixture->n; i++)
    fixture->rowsum[i] = -fixture->Y[i];

  long max_num_threads = omp_get_max_threads();
  long **thread_column_caches = malloc(max_num_threads * sizeof(long *));
  for (long i = 0; i < max_num_threads; i++) {
    thread_column_caches[i] = malloc(fixture->n * sizeof(int));
  }
  fixture->column_caches = thread_column_caches;
  printf("done test init\n");
}

static void test_simple_coordinate_descent_tear_down(UpdateFixture *fixture,
                                                     gconstpointer user_data) {
  for (long i = 0; i < fixture->p; i++) {
    free(fixture->xmatrix.X[i]);
  }
  free(fixture->xmatrix.X);
  free(fixture->Y);
  free(fixture->rowsum);
  free(fixture->beta);
  free(fixture->precalc_get_num);
  free_sparse_matrix(fixture->xmatrix_sparse);
}

static void test_simple_coordinate_descent_int(UpdateFixture *fixture,
                                               gconstpointer user_data) {
  // are we running the shuffle test, or sequential?
  double acceptable_diff = 0.1;
  long shuffle = FALSE;
  if (user_data == TRUE) {
    printf("\nrunning shuffle test!\n");
    acceptable_diff = 10;
    shuffle = TRUE;
  }
  double *glmnet_beta = read_y_csv(
      "/home/kieran/work/lasso_testing/glmnet_small_output.csv", 630);
  printf("starting interaction test\n");
  fixture->xmatrix = read_x_csv(
      "/home/kieran/work/lasso_testing/testXSmall.csv", fixture->n, fixture->p);
  fixture->X = fixture->xmatrix.X;
  fixture->xmatrix_sparse =
      sparse_X2_from_X(fixture->X, fixture->n, fixture->p, -1, shuffle);
  long p_long = fixture->p * (fixture->p + 1) / 2;
  double *beta = fixture->beta;

  double dBMax;
  for (long j = 0; j < 10; j++)
    for (long i = 0; i < p_int; i++) {
      // long k = gsl_permutation_get(fixture->xmatrix_sparse.permutation, i);
      long k = fixture->xmatrix_sparse.permutation->data[i];
      // long k = i;
      Changes changes = update_beta_cyclic_old(
          fixture->xmatrix_sparse, fixture->Y, fixture->rowsum, fixture->n,
          fixture->p, fixture->lambda, beta, k, 0, fixture->precalc_get_num,
          fixture->column_caches[0]);
      dBMax = changes.actual_diff;
    }

  long no_agreeing = 0;
  for (long i = 0; i < p_int; i++) {
    long k = gsl_permutation_get(fixture->xmatrix_sparse.permutation, i);
    // long k = i;
    printf("testing beta[%d] (%f) ~ %f [", i, beta[i],
           small_X2_correct_beta[k]);

    if ((beta[i] < small_X2_correct_beta[k] + acceptable_diff) &&
        (beta[i] > small_X2_correct_beta[k] - acceptable_diff)) {
      no_agreeing++;
      printf("x]\n");
    } else {
      printf(" ]\n");
    }
  }
  printf("frac agreement: %f\n", (double)no_agreeing / p_int);
  g_assert_true(no_agreeing == p_int);
}

static void test_simple_coordinate_descent_vs_glmnet(UpdateFixture *fixture,
                                                     gconstpointer user_data) {
  double *glmnet_beta = read_y_csv(
      "/home/kieran/work/lasso_testing/glmnet_small_output.csv", 630);
  printf("starting interaction test\n");
  fixture->p = 35;
  fixture->xmatrix = read_x_csv(
      "/home/kieran/work/lasso_testing/testXSmall.csv", fixture->n, fixture->p);
  fixture->X = fixture->xmatrix.X;
  fixture->xmatrix_sparse =
      sparse_X2_from_X(fixture->X, fixture->n, fixture->p, -1, FALSE);
  long p_long = fixture->p * (fixture->p + 1) / 2;
  double *beta = fixture->beta;

  beta = simple_coordinate_descent_lasso(
      fixture->xmatrix, fixture->Y, fixture->n, fixture->p, -1, 0.05, 1000, 100,
      0, 0.01, 1.0001, FALSE, 1, "test", FALSE, -1);

  double acceptable_diff = 10;
  long no_agreeing = 0;
  for (long i = 0; i < p_int; i++) {
    long k = gsl_permutation_get(fixture->xmatrix_sparse.permutation, i);
    printf("testing beta[%d] (%f) ~ %f [", i, beta[k], glmnet_beta[i]);

    if ((beta[k] < glmnet_beta[i] + acceptable_diff) &&
        (beta[k] > glmnet_beta[i] - acceptable_diff)) {
      no_agreeing++;
      printf("x]\n");
    } else {
      printf(" ]\n");
    }
  }
  printf("frac agreement: %f\n", (double)no_agreeing / p_int);
  g_assert_true(no_agreeing >= 0.8 * p_int);
}

// will fail if Y has been normalised
static void test_read_y_csv() {
  long n = 1000;
  double *Y = read_y_csv("/home/kieran/work/lasso_testing/testYSmall.csv", n);
  g_assert_true(Y[0] == -133.351709197933);
  g_assert_true(Y[999] == -352.293608898344);
}

// assumes little endian
void printBits(size_t const size, void const *const ptr) {
  unsigned char *b = (unsigned char *)ptr;
  unsigned char byte;
  long i, j;

  for (i = size - 1; i >= 0; i--) {
    for (j = 7; j >= 0; j--) {
      byte = (b[i] >> j) & 1;
      printf("%u", byte);
    }
  }
  puts("");
}

static void check_X2_encoding() {
  long n = 1000;
  long p = 35;
  long p_long = p * (p + 1) / 2;
  XMatrix xmatrix =
      read_x_csv("/home/kieran/work/lasso_testing/testXSmall.csv", n, p);
  XMatrix X2 = read_x_csv("../testX2Small.csv", n, p_int);
  XMatrixSparse xmatrix_sparse = sparse_X2_from_X(xmatrix.X, n, p, -1, FALSE);

  int_pair *nums = get_all_nums(p, -1);
  // create uncompressed sparse version of X2.
  long **col_nz_indices = malloc(sizeof *col_nz_indices * p_int);
  for (long j = 0; j < p_int; j++) {
  }
  long *col_sizes = malloc(sizeof *col_sizes * p_int);
  for (long j = 0; j < p_int; j++) {
    Queue *col_q = queue_new();
    for (long i = 0; i < n; i++) {
      if (X2.X[j][i] != 0) {
        queue_push_tail(col_q, i);
      }
    }
    col_sizes[j] = queue_get_length(col_q);
    col_nz_indices[j] = malloc(sizeof *col_nz_indices[j] * col_sizes[j]);
    printf(" real col size: %d, \tcompressed col contains %d entries \n",
           col_sizes[j], xmatrix_sparse.cols[j].nz);
    g_assert_true(col_sizes[j] == xmatrix_sparse.cols[j].nz);
    long pos = 0;
    while (!queue_is_empty(col_q)) {
      col_nz_indices[j][pos] = queue_pop_head(col_q);
      pos++;
      g_assert_true(pos <= col_sizes[j]);
    }
    g_assert_true(pos == col_sizes[j]);
    queue_free(col_q);
  }

  // mean entry size
  long total = 0;
  long no_entries = 0;
  for (long i = 0; i < p_int; i++) {
    no_entries += xmatrix_sparse.cols[i].nz;
    for (long j = 0; j < xmatrix_sparse.cols[i].nz; j++) {
      total += col_nz_indices[i][j];
    }
  }
  printf("\nmean entry size: %f\n", (double)total / (double)no_entries);

  // mean diff size
  total = 0;
  long prev_entry = 0;
  for (long i = 0; i < p_int; i++) {
    prev_entry = 0;
    for (long j = 0; j < xmatrix_sparse.cols[i].nz; j++) {
      total += col_nz_indices[i][j] - prev_entry;
      prev_entry = col_nz_indices[i][j];
    }
  }
  printf("mean diff size: %f\n", (double)total / (double)no_entries);

  printf("size of s8bword struct: %d (long is %ld)\n", sizeof(S8bWord),
         sizeof(int));

  S8bWord test_word;
  test_word.selector = 7;
  test_word.values = 0;
  unsigned long numbers[10] = {3, 2, 4, 20, 1, 14, 30, 52, 10, 63};
  for (long i = 0; i < 10; i++) {
    test_word.values |= numbers[9 - i];
    if (i < 9)
      test_word.values <<= item_width[test_word.selector];
  }

  S8bWord w2 = to_s8b(10, numbers);

  g_assert_true(sizeof(S8bWord) == 8);
  g_assert_true(test_word.selector == w2.selector);
  g_assert_true(test_word.values == w2.values);

  long max_size_given_entries[61];
  for (long i = 0; i < 60; i++) {
    max_size_given_entries[i] = 60 / (i + 1);
  }
  max_size_given_entries[60] = 0;

  printf("num entries in col 0: %d\n", xmatrix_sparse.cols[0].nz);
  long *col_entries = malloc(60 * sizeof(int));
  long count = 0;
  // GList *s8b_col = NULL;
  GQueue *s8b_col = g_queue_new();
  // work out s8b compressed equivalent of col 0
  long largest_entry = 0;
  long max_bits = max_size_given_entries[0];
  long diff = col_nz_indices[0][0] + 1;
  for (long i = 0; i < xmatrix_sparse.cols[0].nz; i++) {
    if (i != 0)
      diff = col_nz_indices[0][i] - col_nz_indices[0][i - 1];
    // printf("current no. %d, diff %d. available bits %d\n",
    // col_nz_indices[0][i], diff, max_bits); update max bits.
    long used = 0;
    long tdiff = diff;
    while (tdiff > 0) {
      used++;
      tdiff >>= 1;
    }
    max_bits = max_size_given_entries[count + 1];
    // if the current diff won't fit in the s8b word, push the word and start a
    // new one
    if (diff > 1 << max_bits ||
        largest_entry > max_size_given_entries[count + 1]) {
      // if (diff > 1<<max_bits)
      //  printf(" b ");
      // if (largest_entry > max_size_given_entries[count+1])
      //  printf(" c ");
      // printf("pushing word with %d entries: ", count);
      // for (long j = 0; j < count; j++)
      //  printf("%d ", col_entries[j]);
      // printf("\n");
      S8bWord *word = malloc(sizeof(S8bWord));
      S8bWord tempword = to_s8b(count, col_entries);
      memcpy(word, &tempword, sizeof(S8bWord));
      g_queue_push_tail(s8b_col, word);
      count = 0;
      largest_entry = 0;
      max_bits = max_size_given_entries[1];
    }
    col_entries[count] = diff;
    count++;
    if (used > largest_entry)
      largest_entry = used;
  }
  // push the last (non-full) word
  S8bWord *word = malloc(sizeof(S8bWord));
  S8bWord tempword = to_s8b(count, col_entries);
  memcpy(word, &tempword, sizeof(S8bWord));
  g_queue_push_tail(s8b_col, word);

  free(col_entries);
  long length = g_queue_get_length(s8b_col);

  S8bWord *actual_col = malloc(length * sizeof(S8bWord));
  count = 0;
  while (!g_queue_is_empty(s8b_col)) {
    S8bWord *current_word = g_queue_pop_head(s8b_col);
    memcpy(&actual_col[count], current_word, sizeof(S8bWord));
    count++;
  }

  printf("checking [s8b] == [int]\n");
  for (long k = 0; k < p_int; k++) {
    printf("col %d (interaction %d,%d)\n", k, nums[k].i, nums[k].j);
    long checked = 0;
    long col_entry_pos = 0;
    long entry = -1;
    for (long i = 0; i < xmatrix_sparse.cols[k].nwords; i++) {
      S8bWord word = xmatrix_sparse.cols[k].compressed_indices[i];
      unsigned long values = word.values;
      for (long j = 0; j <= group_size[word.selector]; j++) {
        long diff = values & masks[word.selector];
        if (diff != 0) {
          entry += diff;
          printf("pos %d, %d == %d\n", col_entry_pos, entry,
                 col_nz_indices[k][col_entry_pos]);
          g_assert_true(entry == col_nz_indices[k][col_entry_pos]);
          col_entry_pos++;
          checked++;
        }
        values >>= item_width[word.selector];
      }
    }
    printf("col %d, checked %d out of %d present\n", k, checked, col_sizes[k]);
    g_assert_true(checked == xmatrix_sparse.cols[k].nz);
  }

  long bytes = length * sizeof(S8bWord);
  printf("col[0] contains %d words, for a toal of %d bytes, instead of %d "
         "shorts (%d bytes). Effective reduction %f\n",
         length, bytes, xmatrix_sparse.cols[0].nz,
         xmatrix_sparse.cols[0].nz * sizeof(short),
         (double)bytes / (xmatrix_sparse.cols[0].nz * sizeof(short)));

  printf("liblasso vs test compressed first col:\n");
  for (long i = 0; i < xmatrix_sparse.cols[0].nwords; i++) {
    printf("%d == %d\n", xmatrix_sparse.cols[0].compressed_indices[i].selector,
           actual_col[i].selector);
    g_assert_true(xmatrix_sparse.cols[0].compressed_indices[i].selector ==
                  actual_col[i].selector);
    printf("%d == %d\n", xmatrix_sparse.cols[0].compressed_indices[i].values,
           actual_col[i].values);
    g_assert_true(xmatrix_sparse.cols[0].compressed_indices[i].values ==
                  actual_col[i].values);
  }
  g_assert_true(xmatrix_sparse.cols[0].nwords == length);
  printf("correct number of words\n");

  for (long j = 0; j < p_int; j++) {
    free(col_nz_indices[j]);
  }
  free(col_nz_indices);
  free(nums);
  free(col_sizes);
}

static void check_permutation() {
  long threads = omp_get_num_procs();
  gsl_rng **thread_r = malloc(threads * sizeof(gsl_rng *));
  for (long i = 0; i < threads; i++)
    thread_r[i] = gsl_rng_alloc(gsl_rng_taus2);

  long perm_size = 3235; //<< 12 + 67;
  printf("perm_size %ld\n", perm_size);
  gsl_permutation *perm = gsl_permutation_alloc(perm_size);
  gsl_permutation_init(perm);

  parallel_shuffle(perm, perm_size / threads, perm_size % threads, threads);

  long *found = malloc(perm_size * sizeof(int));
  memset(found, 0, perm_size * sizeof(int));
  for (long i = 0; i < perm_size; i++) {
    size_t val = perm->data[i];
    found[val] = 1;
    printf("found %d\n", val);
  }
  for (long i = 0; i < perm_size; i++) {
    printf("checking %d is present\n", i);
    printf("found[%d] = %d\n", i, found[i]);
    printf("found[%d+1] = %d\n", i, found[i + 1]);
    g_assert_true(found[i] == 1);
  }
  free(found);
  gsl_permutation_free(perm);

  perm_size = 123123; //<< 12 + 67;
  printf("perm_size %ld\n", perm_size);
  perm = gsl_permutation_alloc(perm_size);
  gsl_permutation_init(perm);

  parallel_shuffle(perm, perm_size / threads, perm_size % threads, threads);

  found = malloc(perm_size * sizeof(int));
  memset(found, 0, perm_size);
  for (long i = 0; i < perm_size; i++) {
    long val = perm->data[i];
    found[val] = 1;
    printf("found %d\n", val);
  }
  for (long i = 0; i < perm_size; i++) {
    printf("checking %d is present\n", i);
    printf("found[%d] = %d\n", i, found[i]);
    printf("found[%d+1] = %d\n", i, found[i + 1]);
    g_assert_true(found[i] == 1);
  }
  free(found);
  gsl_permutation_free(perm);
}

long check_didnt_update(long p, long p_int, long *wont_update, double *beta) {
  long no_disagreeing = 0;
  for (long i = 0; i < p_int; i++) {
    // long k = gsl_permutation_get(fixture->xmatrix_sparse.permutation, i);
    long k = i;
    // long k = fixture->xmatrix_sparse.permutation->data[i];
    // printf("testing beta[%d] (%f)\n", k, beta[k]);
    int_pair ip = get_num(k, p_int);
    // TODO: we should only check against later items not in the working set,
    // this needs to be udpated
    if (wont_update[ip.i] || wont_update[ip.j]) {
      // printf("checking interaction %d,%d is zero\n", ip.i, ip.j);
      if (beta[k] != 0.0) {
        printf("beta %d (interaction %d,%d) should be zero according to "
               "will_update_effect(), but is in fact %f\n",
               k, ip.i, ip.j, beta[k]);
        no_disagreeing++;
      }
    }
  }
  // printf("frac disagreement: %f\n", (double)no_disagreeing/p);
  if (no_disagreeing == 0) {
    // printf("no disagreements\n");
  }
  return no_disagreeing;
}

static void pruning_fixture_set_up(UpdateFixture *fixture,
                                   gconstpointer use_big) {
  test_simple_coordinate_descent_set_up(fixture, use_big);
  printf("getting sparse X\n");
  XMatrixSparse Xc = sparsify_X(fixture->X, fixture->n, fixture->p);
  printf("getting sparse X2\n");
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  if (use_big < 2) {
    XMatrixSparse X2c =
        sparse_X2_from_X(fixture->X, fixture->n, fixture->p, -1, FALSE);
    fixture->X2c = X2c;
  }
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  x2_conversion_time = ((double)(end.tv_nsec - start.tv_nsec)) / 1e9 +
                       (end.tv_sec - start.tv_sec);
  fixture->Xc = Xc;
}

static void pruning_fixture_tear_down(UpdateFixture *fixture,
                                      gconstpointer user_data) {
  free_sparse_matrix(fixture->Xc);
  free_sparse_matrix(fixture->X2c);

  test_simple_coordinate_descent_tear_down(fixture, NULL);
}

long get_wont_update(char *working_set, long *wont_update, long p,
                    XMatrixSparse Xc, double lambda, double *last_max,
                    double **last_rowsum, double *rowsum, long *column_cache,
                    long n, double *beta) {
  long ruled_out = 0;
  for (long j = 0; j < p; j++) {
    double sum = 0.0;
    wont_update[j] = wont_update_effect(
        Xc, lambda, j, last_max[j], last_rowsum[j], rowsum, column_cache, beta);
    if (wont_update[j])
      ruled_out++;
  }

  // for (long i = 0; i < p; i++) {
  //  if (wont_update[i]) {
  //      printf("%d supposedly wont update\n", i);
  //  }
  //}
  // printf("ruled out %d branch(es)\n", ruled_out);
  return ruled_out;
}
// run branch_prune check, then full regression step without pruning.
// the beta values that would have been pruned should be 0.
static void check_branch_pruning(UpdateFixture *fixture,
                                 gconstpointer user_data) {
  printf("\nstarting branch pruning test\n");
  long n = fixture->n;
  long p = fixture->p;
  double *rowsum = fixture->rowsum;
  // double lambda = fixture->lambda;
  long shuffle = FALSE;
  printf("starting interaction test\n");
  // fixture->xmatrix = read_x_csv("/ho/testXSmall.csv", fixture->n,
  // fixture->p);
  printf("creating X2\n");
  long p_long = fixture->p * (fixture->p + 1) / 2;
  double *beta = fixture->beta;
  char *working_set = malloc(sizeof *working_set * p_int);
  memset(working_set, 0, sizeof *working_set * p_int);

  XMatrixSparse Xc = fixture->Xc;
  XMatrixSparse X2c = fixture->X2c;
  long column_cache[n];

  long wont_update[p];
  for (long j = 0; j < p; j++)
    wont_update[j] = 0;

  double **last_rowsum = malloc(sizeof *last_rowsum * p);
#pragma omp parallel for schedule(static)
  for (long i = 0; i < p; i++) {
    last_rowsum[i] = malloc(sizeof *last_rowsum[i] * n + PADDING);
    memset(last_rowsum[i], 0, sizeof *last_rowsum[i] * n);
  }

  for (long j = 0; j < p; j++)
    for (long i = 0; i < n; i++)
      g_assert_true(last_rowsum[j][i] == 0.0);

  double last_max[n];
  memset(last_max, 0, sizeof(last_max));

  // start running tests with decreasing lambda
  // double lambda_sequence[] = {10000,500, 400, 300, 200, 100, 50, 25, 10, 5,
  // 2, 1, 0.5, 0.2, 0.1, 0.05, 0.01};
  double lambda_sequence[] = {10000, 500, 400, 300, 200, 100, 50, 25,
                              10,    5,   2,   1,   0.5, 0.2, 0.1};
  long seq_length = sizeof(lambda_sequence) / sizeof(*lambda_sequence);
  double lambda = lambda_sequence[0];
  long ruled_out = 0;
  double *old_rowsum = malloc(sizeof *old_rowsum * n);

  double error = 0.0;
  for (long i = 0; i < n; i++) {
    error += rowsum[i] * rowsum[i];
  }
  error = sqrt(error);
  printf("initial error: %f\n", error);

  double max_int_delta[p];
  memset(max_int_delta, 0, sizeof *max_int_delta * p);
  for (long lambda_ind = 0; lambda_ind < seq_length; lambda_ind++) {
    memcpy(old_rowsum, rowsum, sizeof *rowsum * n);
    lambda = lambda_sequence[lambda_ind];
    printf("\nrunning lambda %f, current error: %f\n", lambda, error);
    double dBMax;
    // TODO: implement working set and update test
    long iter = 0;
    for (iter = 0; iter < 50; iter++) {
      double prev_error = error;

      ruled_out =
          get_wont_update(working_set, wont_update, p, Xc, lambda, last_max,
                          last_rowsum, rowsum, column_cache, n, beta);
      printf("iter %d ruled out %d\n", iter, ruled_out);
      long k = 0;
      for (long main_effect = 0; main_effect < p; main_effect++) {
        for (long interaction = main_effect; interaction < p; interaction++) {
          double old = beta[k];
          Changes changes = update_beta_cyclic_old(
              X2c, fixture->Y, rowsum, n, p, lambda, beta, k, 0,
              fixture->precalc_get_num, column_cache);
          dBMax = changes.actual_diff;
          double new = beta[k];
          if (!working_set[k] && fabs(changes.pre_lambda_diff) >
                                     fabs(max_int_delta[main_effect])) {
            max_int_delta[main_effect] = changes.pre_lambda_diff;
          }
          // TODO: maybe reasonable?
          if (changes.actual_diff != 0.0) {
            working_set[k] = TRUE;
          } else {
            working_set[k] = FALSE;
          }
          k++;
        }
      }
      long no_disagreeing = check_didnt_update(p, p_int, wont_update, beta);
      g_assert_true(no_disagreeing == 0);

      for (long i = 0; i < p; i++) {
        if (!wont_update[i]) {
          if (last_max[i] != max_int_delta[i]) {
            printf("main effect %d new last_max is %f\n", i, max_int_delta[i]);
          }
          last_max[i] = max_int_delta[i];
        }
      }
      error = 0.0;
      for (long i = 0; i < n; i++) {
        error += rowsum[i] * rowsum[i];
      }
      error = sqrt(error);
      if (prev_error / error < 1.001) {
        break;
      }
    }
    printf("done lambda %f in %d iters\n", lambda, iter + 1);
    for (long i = 0; i < p; i++) {
      // we did check these anyway, but since we ordinarily wouldn't they don't
      // get updated.
      if (!wont_update[i]) {
        // printf("updating last_rowsum for %d\n", i);
        memcpy(last_rowsum[i], old_rowsum, sizeof *old_rowsum * n);
      }
    }
    printf("new error: %f\n", error);
  }
}

typedef struct {
  XMatrixSparse Xc;
  double **last_rowsum;
  Thread_Cache *thread_caches;
  long n;
  double *beta;
  double *last_max;
  long *wont_update;
  long p;
  long p_int;
  XMatrixSparse X2c;
  double *Y;
  double *max_int_delta;
  int_pair *precalc_get_num;
  gsl_permutation *iter_permutation;
} Iter_Vars;

// nothing in vars is changed
double run_lambda_iters(Iter_Vars *vars, double lambda, double *rowsum) {
  XMatrixSparse Xc = vars->Xc;
  double **last_rowsum = vars->last_rowsum;
  // long **thread_column_caches = vars->thread_column_caches;
  Thread_Cache *thread_caches = vars->thread_caches;
  long n = vars->n;
  double *beta = vars->beta;
  double *last_max = vars->last_max;
  long *wont_update = vars->wont_update;
  long p = vars->p;
  long p_long = vars->p_int;
  XMatrixSparse X2c = vars->X2c;
  double *Y = vars->Y;
  double *max_int_delta = vars->max_int_delta;
  int_pair *precalc_get_num = vars->precalc_get_num;
  gsl_permutation *iter_permutation = vars->iter_permutation;

  double error = 0.0;
  for (long i = 0; i < n; i++) {
    error += rowsum[i] * rowsum[i];
  }
  error = sqrt(error);
  for (long iter = 0; iter < 100; iter++) {
    // printf("iter %d\n", iter);
    // last_iter_count = iter;
    double prev_error = error;

    parallel_shuffle(iter_permutation, permutation_split_size, final_split_size,
                     permutation_splits);
#pragma omp parallel for num_threads(NumCores)                                 \
    shared(X2c, Y, rowsum, beta, precalc_get_num) schedule(static)
    for (long k = 0; k < p_int; k++) {
      // for (long main_effect = 0; main_effect < p; main_effect++) {
      // for (long interaction = main_effect; interaction < p; interaction++) {
      Changes changes = update_beta_cyclic_old(
          X2c, Y, rowsum, n, p, lambda, beta, k, 0, precalc_get_num,
          thread_caches[omp_get_thread_num()].col_i);
      // k++;
      // }
    }
    // g_assert_true(k == p_int);

    error = 0.0;
    for (long i = 0; i < n; i++) {
      error += rowsum[i] * rowsum[i];
    }
    error = sqrt(error);
    // printf("prev_error: %f \t error: %f\n", prev_error, error);
    if (prev_error / error < HALT_ERROR_DIFF) {
      printf("done lambda %.2f after %d iters\n", lambda, iter + 1);
      break;
    } else if (iter == 99) {
      printf("halting lambda %.2f after 100 iters\n", lambda);
    }
  }
  // return dBMax;
  return 0.0;
}

struct AS_Properties {
  long was_present : 1;
  long present : 1;
};

typedef struct {
  // long *entries;
  // struct AS_Properties *properties;
  struct AS_Entry *entries;
  long length;
  long max_length;
  gsl_permutation *permutation;
  // S8bCol *compressed_cols;
} Active_Set;

/*
 * Fits 6 to a cache line. As long as schedule is static, this should be fine.
 */
struct AS_Entry {
  long val : 62;
  long was_present : 1;
  long present : 1;
  S8bCol col;
  // TODO: shouldn't need this
  // char padding[39];
};

Active_Set active_set_new(long max_length) {
  struct AS_Entry *entries = malloc(sizeof *entries * max_length);
  // N.B this is what sets was_present to false for every entry
  memset(entries, 0,
         sizeof *entries *
             max_length); // not strictly necessary, but probably safer.
  long length = 0;
  Active_Set as = {entries, length, max_length, NULL};
  return as;
}

void active_set_free(Active_Set as) {
  for (long i = 0; i < as.length; i++) {
    struct AS_Entry *e = &as.entries[i];
    if (NULL != e->col.compressed_indices) {
      free(e->col.compressed_indices);
    }
  }
  free(as.entries);
  if (NULL != as.permutation)
    free(as.permutation);
}

void active_set_append(Active_Set *as, long value, long *col, long len) {
  struct AS_Entry *e = &as->entries[value];
  if (e->was_present) {
    e->present = TRUE;
  } else {
    long i = as->length;
    e->val = value;
    e->present = TRUE;
    e->was_present = TRUE;
    g_assert_true(as->length < as->max_length);
    // printf("new col has %d words, %d entries\n", s8bCol.nwords,
    // s8bCol.nz); printf("adding col for effect %d\n", value);
    e->col = col_to_s8b_col(len, col);
  }
}

void active_set_remove(Active_Set *as, long index) {
  as->entries[index].present = FALSE;
}

long active_set_get_index(Active_Set *as, long index) {
  struct AS_Entry *e = &as->entries[index];
  if (e->present) {
    return e->val;
  } else {
    return -INT_MAX;
  }
}

struct RawCol {
  long len;
  long *entries;
};

struct timespec sub_start;
struct timespec sub_end;

double reused_col_time = 0.0;
double main_col_time = 0.0;
double int_col_time = 0.0;

// struct RawCol get_raw_interaction_col(XMatrixSparse Xc, long i, long j) {}

/*
 * Returns true if and only if something was added to the active set.
 */
char update_working_set(XMatrixSparse Xc, double *rowsum, long *wont_update,
                        Active_Set *as, double *last_max, long p, long n,
                        int_pair *precalc_get_num, double lambda, double *beta,
                        Thread_Cache *thread_caches, XMatrixSparse X2c) {
  char increased_set = FALSE;
  long length_increase = 0;
#pragma omp parallel for reduction(& : increased_set) shared(last_max) schedule(static) reduction(+: reused_col_time, main_col_time, int_col_time, length_increase)
  for (long main = 0; main < p; main++) {
    Thread_Cache thread_cache = thread_caches[omp_get_thread_num()];
    long *col_i_cache = thread_cache.col_i;
    long *col_j_cache = thread_cache.col_j;
    clock_gettime(CLOCK_MONOTONIC_RAW, &sub_start);
    long main_col_len = 0;
    if (!wont_update[main]) {
      {
        long *column_entries = col_i_cache;
        long col_entry_pos = 0;
        long entry = -1;
        for (long r = 0; r < Xc.cols[main].nwords; r++) {
          S8bWord word = Xc.cols[main].compressed_indices[r];
          unsigned long values = word.values;
          for (long j = 0; j <= group_size[word.selector]; j++) {
            long diff = values & masks[word.selector];
            if (diff != 0) {
              entry += diff;
              column_entries[col_entry_pos] = entry;
              col_entry_pos++;
            }
            values >>= item_width[word.selector];
          }
        }
        main_col_len = col_entry_pos;
        g_assert_true(main_col_len == Xc.cols[main].nz);
      }
      long read_loops = 0;
      clock_gettime(CLOCK_MONOTONIC_RAW, &sub_end);
      main_col_time += ((double)(sub_end.tv_nsec - sub_start.tv_nsec)) / 1e9 +
                       (sub_end.tv_sec - sub_start.tv_sec);
      for (long inter = main; inter < p; inter++) {
        // TODO: no need to re-read the main column when inter == main.
        // worked out by hand as being equivalent to the offset we would have
        // reached. sumb is the amount we would have reached w/o the limit -
        // the amount that was actually covered by the limit.
        long k = (2 * (p - 1) + 2 * (p - 1) * (main - 1) -
                 (main - 1) * (main - 1) - (main - 1)) /
                    2 +
                inter;
        double sumn = 0.0;
        long col_nz = 0;
        // g_assert_true(precalc_get_num[k].i == j);
        g_assert_true(k <= (Xc.p * (Xc.p + 1) / 2));
        if (!wont_update[inter]) {

          // we've already calculated the interaction, re-use it.
          if (as->entries[k].was_present) {
            clock_gettime(CLOCK_MONOTONIC_RAW, &sub_start);
            // printf("re-using column %d\n", k);
            S8bCol s8bCol = as->entries[k].col;
            col_nz = s8bCol.nz;
            long entry = -1;
            long tmpCount = 0;
            for (long i = 0; i < s8bCol.nwords; i++) {
              S8bWord word = s8bCol.compressed_indices[i];
              unsigned long values = word.values;
              for (long j = 0; j <= group_size[word.selector]; j++) {
                tmpCount++;
                long diff = values & masks[word.selector];
                if (diff != 0) {
                  entry += diff;
                  if (entry > Xc.n) {
                    printf("entry: %d\n", entry);
                    printf("col %d col_nz: %d, tmpCount: %d\n", k, col_nz,
                           tmpCount);
                  }
                  g_assert_true(entry < Xc.n);
                  sumn += rowsum[entry];
                }
                values >>= item_width[word.selector];
              }
            }
            clock_gettime(CLOCK_MONOTONIC_RAW, &sub_end);
            reused_col_time +=
                ((double)(sub_end.tv_nsec - sub_start.tv_nsec)) / 1e9 +
                (sub_end.tv_sec - sub_start.tv_sec);
          } else {
            clock_gettime(CLOCK_MONOTONIC_RAW, &sub_start);
            // printf("calculating new column\n");
            // this column has never been in the working set before, therefore
            // its beta value is zero and so is sumn.
            // calculate the interaction
            // and maybe store it read columns i and j simultaneously
            // long entry_i = -1;
            long i_pos = 0;
            long entry_j = -1;
            long pos = 0;
            // sum of rowsums for this column
            // long i_w = 0;
            long j_w = 0;
            // S8bWord word_i = Xc.compressed_indices[i][i_w];
            S8bWord word_j = Xc.cols[inter].compressed_indices[j_w];
            // long i_wpos = 0;
            long j_wpos = 0;
            // unsigned long values_i = word_i.values;
            unsigned long values_j = word_j.values;
            // long i_size = group_size[word_i.selector];
            long j_size = group_size[word_j.selector];
            // This whole loop is a bit awkward, but what can you do.
            // printf("interaction between %d (len %d) and %d (len %d)\n", i,
            //  Xc.col_nz[i], j, Xc.col_nz[j]);
            long entry_i = -2;
            while (i_pos < main_col_len && j_w <= Xc.cols[inter].nwords) {
            read:
              if (entry_i == entry_j) {
                // update interaction and move to next entry of each word
                sumn += rowsum[entry_i];
                col_j_cache[pos] = entry_i;
                pos++;
                g_assert_true(pos < Xc.n);
              }
              while (entry_i <= entry_j && i_pos < main_col_len) {
                entry_i = col_i_cache[i_pos];
                i_pos++;
                if (entry_i == entry_j) {
                  goto read;
                }
              }
              if (entry_j <= entry_i) {
                // read through j until we hit the end, or reach or exceed
                // i.
                while (j_w <= Xc.cols[inter].nwords) {
                  // current word
                  while (j_wpos <= j_size) {
                    long diff = values_j & masks[word_j.selector];
                    j_wpos++;
                    values_j >>= item_width[word_j.selector];
                    if (diff != 0) {
                      entry_j += diff;
                      // we've found the next value of j
                      // if it's equal we'll handle it earlier in the loop,
                      // otherwise go to the j read loop.
                      if (entry_j >= entry_i) {
                        goto read;
                        // break;
                      }
                    }
                  }
                  // switch to the next word
                  j_w++;
                  if (j_w < Xc.cols[inter].nwords) {
                    word_j = Xc.cols[inter].compressed_indices[j_w];
                    values_j = word_j.values;
                    j_size = group_size[word_j.selector];
                    j_wpos = 0;
                  }
                }
              }
            }
            col_nz = pos;
            // g_assert_true(pos == X2c.nz[k]);
            // N.B. putting everything we use in the active set, but marking it
            // as inactive. this speeds up future checks quite significantly, at
            // the cost of increased memory use. For a gpu implenentation we
            // probably don't want this.
            //active_set_append(as, k, col_j_cache, col_nz);
            //length_increase++;
            //active_set_remove(as, k);
            clock_gettime(CLOCK_MONOTONIC_RAW, &sub_end);
            int_col_time +=
                ((double)(sub_end.tv_nsec - sub_start.tv_nsec)) / 1e9 +
                (sub_end.tv_sec - sub_start.tv_sec);
          }
          // if (k == 85)
          //  printf("interaction contains %d cols, sumn: %f\n", col_nz, sumn);
          // either way, we now have sumn
          sumn = fabs(sumn);
          sumn += fabs(beta[k] * col_nz);
          if (sumn > last_max[inter]) {
            last_max[inter] = sumn;
          }
          if (sumn > lambda * n / 2) {
            active_set_append(as, k, col_j_cache, col_nz);
            increased_set = TRUE;
          } else {
            // active_set_remove(as, k);
          }
          // TODO: store column for re-use even if it's not really added to
          // the active set?
        } else {
          // since we don't update any of this columns interactions, they
          // shouldn't be in the working set
          // active_set_remove(as, k);
        }
      }
    }
  }
  as->length += length_increase;
  return increased_set;
}

static double pruning_time = 0.0;
static double working_set_update_time = 0.0;
static double subproblem_time = 0.0;

long run_lambda_iters_pruned(Iter_Vars *vars, double lambda, double *rowsum,
                            double *old_rowsum, Active_Set *active_set) {
  XMatrixSparse Xc = vars->Xc;
  double **last_rowsum = vars->last_rowsum;
  Thread_Cache *thread_caches = vars->thread_caches;
  long n = vars->n;
  double *beta = vars->beta;
  double *last_max = vars->last_max;
  long *wont_update = vars->wont_update;
  long p = vars->p;
  long p_long = vars->p_int;
  XMatrixSparse X2c = vars->X2c;
  double *Y = vars->Y;
  double *max_int_delta = vars->max_int_delta;
  int_pair *precalc_get_num = vars->precalc_get_num;
  long new_nz_beta = 0;
  // active_set[i] if and only if the pair precalc_get_num[i] is in the
  // active set.
  gsl_permutation *iter_permutation = vars->iter_permutation;
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
  gsl_permutation *perm;
  // char *new_active_branch = malloc(sizeof *new_active_branch * p);
  // char new_active_branch[p];
  //#pragma omp parallel for schedule(static)
  //  for (long i = 0; i < p; i++) {
  //    new_active_branch[i] = FALSE;
  //  }

  double error = 0.0;
  for (long i = 0; i < n; i++) {
    error += rowsum[i] * rowsum[i];
  }

  // allocate a local copy of rowsum for each thread
  long **thread_rowsums[NumCores];
  //#pragma omp parallel for
  //  for (long i = 0; i < NumCores; i++) {
  //    long *tr = malloc(sizeof *rowsum * n + 64);
  //    memcpy(tr, rowsum, sizeof *rowsum * n);
  //    thread_rowsums[omp_get_thread_num()] = tr;
  //  }

  printf("\nrunning lambda %f\n", lambda);
  // run several iterations of will_update to make sure we catch any new
  // columns
  /*TODO: in principle we should allow more than one, but it seems to
   * only slow things down. maybe this is because any effects not chosen
   * for the first iter will be small, and therefore not really worht
   * it? (i.e. slow and unreliable).
   * TODO: suffers quite badly when numa updates are allowed
   *        - check if this is only true for small p (openmp tests seem to only
   * improve with p >= 5k)
   */
  // TODO: with multiple iters, many branches are added on the second iter. This
  // doesn't seem right.
  for (long retests = 0; retests < 100; retests++) {
    printf("test %d\n", retests + 1);
    long total_changed = 0;
    long total_unchanged = 0;
    long total_changes = 0;
    long total_present = 0;
    long total_notpresent = 0;
// memset(max_int_delta, 0, sizeof *max_int_delta * p);
// memset(last_max, 0, sizeof *last_max * p);
#pragma omp parallel for schedule(static)
    for (long i = 0; i < p; i++) {
      max_int_delta[i] = 0;
      last_max[i] = 0;
    }

    //********** Branch Pruning       *******************
    printf("branch pruning. ");
    long active_branches = 0;
    long new_active_branches = 0;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    // make a local copy of rowsum for each thread
    //#pragma omp parallel for
    //    for (long i = 0; i < NumCores; i++) {
    //      memcpy(thread_rowsums[omp_get_thread_num()], rowsum, sizeof *rowsum
    //      * n);
    //    }

#pragma omp parallel for schedule(static) reduction(+ : new_active_branches)
    for (long j = 0; j < p; j++) {
      long old_wont_update = wont_update[j];
      wont_update[j] =
          wont_update_effect(Xc, lambda, j, last_max[j], last_rowsum[j], rowsum,
                             thread_caches[omp_get_thread_num()].col_j, beta);
      char new_active_branch = old_wont_update && !wont_update[j];
      if (new_active_branch)
        new_active_branches++;
    }
    // this slows things down on multiple numa nodes. There must be something
    // going on with rowsum/last_rowsum?
    // #pragma omp threadprivate(local_rowsum) num_threads(NumCores)
    // #pragma omp parallel num_threads(NumCores) shared(last_rowsum)
    {
// long *local_rowsum = malloc(n * sizeof *rowsum);
// printf("local_rowsum: %x\n");
// memcpy(local_rowsum, rowsum, n * sizeof *rowsum);
// TODO: parallelising this loop slows down numa updates.
#pragma omp parallel for schedule(static) reduction(+ : active_branches)
      for (long j = 0; j < p; j++) {
        // if the branch hasn't been pruned then we'll get an accurate estimate
        // for this rowsum from update_working_set.
        if (!wont_update[j]) {
          memcpy(last_rowsum[j], rowsum,
                 sizeof *rowsum * n); // TODO: probably overkill
          active_branches++;
        }
      }
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    pruning_time += ((double)(end.tv_nsec - start.tv_nsec)) / 1e9 +
                    (end.tv_sec - start.tv_sec);
    printf("(%d active branches, %d new)\n", active_branches,
           new_active_branches);
    if (active_branches == 0) {
      break;
    }
    //********** Identify Working Set *******************
    // TODO: is it worth constructing a new set with no 'blank'
    // elements?
    printf("updating working set.\n");
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    char increased_set =
        update_working_set(Xc, rowsum, wont_update, active_set, last_max, p, n,
                           precalc_get_num, lambda, beta, thread_caches, X2c);
    // update_working_set(Xc, rowsum, new_active_branch, active_set, last_max,
    //                   p, n, precalc_get_num, lambda, beta, thread_caches);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    working_set_update_time += ((double)(end.tv_nsec - start.tv_nsec)) / 1e9 +
                               (end.tv_sec - start.tv_sec);
    if (retests > 0 && !increased_set) {
      // there's no need to re-run on the same set. Nothing has changed
      // and the remaining retests will all do nothing.
      printf("didn't increase set, no further iters\n");
      break;
    }
    printf("active set size: %d, or %.2f \%\n", active_set->length,
           100 * (double)active_set->length / (double)p_int);
    permutation_splits = max(NumCores, active_set->length / NumCores);
    permutation_split_size = active_set->length / permutation_splits;
    if (active_set->length > NumCores) {
      final_split_size = active_set->length % NumCores;
    } else {
      final_split_size = 0;
    }
    if (active_set->length > 0) {
      // printf("allocation permutation of size %d\n",
      // active_set->length);
      perm = gsl_permutation_calloc(
          active_set->length); // TODO: don't alloc/free in this loop
      // printf("permutation has size %d\n", perm->size);
    }
    //********** Solve subproblem     *******************
    printf("solving subproblem.\n");
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    long iter = 0;
    for (iter = 0; iter < 100; iter++) {
      double prev_error = error;
      // update entire working set
      // TODO: we should shuffle the active set, not the matrix
      // if (active_set->length > NumCores) {
      //  parallel_shuffle(perm, permutation_split_size, final_split_size,
      //                   permutation_splits);
      //}
      // parallel_shuffle(iter_permutation, permutation_split_size,
      //                 final_split_size, permutation_splits);
#pragma omp parallel for num_threads(NumCores) schedule(static) shared(X2c, Y, rowsum, beta, precalc_get_num, perm) reduction(+:total_unchanged, total_changed, total_present, total_notpresent, new_nz_beta)
      for (long i = 0; i < p; i++) {
        for (long j = i; j < p; j++) {
          long k = (2 * (p - 1) + 2 * (p - 1) * (i - 1) - (i - 1) * (i - 1) -
                   (i - 1)) /
                      2 +
                  j;
          if (active_set->entries[k].present) {
            // TODO: apply permutation here.
            total_present++;
            long was_zero = FALSE;
            if (beta[k] == 0.0) {
              was_zero = TRUE;
            }
            Changes changes = update_beta_cyclic(
                active_set->entries[k].col, Y, rowsum, n, p, lambda, beta, k, 0,
                precalc_get_num, thread_caches[omp_get_thread_num()].col_i);
            if (changes.actual_diff == 0.0) {
              total_unchanged++;
            } else {
              total_changed++;
            }
            if (was_zero && changes.actual_diff != 0) {
              new_nz_beta++;
            }
            if (!was_zero && beta[k] == 0) {
              new_nz_beta--;
            }
          } else {
            total_notpresent++;
          }
        }
      }
      // for (long ki = 0; ki < active_set->length; ki++) {
      //  long k = active_set->entries[perm->data[ki]];
      //  if (active_set->properties[k].present) {
      //    total_present++;
      //    Changes changes = update_beat_cyclic_old(
      //        X2c, Y, rowsum, n, p, lambda, beta, k, 0, precalc_get_num,
      //        thread_caches[omp_get_thread_num()].col_i);
      //    if (changes.actual_diff == 0.0) {
      //      total_unchanged++;
      //    } else {
      //      total_changed++;
      //    }
      //  } else {
      //    total_notpresent++;
      //  }
      //}
      // check whether we need another iteration
      error = 0.0;
      for (long i = 0; i < n; i++) {
        error += rowsum[i] * rowsum[i];
      }
      error = sqrt(error);
      if (prev_error / error < HALT_ERROR_DIFF) {
        // printf("done after %d iters\n", lambda, iter+1);
        break;
      }
    }
    // printf("active set length: %d, present: %d not: %d\n",
    // active_set->length, total_present, total_notpresent);
    // g_assert_true(total_present/iter+total_notpresent/iter ==
    // active_set->length-1);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    subproblem_time += ((double)(end.tv_nsec - start.tv_nsec)) / 1e9 +
                       (end.tv_sec - start.tv_sec);
    // printf("%.1f%% of active set updates didn't change\n",
    // (double)(total_changed*100)/(double)(total_changed+total_unchanged));
    // printf("%.1f%% of active set was blank\n",
    // (double)total_present/(double)(total_present+total_notpresent));
    if (active_set->length > 0) {
      gsl_permutation_free(perm);
    }
  }

  //#pragma omp parallel for
  //  for (long i = 0; i < NumCores; i++) {
  //    free(thread_rowsums[omp_get_thread_num()]);
  //  }

  // free(new_active_branch);
  gsl_rng_free(rng);
  return new_nz_beta;
}

static void check_branch_pruning_faster(UpdateFixture *fixture,
                                        gconstpointer user_data) {
  printf("starting branch pruning speed test\n");
  double acceptable_diff = 1.05;
  long n = fixture->n;
  long p = fixture->p;
  double *rowsum = fixture->rowsum;
  long shuffle = FALSE;
  printf("starting interaction test\n");
  printf("creating X2\n");
  long p_long = fixture->p * (fixture->p + 1) / 2;
  double *beta = fixture->beta;
  double *Y = fixture->Y;
  printf("test\n");
  const double LAMBDA_MIN = 5;
  gsl_permutation *iter_permutation = gsl_permutation_alloc(p_int);

  Thread_Cache thread_caches[NumCores];

  for (long i = 0; i < NumCores; i++) {
    thread_caches[i].col_i = malloc(sizeof(int) * n);
    thread_caches[i].col_j = malloc(sizeof(int) * n);
  }

  XMatrixSparse Xc = fixture->Xc;
  XMatrixSparse X2c = fixture->X2c;

  long *wont_update = malloc(sizeof *wont_update * p);
#pragma omp parallel for schedule(static)
  for (long j = 0; j < p; j++)
    wont_update[j] = 0;

  double **last_rowsum = malloc(sizeof *last_rowsum * p);
#pragma omp parallel for schedule(static)
  for (long i = 0; i < p; i++) {
    last_rowsum[i] = malloc(sizeof *last_rowsum[i] * n + 64);
    memset(last_rowsum[i], 0, sizeof *last_rowsum[i] * n);
  }

  for (long j = 0; j < p; j++)
    for (long i = 0; i < n; i++)
      g_assert_true(last_rowsum[j][i] == 0.0);

  double last_max[n];
  memset(last_max, 0, sizeof(last_max));

  // start running tests with decreasing lambda
  // double lambda_sequence[] = {10000,500, 400, 300, 200, 100, 50, 25,
  // 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.01};
  double lambda_sequence[] = {10000, 500, 400, 300, 200, 100, 50,  25,
                              10,    5,   2,   1,   0.5, 0.2, 0.1, 0.05};
  // double lambda_sequence[] = {10000,500, 400, 300};
  long seq_length = sizeof(lambda_sequence) / sizeof(*lambda_sequence);
  double *old_rowsum = malloc(sizeof *old_rowsum * n);

  double error = 0.0;
  for (long i = 0; i < n; i++) {
    error += rowsum[i] * rowsum[i];
  }
  error = sqrt(error);
  // printf("initial error: %f\n", error);

  double *max_int_delta = malloc(sizeof *max_int_delta * p);
  memset(max_int_delta, 0, sizeof *max_int_delta * p);

  Iter_Vars iter_vars_basic = {
      Xc,
      last_rowsum,
      thread_caches,
      n,
      beta,
      last_max,
      NULL,
      p,
      p_int,
      X2c,
      fixture->Y,
      max_int_delta,
      fixture->precalc_get_num,
      iter_permutation,
  };
  struct timespec start, end;
  double basic_cpu_time_used, pruned_cpu_time_used;
  printf("getting time for un-pruned version\n");
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  // for (long lambda_ind = 0; lambda_ind < seq_length; lambda_ind ++) {
  for (double lambda = 10000; lambda > LAMBDA_MIN; lambda *= 0.95) {
    // double lambda = lambda_sequence[lambda_ind];
    printf("lambda: %f\n", lambda);
    double dBMax;
    // TODO: implement working set and update test
    long last_iter_count = 0;

    if (Xc.p <= 1000) {
      run_lambda_iters(&iter_vars_basic, lambda, rowsum);
    }
  }
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  basic_cpu_time_used = ((double)(end.tv_nsec - start.tv_nsec)) / 1e9 +
                        (end.tv_sec - start.tv_sec);
  printf("time: %f s\n", basic_cpu_time_used);

  double *beta_pruning = malloc(sizeof *beta_pruning * p_int);
#pragma omp parallel for schedule(static)
  for (long i = 0; i < p; i++) {
    memset(last_rowsum[i], 0, sizeof *last_rowsum[i] * n);
    last_max[i] = 0.0;
    max_int_delta[i] = 0;
  }
  for (long i = 0; i < n; i++) {
    rowsum[i] = -Y[i];
  }
  for (long i = 0; i < p_int; i++) {
    beta_pruning[i] = 0;
  }
  Iter_Vars iter_vars_pruned = {
      Xc,
      last_rowsum,
      thread_caches,
      n,
      beta_pruning,
      last_max,
      wont_update,
      p,
      p_int,
      X2c,
      fixture->Y,
      max_int_delta,
      fixture->precalc_get_num,
      iter_permutation,
  };

  printf("getting time for pruned version\n");
  double *p_rowsum = malloc(sizeof *p_rowsum * n);
  for (long i = 0; i < n; i++) {
    p_rowsum[i] = -Y[i];
  }
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  // for (long lambda_ind = 0; lambda_ind < seq_length; lambda_ind ++) {
  Active_Set active_set = active_set_new(p_int);
  for (double lambda = 10000; lambda > LAMBDA_MIN; lambda *= 0.95) {
    // memcpy(old_rowsum, p_rowsum, sizeof *p_rowsum *n);
    // double lambda = lambda_sequence[lambda_ind];
    double dBMax;
    // TODO: implement working set and update test
    long last_iter_count = 0;

    run_lambda_iters_pruned(&iter_vars_pruned, lambda, p_rowsum, old_rowsum,
                            &active_set);
  }

  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  printf("addresses maybe of interest:\n");
  printf("Xc.cols:       %lx\n", Xc.cols);
  printf("last_rowsum:   %lx\n", last_rowsum);
  printf("p_rowsum:      %lx\n", p_rowsum);
  printf("wont_update:   %lx\n", wont_update);
  active_set_free(active_set);
  pruned_cpu_time_used = ((double)(end.tv_nsec - start.tv_nsec)) / 1e9 +
                         (end.tv_sec - start.tv_sec);
  printf("basic time: %.2fs (%.2f X2 conversion) \t pruned time %.2f s\n",
         basic_cpu_time_used + x2_conversion_time, x2_conversion_time,
         pruned_cpu_time_used);
  printf("pruning time is composed of %.2f pruning, %.2f working set "
         "updates, "
         "and %.2f subproblem time\n",
         pruning_time, working_set_update_time, subproblem_time);

  printf("working set upates were: %.2f main effect col, %.2f long col, %.2f "
         "reused col\n",
         main_col_time, int_col_time, reused_col_time);

  // g_assert_true(pruned_cpu_time_used < 0.9 * basic_cpu_time_used);
  double *basic_rowsum = malloc(sizeof *basic_rowsum * n);
  double *pruned_rowsum = malloc(sizeof *pruned_rowsum * n);
  for (long i = 0; i < n; i++) {
    basic_rowsum[i] = -Y[i];
    pruned_rowsum[i] = -Y[i];
  }
  if (Xc.p > 1000) {
    return;
  }
  for (long k = 0; k < p_int; k++) {
    long entry = -1;
    for (long i = 0; i < X2c.cols[k].nwords; i++) {
      S8bWord word = X2c.cols[k].compressed_indices[i];
      unsigned long values = word.values;
      for (long j = 0; j <= group_size[word.selector]; j++) {
        long diff = values & masks[word.selector];
        if (diff != 0) {
          entry += diff;

          // do whatever we need here with the index below:
          basic_rowsum[entry] += beta[k];
          pruned_rowsum[entry] += beta_pruning[k];
          if (basic_rowsum[entry] > 1e20) {
            printf("1. basic_rowsum[%d] = %f\n", entry, basic_rowsum[entry]);
          }
          if (beta[k] > 1e20) {
            printf("beta[%d] = %f\n", k, beta[k]);
          }
        }
        values >>= item_width[word.selector];
      }
    }
  }
  double basic_error = 0.0;
  double pruned_error = 0.0;
  for (long i = 0; i < n; i++) {
    basic_error += basic_rowsum[i] * basic_rowsum[i];
    pruned_error += pruned_rowsum[i] * pruned_rowsum[i];
    if (basic_rowsum[i] > 1e20) {
      printf("2. basic_rowsum[%d] = %f\n", i, basic_rowsum[i]);
    }
  }
  basic_error = sqrt(basic_error);
  pruned_error = sqrt(pruned_error);

  printf("basic error %.2f \t pruned err %.2f\n", basic_error, pruned_error);
  printf("pruning time is composed of %.2f pruning, %.2f working set "
         "updates, "
         "and %.2f subproblem time\n",
         pruning_time, working_set_update_time, subproblem_time);
  g_assert_true(
      fmax(basic_error, pruned_error) / fmin(basic_error, pruned_error) < 1.2);

  printf("working set upates were: %.2f main effect col, %.2f long col, %.2f "
         "reused col\n",
         main_col_time, int_col_time, reused_col_time);

  printf("checking beta values come out the same\n");
  // for (long k = 0; k < p_int; k++) {
  for (long k = 0; k < 10; k++) {
    double basic_beta = fabs(beta[k]);
    double pruned_beta = fabs(beta[k]);
    double max = fmax(basic_beta, pruned_beta);
    double min = fmin(basic_beta, pruned_beta);

    if (max / min > acceptable_diff) {
      printf("basic[%d] \t   %.2f \t =\\= \t %.2f \t pruning[%d]\n", k, beta[k],
             beta_pruning[k], k);
    }
  }
}

long main(long argc, char *argv[]) {
  initialise_static_resources();
  setlocale(LC_ALL, "");
  g_test_init(&argc, &argv, NULL);

  g_test_add_func("/func/test-read-y-csv", test_read_y_csv);
  g_test_add_func("/func/test-read-x-csv", test_read_x_csv);
  g_test_add_func("/func/test-soft-threshol", test_soft_threshold);
  g_test_add("/func/test-update-beta-cyclic", UpdateFixture, NULL,
             update_beta_fixture_set_up, test_update_beta_cyclic,
             update_beta_fixture_tear_down);
  g_test_add_func("/func/test-update-intercept-cyclic",
                  test_update_intercept_cyclic);
  g_test_add_func("/func/test-X2_from_X", test_X2_from_X);
  g_test_add_func("/func/test-compressed-main-X", test_compressed_main_X);
  g_test_add("/func/test-simple-coordinate-descent-int", UpdateFixture, FALSE,
             test_simple_coordinate_descent_set_up,
             test_simple_coordinate_descent_int,
             test_simple_coordinate_descent_tear_down);
  g_test_add("/func/test-simple-coordinate-descent-int-shuffle", UpdateFixture,
             TRUE, test_simple_coordinate_descent_set_up,
             test_simple_coordinate_descent_int,
             test_simple_coordinate_descent_tear_down);
  g_test_add("/func/test-simple-coordinate-descent-vs-glmnet", UpdateFixture,
             TRUE, test_simple_coordinate_descent_set_up,
             test_simple_coordinate_descent_vs_glmnet,
             test_simple_coordinate_descent_tear_down);
  g_test_add_func("/func/test-X2-encoding", check_X2_encoding);
  g_test_add_func("/func/test-permutation", check_permutation);
  g_test_add("/func/test-branch-pruning", UpdateFixture, FALSE,
             pruning_fixture_set_up, check_branch_pruning,
             pruning_fixture_tear_down);
  g_test_add("/func/test-branch-pruning-faster", UpdateFixture, 0,
             pruning_fixture_set_up, check_branch_pruning_faster,
             pruning_fixture_tear_down);
  g_test_add("/func/test-branch-pruning-faster-big", UpdateFixture, 1,
             pruning_fixture_set_up, check_branch_pruning_faster,
             pruning_fixture_tear_down);
  g_test_add("/func/test-branch-pruning-faster-bigger", UpdateFixture, 2,
             pruning_fixture_set_up, check_branch_pruning_faster,
             pruning_fixture_tear_down);
  // g_test_add("/func/test-branch-pruning", UpdateFixture, FALSE,
  // test_simple_coordinate_descent_set_up,
  // test_simple_coordinate_descent_int,
  // test_simple_coordinate_descent_tear_down);
  // g_test_add("/func/test-branch-pruning", UpdateFixture, FALSE,
  // test_simple_coordinate_descent_set_up,
  // test_simple_coordinate_descent_int,
  // test_simple_coordinate_descent_tear_down);

  return g_test_run();
}
