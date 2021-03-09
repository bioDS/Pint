#include <glib-2.0/glib.h>
#include "../src/liblasso.h"
#include <locale.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_permutation.h>
#include <glib-2.0/glib.h>
#include <omp.h>

/* int **X2_from_X(int **X, int n, int p); */
/* double *simple_coordinate_descent_lasso(int **X, double *Y, int n, int p, double lambda, char *method); */
/* double update_beta_greedy_l1(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax); */
/* double update_intercept_cyclic(double intercept, int **X, double *Y, double *beta, int n, int p); */
/* double update_beta_cyclic(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax, double intercept); */
/* double update_beta_glmnet(int **X, double *Y, int n, int p, double lambda, double *beta, int k, double dBMax, double intercept); */
/* double soft_threshold(double z, double gamma); */
/* double *read_y_csv(char *fn, int n); */
/* XMatrix read_x_csv(char *fn, int n, int p); */

#define NumCores 4

typedef struct {
    int n;
    int p;
    XMatrix xmatrix;
    int **X;
    double *Y;
    double *rowsum;
    double lambda;
    double *beta;
    int k;
    double dBMax;
    double intercept;
    XMatrixSparse xmatrix_sparse;
    int_pair *precalc_get_num;
    int **column_caches;
    XMatrixSparse Xc;
    XMatrixSparse X2c;
} UpdateFixture;

const static double small_X2_correct_beta[630] = {-83.112248,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-39.419762,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-431.597831,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-56.125867,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-54.818886,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-144.076649,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-64.023489,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-33.646329,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-62.705188,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-334.676519,0.000000,0.000000,-215.196793,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-165.866118,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-112.678381,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-1.284220,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-58.031513,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,3.916624,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-73.009253,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,6.958046,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-120.529141,0.000000,0.000000,0.000000,0.000000,-80.263024};

static void test_update_beta_greedy_l1() {
    printf("not implemented yet\n");
}

static void test_update_intercept_cyclic() {
    printf("not implemented yet\n");
}

static void update_beta_fixture_set_up(UpdateFixture *fixture, gconstpointer user_data) {
    fixture->n = 1000;
    fixture->p = 35;
    fixture->xmatrix = read_x_csv("../testXSmall.csv", fixture->n, fixture->p);
    fixture->X = fixture->xmatrix.X;
    fixture->Y = read_y_csv("../testYSmall.csv", fixture->n);
    fixture->rowsum = malloc(fixture->n*sizeof(double));
    fixture->lambda = 6.46;
    fixture->beta = malloc(fixture->p*sizeof(double));
    memset(fixture->beta, 0, fixture->p*sizeof(double));
    fixture->k = 27;
    fixture->dBMax = 0;
    fixture->intercept = 0;
    printf("%d\n", fixture->X[1][0]);
    int p_int = (fixture->p*(fixture->p+1))/2;
    int_pair *precalc_get_num = malloc(p_int*sizeof(int_pair));
    int offset = 0;
    for (int i = 0; i < fixture->p; i++) {
        for (int j = i; j < fixture->p; j++) {
            precalc_get_num[offset].i = i;
            precalc_get_num[offset].j = j;
            offset++;
        }
    }
    fixture->precalc_get_num = precalc_get_num;

    int **thread_column_caches = malloc(NumCores*sizeof(int*));
    for (int i = 0; i <  NumCores; i++) {
        thread_column_caches[i] = malloc(fixture->n*sizeof(int));
    }
    fixture->column_caches=thread_column_caches;
}

static void update_beta_fixture_tear_down(UpdateFixture *fixture, gconstpointer user_data) {
    for (int i = 0; i < fixture->p; i++) {
        free(fixture->xmatrix.X[i]);
    }
    free(fixture->Y);
    free(fixture->rowsum);
    free(fixture->beta);
    free(fixture->precalc_get_num);
    for (int i = 0; i < NumCores; i++) {
        free(fixture->column_caches[i]);
    }
    free(fixture->column_caches);
}

static void test_update_beta_cyclic(UpdateFixture *fixture, gconstpointer user_data) {
    printf("beta[27]: %f\n", fixture->beta[27]);
    fixture->xmatrix_sparse = sparse_X2_from_X(fixture->X, fixture->n, fixture->p, 0, -1);
    update_beta_cyclic(fixture->xmatrix_sparse, fixture->Y, fixture->rowsum, fixture->n, fixture->p, fixture->lambda, fixture->beta, fixture->k, fixture->intercept, fixture->precalc_get_num, fixture->column_caches);
    printf("beta[27]: %f\n", fixture->beta[27]);
    g_assert_true(fixture->beta[27] != 0.0);
    g_assert_true(fixture->beta[27] < -263.94);
    g_assert_true(fixture->beta[27] > -263.941);
}


static void test_soft_threshold() {
    printf("not implemented yet\n");
}

static void test_read_x_csv() {
    int n = 1000;
    int p = 100;
    XMatrix xmatrix = read_x_csv("../testX.csv", n, p);
    g_assert_true(xmatrix.actual_cols == 100);
    g_assert_true(xmatrix.X[0][0] == 0);
    g_assert_true(xmatrix.X[99][999] == 0);
    g_assert_true(xmatrix.X[16][575] == 1);

    int sum = 0;
    for (int i = 0; i < p; i++) {
        sum += xmatrix.X[i][321];
    }
    g_assert_true(sum == 8);
}

static void test_compressed_main_X() {
    int n = 1000;
    int p = 100;
    XMatrix xm = read_x_csv("../testX.csv", n, p);

    XMatrixSparse Xs = sparsify_X(xm.X, n, p);

    g_assert_true(Xs.n == n);
    g_assert_true(Xs.p == p);

    int *column_entries[n];

    long agreed_on = 0;
    for (int k = 0; k < p; k++) {
        long col_entry_pos = 0;
        long entry = -1;
        memset(column_entries, 0, sizeof *column_entries *n);
        for (int i = 0; i < Xs.col_nwords[k]; i++) {
            S8bWord word = Xs.compressed_indices[k][i];
            unsigned long values = word.values;
            for (int j = 0; j <= group_size[word.selector]; j++) {
                int diff = values & masks[word.selector];
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
        for (int i = 0; i < n; i++) {
            //printf("\ncolumn %d contains %d entries", k, X2s.col_nz[k]);
            if (col_entry_pos > Xs.col_nz[k] || column_entries[col_entry_pos] < i) {
                if (xm.X[k][i] != 0) {
                    printf("\n[%d][%d] is not in the index but should be", k, i);
                    g_assert_true(FALSE);
                }
            } else if (Xs.col_nz[k] > 0 && column_entries[col_entry_pos] == i) {
                if (xm.X[k][i] != 1) {
                    printf("\n[%d][%d] missing from \n", k, i);
                    g_assert_true(FALSE);
                } else {
                    agreed_on++;
                }
                col_entry_pos++;
            }
        }
      //printf("\nfinished column %d", k);
    }
    printf("agreed on %d\n", agreed_on);
}

static void test_X2_from_X() {
    int n = 1000;
    int p = 100;
    int p_int = p*(p+1)/2;
    XMatrix xm = read_x_csv("../testX.csv", n, p);
    XMatrix xm2 = read_x_csv("../testX2.csv", n, p_int);
    
    XMatrixSparse X2s = sparse_X2_from_X(xm.X, n, p, -1, FALSE);

    g_assert_true(X2s.n == n);
    g_assert_true(X2s.p == p_int);

    // print X2s
    //printf("X2s:\n");
    //for (int k = 0; k < p_int; k++) {
    //  long entry = -1;
    //  printf("%d: ", k);
    //  for (int i = 0; i < X2s.col_nwords[k]; i++) {
    //    S8bWord word = X2s.compressed_indices[k][i];
    //    unsigned long values = word.values;
    //    for (int j = 0; j <= group_size[word.selector]; j++) {
    //      int diff = values & masks[word.selector];
    //      if (diff != 0) {
    //        entry += diff;
    //        printf(" %d", entry);
    //      }
    //      values >>= item_width[word.selector];
    //    }
    //  }
    //  printf("\n");
    //}

    //printf("X2 (printed rows are file columns)\n");
    //for (int j = 0; j < p_int; j++) {
    //  for (int i = 0; i < n; i++) {
    //    printf(" %d", xm2.X[j][i]);
    //  }
    //  printf("\n");
    //}

    int *column_entries[n];

    for (int k = 0; k < p_int; k++) {
        if (k == 2905) {
            printf("xm2.X[%d][0] == %d\n", k, xm2.X[k][0]);
        }
        long col_entry_pos = 0;
        long entry = -1;
        memset(column_entries, 0, sizeof *column_entries *n);
        for (int i = 0; i < X2s.col_nwords[k]; i++) {
            S8bWord word = X2s.compressed_indices[k][i];
            unsigned long values = word.values;
            for (int j = 0; j <= group_size[word.selector]; j++) {
                int diff = values & masks[word.selector];
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
      for (int i = 0; i < n; i++) {
        //printf("\ncolumn %d contains %d entries", k, X2s.col_nz[k]);
        if (col_entry_pos > X2s.col_nz[k] || column_entries[col_entry_pos] < i) {
            if (xm2.X[k][i] != 0) {
                printf("\n[%d][%d] is not in the index but should be", k, i);
                g_assert_true(FALSE);
            }
        } else if (X2s.col_nz[k] > 0 && column_entries[col_entry_pos] == i) {
            if (xm2.X[k][i] != 1) {
                printf("\n[%d][%d] missing from \n", k, i);
                g_assert_true(FALSE);
            }
            col_entry_pos++;
        }
      }
      //printf("\nfinished column %d", k);
    }
}

static void test_simple_coordinate_descent_set_up(UpdateFixture *fixture, gconstpointer user_data) {
    fixture->n = 1000;
    fixture->p = 100;
    printf("reading X\n");
    fixture->xmatrix = read_x_csv("../testX.csv", fixture->n, fixture->p);
    fixture->X = fixture->xmatrix.X;
    printf("reading Y\n");
    fixture->Y = read_y_csv("../testY.csv", fixture->n);
    fixture->rowsum = malloc(fixture->n*sizeof(double));
    fixture->lambda = 20;
    int p_int = fixture->p*(fixture->p+1)/2;
    fixture->beta = malloc(p_int*sizeof(double));
    memset(fixture->beta, 0, p_int*sizeof(double));
    fixture->k = 27;
    fixture->dBMax = 0;
    fixture->intercept = 0;
    int_pair *precalc_get_num = malloc(p_int*sizeof(int_pair));
    int offset = 0;
    for (int i = 0; i < fixture->p; i++) {
        for (int j = i; j < fixture->p; j++) {
            precalc_get_num[offset].i = i;
            precalc_get_num[offset].j = j;
            offset++;
        }
    }
    fixture->precalc_get_num = precalc_get_num;

    cached_nums = get_all_nums(fixture->p, p_int);

    for (int i = 0; i < fixture->n; i++)
        fixture->rowsum[i] = -fixture->Y[i];

    int **thread_column_caches = malloc(NumCores*sizeof(int*));
    for (int i = 0; i <  NumCores; i++) {
        thread_column_caches[i] = malloc(fixture->n*sizeof(int));
    }
    fixture->column_caches=thread_column_caches;
    printf("done test init\n");
}

static void test_simple_coordinate_descent_tear_down(UpdateFixture *fixture, gconstpointer user_data) {
    for (int i = 0; i < fixture->p; i++) {
        free(fixture->xmatrix.X[i]);
    }
    free(fixture->xmatrix.X);
    free(fixture->Y);
    free(fixture->rowsum);
    free(fixture->beta);
    free(fixture->precalc_get_num);
    free(fixture->xmatrix_sparse.compressed_indices);
    free(fixture->xmatrix_sparse.col_nz);
    free(fixture->xmatrix_sparse.col_nwords);
}

static void test_simple_coordinate_descent_int(UpdateFixture *fixture, gconstpointer user_data) {
    // are we running the shuffle test, or sequential?
    double acceptable_diff = 0.1;
    int shuffle = FALSE;
    if (user_data == TRUE) {
        printf("\nrunning shuffle test!\n");
        acceptable_diff = 10;
        shuffle = TRUE;
    }
    double *glmnet_beta = read_y_csv("/home/kieran/work/lasso_testing/glmnet_small_output.csv", 630);
    printf("starting interaction test\n");
    fixture->xmatrix = read_x_csv("/home/kieran/work/lasso_testing/testXSmall.csv", fixture->n, fixture->p);
    fixture->X = fixture->xmatrix.X;
    fixture->xmatrix_sparse = sparse_X2_from_X(fixture->X, fixture->n, fixture->p, -1, shuffle);
    int p_int = fixture->p*(fixture->p+1)/2;
    double *beta = fixture->beta;

    double dBMax;
    for (int j = 0; j < 10; j++)
        for (int i = 0; i < p_int; i++) {
            //int k = gsl_permutation_get(fixture->xmatrix_sparse.permutation, i);
            int k = fixture->xmatrix_sparse.permutation->data[i];
            //int k = i;
            Changes changes = update_beta_cyclic(fixture->xmatrix_sparse, fixture->Y, fixture->rowsum, fixture->n, fixture->p, fixture->lambda, beta, k, 0, fixture->precalc_get_num, fixture->column_caches[0]);
            dBMax = changes.actual_diff;
        }

    int no_agreeing = 0;
    for (int i = 0; i < p_int; i++) {
        int k = gsl_permutation_get(fixture->xmatrix_sparse.permutation, i);
        //int k = i;
        printf("testing beta[%d] (%f) ~ %f [", i, beta[i], small_X2_correct_beta[k]);

        if (    (beta[i] < small_X2_correct_beta[k] + acceptable_diff)
            &&  (beta[i] > small_X2_correct_beta[k] - acceptable_diff)) {
                no_agreeing++;
                printf("x]\n");
            } else {
                printf(" ]\n");
            }
    }
    printf("frac agreement: %f\n", (double)no_agreeing/p_int);
    g_assert_true(no_agreeing == p_int);
}

static void test_simple_coordinate_descent_vs_glmnet(UpdateFixture *fixture, gconstpointer user_data) {
    double *glmnet_beta = read_y_csv("/home/kieran/work/lasso_testing/glmnet_small_output.csv", 630);
    printf("starting interaction test\n");
    fixture->p = 35;
    fixture->xmatrix = read_x_csv("/home/kieran/work/lasso_testing/testXSmall.csv", fixture->n, fixture->p);
    fixture->X = fixture->xmatrix.X;
    fixture->xmatrix_sparse = sparse_X2_from_X(fixture->X, fixture->n, fixture->p, -1, FALSE);
    int p_int = fixture->p*(fixture->p+1)/2;
    double *beta = fixture->beta;

    beta = simple_coordinate_descent_lasso(fixture->xmatrix, fixture->Y, fixture->n, fixture->p, -1, 0.05, 1000, 100, 0, 0.01, 1.0001, FALSE, 1, "test", FALSE, -1);

    double acceptable_diff = 10;
    int no_agreeing = 0;
    for (int i = 0; i < p_int; i++) {
        int k = gsl_permutation_get(fixture->xmatrix_sparse.permutation, i);
        printf("testing beta[%d] (%f) ~ %f [", i, beta[k], glmnet_beta[i]);

        if (    (beta[k] < glmnet_beta[i] + acceptable_diff)
            &&  (beta[k] > glmnet_beta[i] - acceptable_diff)) {
                no_agreeing++;
                printf("x]\n");
            } else {
                printf(" ]\n");
            }
    }
    printf("frac agreement: %f\n", (double)no_agreeing/p_int);
    g_assert_true(no_agreeing >= 0.8*p_int);
}

// will fail if Y has been normalised
static void test_read_y_csv() {
    int n = 1000;
    double *Y = read_y_csv("/home/kieran/work/lasso_testing/testYSmall.csv", n);
    g_assert_true(Y[0] == -133.351709197933);
    g_assert_true(Y[999] == -352.293608898344);
}

//assumes little endian
void printBits(size_t const size, void const * const ptr) {
    unsigned char *b = (unsigned char*) ptr;
    unsigned char byte;
    int i, j;

    for (i=size-1;i>=0;i--)
    {
        for (j=7;j>=0;j--)
        {
            byte = (b[i] >> j) & 1;
            printf("%u", byte);
        }
    }
    puts("");
}

static void check_X2_encoding() {
    int n = 1000;
    int p = 35;
    int p_int = p*(p+1)/2;
    XMatrix xmatrix = read_x_csv("/home/kieran/work/lasso_testing/testXSmall.csv", n, p);
    XMatrix X2 = read_x_csv("../testX2Small.csv", n, p_int);
    XMatrixSparse xmatrix_sparse = sparse_X2_from_X(xmatrix.X, n, p, -1, FALSE);

    int_pair *nums = get_all_nums(p, -1);
    // create uncompressed sparse version of X2.
    int **col_nz_indices = malloc(sizeof *col_nz_indices * p_int);
    for (int j = 0; j < p_int; j++) {
    }
    int *col_sizes = malloc(sizeof *col_sizes *p_int);
    for (int j = 0; j < p_int; j++) {
        Queue *col_q = queue_new();
        for (int i = 0; i < n; i++) {
            if (X2.X[j][i] != 0) {
                queue_push_tail(col_q, i);
            }
        }
        col_sizes[j] = queue_get_length(col_q);
        col_nz_indices[j] = malloc(sizeof *col_nz_indices[j] * col_sizes[j]);
        printf(" real col size: %d, \tcompressed col contains %d entries \n", col_sizes[j], xmatrix_sparse.col_nz[j]);
        g_assert_true(col_sizes[j] == xmatrix_sparse.col_nz[j]);
        int pos = 0;
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
    int no_entries = 0;
    for (int i = 0; i < p_int; i++) {
        no_entries += xmatrix_sparse.col_nz[i];
        for (int j = 0; j < xmatrix_sparse.col_nz[i]; j++) {
            total += col_nz_indices[i][j];
        }
    }
    printf("\nmean entry size: %f\n", (double)total/(double)no_entries);

    // mean diff size
    total = 0;
    int prev_entry = 0;
    for (int i = 0; i < p_int; i++) {
        prev_entry = 0;
        for (int j = 0; j < xmatrix_sparse.col_nz[i]; j++) {
            total += col_nz_indices[i][j] - prev_entry;
            prev_entry = col_nz_indices[i][j];
        }
    }
    printf("mean diff size: %f\n", (double)total/(double)no_entries);

    printf("size of s8bword struct: %d (int is %ld)\n", sizeof(S8bWord), sizeof(int));

    S8bWord test_word;
    test_word.selector = 7;
    test_word.values = 0;
    unsigned int numbers[10] = {3,2,4,20,1,14,30,52,10,63};
    for (int i = 0; i < 10; i++) {
        test_word.values |= numbers[9-i];
        if (i < 9)
            test_word.values <<= item_width[test_word.selector];
    }

    S8bWord w2 =  to_s8b(10, numbers);

    g_assert_true(sizeof(S8bWord) == 8);
    g_assert_true(test_word.selector == w2.selector);
    g_assert_true(test_word.values == w2.values);

    int max_size_given_entries[61];
    for (int i = 0; i < 60; i++) {
        max_size_given_entries[i] = 60/(i+1);
    }
    max_size_given_entries[60] = 0;

    printf("num entries in col 0: %d\n", xmatrix_sparse.col_nz[0]);
    int *col_entries = malloc(60*sizeof(int));
    int count = 0;
    //GList *s8b_col = NULL;
    GQueue *s8b_col = g_queue_new();
    // work out s8b compressed equivalent of col 0
    int largest_entry = 0;
    int max_bits = max_size_given_entries[0];
    int diff = col_nz_indices[0][0] + 1;
    for (int i = 0; i < xmatrix_sparse.col_nz[0]; i++) {
        if (i != 0)
            diff = col_nz_indices[0][i] - col_nz_indices[0][i-1];
        // printf("current no. %d, diff %d. available bits %d\n", col_nz_indices[0][i], diff, max_bits);
        // update max bits.
        int used = 0;
        int tdiff = diff;
        while (tdiff > 0) {
            used++;
            tdiff >>= 1;
        }
        max_bits = max_size_given_entries[count+1];
        // if the current diff won't fit in the s8b word, push the word and start a new one
        if (diff > 1<<max_bits || largest_entry > max_size_given_entries[count+1]) {
            //if (diff > 1<<max_bits)
            //  printf(" b ");
            //if (largest_entry > max_size_given_entries[count+1])
            //  printf(" c ");
            // printf("pushing word with %d entries: ", count);
            //for (int j = 0; j < count; j++)
            //  printf("%d ", col_entries[j]);
            //printf("\n");
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
    //push the last (non-full) word
    S8bWord *word = malloc(sizeof(S8bWord));
    S8bWord tempword = to_s8b(count, col_entries);
    memcpy(word, &tempword, sizeof(S8bWord));
    g_queue_push_tail(s8b_col, word);

    free(col_entries);
    int length = g_queue_get_length(s8b_col);

    S8bWord *actual_col = malloc(length*sizeof(S8bWord));
    count = 0;
    while (!g_queue_is_empty(s8b_col)) {
        S8bWord *current_word = g_queue_pop_head(s8b_col);
        memcpy(&actual_col[count], current_word, sizeof(S8bWord));
        count++;
    }

    printf("checking [s8b] == [int]\n");
    for (int k = 0; k < p_int; k++) {
        printf("col %d (interaction %d,%d)\n", k, nums[k].i, nums[k].j);
        int checked = 0;
        int col_entry_pos = 0;
        int entry = -1;
        for (int i = 0; i < xmatrix_sparse.col_nwords[k]; i++) {
            S8bWord word = xmatrix_sparse.compressed_indices[k][i];
            unsigned long values = word.values;
            for (int j = 0; j <= group_size[word.selector]; j++) {
                int diff = values & masks[word.selector];
                if (diff != 0) {
                    entry += diff;
                    printf("pos %d, %d == %d\n", col_entry_pos, entry, col_nz_indices[k][col_entry_pos]);
                    g_assert_true(entry == col_nz_indices[k][col_entry_pos]);
                    col_entry_pos++;
                    checked++;
                }
                values >>= item_width[word.selector];
            }
        }
        printf("col %d, checked %d out of %d present\n", k, checked, col_sizes[k]);
        g_assert_true(checked == xmatrix_sparse.col_nz[k]);
    }

    int bytes = length*sizeof(S8bWord);
    printf("col[0] contains %d words, for a toal of %d bytes, instead of %d shorts (%d bytes). Effective reduction %f\n",
        length, bytes, xmatrix_sparse.col_nz[0], xmatrix_sparse.col_nz[0]*sizeof(short), (double)bytes/(xmatrix_sparse.col_nz[0]*sizeof(short)));

    printf("liblasso vs test compressed first col:\n");
    for (int i = 0; i < xmatrix_sparse.col_nwords[0]; i++) {
        printf("%d == %d\n", xmatrix_sparse.compressed_indices[0][i].selector, actual_col[i].selector);
        g_assert_true(xmatrix_sparse.compressed_indices[0][i].selector == actual_col[i].selector);
        printf("%d == %d\n", xmatrix_sparse.compressed_indices[0][i].values, actual_col[i].values);
        g_assert_true(xmatrix_sparse.compressed_indices[0][i].values == actual_col[i].values);
    }
    g_assert_true(xmatrix_sparse.col_nwords[0] == length);
    printf("correct number of words\n");

    for (int j = 0; j < p_int; j++) {
        free(col_nz_indices[j]);
    }
    free(col_nz_indices);
    free(nums);
    free(col_sizes);
}

static void check_permutation() {
    int threads = omp_get_num_procs();
    gsl_rng **thread_r = malloc(threads*sizeof(gsl_rng*));
    for (int i = 0; i < threads; i++)
        thread_r[i] = gsl_rng_alloc(gsl_rng_taus2);

    long perm_size = 3235; //<< 12 + 67;
    printf("perm_size %ld\n", perm_size);
    gsl_permutation *perm = gsl_permutation_alloc(perm_size);
    gsl_permutation_init(perm);

    parallel_shuffle(perm, perm_size/threads, perm_size%threads, threads);

    int *found = malloc(perm_size*sizeof(int));
    memset(found, 0, perm_size*sizeof(int));
    for (int i = 0; i < perm_size; i++) {
        size_t val = perm->data[i];
        found[val] = 1;
        printf("found %d\n", val);
    }
    for (int i = 0; i < perm_size; i++) {
        printf("checking %d is present\n", i);
        printf("found[%d] = %d\n", i, found[i]);
        printf("found[%d+1] = %d\n", i, found[i+1]);
        g_assert_true(found[i] == 1);
    }
    free(found);
    gsl_permutation_free(perm);

    perm_size = 123123; //<< 12 + 67;
    printf("perm_size %ld\n", perm_size);
    perm = gsl_permutation_alloc(perm_size);
    gsl_permutation_init(perm);

    parallel_shuffle(perm, perm_size/threads, perm_size%threads, threads);

    found = malloc(perm_size*sizeof(int));
    memset(found, 0, perm_size);
    for (long i = 0; i < perm_size; i++) {
        long val = perm->data[i];
        found[val] = 1;
        printf("found %d\n", val);
    }
    for (long i = 0; i < perm_size; i++) {
        printf("checking %d is present\n", i);
        printf("found[%d] = %d\n", i, found[i]);
        printf("found[%d+1] = %d\n", i, found[i+1]);
        g_assert_true(found[i] == 1);
    }
    free(found);
    gsl_permutation_free(perm);
}

int check_didnt_update(int p, int p_int, int *wont_update, double *beta) {
    int no_disagreeing = 0;
    for (int i = 0; i < p_int; i++) {
        // int k = gsl_permutation_get(fixture->xmatrix_sparse.permutation, i);
        int k = i;
        // int k = fixture->xmatrix_sparse.permutation->data[i];
        // printf("testing beta[%d] (%f)\n", k, beta[k]);
        int_pair ip = get_num(k, p_int);
        //TODO: we should only check against later items not in the working set, this needs to be udpated
        if (wont_update[ip.i] || wont_update[ip.j]) {
            // printf("checking interaction %d,%d is zero\n", ip.i, ip.j);
            if (beta[k] != 0.0) {
                printf("beta %d (interaction %d,%d) should be zero according to will_update_effect(), but is in fact %f\n", k, ip.i, ip.j, beta[k]);
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

static void pruning_fixture_set_up(UpdateFixture *fixture, gconstpointer user_data) {
    test_simple_coordinate_descent_set_up(fixture, NULL);
    XMatrixSparse Xc = sparsify_X(fixture->X, fixture->n, fixture->p);
    XMatrixSparse X2c = sparse_X2_from_X(fixture->X, fixture->n, fixture->p, -1, FALSE);
    fixture->Xc = Xc;
    fixture-> X2c = X2c;
}

static void pruning_fixture_tear_down(UpdateFixture *fixture, gconstpointer user_data) {
    free_sparse_matrix(fixture->Xc);
    free_sparse_matrix(fixture->X2c);

    test_simple_coordinate_descent_tear_down(fixture, NULL);
}

int get_wont_update(int *wont_update, int p, XMatrixSparse Xc, double lambda, double *last_max, double **last_rowsum, double *rowsum, int *column_cache, int n, double *beta) {
    int ruled_out = 0;
    for (int j = 0; j < p; j++) {
        double sum = 0.0;
        wont_update[j] = wont_update_effect(Xc, lambda, j, last_max[j], last_rowsum[j], rowsum, column_cache, beta);
        if (wont_update[j])
            ruled_out++;
    }

    //for (int i = 0; i < p; i++) {
    //  if (wont_update[i]) {
    //      printf("%d supposedly wont update\n", i);
    //  }
    //}
    // printf("ruled out %d branch(es)\n", ruled_out);
    return ruled_out;
}
// run branch_prune check, then full regression step without pruning.
// the beta values that would have been pruned should be 0.
static void check_branch_pruning(UpdateFixture *fixture, gconstpointer user_data) {
    printf("starting branch pruning test\n");
    int n = fixture->n;
    int p = fixture->p;
    double *rowsum = fixture->rowsum;
    // double lambda = fixture->lambda;
    int shuffle = FALSE;
    printf("starting interaction test\n");
    //fixture->xmatrix = read_x_csv("/ho/testXSmall.csv", fixture->n, fixture->p);
    printf("creating X2\n");
    int p_int = fixture->p*(fixture->p+1)/2;
    double *beta = fixture->beta;
    printf("test\n");

    XMatrixSparse Xc = fixture->Xc;
    XMatrixSparse X2c = fixture->X2c;
    int column_cache[n];

    int wont_update[p];
    for (int j = 0; j < p; j++)
        wont_update[j] = 0;

    double **last_rowsum = malloc(sizeof *last_rowsum * p);
    for (int i = 0; i < p; i++) {
        last_rowsum[i] = malloc(sizeof *last_rowsum[i] * n);
        memset(last_rowsum[i], 0, sizeof *last_rowsum[i] * n);
    }

    for (int j = 0; j < p; j++)
        for (int i = 0; i < n; i++)
            g_assert_true(last_rowsum[j][i] == 0.0);

    double last_max[n];
    memset(last_max, 0, sizeof(last_max));

    // start running tests with decreasing lambda
    double lambda_sequence[] = {10000,500, 400, 300, 200, 100, 50, 25, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.01};
    int seq_length = sizeof(lambda_sequence)/sizeof(*lambda_sequence);
    double lambda = lambda_sequence[0];
    int ruled_out = 0;
    double old_rowsum[n];

    double error = 0.0;
    for (int i = 0; i < n; i++) {
        error += rowsum[i]*rowsum[i];
    }
    error = sqrt(error);
    printf("initial error: %f\n", error);

    double max_int_delta[p];
    memset(max_int_delta, 0, sizeof *max_int_delta * p);
    for (int lambda_ind = 0; lambda_ind < seq_length; lambda_ind ++) {
        memcpy(old_rowsum, rowsum, sizeof *rowsum *n);
        lambda = lambda_sequence[lambda_ind];
        printf("\nrunning lambda %f, current error: %f\n", lambda, error);
        double dBMax;
        //TODO: implement working set and update test
        int last_iter_count = 0;
        for (int iter = 0; iter < 50; iter++) {
            printf("iter %d\n", iter);
            last_iter_count = iter;
            double prev_error = error;

            int k = 0;
            double rowsum_delta[n];
            memset(rowsum_delta, 0, sizeof *rowsum_delta * n);
            ruled_out += get_wont_update(wont_update, p, Xc, lambda, last_max, last_rowsum, rowsum, column_cache, n, beta);
            for (int main_effect = 0; main_effect < p; main_effect++) {
                for (int interaction = main_effect; interaction < p; interaction++) {
                    double old = beta[k];
                    Changes changes = update_beta_cyclic(X2c, fixture->Y, rowsum_delta, n, p, lambda, beta, k, 0, fixture->precalc_get_num, column_cache);
                    dBMax = changes.actual_diff;
                    double new = beta[k];
                    if (fabs(changes.pre_lambda_diff) > fabs(max_int_delta[main_effect])) {
                        max_int_delta[main_effect] = changes.pre_lambda_diff;
                    }
                    k++;
                }
            }
            for (int i = 0; i < n; i++) {
                rowsum[i] += rowsum_delta[i];
            }
            int no_disagreeing = check_didnt_update(p, p_int, wont_update, beta);
            g_assert_true(no_disagreeing == 0);

            for (int i = 0; i < p; i++) {
                if (!wont_update[i]) {
                    if (last_max[i] != max_int_delta[i]) {
                        printf("main effect %d new last_max is %f\n", i, max_int_delta[i]);
                    }
                    last_max[i] = max_int_delta[i];
                }
            }
            error = 0.0;
            for (int i = 0; i < n; i++) {
                error += rowsum[i]*rowsum[i];
            }
            error = sqrt(error);
            if (prev_error/error < 1.01) {
                break;
            }
        }
        printf("done lambda %f in %d iters\n", lambda, last_iter_count+1);
        for (int i = 0; i < p; i++) {
            // we did check these anyway, but since we ordinarily wouldn't they don't get updated.
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
    double *rowsum;
    int *column_cache;
    int n;
    double *beta;
    double lambda;
    double *last_max;
    int *wont_update;
    int p;
    int p_int;
    XMatrixSparse X2c;
    double *Y;
    double *max_int_delta;
    int_pair *precalc_get_num;
} Iter_Vars;

double run_lambda_iters(Iter_Vars *vars) {
    XMatrixSparse Xc = vars->Xc;
    double **last_rowsum = vars->last_rowsum;
    double *rowsum = vars->rowsum;
    int *column_cache = vars->column_cache;
    int n = vars->n;
    double *beta = vars->beta;
    double lambda = vars->lambda;
    double *last_max = vars->last_max;
    int *wont_update = vars->wont_update;
    int p = vars->p;
    int p_int = vars->p_int;
    XMatrixSparse X2c = vars->X2c;
    double *Y = vars->Y;
    double *max_int_delta = vars->max_int_delta;
    int_pair *precalc_get_num = vars->precalc_get_num;

    double error = 0.0;
    for (int i = 0; i < n; i++) {
        error += rowsum[i]*rowsum[i];
    }
    for (int iter = 0; iter < 50; iter++) {
        // printf("iter %d\n", iter);
        // last_iter_count = iter;
        double prev_error = error;


        int k = 0;
        double rowsum_delta[n];
        memset(rowsum_delta, 0, sizeof *rowsum_delta * n);
        for (int main_effect = 0; main_effect < p; main_effect++) {
            for (int interaction = main_effect; interaction < vars->p; interaction++) {
                Changes changes = update_beta_cyclic(X2c, Y, rowsum_delta, n, p, lambda, beta, k, 0, precalc_get_num, column_cache);
                // dBMax = changes.actual_diff;
                k++;
            }
        }
        for (int i = 0; i < n; i++) {
            rowsum[i] += rowsum_delta[i];
        }

        error = 0.0;
        for (int i = 0; i < n; i++) {
            error += rowsum[i]*rowsum[i];
        }
        error = sqrt(error);
        if (prev_error/error < 1.01) {
            break;
        }
    }
    // return dBMax;
    return 0.0;
}

void run_lambda_iters_pruned(Iter_Vars *vars) {
    XMatrixSparse Xc = vars->Xc;
    double **last_rowsum = vars->last_rowsum;
    double *rowsum = vars->rowsum;
    int *column_cache = vars->column_cache;
    int n = vars->n;
    double *beta = vars->beta;
    double lambda = vars->lambda;
    double *last_max = vars->last_max;
    int *wont_update = vars->wont_update;
    int p = vars->p;
    int p_int = vars->p_int;
    XMatrixSparse X2c = vars->X2c;
    double *Y = vars->Y;
    double *max_int_delta = vars->max_int_delta;
    int_pair *precalc_get_num = vars->precalc_get_num;

    double error = 0.0;
    for (int i = 0; i < n; i++) {
        error += rowsum[i]*rowsum[i];
    }
    double prev_error = error;
    for (int iter = 0; iter < 50; iter++) {
        int k = 0;
        double rowsum_delta[n];
        memset(rowsum_delta, 0, sizeof *rowsum_delta * n);
        get_wont_update(wont_update, p, Xc, lambda, last_max, last_rowsum, rowsum, column_cache, n, beta);
        for (int main_effect = 0; main_effect < vars->p; main_effect++) {
            for (int interaction = main_effect; interaction < vars->p; interaction++) {
                double old = beta[k];
                Changes changes = update_beta_cyclic(X2c, Y, rowsum_delta, n, p, lambda, beta, k, 0, precalc_get_num, column_cache);
                // dBMax = changes.actual_diff;
                double new = beta[k];
                if (fabs(changes.pre_lambda_diff) > fabs(max_int_delta[main_effect])) {
                    max_int_delta[main_effect] = changes.pre_lambda_diff;
                }
                k++;
            }
        }
        for (int i = 0; i < n; i++) {
            rowsum[i] += rowsum_delta[i];
        }
        //int no_disagreeing = check_didnt_update(p, p_int, wont_update, beta);
        int no_disagreeing = 0;
        g_assert_true(no_disagreeing == 0);

        for (int i = 0; i < p; i++) {
            if (!wont_update[i]) {
                //if (last_max[i] != max_int_delta[i]) {
                //  printf("main effect %d new last_max is %f\n", i, max_int_delta[i]);
                //}
                last_max[i] = max_int_delta[i];
            }
        }
        error = 0.0;
        for (int i = 0; i < n; i++) {
            error += rowsum[i]*rowsum[i];
        }
        error = sqrt(error);
        if (prev_error/error < 1.01) {
            break;
        }
    }
}

static void check_branch_pruning_faster(UpdateFixture *fixture, gconstpointer user_data) {
    printf("starting branch pruning speed test\n");
    int n = fixture->n;
    int p = fixture->p;
    double *rowsum = fixture->rowsum;
    int shuffle = FALSE;
    printf("starting interaction test\n");
    printf("creating X2\n");
    int p_int = fixture->p*(fixture->p+1)/2;
    double *beta = fixture->beta;
    printf("test\n");

    XMatrixSparse Xc = fixture->Xc;
    XMatrixSparse X2c = fixture->X2c;
    int column_cache[n];

    int wont_update[p];
    for (int j = 0; j < p; j++)
        wont_update[j] = 0;

    double **last_rowsum = malloc(sizeof *last_rowsum * p);
    for (int i = 0; i < p; i++) {
        last_rowsum[i] = malloc(sizeof *last_rowsum[i] * n);
        memset(last_rowsum[i], 0, sizeof *last_rowsum[i] * n);
    }

    for (int j = 0; j < p; j++)
        for (int i = 0; i < n; i++)
            g_assert_true(last_rowsum[j][i] == 0.0);

    double last_max[n];
    memset(last_max, 0, sizeof(last_max));

    // start running tests with decreasing lambda
    double lambda_sequence[] = {10000,500, 400, 300, 200, 100, 50, 25, 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.01};
    int seq_length = sizeof(lambda_sequence)/sizeof(*lambda_sequence);
    double lambda = lambda_sequence[0];
    int ruled_out = 0;
    double old_rowsum[n];

    double error = 0.0;
    for (int i = 0; i < n; i++) {
        error += rowsum[i]*rowsum[i];
    }
    error = sqrt(error);
    // printf("initial error: %f\n", error);

    double max_int_delta[p];
    memset(max_int_delta, 0, sizeof *max_int_delta * p);

    Iter_Vars iter_vars_basic = {
        Xc,
        last_rowsum,
        rowsum,
        column_cache,
        n,
        beta,
        lambda,
        last_max,
        NULL,
        p,
        p_int,
        X2c,
        fixture->Y,
        max_int_delta,
        fixture->precalc_get_num,
    };
    struct timespec start, end;
    double basic_cpu_time_used, pruned_cpu_time_used;
    printf("getting time for un-pruned version\n");
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    for (int lambda_ind = 0; lambda_ind < seq_length; lambda_ind ++) {
        memcpy(old_rowsum, rowsum, sizeof *rowsum *n);
        lambda = lambda_sequence[lambda_ind];
        double dBMax;
        //TODO: implement working set and update test
        int last_iter_count = 0;

        run_lambda_iters(&iter_vars_basic);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    basic_cpu_time_used = ((double)(end.tv_nsec-start.tv_nsec))/1e9 + (end.tv_sec - start.tv_sec);
    printf("time: %f s\n", basic_cpu_time_used);

    double beta_pruning[p_int];
    for (int i = 0; i < p; i++) {
        last_rowsum[i] = malloc(sizeof *last_rowsum[i] * n);
        memset(last_rowsum[i], 0, sizeof *last_rowsum[i] * n);
    }
    for (int i = 0; i < n; i++) {
        rowsum[i] = 0;
    }
    Iter_Vars iter_vars_pruned = {
        Xc,
        last_rowsum,
        rowsum,
        column_cache,
        n,
        beta_pruning,
        lambda,
        last_max,
        wont_update,
        p,
        p_int,
        X2c,
        fixture->Y,
        max_int_delta,
        fixture->precalc_get_num,
    };

    printf("getting time for pruned version\n");
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    for (int lambda_ind = 0; lambda_ind < seq_length; lambda_ind ++) {
        memcpy(old_rowsum, rowsum, sizeof *rowsum *n);
        lambda = lambda_sequence[lambda_ind];
        double dBMax;
        //TODO: implement working set and update test
        int last_iter_count = 0;

        run_lambda_iters_pruned(&iter_vars_pruned);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    pruned_cpu_time_used = ((double)(end.tv_nsec-start.tv_nsec))/1e9 + (end.tv_sec - start.tv_sec);
    printf("time: %f s\n", pruned_cpu_time_used);

    g_assert_true(pruned_cpu_time_used < 0.5 * basic_cpu_time_used);
}

int main (int argc, char *argv[]) {
    initialise_static_resources();
    setlocale (LC_ALL, "");
    g_test_init (&argc, &argv, NULL);

    g_test_add_func("/func/test-read-y-csv", test_read_y_csv);
    g_test_add_func("/func/test-read-x-csv", test_read_x_csv);
    g_test_add_func("/func/test-soft-threshol", test_soft_threshold);
    g_test_add("/func/test-update-beta-cyclic", UpdateFixture, NULL, update_beta_fixture_set_up, test_update_beta_cyclic, update_beta_fixture_tear_down);
    g_test_add_func("/func/test-update-intercept-cyclic", test_update_intercept_cyclic);
    g_test_add_func("/func/test-X2_from_X", test_X2_from_X);
    g_test_add_func("/func/test-compressed-main-X", test_compressed_main_X);
    g_test_add("/func/test-simple-coordinate-descent-int", UpdateFixture, FALSE, test_simple_coordinate_descent_set_up, test_simple_coordinate_descent_int, test_simple_coordinate_descent_tear_down);
    g_test_add("/func/test-simple-coordinate-descent-int-shuffle", UpdateFixture, TRUE, test_simple_coordinate_descent_set_up, test_simple_coordinate_descent_int, test_simple_coordinate_descent_tear_down);
    g_test_add("/func/test-simple-coordinate-descent-vs-glmnet", UpdateFixture, TRUE, test_simple_coordinate_descent_set_up, test_simple_coordinate_descent_vs_glmnet, test_simple_coordinate_descent_tear_down);
    g_test_add_func("/func/test-X2-encoding", check_X2_encoding);
    g_test_add_func("/func/test-permutation", check_permutation);
    g_test_add("/func/test-branch-pruning", UpdateFixture, FALSE, pruning_fixture_set_up, check_branch_pruning, pruning_fixture_tear_down);
    g_test_add("/func/test-branch-pruning-faster", UpdateFixture, FALSE, pruning_fixture_set_up, check_branch_pruning_faster, pruning_fixture_tear_down);
    //g_test_add("/func/test-branch-pruning", UpdateFixture, FALSE, test_simple_coordinate_descent_set_up, test_simple_coordinate_descent_int, test_simple_coordinate_descent_tear_down);
    // g_test_add("/func/test-branch-pruning", UpdateFixture, FALSE, test_simple_coordinate_descent_set_up, test_simple_coordinate_descent_int, test_simple_coordinate_descent_tear_down);

    return g_test_run();
}
