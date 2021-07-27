#include "../src/liblasso.h"
#include <glib-2.0/glib.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_rng.h>
#include <locale.h>
#include <omp.h>
#include <stdlib.h>
#include <tuple>
#define _POSIX_C_SOURCE 199309L
#include <iostream>
#include <time.h>
#include <vector>

#include <algorithm>

using namespace std;
//using namespace boost;

#define HALT_ERROR_DIFF 1.01

struct timespec start_time, end_time;
static float x2_conversion_time = 0.0;
extern long run_lambda_iters_pruned(Iter_Vars* vars, float lambda, float* rowsum,
    float* old_rowsum, Active_Set* active_set, struct OpenCL_Setup* ocl_setup, long depth);
static long total_basic_beta_updates = 0;
static long total_basic_beta_nz_updates = 0;
static float LAMBDA_MIN = 15.0;

// #pragma omp declare target
// float fabs(float a) {
//  if (-a > a)
//    return -a;
//  return a;
//}
// #pragma omp end declare target

typedef struct {
    long n;
    long p;
    XMatrix xmatrix;
    long** X;
    float* Y;
    float* rowsum;
    float lambda;
    robin_hood::unordered_flat_map<long, float> beta;
    long k;
    float dBMax;
    float intercept;
    XMatrixSparse xmatrix_sparse;
    int_pair* precalc_get_num;
    long** column_caches;
    XMatrixSparse Xc;
    XMatrixSparse X2c;
} UpdateFixture;

const static float small_X2_correct_beta[630] = {
    -83.112248, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, -39.419762, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -431.597831,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, -56.125867, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    -54.818886, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, -144.076649, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, -64.023489, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, -33.646329, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -62.705188,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    -334.676519, 0.000000, 0.000000, -215.196793, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    -165.866118, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, -112.678381, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, -1.284220, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    -58.031513, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 3.916624, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, -73.009253,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 6.958046, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    -120.529141, 0.000000, 0.000000, 0.000000, 0.000000, -80.263024
};

static void test_update_beta_greedy_l1() { printf("not implemented yet\n"); }

static void test_update_intercept_cyclic() { printf("not implemented yet\n"); }

static void update_beta_fixture_set_up(UpdateFixture* fixture,
    gconstpointer user_data)
{
    fixture->n = 1000;
    fixture->p = 35;
    fixture->xmatrix = read_x_csv("../testXSmall.csv", fixture->n, fixture->p);
    fixture->X = fixture->xmatrix.X;
    fixture->Y = read_y_csv("../testYSmall.csv", fixture->n);
    fixture->rowsum = (float*)malloc(fixture->n * sizeof(float));
    fixture->lambda = 6.46;
    // fixture->beta = (float*)malloc(fixture->p * sizeof(float));
    // memset(fixture->beta, 0, fixture->p * sizeof(float));
    fixture->k = 27;
    fixture->dBMax = 0;
    fixture->intercept = 0;
    printf("%ld\n", fixture->X[1][0]);
    long p_int = (fixture->p * (fixture->p + 1)) / 2;
    int_pair* precalc_get_num = (int_pair*)malloc(p_int * sizeof(int_pair));
    long offset = 0;
    for (long i = 0; i < fixture->p; i++) {
        for (long j = i; j < fixture->p; j++) {
            precalc_get_num[offset].i = i;
            precalc_get_num[offset].j = j;
            offset++;
        }
    }
    fixture->precalc_get_num = precalc_get_num;

    long** thread_column_caches = (long**)malloc(NumCores * sizeof(long*));
    for (long i = 0; i < NumCores; i++) {
        thread_column_caches[i] = (long*)malloc(fixture->n * sizeof(int));
    }
    fixture->column_caches = thread_column_caches;
}

static void update_beta_fixture_tear_down(UpdateFixture* fixture,
    gconstpointer user_data)
{
    for (long i = 0; i < fixture->p; i++) {
        free(fixture->xmatrix.X[i]);
    }
    free(fixture->Y);
    free(fixture->rowsum);
    // free(fixture->beta);
    fixture->beta.clear();
    free(fixture->precalc_get_num);
    for (long i = 0; i < NumCores; i++) {
        free(fixture->column_caches[i]);
    }
    free(fixture->column_caches);
}

static void test_update_beta_cyclic(UpdateFixture* fixture,
    gconstpointer user_data)
{
    printf("beta[27]: %f\n", fixture->beta[27]);
    fixture->xmatrix_sparse = sparse_X2_from_X(fixture->X, fixture->n, fixture->p, 0, -1);
    update_beta_cyclic_old(fixture->xmatrix_sparse, fixture->Y, fixture->rowsum,
        fixture->n, fixture->p, fixture->lambda, &fixture->beta,
        fixture->k, fixture->intercept,
        fixture->precalc_get_num, fixture->column_caches[0]);
    printf("beta[27]: %f\n", fixture->beta[27]);
    g_assert_true(fixture->beta[27] != 0.0);
    g_assert_true(fixture->beta[27] < -263.94);
    g_assert_true(fixture->beta[27] > -263.941);
}

static void test_soft_threshold() { printf("not implemented yet\n"); }

static void test_read_x_csv()
{
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

static void test_compressed_main_X()
{
    long n = 1000;
    long p = 100;
    XMatrix xm = read_x_csv("../testX.csv", n, p);

    XMatrixSparse Xs = sparsify_X(xm.X, n, p);

    g_assert_true(Xs.n == n);
    g_assert_true(Xs.p == p);

    long column_entries[n];

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
                    column_entries[col_entry_pos] = (int)entry;
                    col_entry_pos++;
                }
                values >>= item_width[word.selector];
            }
        }
        // check the read column agrees with k of testX2.csv
        // n.b. XMatrix.X is column-major
        col_entry_pos = 0;
        for (long i = 0; i < n; i++) {
            // printf("\ncolumn %ld contains %ld entries", k, X2s.nz[k]);
            if (col_entry_pos > Xs.cols[k].nz || column_entries[col_entry_pos] < i) {
                if (xm.X[k][i] != 0) {
                    printf("\n[%ld][%ld] is not in the index but should be", k, i);
                    g_assert_true(FALSE);
                }
            } else if (Xs.cols[k].nz > 0 && column_entries[col_entry_pos] == i) {
                if (xm.X[k][i] != 1) {
                    printf("\n[%ld][%ld] missing from \n", k, i);
                    g_assert_true(FALSE);
                } else {
                    agreed_on++;
                }
                col_entry_pos++;
            }
        }
        // printf("\nfinished column %ld", k);
    }
    printf("agreed on %ld\n", agreed_on);
}

static void test_X2_from_X()
{
    long n = 1000;
    long p = 100;
    long p_int = p * (p + 1) / 2;
    XMatrix xm = read_x_csv("../testX.csv", n, p);
    XMatrix xm2 = read_x_csv("../testX2.csv", n, p_int);

    XMatrixSparse X2s = sparse_X2_from_X(xm.X, n, p, -1, FALSE);

    g_assert_true(X2s.n == n);
    g_assert_true(X2s.p == p_int);

    // print X2s
    // printf("X2s:\n");
    // for (long k = 0; k < p_int; k++) {
    //  long entry = -1;
    //  printf("%ld: ", k);
    //  for (long i = 0; i < X2s.cols[k].nwords; i++) {
    //    S8bWord word = X2s.cols[k].compressed_indices[i];
    //    unsigned long values = word.values;
    //    for (long j = 0; j <= group_size[word.selector]; j++) {
    //      long diff = values & masks[word.selector];
    //      if (diff != 0) {
    //        entry += diff;
    //        printf(" %ld", entry);
    //      }
    //      values >>= item_width[word.selector];
    //    }
    //  }
    //  printf("\n");
    //}

    // printf("X2 (printed rows are file columns)\n");
    // for (long j = 0; j < p_int; j++) {
    //  for (long i = 0; i < n; i++) {
    //    printf(" %ld", xm2.X[j][i]);
    //  }
    //  printf("\n");
    //}

    long column_entries[n];

    for (long k = 0; k < p_int; k++) {
        if (k == 2905) {
            printf("xm2.X[%ld][0] == %ld\n", k, xm2.X[k][0]);
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
            // printf("\ncolumn %ld contains %ld entries", k, X2s.nz[k]);
            if (col_entry_pos > X2s.cols[k].nz || column_entries[col_entry_pos] < i) {
                if (xm2.X[k][i] != 0) {
                    printf("\n[%ld][%ld] is not in the index but should be", k, i);
                    g_assert_true(FALSE);
                }
            } else if (X2s.cols[k].nz > 0 && column_entries[col_entry_pos] == i) {
                if (xm2.X[k][i] != 1) {
                    printf("\n[%ld][%ld] missing from \n", k, i);
                    g_assert_true(FALSE);
                }
                col_entry_pos++;
            }
        }
        // printf("\nfinished column %ld", k);
    }
}

static void test_simple_coordinate_descent_set_up(UpdateFixture* fixture,
    gconstpointer use_big)
{
    halt_error_diff = HALT_ERROR_DIFF;
    char *xfile, *yfile;
    if ((long)use_big == 2) {
        printf("\nusing huge test case\n");
        fixture->n = 10000;
        fixture->p = 5000;
        xfile = "../n10000_p5000_SNR5_nbi50_nbij1000_nlethals250_viol0_40452/X.csv";
        yfile = "../n10000_p5000_SNR5_nbi50_nbij1000_nlethals250_viol0_40452/Y.csv";
        LAMBDA_MIN = 0.01 * fixture->n / 2;
        //fixture->n = 8000;
        //fixture->p = 4000;
        //xfile = "../testcase//n8000_p4000_SNR5_nbi40_nbij800_nlethals200_viol0_91159/X.csv";
        //yfile = "../testcase//n8000_p4000_SNR5_nbi40_nbij800_nlethals200_viol0_91159/Y.csv";
    } else if ((long)use_big == 1) {
        printf("\nusing large test case\n");
        fixture->n = 2000;
        fixture->p = 1000;
        // xfile = "../X_nlethals50_v15803.csv";
        // yfile = "../Y_nlethals50_v15803.csv";
        xfile = "../testcase/n2000_p1000_SNR5_nbi10_nbij200_nlethals50_viol0_11057/X.csv";
        yfile = "../testcase/n2000_p1000_SNR5_nbi10_nbij200_nlethals50_viol0_11057/Y.csv";
        LAMBDA_MIN = 0.01 * fixture->n / 2;
        //xfile = "../broken_case/n10000_p1000_SNR2_nbi0_nbij200_nlethals0_viol0_89257/X.csv";
        //yfile = "../broken_case/n10000_p1000_SNR2_nbi0_nbij200_nlethals0_viol0_89257/Y.csv";
    } else if (use_big == 0) {
        printf("\nusing small test case\n");
        fixture->n = 1000;
        fixture->p = 100;
        LAMBDA_MIN = 0.000001 * fixture->n / 2;
        // xfile = "../testX.csv";
        // yfile = "../testY.csv";
        xfile = "../testcase/n1000_p100_SNR5_nbi100_nbij50_nlethals0_viol0_3231/X.csv";
        yfile = "../testcase/n1000_p100_SNR5_nbi100_nbij50_nlethals0_viol0_3231/Y.csv";
    }
    printf("reading X from %s\n", xfile);
    fixture->xmatrix = read_x_csv(xfile, fixture->n, fixture->p);
    fixture->X = fixture->xmatrix.X;
    printf("reading Y from %s\n", yfile);
    fixture->Y = read_y_csv(yfile, fixture->n);
    fixture->rowsum = (float*)malloc(fixture->n * sizeof(float));
    fixture->lambda = 20;
    long p_int = fixture->p * (fixture->p + 1) / 2;
    // fixture->beta = (float*)malloc(p_int * sizeof(float));
    // memset(fixture->beta, 0, p_int * sizeof(float));
    fixture->k = 27;
    fixture->dBMax = 0;
    fixture->intercept = 0;
    int_pair* precalc_get_num = (int_pair*)malloc(p_int * sizeof(int_pair));
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
    long** thread_column_caches = (long**)malloc(max_num_threads * sizeof(long*));
    for (long i = 0; i < max_num_threads; i++) {
        thread_column_caches[i] = (long*)malloc(fixture->n * sizeof(int));
    }
    fixture->column_caches = thread_column_caches;
    printf("done test init\n");
}

static void test_simple_coordinate_descent_tear_down(UpdateFixture* fixture,
    gconstpointer user_data)
{
    for (long i = 0; i < fixture->p; i++) {
        free(fixture->xmatrix.X[i]);
    }
    free(fixture->xmatrix.X);
    free(fixture->Y);
    free(fixture->rowsum);
    // free(fixture->beta);
    free(fixture->precalc_get_num);
    free_sparse_matrix(fixture->xmatrix_sparse);
}

static void test_simple_coordinate_descent_int(UpdateFixture* fixture,
    gconstpointer user_data)
{
    // are we running the shuffle test, or sequential?
    float acceptable_diff = 0.1;
    long shuffle = FALSE;
    if ((long)user_data == TRUE) {
        printf("\nrunning shuffle test!\n");
        acceptable_diff = 10;
        shuffle = TRUE;
    }
    float* glmnet_beta = read_y_csv(
        "/home/kieran/work/lasso_testing/glmnet_small_output.csv", 630);
    printf("starting interaction test\n");
    fixture->xmatrix = read_x_csv(
        "/home/kieran/work/lasso_testing/testXSmall.csv", fixture->n, fixture->p);
    fixture->X = fixture->xmatrix.X;
    fixture->xmatrix_sparse = sparse_X2_from_X(fixture->X, fixture->n, fixture->p, -1, shuffle);
    long p_int = fixture->p * (fixture->p + 1) / 2;
    robin_hood::unordered_flat_map<long, float> beta = fixture->beta;

    float dBMax;
    for (long j = 0; j < 10; j++)
        for (long i = 0; i < p_int; i++) {
            // long k = gsl_permutation_get(fixture->xmatrix_sparse.permutation, i);
            long k = fixture->xmatrix_sparse.permutation->data[i];
            // long k = i;
            Changes changes = update_beta_cyclic_old(
                fixture->xmatrix_sparse, fixture->Y, fixture->rowsum, fixture->n,
                fixture->p, fixture->lambda, &beta, k, 0, fixture->precalc_get_num,
                fixture->column_caches[0]);
            dBMax = changes.actual_diff;
        }

    long no_agreeing = 0;
    for (long i = 0; i < p_int; i++) {
        long k = gsl_permutation_get(fixture->xmatrix_sparse.permutation, i);
        // long k = i;
        printf("testing beta[%ld] (%f) ~ %f [", i, beta[i],
            small_X2_correct_beta[k]);

        if ((beta[i] < small_X2_correct_beta[k] + acceptable_diff) && (beta[i] > small_X2_correct_beta[k] - acceptable_diff)) {
            no_agreeing++;
            printf("x]\n");
        } else {
            printf(" ]\n");
        }
    }
    printf("frac agreement: %f\n", (float)no_agreeing / p_int);
    g_assert_true(no_agreeing == p_int);
}

static void test_simple_coordinate_descent_vs_glmnet(UpdateFixture* fixture,
    gconstpointer user_data)
{
    float* glmnet_beta = read_y_csv(
        "/home/kieran/work/lasso_testing/glmnet_small_output.csv", 630);
    printf("starting interaction test\n");
    fixture->p = 35;
    fixture->xmatrix = read_x_csv(
        "/home/kieran/work/lasso_testing/testXSmall.csv", fixture->n, fixture->p);
    fixture->X = fixture->xmatrix.X;
    fixture->xmatrix_sparse = sparse_X2_from_X(fixture->X, fixture->n, fixture->p, -1, FALSE);
    long p_int = fixture->p * (fixture->p + 1) / 2;
    robin_hood::unordered_flat_map<long, float> beta = fixture->beta;

    Beta_Value_Sets beta_sets = simple_coordinate_descent_lasso(
        fixture->xmatrix, fixture->Y, fixture->n, fixture->p, -1, 0.05, 1000, 100,
        0, 0.01, 1.0001, FALSE, 1, "test", FALSE, -1, "test.log", 2);
    beta = beta_sets.beta3; //TODO: don't

    float acceptable_diff = 10;
    long no_agreeing = 0;
    for (long i = 0; i < p_int; i++) {
        long k = gsl_permutation_get(fixture->xmatrix_sparse.permutation, i);
        printf("testing beta[%ld] (%f) ~ %f [", i, beta[k], glmnet_beta[i]);

        if ((beta[k] < glmnet_beta[i] + acceptable_diff) && (beta[k] > glmnet_beta[i] - acceptable_diff)) {
            no_agreeing++;
            printf("x]\n");
        } else {
            printf(" ]\n");
        }
    }
    printf("frac agreement: %f\n", (float)no_agreeing / p_int);
    g_assert_true(no_agreeing >= 0.8 * p_int);
}

// will fail if Y has been normalised
static void test_read_y_csv()
{
    long n = 1000;
    float* Y = read_y_csv("../testYSmall.csv", n);
    printf("%f\n", Y[0]);
    g_assert_true(Y[0] >= -133.351709197933 - 0.0001);
    g_assert_true(Y[0] <= -133.351709197933 + 0.0001);
    g_assert_true(Y[999] >= -352.293608898344 - 0.0001);
    g_assert_true(Y[999] <= -352.293608898344 + 0.0001);
}

// assumes little endian
void printBits(size_t const size, void const* const ptr)
{
    unsigned char* b = (unsigned char*)ptr;
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

static void check_X2_encoding()
{
    long n = 1000;
    long p = 35;
    long p_int = p * (p + 1) / 2;
    XMatrix xmatrix = read_x_csv("../testXSmall.csv", n, p);
    XMatrix X2 = read_x_csv("../testX2Small.csv", n, p_int);
    XMatrixSparse xmatrix_sparse = sparse_X2_from_X(xmatrix.X, n, p, -1, FALSE);

    int_pair* nums = get_all_nums(p, -1);
    // create uncompressed sparse version of X2.
    long** col_nz_indices = malloc(sizeof *col_nz_indices * p_int);
    for (long j = 0; j < p_int; j++) {
    }
    long* col_sizes = malloc(sizeof *col_sizes * p_int);
    for (long j = 0; j < p_int; j++) {
        Queue* col_q = queue_new();
        for (long i = 0; i < n; i++) {
            if (X2.X[j][i] != 0) {
                queue_push_tail(col_q, i);
            }
        }
        col_sizes[j] = queue_get_length(col_q);
        col_nz_indices[j] = malloc(sizeof *col_nz_indices[j] * col_sizes[j]);
        printf(" real col size: %ld, \tcompressed col contains %ld entries \n",
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
    printf("\nmean entry size: %f\n", (float)total / (float)no_entries);

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
    printf("mean diff size: %f\n", (float)total / (float)no_entries);

    printf("size of s8bword struct: %ld (long is %ld)\n", sizeof(S8bWord),
        sizeof(int));

    S8bWord test_word;
    test_word.selector = 7;
    test_word.values = 0;
    unsigned long numbers[10] = { 3, 2, 4, 20, 1, 14, 30, 52, 10, 63 };
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

    printf("num entries in col 0: %ld\n", xmatrix_sparse.cols[0].nz);
    long* col_entries = malloc(60 * sizeof(int));
    long count = 0;
    // GList *s8b_col = NULL;
    GQueue* s8b_col = g_queue_new();
    // work out s8b compressed equivalent of col 0
    long largest_entry = 0;
    long max_bits = max_size_given_entries[0];
    long diff = col_nz_indices[0][0] + 1;
    for (long i = 0; i < xmatrix_sparse.cols[0].nz; i++) {
        if (i != 0)
            diff = col_nz_indices[0][i] - col_nz_indices[0][i - 1];
        // printf("current no. %ld, diff %ld. available bits %ld\n",
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
        if (diff > 1 << max_bits || largest_entry > max_size_given_entries[count + 1]) {
            // if (diff > 1<<max_bits)
            //  printf(" b ");
            // if (largest_entry > max_size_given_entries[count+1])
            //  printf(" c ");
            // printf("pushing word with %ld entries: ", count);
            // for (long j = 0; j < count; j++)
            //  printf("%ld ", col_entries[j]);
            // printf("\n");
            S8bWord* word = malloc(sizeof(S8bWord));
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
    S8bWord* word = malloc(sizeof(S8bWord));
    S8bWord tempword = to_s8b(count, col_entries);
    memcpy(word, &tempword, sizeof(S8bWord));
    g_queue_push_tail(s8b_col, word);

    free(col_entries);
    long length = g_queue_get_length(s8b_col);

    S8bWord* actual_col = malloc(length * sizeof(S8bWord));
    count = 0;
    while (!g_queue_is_empty(s8b_col)) {
        S8bWord* current_word = g_queue_pop_head(s8b_col);
        memcpy(&actual_col[count], current_word, sizeof(S8bWord));
        count++;
    }

    printf("checking [s8b] == [int]\n");
    for (long k = 0; k < p_int; k++) {
        printf("col %ld (interaction %ld,%ld)\n", k, nums[k].i, nums[k].j);
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
                    printf("pos %ld, %ld == %ld\n", col_entry_pos, entry,
                        col_nz_indices[k][col_entry_pos]);
                    g_assert_true(entry == col_nz_indices[k][col_entry_pos]);
                    col_entry_pos++;
                    checked++;
                }
                values >>= item_width[word.selector];
            }
        }
        printf("col %ld, checked %ld out of %ld present\n", k, checked, col_sizes[k]);
        g_assert_true(checked == xmatrix_sparse.cols[k].nz);
    }

    long bytes = length * sizeof(S8bWord);
    printf("col[0] contains %ld words, for a toal of %ld bytes, instead of %ld "
           "shorts (%ld bytes). Effective reduction %f\n",
        length, bytes, xmatrix_sparse.cols[0].nz,
        xmatrix_sparse.cols[0].nz * sizeof(short),
        (float)bytes / (xmatrix_sparse.cols[0].nz * sizeof(short)));

    printf("liblasso vs test compressed first col:\n");
    for (long i = 0; i < xmatrix_sparse.cols[0].nwords; i++) {
        printf("%ld == %ld\n", xmatrix_sparse.cols[0].compressed_indices[i].selector,
            actual_col[i].selector);
        g_assert_true(xmatrix_sparse.cols[0].compressed_indices[i].selector == actual_col[i].selector);
        printf("%ld == %ld\n", xmatrix_sparse.cols[0].compressed_indices[i].values,
            actual_col[i].values);
        g_assert_true(xmatrix_sparse.cols[0].compressed_indices[i].values == actual_col[i].values);
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

static void check_permutation()
{
    long threads = omp_get_num_procs();
    gsl_rng** thread_r = malloc(threads * sizeof(gsl_rng*));
    for (long i = 0; i < threads; i++)
        thread_r[i] = gsl_rng_alloc(gsl_rng_taus2);

    long perm_size = 3235; //<< 12 + 67;
    printf("perm_size %ld\n", perm_size);
    gsl_permutation* perm = gsl_permutation_alloc(perm_size);
    gsl_permutation_init(perm);

    parallel_shuffle(perm, perm_size / threads, perm_size % threads, threads);

    long* found = malloc(perm_size * sizeof(int));
    memset(found, 0, perm_size * sizeof(int));
    for (long i = 0; i < perm_size; i++) {
        size_t val = perm->data[i];
        found[val] = 1;
        printf("found %ld\n", val);
    }
    for (long i = 0; i < perm_size; i++) {
        printf("checking %ld is present\n", i);
        printf("found[%ld] = %ld\n", i, found[i]);
        printf("found[%ld+1] = %ld\n", i, found[i + 1]);
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
        printf("found %ld\n", val);
    }
    for (long i = 0; i < perm_size; i++) {
        printf("checking %ld is present\n", i);
        printf("found[%ld] = %ld\n", i, found[i]);
        printf("found[%ld+1] = %ld\n", i, found[i + 1]);
        g_assert_true(found[i] == 1);
    }
    free(found);
    gsl_permutation_free(perm);
}

long check_didnt_update(long p, long p_int, bool* wont_update, robin_hood::unordered_flat_map<long, float> beta)
{
    long no_disagreeing = 0;
    for (long i = 0; i < p_int; i++) {
        // long k = gsl_permutation_get(fixture->xmatrix_sparse.permutation, i);
        long k = i;
        // long k = fixture->xmatrix_sparse.permutation->data[i];
        // printf("testing beta[%ld] (%f)\n", k, beta[k]);
        int_pair ip = get_num(k, p_int);
        // TODO: we should only check against later items not in the working set,
        // this needs to be udpated
        if (wont_update[ip.i] || wont_update[ip.j]) {
            // printf("checking interaction %ld,%ld is zero\n", ip.i, ip.j);
            if (beta[k] != 0.0) {
                printf("beta %ld (interaction %ld,%ld) should be zero according to "
                       "will_update_effect(), but is in fact %f\n",
                    k, ip.i, ip.j, beta[k]);
                no_disagreeing++;
            }
        }
    }
    // printf("frac disagreement: %f\n", (float)no_disagreeing/p);
    if (no_disagreeing == 0) {
        // printf("no disagreements\n");
    }
    return no_disagreeing;
}

static void pruning_fixture_set_up(UpdateFixture* fixture,
    gconstpointer use_big)
{
    test_simple_coordinate_descent_set_up(fixture, use_big);
    printf("getting sparse X\n");
    XMatrixSparse Xc = sparsify_X(fixture->X, fixture->n, fixture->p);
    printf("getting sparse X2\n");
    clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);
    if (use_big < 2) {
        XMatrixSparse X2c = sparse_X2_from_X(fixture->X, fixture->n, fixture->p, -1, FALSE);
        fixture->X2c = X2c;
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_time);
    x2_conversion_time = ((float)(end_time.tv_nsec - start_time.tv_nsec)) / 1e9 + (end_time.tv_sec - start_time.tv_sec);
    fixture->Xc = Xc;
}

static void pruning_fixture_tear_down(UpdateFixture* fixture,
    gconstpointer user_data)
{
    free_sparse_matrix(fixture->Xc);
    free_sparse_matrix(fixture->X2c);

    test_simple_coordinate_descent_tear_down(fixture, NULL);
}

bool get_wont_update(char* working_set, bool* wont_update, long p,
    X_uncompressed Xu, float lambda, float* last_max,
    float** last_rowsum, float* rowsum, long* column_cache,
    long n, robin_hood::unordered_flat_map<long, float> beta)
{
    bool ruled_out = false;
    for (long j = 0; j < p; j++) {
        float sum = 0.0;
        wont_update[j] = wont_update_effect(
            Xu, lambda, j, last_max[j], last_rowsum[j], rowsum, column_cache);
        if (wont_update[j])
            ruled_out = true;
    }

    // for (long i = 0; i < p; i++) {
    //  if (wont_update[i]) {
    //      printf("%ld supposedly wont update\n", i);
    //  }
    //}
    // printf("ruled out %ld branch(es)\n", ruled_out);
    return ruled_out;
}
// run branch_prune check, then full regression step without pruning.
// the beta values that would have been pruned should be 0.
static void check_branch_pruning(UpdateFixture* fixture,
    gconstpointer user_data)
{
    printf("\nstarting branch pruning test\n");
    long n = fixture->n;
    long p = fixture->p;
    float* rowsum = fixture->rowsum;
    // float lambda = fixture->lambda;
    long shuffle = FALSE;
    printf("starting interaction test\n");
    // fixture->xmatrix = read_x_csv("/ho/testXSmall.csv", fixture->n,
    // fixture->p);
    printf("creating X2\n");
    long p_int = fixture->p * (fixture->p + 1) / 2;
    robin_hood::unordered_flat_map<long, float> beta = fixture->beta;
    char* working_set = malloc(sizeof *working_set * p_int);
    memset(working_set, 0, sizeof *working_set * p_int);

    XMatrixSparse Xc = fixture->Xc;
    X_uncompressed Xu = construct_host_X(&Xc);
    XMatrixSparse X2c = fixture->X2c;
    long column_cache[n];

    bool wont_update[p];
    for (long j = 0; j < p; j++)
        wont_update[j] = 0;

    float** last_rowsum = malloc(sizeof *last_rowsum * p);
#pragma omp parallel for schedule(static)
    for (long i = 0; i < p; i++) {
        last_rowsum[i] = malloc(sizeof *last_rowsum[i] * n + PADDING);
        memset(last_rowsum[i], 0, sizeof *last_rowsum[i] * n);
    }

    for (long j = 0; j < p; j++)
        for (long i = 0; i < n; i++)
            g_assert_true(last_rowsum[j][i] == 0.0);

    float last_max[n];
    memset(last_max, 0, sizeof(last_max));

    // start running tests with decreasing lambda
    // float lambda_sequence[] = {10000,500, 400, 300, 200, 100, 50, 25, 10, 5,
    // 2, 1, 0.5, 0.2, 0.1, 0.05, 0.01};
    float lambda_sequence[] = { 10000, 500, 400, 300, 200, 100, 50, 25,
        10, 5, 2, 1, 0.5, 0.2, 0.1 };
    long seq_length = sizeof(lambda_sequence) / sizeof(*lambda_sequence);
    float lambda = lambda_sequence[0] * n / 2;
    bool ruled_out = 0;
    float* old_rowsum = malloc(sizeof *old_rowsum * n);

    float error = 0.0;
    for (long i = 0; i < n; i++) {
        error += rowsum[i] * rowsum[i];
    }
    error = sqrt(error);
    printf("initial error: %f\n", error);

    float max_int_delta[p];
    memset(max_int_delta, 0, sizeof *max_int_delta * p);
    for (long lambda_ind = 0; lambda_ind < seq_length; lambda_ind++) {
        memcpy(old_rowsum, rowsum, sizeof *rowsum * n);
        lambda = lambda_sequence[lambda_ind] * n / 2;
        printf("\nrunning lambda %f, current error: %f\n", lambda, error);
        float dBMax;
        // TODO: implement working set and update test
        long iter = 0;
        for (iter = 0; iter < 50; iter++) {
            float prev_error = error;

            ruled_out = get_wont_update(working_set, wont_update, p, Xu, lambda, last_max,
                last_rowsum, rowsum, column_cache, n, beta);
            printf("iter %ld ruled out %ld\n", iter, ruled_out);
            long k = 0;
            for (long main_effect = 0; main_effect < p; main_effect++) {
                for (long interaction = main_effect; interaction < p; interaction++) {
                    float old = beta[k];
                    Changes changes = update_beta_cyclic_old(
                        X2c, fixture->Y, rowsum, n, p, lambda, &beta, k, 0,
                        fixture->precalc_get_num, column_cache);
                    dBMax = changes.actual_diff;
                    // float new = beta[k];
                    if (!working_set[k] && fabs(changes.pre_lambda_diff) > fabs(max_int_delta[main_effect])) {
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
                        printf("main effect %ld new last_max is %f\n", i, max_int_delta[i]);
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
        printf("done lambda %f in %ld iters\n", lambda, iter + 1);
        for (long i = 0; i < p; i++) {
            // we did check these anyway, but since we ordinarily wouldn't they don't
            // get updated.
            if (!wont_update[i]) {
                // printf("updating last_rowsum for %ld\n", i);
                memcpy(last_rowsum[i], old_rowsum, sizeof *old_rowsum * n);
            }
        }
        printf("new error: %f\n", error);
    }
}

static long last_updated_val = -1;
// nothing in vars is changed
float run_lambda_iters(Iter_Vars* vars, float lambda, float* rowsum)
{
    XMatrixSparse Xc = vars->Xc;
    float** last_rowsum = vars->last_rowsum;
    // long **thread_column_caches = vars->thread_column_caches;
    Thread_Cache* thread_caches = vars->thread_caches;
    long n = vars->n;
    Beta_Value_Sets* beta_sets = vars->beta_sets;
    robin_hood::unordered_flat_map<long, float>* beta1 = &beta_sets->beta1;
    robin_hood::unordered_flat_map<long, float>* beta2 = &beta_sets->beta2;
    robin_hood::unordered_flat_map<long, float>* beta3 = &beta_sets->beta3;
    robin_hood::unordered_flat_map<long, float>* beta = &beta_sets->beta2; //TODO: dont
    float* last_max = vars->last_max;
    bool* wont_update = vars->wont_update;
    long p = vars->p;
    long p_int = vars->p_int;
    XMatrixSparse X2c = vars->X2c;
    float* Y = vars->Y;
    float* max_int_delta = vars->max_int_delta;
    int_pair* precalc_get_num = vars->precalc_get_num;
    gsl_permutation* iter_permutation = vars->iter_permutation;

    float error = 0.0;
    for (long i = 0; i < n; i++) {
        error += rowsum[i] * rowsum[i];
    }
    error = sqrt(error);
    for (long iter = 0; iter < 100; iter++) {
        // printf("iter %ld\n", iter);
        // last_iter_count = iter;
        float prev_error = error;

        parallel_shuffle(iter_permutation, permutation_split_size, final_split_size,
            permutation_splits);
#pragma omp parallel for num_threads(NumCores)                                 \
    shared(X2c, Y, rowsum, beta, precalc_get_num) schedule(static) reduction(+ \
                                                                             : total_basic_beta_updates, total_basic_beta_nz_updates)
        for (long k = 0; k < p_int; k++) {
            // for (long main_effect = 0; main_effect < p; main_effect++) {
            // for (long interaction = main_effect; interaction < p; interaction++) {
            total_basic_beta_updates++;
            Changes changes = update_beta_cyclic_old(
                X2c, Y, rowsum, n, p, lambda, beta, k, 0, precalc_get_num,
                thread_caches[omp_get_thread_num()].col_i);
            if (changes.actual_diff != 0.0) {
                last_updated_val = k;
                total_basic_beta_nz_updates++;
            }
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
            printf("done lambda %.2f after %ld iters\n", lambda, iter + 1);
            break;
        } else if (iter == 99) {
            printf("halting lambda %.2f after 100 iters\n", lambda);
        }
    }
    // return dBMax;
    return 0.0;
}

struct RawCol {
    long len;
    long* entries;
};

struct timespec sub_start;
struct timespec sub_end;

float reused_col_time = 0.0;
float main_col_time = 0.0;
float int_col_time = 0.0;

// struct RawCol get_raw_interaction_col(XMatrixSparse Xc, long i, long j) {}

/**
 * set will be null iff length is zero.
 */
struct to_append {
    char* set;
    long length;
};

static unsigned long* target_X;
static unsigned long* target_size;
static unsigned long* target_col_offsets;
static float* target_sumn;
static unsigned long* target_col_nz;

//void target_setup(XMatrixSparse *Xc, XMatrix *xm) {
//  long n = Xc->n;
//  long p = Xc->p;
//  long p_int = p*(p+1)/2;

//  target_X = omp_target_alloc(Xc->total_entries * sizeof *target_X,
//                              omp_get_default_device());
//  target_size =
//      omp_target_alloc(p * sizeof *target_size, omp_get_default_device());
//  target_col_offsets = omp_target_alloc(p * sizeof *target_col_offsets,
//                                        omp_get_default_device());
//  //target_sumn =
//  //    omp_target_alloc(p_int * sizeof *target_sumn, omp_get_default_device());
//  //target_col_nz =
//  //    omp_target_alloc(p_int * sizeof *target_col_nz, omp_get_default_device());
//  printf("total entries: %ld\n", Xc->total_entries);
//  unsigned long *temp_X = malloc(Xc->total_entries * sizeof *temp_X);
//  unsigned long *temp_offset = malloc(p * sizeof *temp_offset);
//  unsigned long *temp_size = malloc(p * sizeof *temp_offset);
//  memset(temp_X, 0, Xc->total_entries * sizeof *temp_X);
//  long offset = 0;
//  for (long k = 0; k < p; k++) {
//    temp_offset[k] = offset;
//    temp_size[k] = Xc->cols[k].nz;
//    long *col = &temp_X[offset];
//    // read column
//    {
//      long col_entry_pos = 0;
//      long entry = -1;
//      for (long i = 0; i < Xc->cols[k].nwords; i++) {
//        S8bWord word = Xc->cols[k].compressed_indices[i];
//        unsigned long values = word.values;
//        for (long j = 0; j <= group_size[word.selector]; j++) {
//          long diff = values & masks[word.selector];
//          if (diff != 0) {
//            entry += diff;
//            col[col_entry_pos] = entry;
//            col_entry_pos++;
//            offset++;
//          }
//          values >>= item_width[word.selector];
//        }
//      }
//      // offset += col_entry_pos;
//    }
//  }
//  // memcpy(target_size, Xc.col_nz, n * sizeof *target_size);
//  omp_target_memcpy(target_size, temp_size, p * sizeof *target_size, 0, 0,
//                    omp_get_default_device(), omp_get_initial_device());
//  omp_target_memcpy(target_col_offsets, temp_offset, p * sizeof *temp_offset, 0,
//                    0, omp_get_default_device(), omp_get_initial_device());
//  omp_target_memcpy(target_X, temp_X, Xc->total_entries * sizeof *target_size,
//                    0, 0, omp_get_default_device(), omp_get_initial_device());
//  free(temp_X);
//  // target_X = temp_X;
//  free(temp_offset);
//  free(temp_size);
//  // target_col_offsets = temp_offset;
//}

// struct X_uncompressed construct_host_X(XMatrixSparse *Xc) {

static void check_branch_pruning_accuracy(UpdateFixture* fixture,
    gconstpointer user_data)
{
    printf("starting accuracy test\n");
    long use_adcal = TRUE;
    Beta_Value_Sets beta_sets = simple_coordinate_descent_lasso(fixture->xmatrix, fixture->Y, fixture->n, fixture->p,
        -1, 0.01, 47504180,
        200, FALSE, -1, 1.01,
        NONE, NULL, 0, use_adcal,
        77, "test.log", 2);

    vector<pair<int, int>> true_effects = {
        { 79, 432 },
        { 107, 786 },
        { 265, 522 },
        { 265, 630 },
        { 293, 432 },
        { 314, 779 },
        { 314, 812 },
        { 382, 816 },
        { 522, 811 },
        { 585, 939 },
        { 630, 786 },
        { 630, 820 },
        { 656, 812 },
        { 107, 382 }
    };

    auto check_results = [&]() {
        g_assert_true(beta_sets.beta3.size() == 0);

        printf("found:\n");
        for (auto& b1 : beta_sets.beta1) {
            long val = b1.first;
            printf(" %ld\n", val);
        }
        for (auto& b2 : beta_sets.beta2) {
            long val = b2.first;
            std::tuple<long, long> inter = val_to_pair(val, fixture->p);
            printf(" %ld,%ld\n", std::get<0>(inter), std::get<1>(inter));
        }

        for (auto it = true_effects.begin(); it != true_effects.end(); it++) {
            long first = it->first - 1;
            long second = it->second - 1;
            printf("checking %ld,%ld\n", first, second);
            long val = pair_to_val(std::make_tuple(first, second), fixture->p);
            printf("beta[%ld] (%ld,%ld) = %f\n", val, first, second, beta_sets.beta2[val]);
            g_assert_true(fabs(beta_sets.beta2[val]) > 0.0);
        }
    };

    check_results();

    // these are the values that the previous version reliably finds
    true_effects = {
        { 4, 195 },
        { 24, 109 },
        { 52, 881 },
        { 67, 778 },
        { 79, 432 },
        { 107, 786 },
        { 121, 325 },
        { 137, 347 },
        { 141, 300 },
        { 179, 246 },
        { 197, 443 },
        { 265, 522 },
        { 265, 630 },
        { 293, 432 },
        { 293, 849 },
        { 314, 779 },
        { 314, 812 },
        { 314, 990 },
        { 352, 645 },
        { 355, 560 },
        { 382, 816 },
        { 390, 600 },
        { 415, 987 },
        { 432, 586 },
        { 506, 560 },
        { 511, 642 },
        { 522, 811 },
        { 526, 905 },
        { 544, 585 },
        { 565, 881 },
        { 585, 939 },
        { 593, 630 },
        { 602, 859 },
        { 630, 786 },
        { 630, 820 },
        { 631, 704 },
        { 656, 812 },
        { 703, 952 },
        { 973, 990 },
        { 36, 195 },
        { 93, 862 },
        { 107, 382 },
    };

    beta_sets = simple_coordinate_descent_lasso(fixture->xmatrix, fixture->Y, fixture->n, fixture->p,
        -1, 0.01, 47504180,
        200, FALSE, -1, 1.01,
        NONE, NULL, 0, use_adcal,
        500, "test.log", 2);

    check_results();
    // beta_sets.beta2;
    // g_assert_true(beta_sets.beta2[])
}

static void check_branch_pruning_faster(UpdateFixture* fixture,
    gconstpointer user_data)
{
    printf("starting branch pruning speed test\n");
    float acceptable_diff = 1.05;
    long n = fixture->n;
    long p = fixture->p;
    float* rowsum = fixture->rowsum;
    long shuffle = FALSE;
    printf("starting interaction test\n");
    printf("creating X2\n");
    long p_int = fixture->p * (fixture->p + 1) / 2;
    robin_hood::unordered_flat_map<long, float> beta1;
    robin_hood::unordered_flat_map<long, float> beta2;
    robin_hood::unordered_flat_map<long, float> beta3;
    Beta_Value_Sets beta_sets = { beta1, beta2, beta3, p };
    robin_hood::unordered_flat_map<long, float> pruning_beta1;
    robin_hood::unordered_flat_map<long, float> pruning_beta2;
    robin_hood::unordered_flat_map<long, float> pruning_beta3;
    Beta_Value_Sets pruning_beta_sets = { pruning_beta1, pruning_beta2, pruning_beta3, p };

    for (long i = 0; i < p_int; i++) {
        beta_sets.beta2[i] = 0.0;
    }

    float* Y = fixture->Y;

    const long MAX_NZ_BETA = 2000;
    gsl_permutation* iter_permutation = gsl_permutation_alloc(p_int);

    Thread_Cache thread_caches[NumCores];

    for (long i = 0; i < NumCores; i++) {
        thread_caches[i].col_i = malloc(sizeof(int) * max(n, p));
        thread_caches[i].col_j = malloc(sizeof(int) * n);
    }

    XMatrixSparse Xc = fixture->Xc;
    XMatrixSparse X2c = fixture->X2c;

    bool* wont_update = malloc(sizeof *wont_update * p);
#pragma omp parallel for schedule(static)
    for (long j = 0; j < p; j++)
        wont_update[j] = 0;

    // float last_rowsum[p][n];
    float** last_rowsum = malloc(sizeof *last_rowsum * p);
    last_rowsum[0] = malloc(sizeof(float) * p * n);
#pragma omp parallel for schedule(static)
    for (long i = 0; i < p; i++) {
        last_rowsum[i] = malloc(sizeof *last_rowsum[i] * n + 64);
        // printf("i: %ld\n", i);
        // last_rowsum[i] = &last_rowsum[0][i * n];
        memset(last_rowsum[i], 0, sizeof *last_rowsum[i] * n);
    }

    for (long j = 0; j < p; j++)
        for (long i = 0; i < n; i++)
            g_assert_true(last_rowsum[j][i] == 0.0);

    // float last_max[n];
    float* last_max = calloc(n, sizeof(float));
    // printf("last_max: %p\n", last_max);
    memset(last_max, 0, sizeof(*last_max));

    // start running tests with decreasing lambda
    // float lambda_sequence[] = {10000,500, 400, 300, 200, 100, 50, 25,
    // 10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.01};
    float lambda_sequence[] = { 10000, 500, 400, 300, 200, 100, 50, 25,
        10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05 };
    // float lambda_sequence[] = {10000,500, 400, 300};
    long seq_length = sizeof(lambda_sequence) / sizeof(*lambda_sequence);
    float* old_rowsum = malloc(sizeof *old_rowsum * n);

    float error = 0.0;
    for (long i = 0; i < n; i++) {
        error += rowsum[i] * rowsum[i];
    }
    error = sqrt(error);
    // printf("initial error: %f\n", error);

    float* max_int_delta = malloc(sizeof *max_int_delta * p);
    memset(max_int_delta, 0, sizeof *max_int_delta * p);
    struct X_uncompressed Xu = construct_host_X(&Xc);

    Iter_Vars iter_vars_basic = {
        Xc,
        last_rowsum,
        thread_caches,
        n,
        &beta_sets,
        last_max,
        NULL,
        p,
        p_int,
        X2c,
        fixture->Y,
        max_int_delta,
        fixture->precalc_get_num,
        iter_permutation,
        Xu,
    };
    struct timespec start, end;
    float basic_cpu_time_used, pruned_cpu_time_used;
    printf("getting time for un-pruned version: lambda min = %f\n", LAMBDA_MIN);
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    // for (long lambda_ind = 0; lambda_ind < seq_length; lambda_ind ++) {
    for (float lambda = 10000 * fixture->n / 2; lambda > LAMBDA_MIN; lambda *= 0.9) {
        //long nz_beta = beta_sets.beta1.size()
        //+beta_sets.beta2.size()
        //+beta_sets.beta3.size();
        long nz_beta = 0;
        for (auto it = beta_sets.beta2.begin(); it != beta_sets.beta2.end(); it++) {
            if (fabs(it->second) != 0.0) {
                nz_beta++;
            }
        }

        if (nz_beta > MAX_NZ_BETA) {
            break;
        }
        // float lambda = lambda_sequence[lambda_ind];
        printf("lambda: %f\n", lambda);
        float dBMax;
        // TODO: implement working set and update test
        long last_iter_count = 0;

        if (Xc.p <= 1000) {
            run_lambda_iters(&iter_vars_basic, lambda, rowsum);
        }
    }
    printf("last updated val: %ld\n", last_updated_val);
    int_pair pair = fixture->precalc_get_num[last_updated_val];
    printf("%ld,%ld\n", pair.i, pair.j);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    basic_cpu_time_used = ((float)(end.tv_nsec - start.tv_nsec)) / 1e9 + (end.tv_sec - start.tv_sec);
    printf("time: %f s\n", basic_cpu_time_used);

#pragma omp parallel for schedule(static)
    for (long i = 0; i < p; i++) {
        memset(last_rowsum[i], 0, sizeof *last_rowsum[i] * n);
        last_max[i] = 0.0;
        max_int_delta[i] = 0;
    }
    for (long i = 0; i < n; i++) {
        rowsum[i] = -Y[i];
    }
    XMatrixSparse X2c_fake;
    Iter_Vars iter_vars_pruned = {
        Xc,
        last_rowsum,
        thread_caches,
        n,
        &pruning_beta_sets,
        last_max,
        wont_update,
        p,
        p_int,
        X2c_fake,
        fixture->Y,
        max_int_delta,
        fixture->precalc_get_num,
        iter_permutation,
        Xu,
    };

    printf("getting time for pruned version\n");
    float* p_rowsum = malloc(sizeof *p_rowsum * n);
    for (long i = 0; i < n; i++) {
        p_rowsum[i] = -Y[i];
    }
    //target_setup(&Xc, &fixture->xmatrix);
    //TODO: setup OpenCL device.
    // struct OpenCL_Setup ocl_setup = setup_working_set_kernel(Xu, n, p);
    struct OpenCL_Setup ocl_setup; //TODO: remove this
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    Active_Set active_set = active_set_new(p_int, p);
    for (float lambda = 10000 * fixture->n / 2; lambda > LAMBDA_MIN; lambda *= 0.9) {
        long nz_beta = 0;
        // #pragma omp parallel for schedule(static) reduction(+:nz_beta)
        //for (auto it = beta_pruning.begin(); it != beta_pruning.end(); it++) {
        //  if (it->second != 0) {
        //    nz_beta++;
        //  }
        //}
        nz_beta = pruning_beta_sets.beta1.size() + pruning_beta_sets.beta2.size() + pruning_beta_sets.beta3.size();
        if (nz_beta > MAX_NZ_BETA) {
            break;
        }
        // memcpy(old_rowsum, p_rowsum, sizeof *p_rowsum *n);
        // float lambda = lambda_sequence[lambda_ind];
        float dBMax;
        // TODO: implement working set and update test
        long last_iter_count = 0;

        //TODO: probably best to remove lambda scalling in all the tests too.
        run_lambda_iters_pruned(&iter_vars_pruned, lambda, p_rowsum, old_rowsum,
            &active_set, &ocl_setup, 2);
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    printf("\n");
    printf("col 0,1 length: %ld\n", X2c.cols[1].nz);
    printf("classic 0,1: %f\n", beta_sets.beta2[1]);
    printf("pruned  0,1: %f\n", pruning_beta_sets.beta2[101]);
    printf("addresses maybe of interest:\n");
    printf("Xc.cols:       %p\n", Xc.cols);
    printf("last_rowsum:   %p\n", last_rowsum);
    printf("p_rowsum:      %p\n", p_rowsum);
    printf("wont_update:   %p\n", wont_update);
    active_set_free(active_set);
    pruned_cpu_time_used = ((float)(end.tv_nsec - start.tv_nsec)) / 1e9 + (end.tv_sec - start.tv_sec);
    printf("basic time: %.2fs (%.2f X2 conversion) \t pruned time %.2f s\n",
        basic_cpu_time_used + x2_conversion_time, x2_conversion_time,
        pruned_cpu_time_used);
    printf("basic updates: %ld, [%ld nz], pruned_updates: %ld [%ld nz]\n", total_basic_beta_updates, total_basic_beta_nz_updates, total_beta_updates, total_beta_nz_updates);
    printf("pruning time is composed of %.2f pruning, %.2f working set "
           "updates, "
           "and %.2f subproblem time\n",
        pruning_time, working_set_update_time, subproblem_time);

    printf("working set upates were: %.2f main effect col, %.2f long col, %.2f "
           "reused col\n",
        main_col_time, int_col_time, reused_col_time);

    printf("used branches:   %ld\n", used_branches);
    printf("pruned branches: %ld\n", pruned_branches);
    printf("total branches:  %ld\n", used_branches + pruned_branches);

    // g_assert_true(pruned_cpu_time_used < 0.9 * basic_cpu_time_used);
    float* basic_rowsum = malloc(sizeof *basic_rowsum * n);
    float* pruned_rowsum = malloc(sizeof *pruned_rowsum * n);
    for (long i = 0; i < n; i++) {
        basic_rowsum[i] = -Y[i];
        pruned_rowsum[i] = -Y[i];
    }
    if (Xc.p > 1000) {
        return;
    }
    long nz_beta_basic = 0;
    long nz_beta_pruning = 0;
    for (long k = 0; k < p_int; k++) {
        nz_beta_basic += beta_sets.beta2[k] != 0.0;
        long entry = -1;
        for (long i = 0; i < X2c.cols[k].nwords; i++) {
            S8bWord word = X2c.cols[k].compressed_indices[i];
            unsigned long values = word.values;
            for (long j = 0; j <= group_size[word.selector]; j++) {
                long diff = values & masks[word.selector];
                if (diff != 0) {
                    entry += diff;

                    // do whatever we need here with the index below:
                    basic_rowsum[entry] += beta_sets.beta2[k];
                    // nz_beta_pruning += beta_pruning[k] != 0.0;
                    // pruned_rowsum[entry] += beta_pruning[k];
                    //if (basic_rowsum[entry] > 1e20) {
                    //  printf("1. basic_rowsum[%ld] = %f\n", entry, basic_rowsum[entry]);
                    //}
                }
                values >>= item_width[word.selector];
            }
        }
    }
    //for (long i = 0; i < Xu.host_col_nz[54]; i++) {
    //  Xu.host_X Xu.host_col_offsets[54]
    //}
    long total_effects = 0;
    printf("reading beta values\n");

    auto check_beta_set = [&](auto beta_set) {
        for (auto it = beta_set->begin(); it != beta_set->end(); it++) {
            long val = it->first;
            float bv = it->second;
            if (fabs(bv) != 0.0) {
                nz_beta_pruning++;
            }
            auto check_columns = [&](long a, long b, long c) {
                long* colA = &Xu.host_X[Xu.host_col_offsets[a]];
                long* colB = &Xu.host_X[Xu.host_col_offsets[b]];
                long* colC = &Xu.host_X[Xu.host_col_offsets[c]];
                long ib = 0, ic = 0;
                long total_entries_found = 0;
                for (long ia = 0; ia < Xu.host_col_nz[a]; ia++) {
                    long cur_row = colA[ia];
                    while (colB[ib] < cur_row && ib < Xu.host_col_nz[b] - 1)
                        ib++;
                    while (colC[ic] < cur_row && ic < Xu.host_col_nz[c] - 1)
                        ic++;
                    if (cur_row == colB[ib] && cur_row == colC[ic]) {
                        pruned_rowsum[cur_row] += bv;
                        total_entries_found++;
                    }
                }
                if (a == b && a == c) {
                    g_assert_true(total_entries_found == Xu.host_col_nz[a]);
                }
            };
            if (val < p) {
                check_columns(val, val, val);
            } else if (val < p * p) {
                auto pair = val_to_pair(val, p);
                long a = std::get<0>(pair);
                long b = std::get<1>(pair);
                g_assert_true(a < b);
                check_columns(a, a, b);
            } else {
                if (val >= p * p * p) {
                    printf("broken val: %ld\n", val);
                }
                g_assert_true(val < p * p * p);
                auto triple = val_to_triplet(val, p);
                long a = std::get<0>(triple);
                long b = std::get<1>(triple);
                long c = std::get<2>(triple);
                g_assert_true(a < b);
                g_assert_true(b < c);
                check_columns(a, b, c);
            }
        }
    };

    printf("checking beta 1\n");
    check_beta_set(&pruning_beta_sets.beta1);
    printf("checking beta 2\n");
    check_beta_set(&pruning_beta_sets.beta2);
    printf("checking beta 3\n");
    check_beta_set(&pruning_beta_sets.beta3);

    printf("found %ld effects\n", total_effects);
    // g_assert_true(nz_beta_pruning >= beta_pruning.size() /2);
    float basic_error = 0.0;
    float pruned_error = 0.0;
    for (long i = 0; i < n; i++) {
        basic_error += basic_rowsum[i] * basic_rowsum[i];
        pruned_error += pruned_rowsum[i] * pruned_rowsum[i];
        if (basic_rowsum[i] > 1e20) {
            printf("2. basic_rowsum[%ld] = %f\n", i, basic_rowsum[i]);
        }
    }
    basic_error = sqrt(basic_error);
    pruned_error = sqrt(pruned_error);

    printf("basic had %ld nz beta, pruning had %ld\n", nz_beta_basic, nz_beta_pruning);

    printf("basic error %.2f \t pruned err %.2f\n", basic_error, pruned_error);
    //printf("pruning time is composed of %.2f pruning, %.2f working set "
    //       "updates, "
    //       "and %.2f subproblem time\n",
    //       pruning_time, working_set_update_time, subproblem_time);
    g_assert_true(
        fmax(basic_error, pruned_error) / fmin(basic_error, pruned_error) < 1.2);

    printf("working set upates were: %.2f main effect col, %.2f long col, %.2f "
           "reused col\n",
        main_col_time, int_col_time, reused_col_time);

    //printf("checking beta values come out the same\n");
    //long offset = 0;
    //for (long main = 0; main < p; main++) {
    //  for (long inter = 0; inter < p; inter++) {
    //    long k = offset;
    //    float basic_beta = fabs(beta_sets.beta2[offset]);
    //    float pruned_beta = 0.0;
    //    if (main == inter) {
    //      pruned_beta = fabs(pruning_beta_sets.beta1[main]);
    //    } else {
    //      pruned_beta = fabs(pruning_beta_sets.beta2[pair_to_val(std::make_tuple(main, inter), p)]);
    //    }
    //    float max = fmax(basic_beta, pruned_beta);
    //    float min = fmin(basic_beta, pruned_beta);

    //    if (max / min > acceptable_diff && fabs(basic_beta) > 0.01) {
    //      printf("basic[%ld,%ld = %ld] \t   %.2e \t =\\= \t %.2e \t pruning[%ld,%ld = %ld]\n", main, inter, k, basic_beta,
    //             pruned_beta, main, inter, pair_to_val(std::make_tuple(main, inter), p));
    //    }
    //    g_assert_true(max/min <= acceptable_diff || fabs(basic_beta) < 0.01);
    //    offset++;
    //  }
    //}
    //for (long k = 0; k < p; k++) {
    //  float basic_beta = fabs( beta_sets.beta2[k]);
    //  float pruned_beta = fabs( pruning_beta_sets->beta1[k]);
    //  float max = fmax(basic_beta, pruned_beta);
    //  float min = fmin(basic_beta, pruned_beta);

    //  if (max / min > acceptable_diff) {
    //    printf("basic[%ld] \t   %.2f \t =\\= \t %.2f \t pruning[%ld]\n", k, beta[k],
    //           beta_pruning[k], k);
    //  }
    //}
    // opencl_cleanup(ocl_setup);
}

void test_row_list_without_columns()
{
    long n = 5, p = 5;
    const long xm_a[n][p] = {
        { 1, 0, 0, 0, 0 },
        { 0, 1, 1, 1, 0 },
        { 0, 0, 0, 0, 1 },
        { 1, 1, 0, 0, 0 },
        { 1, 1, 1, 0, 0 },
    };
    long** xm = new long*[p];
    for (long j = 0; j < p; j++) {
        xm[j] = new long[n];
    }
    for (long i = 0; i < n; i++)
        for (long j = 0; j < p; j++) {
            xm[j][i] = xm_a[i][j];
        }
    XMatrixSparse Xc = sparsify_X(xm, n, p);
    X_uncompressed Xu = construct_host_X(&Xc);
    bool remove[p] = { false, false, false, true, false };

    Thread_Cache thread_caches[NumCores];
    for (long i = 0; i < NumCores; i++) {
        thread_caches[i].col_i = malloc(sizeof(int) * max(n, p));
        thread_caches[i].col_j = malloc(sizeof(int) * n);
    }

    struct row_set rs = row_list_without_columns(Xc, Xu, remove, thread_caches);

    g_assert_true(rs.row_lengths[0] == 1);
    g_assert_true(rs.row_lengths[1] == 2);
    g_assert_true(rs.row_lengths[4] == 3);

    printf("returned rows:\n");
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < rs.row_lengths[i]; j++) {
            long entry = rs.rows[i][j];
            printf("%ld, ", entry);
        }
        printf("\n");
    }
    printf("checking:\n");
    for (long i = 0; i < n; i++) {
        long found = 0;
        for (long j = 0; j < p; j++) {
            if (remove[j])
                continue;
            long entry = xm_a[i][j];
            printf("%ld, ", entry);
            if (entry == 1) {
                if (rs.rows[i][found] != j) {
                    printf("!= found! (%ld)\n", rs.rows[i][found]);
                }
                g_assert_true(rs.rows[i][found] == j);
                found++;
            }
        }
        if (found != rs.row_lengths[i]) {
            printf("\nfound %ld, row length %ld\n", found, rs.row_lengths[i]);
        }
        g_assert_true(found == rs.row_lengths[i]);
        printf("\n");
    }

    for (long i = 0; i < NumCores; i++) {
        free(thread_caches[i].col_i);
        free(thread_caches[i].col_j);
    }
}

void trivial_3way_test()
{
    long n = 5, p = 5;
    const long xm_a[n][p] = {
        { 1, 1, 1, 0, 0 },
        { 1, 0, 1, 1, 0 },
        { 0, 1, 1, 1, 1 },
        { 1, 1, 0, 0, 0 },
        { 1, 1, 1, 0, 0 },
    };
    long** xm = new long*[p];
    for (long j = 0; j < p; j++) {
        xm[j] = new long[n];
    }
    for (long i = 0; i < n; i++)
        for (long j = 0; j < p; j++) {
            xm[i][j] = xm_a[j][i];
        }

    robin_hood::unordered_map<long, float> correct_beta;
    //robin_hood::unordered_map<std::pair<long,long>, float> correct_beta2;
    //robin_hood::unordered_map<std::tuple<long,long,long>, float> correct_beta3;

    //correct_beta1[0] = 2.3;
    //correct_beta2[std::pair(0,1)] = -5;
    //correct_beta3[std::make_tuple(0,1,2)] = 4.6;
    correct_beta[0] = 2.3;
    correct_beta[pair_to_val(std::make_tuple(0, 3), p)] = 5;
    correct_beta[triplet_to_val(std::make_tuple(0, 1, 2), p)] = -14.6;
    // val_to_triplet()

    float Y[n];
    for (long i = 0; i < n; i++) {
        Y[i] = 0.0;
    }
    //auto beta_sets = std::tuple(correct_beta1, correct_beta2, correct_beta3);

    //std::apply([](auto beta_set ...) {

    //}, beta_sets);

    //for_each(beta_sets, [](auto& beta_set) {
    //  auto curr_inter = beta_set.cbegin();
    //  auto last_inter = beta_set.cend();
    //  while (curr_inter != last_inter) {
    //    std::tuple ind = curr_inter->first;
    //    for_each(ind, [](auto& val){
    //      cout << val << ", ";
    //    });
    //    cout << ": ";
    //    float effect = curr_inter->second;
    //    std::cout << effect << "\n";
    //    curr_inter++;
    //  }
    //});

    for (long i = 0; i < n; i++) {
        for (long j = 0; j < p; j++) {
            if (xm[j][i] != 0) {
                for (long j2 = j; j2 < p; j2++) {
                    if (xm[j2][i] != 0) {
                        for (long j3 = j2; j3 < p; j3++) {
                            if (xm[j3][i] != 0) {
                                //if (j == 0 && j2 == 1 && j3 == 2) {
                                //  cout << "b3[0,1,2]: " << correct_beta3[std::tuple{j,j2,j3}] << "\n";
                                //}
                                float cb = correct_beta[triplet_to_val(std::make_tuple(j, j2, j3), p)];
                                if (cb != 0.0) {
                                    printf("Y[%ld] +=  %f (%ld,%ld,%ld)\n", i, cb, j, j2, j3);
                                    Y[i] += cb;
                                }
                            }
                        }
                        Y[i] += correct_beta[pair_to_val(std::make_tuple(j, j2), p)];
                    }
                }
                Y[i] += correct_beta[j];
            }
        }
    }

    cout << "Y:\n";
    for (long i = 0; i < n; i++) {
        cout << i << ": " << Y[i] << "\n";
    }

    XMatrix X;
    X.X = xm;
    X.actual_cols = p;

    XMatrixSparse Xc = sparsify_X(xm, n, p);
    X_uncompressed Xu = construct_host_X(&Xc);

    const long use_adcal = FALSE;
    Beta_Value_Sets beta_sets = simple_coordinate_descent_lasso(X, Y, n, p,
        -1, 0.01, 100,
        100, FALSE, -1, 1.01,
        NONE, NULL, 0, use_adcal,
        -1, "test.log", 3);
    // auto beta = beta_sets.beta3;

    long total_effects = 0;
    auto print_beta_set = [&](auto beta_set) {
        for (auto it = beta_set->begin(); it != beta_set->end(); it++) {
            long val = it->first;
            if (val < p) {
                printf("%ld: %f\n", val, it->second);
            } else if (val < p * p) {
                auto pair = val_to_pair(val, p);
                printf("%ld,%ld: %f\n", std::get<0>(pair), std::get<1>(pair), it->second);
            } else {
                g_assert_true(val < p * p * p);
                auto triple = val_to_triplet(val, p);
                printf("%ld,%ld,%ld: %f\n", std::get<0>(triple), std::get<1>(triple), std::get<2>(triple), it->second);
            }
        }
    };
    printf("Beta values found:\n");
    printf("beta 1:\n");
    print_beta_set(&beta_sets.beta1);
    printf("beta 2:\n");
    print_beta_set(&beta_sets.beta2);
    printf("beta 3:\n");
    print_beta_set(&beta_sets.beta3);

    g_assert_true(beta_sets.beta1.contains(0) && beta_sets.beta1.at(0) > 2.0);
    long tmpval = pair_to_val(std::make_tuple(0, 3), p);
    g_assert_true(beta_sets.beta2.contains(tmpval) && beta_sets.beta2.at(tmpval) != 0.0);
    tmpval = triplet_to_val(std::make_tuple(0, 1, 2), p);
    g_assert_true(beta_sets.beta3.contains(tmpval) && beta_sets.beta3.at(tmpval) < -9.0);

    g_assert_true(beta_sets.beta1.size() + beta_sets.beta2.size() + beta_sets.beta3.size() < 8);

    for (long i = 0; i < p; i++) {
        free(xm[i]);
    }
    free(xm);
}

void test_tuple_vals()
{
    long p = 100;
    for (long a = 0; a < p; a++) {
        for (long b = 0; b < p; b++) {
            for (long c = 0; c < p; c++) {
                long val = triplet_to_val(std::make_tuple(a, b, c), p);
                auto tp = val_to_triplet(val, p);
                long x = std::get<0>(tp);
                long y = std::get<1>(tp);
                long z = std::get<2>(tp);
                if (a != x || b != y || c != z) {
                    printf("assigning %ld %ld %ld ... ", a, b, c);
                    printf("%ld\n", val);
                    printf("found %ld %ld %ld\n", x, y, z);
                }
                g_assert_true(std::get<0>(tp) == a);
                g_assert_true(std::get<1>(tp) == b);
                g_assert_true(std::get<2>(tp) == c);
            }
        }
    }
}

void save_restore_log()
{
    Beta_Value_Sets beta_sets;
    beta_sets.beta1[2] = 0.3;
    beta_sets.beta1[7] = 0.3;
    beta_sets.beta2[12] = 0.7;
    beta_sets.beta2[22] = 123123.4;
    beta_sets.beta3[101] = 123.1;
    beta_sets.beta3[121] = -0.001;
    beta_sets.beta3[999] = -1231.4;
    long num_betas = 7;
    long n = 12;
    long p = 11;
    long iter = 1;
    float lambda_value = 1000.0;
    long lambda_count = 5;
    char* job_args[] = { "arg1", "arg2", "longer_argument_3", "4" };
    char* not_job_args[] = { "arg1", "arg2", "longer_argument_7", "4" };
    long job_args_num = 4;

    char* log_filename = "testlog";

    FILE* logfile = init_log(log_filename, n, p, num_betas, job_args, job_args_num);
    save_log(iter, lambda_value, lambda_count, &beta_sets, logfile);

    printf("\nchecking can_restore_from_log... ");
    g_assert_true(check_can_restore_from_log(log_filename, n, p, num_betas, job_args, job_args_num) == TRUE);
    printf("true\n");
    g_assert_true(check_can_restore_from_log(log_filename, n, p, num_betas, not_job_args, job_args_num) == FALSE);

    long restored_iter = -1;
    long restored_lambda_count = -1;
    float restored_lambda_value = -1.0;
    Beta_Value_Sets restored_beta_sets;

    printf("\t - restoring from log... \n");
    restore_from_log(log_filename, true, n, p, job_args, job_args_num, &restored_iter, &restored_lambda_count, &restored_lambda_value, &restored_beta_sets);
    printf("\t - done\n");

    auto check = [&](auto* set1, auto* set2) {
        for (auto it = set1->begin(); it != set1->end(); it++) {
            long key = it->first;
            float val = it->second;
            if (!set2->contains(key))
                printf("%ld not present in set2\n", key);
            if (!(fabs(set2->at(key) - val) < 0.00001))
                printf("%ld: %f != %f \n", key, set2->at(key), val);
            g_assert_true(fabs(set2->at(key) - val) < 0.00001);
        }
        for (auto it = set2->begin(); it != set2->end(); it++) {
            long key = it->first;
            float val = it->second;
            if (!set1->contains(key))
                printf("%ld not present in set1\n", key);
            if (!(fabs(set1->at(key) - val) < 0.00001))
                printf("%ld: %f != %f \n", key, set2->at(key), val);
            g_assert_true(fabs(set1->at(key) - val) < 0.00001);
        }
    };

    printf("checking beta1\n");
    check(&beta_sets.beta1, &restored_beta_sets.beta1);
    printf("checking beta2\n");
    check(&beta_sets.beta2, &restored_beta_sets.beta2);
    printf("checking beta3\n");
    check(&beta_sets.beta3, &restored_beta_sets.beta3);

    g_assert_true(restored_lambda_count == lambda_count);
    g_assert_true(restored_lambda_value == lambda_value);
    g_assert_true(restored_iter == iter);

    iter += 3;
    lambda_count += 2;
    lambda_value *= 0.75;
    beta_sets.beta3[101] = 103.1;
    beta_sets.beta3[131] = 3.17;
    beta_sets.beta2[21] = 0.17;

    save_log(iter, lambda_value, lambda_count, &beta_sets, logfile);
    g_assert_true(check_can_restore_from_log(log_filename, n, p, num_betas, job_args, job_args_num) == TRUE);
    restore_from_log(log_filename, true, n, p, job_args, job_args_num, &restored_iter, &restored_lambda_count, &restored_lambda_value, &restored_beta_sets);

    printf("checking beta1\n");
    check(&beta_sets.beta1, &restored_beta_sets.beta1);
    printf("checking beta2\n");
    check(&beta_sets.beta2, &restored_beta_sets.beta2);
    printf("checking beta3\n");
    check(&beta_sets.beta3, &restored_beta_sets.beta3);

    g_assert_true(restored_lambda_count == lambda_count);
    g_assert_true(restored_lambda_value == lambda_value);
    g_assert_true(restored_iter == iter);

    iter += 1;
    lambda_count = 0;
    lambda_value *= 0.75;
    beta_sets.beta3[101] = 102.1;
    beta_sets.beta3[131] = 4.17;
    beta_sets.beta2[21] = 0.17;

    save_log(iter, lambda_value, lambda_count, &beta_sets, logfile);
    g_assert_true(check_can_restore_from_log(log_filename, n, p, num_betas, job_args, job_args_num) == TRUE);
    restore_from_log(log_filename, true, n, p, job_args, job_args_num, &restored_iter, &restored_lambda_count, &restored_lambda_value, &restored_beta_sets);

    printf("checking beta1\n");
    check(&beta_sets.beta1, &restored_beta_sets.beta1);
    printf("checking beta2\n");
    check(&beta_sets.beta2, &restored_beta_sets.beta2);
    printf("checking beta3\n");
    check(&beta_sets.beta3, &restored_beta_sets.beta3);

    g_assert_true(restored_lambda_count == lambda_count);
    g_assert_true(restored_lambda_value == lambda_value);
    g_assert_true(restored_iter == iter);

    //TODO: test unfinished log entry

    close_log(logfile);
}

static void test_adcal(UpdateFixture* fixture, gconstpointer user_data)
{
    long depth = 2;
    char* log_filename = "adcal_test.log";
    remove(log_filename);
    long max_interaction_distance = -1;
    float lambda_min = 0.01;
    float lambda_max = 10000;
    long max_iter = 200;
    long VERBOSE = FALSE;
    float frac_overlap_allowed = -1;
    float halt_beta_diff = 1.01;
    LOG_LEVEL log_level = LOG_LEVEL::LAMBDA;
    char* job_args[] = { "adcal", "test", "args" };
    long job_args_num = 3;
    long use_adaptive_calibration = TRUE;
    long max_nz_beta = -1;

    Sparse_Betas beta1;
    beta1.count = 5;
    beta1.indices = new long[5] { 1, 4, 7, 21, 35 };
    beta1.values = new float[5] { 1.2, -2.1, 2.1, 13.0, -11.2 };

    Sparse_Betas beta2;
    beta2.count = 6;
    beta2.indices = new long[6] { 1, 4, 7, 21, 35, 107 };
    beta2.values = new float[6] { 1.7, -2.1, 2.3, 12.0, -11.3, 0.8 };

    g_assert_true(beta2.indices[1] == 4);
    g_assert_true(fabs(beta2.values[2] - 2.3) < 0.001);

    long result = adaptive_calibration_check_beta(0.75, 12.2, &beta1, 10.9, &beta2, fixture->n);
    g_assert_true(result == 1);

    Beta_Value_Sets beta_sets = simple_coordinate_descent_lasso(fixture->xmatrix, fixture->Y, fixture->n, fixture->p, max_interaction_distance, lambda_min, lambda_max, max_iter, VERBOSE, frac_overlap_allowed, halt_beta_diff, log_level, job_args, job_args_num, use_adaptive_calibration, max_nz_beta, log_filename, depth);

    long final_iter, final_lambda_count;
    float final_lambda_value;
    Beta_Value_Sets final_beta_sets;
    restore_from_log(log_filename, true, fixture->n, fixture->p, job_args, job_args_num, &final_iter, &final_lambda_count, &final_lambda_value, &final_beta_sets);

    printf("finished at lambda %ld: %f\n", final_lambda_count, final_lambda_value);
    g_assert_true(fabs(final_lambda_value - 0.556171) < 0.001);
    g_assert_true(final_lambda_count == 192);

    free(beta1.indices);
    free(beta1.values);
    free(beta2.indices);
    free(beta2.values);
}

int main(int argc, char* argv[])
{
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
    g_test_add("/func/test-branch-pruning-accuracy", UpdateFixture, 1,
        pruning_fixture_set_up, check_branch_pruning_accuracy,
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
    g_test_add_func("/func/trivial-3way", trivial_3way_test);
    g_test_add_func("/func/test_tuple_vals", test_tuple_vals);
    g_test_add_func("/func/test_row_list_without_columns", test_row_list_without_columns);
    g_test_add_func("/func/test_save_restore_log", save_restore_log);
    g_test_add("/func/test-adcal", UpdateFixture, 0,
        pruning_fixture_set_up, test_adcal,
        pruning_fixture_tear_down);

    return g_test_run();
}
