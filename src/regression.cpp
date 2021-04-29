#include "liblasso.h"
#include <gsl/gsl_complex.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>

using namespace std;

// check a particular pair of betas in the adaptive calibration scheme
int adaptive_calibration_check_beta(float c_bar, float lambda_1,
                                    Sparse_Betas beta_1, float lambda_2,
                                    Sparse_Betas beta_2, int beta_length,
                                    int n) {
  float max_diff = 0.0;
  float adjusted_max_diff = 0.0;

  int b1_ind = 0;
  int b2_ind = 0;

  while (b1_ind < beta_1.count && b2_ind < beta_2.count) {
    while (beta_1.indices[b1_ind] < beta_2.indices[b2_ind] &&
           b1_ind < beta_1.count)
      b1_ind++;
    while (beta_2.indices[b2_ind] < beta_1.indices[b1_ind] &&
           b2_ind < beta_2.count)
      b2_ind++;
    if (b1_ind < beta_1.count && b2_ind < beta_2.count &&
        beta_1.indices[b1_ind] == beta_2.indices[b2_ind]) {
      float diff = fabs(beta_1.betas[b1_ind] - beta_2.betas[b2_ind]);
      if (diff > max_diff)
        max_diff = diff;
      b1_ind++;
    }
  }

  adjusted_max_diff = max_diff / ((lambda_1 + lambda_2) * (n / 2));

  if (adjusted_max_diff <= c_bar) {
    return 1;
  }
  return 0;
}

// checks whether the last element in the beta_sequence is the one we should
// stop at, according to Chichignoud et als 'Adaptive Calibration Scheme'
// returns TRUE if we are finished, FALSE if we should continue.
int check_adaptive_calibration(float c_bar, Beta_Sequence beta_sequence,
                               int n) {
  // printf("\nchecking %d betas\n", beta_sequence.count);
  for (int i = 0; i < beta_sequence.count; i++) {
    int this_result = adaptive_calibration_check_beta(
        c_bar, beta_sequence.lambdas[beta_sequence.count - 1],
        beta_sequence.betas[beta_sequence.count - 1], beta_sequence.lambdas[i],
        beta_sequence.betas[i], beta_sequence.vec_length, n);
    // printf("result: %d\n", this_result);
    if (this_result == 0) {
      return TRUE;
    }
  }
  return FALSE;
}


float calculate_error(int n, long p_int, XMatrixSparse X2, float *Y, int **X,
                       float *beta, float p, float intercept,
                       float *rowsum) {
  float error = 0.0;
  for (int row = 0; row < n; row++) {
    float row_err = intercept - rowsum[row];
    error += row_err * row_err;
  }
  return error;
}

float *simple_coordinate_descent_lasso(
    XMatrix xmatrix, float *Y, int n, int p, long max_interaction_distance,
    float lambda_min, float lambda_max, int max_iter, int verbose,
    float frac_overlap_allowed, float halt_beta_diff,
    enum LOG_LEVEL log_level, char **job_args, int job_args_num,
    int use_adaptive_calibration, int max_nz_beta) {
  long num_nz_beta = 0;
  long became_zero = 0;
  float lambda = lambda_max;
  VERBOSE = verbose;
  int_pair *precalc_get_num;
  int **X = xmatrix.X;

  // Rprintf("using %d threads\n", NumCores);

  XMatrixSparse X2 = sparse_X2_from_X(X, n, p, max_interaction_distance, FALSE);

  for (int i = 0; i < NUM_MAX_ROWSUMS; i++) {
    max_rowsums[i] = 0;
    max_cumulative_rowsums[i] = 0;
  }

  long p_int = get_p_int(p, max_interaction_distance);
  if (max_interaction_distance == -1) {
    max_interaction_distance = p_int / 2 + 1;
  }
  float *beta;
  beta = malloc(p_int * sizeof(float)); // probably too big in most cases.
  memset(beta, 0, p_int * sizeof(float));

  precalc_get_num = malloc(p_int * sizeof(int_pair));
  int offset = 0;
  for (int i = 0; i < p; i++) {
    for (int j = i; j < min((long)p, i + max_interaction_distance + 1); j++) {
      i = gsl_permutation_get(global_permutation_inverse, offset);
      j = gsl_permutation_get(global_permutation_inverse, offset);
      // printf("i,j: %d,%d\n", i, j);
      precalc_get_num[gsl_permutation_get(global_permutation_inverse, offset)]
          .i = i;
      precalc_get_num[gsl_permutation_get(global_permutation_inverse, offset)]
          .j = j;
      offset++;
    }
  }

  cached_nums = get_all_nums(p, max_interaction_distance);

  float error = 0.0, prev_error;
  for (int i = 0; i < n; i++) {
    error += Y[i] * Y[i];
  }
  float intercept = 0.0;

  float *rowsum = malloc(n * sizeof(float));
  for (int i = 0; i < n; i++)
    rowsum[i] = -Y[i];

  colsum = malloc(p_int * sizeof(float));
  memset(colsum, 0, p_int * sizeof(float));

  col_ysum = malloc(p_int * sizeof(float));
  memset(col_ysum, 0, p_int * sizeof(float));
  for (int col = 0; col < p_int; col++) {
    int entry = -1;
    for (int i = 0; i < X2.cols[col].nwords; i++) {
      S8bWord word = X2.cols[col].compressed_indices[i];
      unsigned long values = word.values;
      for (int j = 0; j <= group_size[word.selector]; j++) {
        int diff = values & masks[word.selector];
        if (diff != 0) {
          entry += diff;
          col_ysum[col] += Y[entry];
        }
        values >>= item_width[word.selector];
      }
    }
  }

  // find largest number of non-zeros in any column
  int largest_col = 0;
  long total_col = 0;
  for (int i = 0; i < p_int; i++) {
    if (X2.cols[i].nz > largest_col) {
      largest_col = X2.cols[i].nz;
    }
    total_col += X2.cols[i].nz;
  }
  int main_sum = 0;
  for (int i = 0; i < p; i++)
    for (int j = 0; j < n; j++)
      main_sum += X[i][j];

  struct timespec start, end;
  float cpu_time_used;

  int set_min_lambda = FALSE;
  gsl_permutation *iter_permutation = gsl_permutation_alloc(p_int);
  gsl_rng *iter_rng;
  gsl_permutation_init(iter_permutation);
  gsl_rng_env_setup();
  const gsl_rng_type *T = gsl_rng_default;
  parallel_shuffle(iter_permutation, permutation_split_size, final_split_size,
                   permutation_splits);
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  float final_lambda = lambda_min;
  int max_lambda_count = 50;
  Rprintf("running from lambda %.2f to lambda %.2f\n", lambda, final_lambda);
  int lambda_count = 1;
  int iter_count = 0;

  int max_num_threads = omp_get_max_threads();
  int **thread_column_caches = malloc(max_num_threads * sizeof(int *));
  for (int i = 0; i < max_num_threads; i++) {
    thread_column_caches[i] = malloc(largest_col * sizeof(int));
  }

  FILE *log_file;
  char *log_filename = "lasso_log.log";
  int iter = 0;
  if (check_can_restore_from_log(log_filename, n, p, p_int, job_args,
                                 job_args_num)) {
    Rprintf("We can restore from a partial log!\n");
    restore_from_log(log_filename, n, p, p_int, job_args, job_args_num, &iter,
                     &lambda_count, &lambda, beta);
    // we need to recalculate the rowsums
    for (int col = 0; col < p_int; col++) {
      int entry = -1;
      for (int i = 0; i < X2.cols[col].nwords; i++) {
        S8bWord word = X2.cols[col].compressed_indices[i];
        unsigned long values = word.values;
        for (int j = 0; j <= group_size[word.selector]; j++) {
          int diff = values & masks[word.selector];
          if (diff != 0) {
            entry += diff;
            rowsum[entry] += beta[col];
          }
          values >>= item_width[word.selector];
        }
      }
    }
  } else {
    Rprintf("no partial log for current job found\n");
  }
  if (log_level != NONE)
    log_file = init_log(log_filename, n, p, p_int, job_args, job_args_num);

  // set-up beta_sequence struct
  float *beta_cache = NULL;
  int *index_cache = NULL;
  Beta_Sequence beta_sequence;
  if (use_adaptive_calibration) {
    Rprintf("Using Adaptive Calibration\n");
    beta_sequence.count = 0;
    beta_sequence.vec_length = p_int;
    beta_sequence.betas = malloc(max_lambda_count * sizeof(Beta_Sequence));
    beta_sequence.lambdas = malloc(max_lambda_count * sizeof(float));
    beta_cache = malloc(p_int * sizeof(beta));
    index_cache = malloc(p_int * sizeof(int));
  }

  // int set_step_size
  for (; lambda > final_lambda; iter++) {
    // save current beta values to log each iteration
    if (log_level == ITER)
      save_log(iter, lambda, lambda_count, beta, p_int, log_file);
    prev_error = error;
    float dBMax = 0.0; // largest beta diff this cycle

    // update intercept (don't for the moment, it should be 0 anyway)
    // intercept = update_intercept_cyclic(intercept, X, Y, beta, n, p);

    // update the predictor \Beta_k
    // TODO: shuffling is single-threaded and significantly-ish (40s -> 60s? on
    // the workstation) slows things down.
    // it might be possible to do something better than this.
    if (set_min_lambda == TRUE) {
      parallel_shuffle(iter_permutation, permutation_split_size,
                       final_split_size, permutation_splits);
    }
#pragma omp parallel for num_threads(NumCores) private(max_rowsums, max_cumulative_rowsums) shared(col_ysum, xmatrix, X2, Y, rowsum, beta, precalc_get_num) reduction(+:total_updates, skipped_updates, skipped_updates_entries, total_updates_entries, error, num_nz_beta) reduction(max: dBMax) schedule(static) reduction(-:became_zero)
    for (long i = 0; i < p_int; i++) {
      long k = iter_permutation->data[i];

      // TODO: in principle this is a problem if beta is ever set back to zero,
      // but that rarely/never happens.
      int was_zero = FALSE;
      if (beta[k] == 0.0) {
        was_zero = TRUE;
      }
      Changes changes = update_beta_cyclic_old(
          X2, Y, rowsum, n, p, lambda, beta, k, intercept, precalc_get_num,
          thread_column_caches[omp_get_thread_num()]);
      float diff = changes.actual_diff;
      // TODO: kills performance
      // if (fabs(diff) < lambda)
      //	diff=0.0;
      if (was_zero && diff != 0) {
        num_nz_beta++;
      }
      if (!was_zero && beta[k] == 0) {
        became_zero++;
      }
      float diff2 = diff * diff;
      if (diff2 > dBMax) {
        dBMax = diff2;
      }
      total_updates++;
      total_updates_entries += X2.cols[k].nz;
    }

    if (!set_min_lambda) {
      if (fabs(dBMax) > 0) {
        set_min_lambda = TRUE;
        // final_lambda = (pow(0.9,50))*lambda;
        Rprintf("first change at lambda %f, stopping at lambda %f\n", lambda,
                final_lambda);
      } else {
        Rprintf("done lambda %d (%f) after %d iteration(s) (dbmax: %f), final "
                "error %.1f\n",
                lambda_count, lambda, iter + 1, dBMax, error);
        lambda_count++;
        lambda *= 0.9;
        iter_count += iter;
        iter = -1;
      }
    } else {
      error = calculate_error(n, p_int, X2, Y, X, beta, p, intercept, rowsum);

      // Be sure to clean up anything extra we allocate
      if (prev_error / error < halt_beta_diff || iter == max_iter) {
        if (prev_error / error < halt_beta_diff) {
          Rprintf("largest change (%f) was less than %f, halting after %d "
                  "iterations\n",
                  prev_error / error, halt_beta_diff, iter + 1);
          Rprintf("done lambda %d (%f) after %d iteration(s) (dbmax: %f), "
                  "final error %.1f\n",
                  lambda_count, lambda, iter + 1, dBMax, error);
        } else {
          Rprintf("stopping after iter (%d) >= max_iter (%d) iterations\n",
                  iter + 1, max_iter);
        }

        if (log_level == LAMBDA)
          save_log(iter, lambda, lambda_count, beta, p_int, log_file);

        Rprintf("%d nz beta\n", num_nz_beta);
        if (max_nz_beta > 0 && num_nz_beta - became_zero >= max_nz_beta) {
          Rprintf("Maximum non-zero beta count reached, stopping after this "
                  "lambda");
          final_lambda = lambda;
        }
        if (use_adaptive_calibration) {
          if (set_min_lambda == TRUE) {
            Sparse_Betas sparse_betas;
            int count = 0;
            for (int b = 0; b < p_int; b++) {
              if (beta[b] != 0) {
                beta_cache[count] = beta[b];
                index_cache[count] = b;
                count++;
              }
            }
            sparse_betas.betas = malloc(count * sizeof(float));
            sparse_betas.indices = malloc(count * sizeof(int));
            memcpy(sparse_betas.betas, beta_cache, count * sizeof(float));
            memcpy(sparse_betas.indices, index_cache, count * sizeof(int));

            sparse_betas.count = count;

            beta_sequence.lambdas[beta_sequence.count] = lambda;
            beta_sequence.betas[beta_sequence.count] = sparse_betas;
            beta_sequence.count++;

            if (check_adaptive_calibration(0.75, beta_sequence, n)) {
              printf("Halting as reccommended by adaptive calibration\n");
              final_lambda = lambda;
            }
          }
        }

        lambda_count++;
        lambda *= 0.9;
        iter_count += iter;
        iter = -1;
      }
    }
  }
  if (log_level != NONE)
    close_log(log_file);
  Rprintf("\nfinished at lambda = %f\n", lambda);
  Rprintf("after %d total iters\n", iter_count);

  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  cpu_time_used = ((float)(end.tv_nsec - start.tv_nsec)) / 1e9 +
                  (end.tv_sec - start.tv_sec);

  Rprintf("lasso done in %.4f seconds\n", cpu_time_used);
  // Rprintf("lasso done in %.4f seconds, columns skipped %ld out of %ld a.k.a
  // (%f%%)\n", cpu_time_used, skipped_updates, total_updates,
  // (skipped_updates*100.0)/((long)total_updates)); Rprintf("cols: performed
  // %ld zero updates (%f%%)\n", zero_updates,
  // ((float)zero_updates/(total_updates))
  // * 100); Rprintf("skipped entries %ld out of %ld a.k.a (%f%%)\n",
  // skipped_updates_entries, total_updates_entries,
  // (skipped_updates_entries*100.0)/((long)total_updates_entries));
  free(precalc_get_num);
  // Rprintf("entries: performed %d zero updates (%f%%)\n",
  // zero_updates_entries, ((float)zero_updates_entries/(total_updates_entries))
  // * 100);

  // TODO: this really should be 0. Fix things until it is.
  Rprintf("checking how much rowsums have diverged:\n");
  float *temp_rowsum = malloc(n * sizeof(float));
  memset(temp_rowsum, 0, n * sizeof(float));
  for (long col = 0; col < p_int; col++) {
    int entry = -1;
    for (int i = 0; i < X2.cols[col].nwords; i++) {
      S8bWord word = X2.cols[col].compressed_indices[i];
      unsigned long values = word.values;
      for (int j = 0; j <= group_size[word.selector]; j++) {
        int diff = values & masks[word.selector];
        if (diff != 0) {
          entry += diff;
          temp_rowsum[entry] += beta[col];
        }
        values >>= item_width[word.selector];
      }
    }
  }
  float total_rowsum_diff = 0;
  float frac_rowsum_diff = 0;
  for (int i = 0; i < n; i++) {
    total_rowsum_diff += fabs((temp_rowsum[i] - rowsum[i]));
    if (fabs(rowsum[i]) > 1)
      frac_rowsum_diff += fabs((temp_rowsum[i] - rowsum[i]) / rowsum[i]);
  }
  Rprintf("mean diff: %.2f (%.2f%%)\n", total_rowsum_diff / n,
          (frac_rowsum_diff * 100));
  free(temp_rowsum);

  if (use_adaptive_calibration) {
    for (int i = 0; i < beta_sequence.count; i++) {
      free(beta_sequence.betas[i].betas);
      free(beta_sequence.betas[i].indices);
    }
    free(beta_sequence.betas);
    free(beta_sequence.lambdas);

    free(beta_cache);
    free(index_cache);
  }

  // free beta sets
  // free X2
  free_sparse_matrix(X2);
  free(col_ysum);
  gsl_permutation_free(iter_permutation);
  gsl_rng_free(iter_rng);
  for (int i = 0; i < max_num_threads; i++) {
    free(thread_column_caches[i]);
  }
  free(thread_column_caches);
  free(rowsum);

  printf("checking nz beta count\n");
  int nonzero = 0;
  for (int i = 0; i < p_int; i++) {
    if (beta[i] != 0) {
      nonzero++;
    }
  }
  printf("%d found\n", nonzero);
  printf("nz = %d, became_zero = %d\n", num_nz_beta, became_zero);

  return beta;
}

static int firstchanged = FALSE;

Changes update_beta_cyclic_old(XMatrixSparse xmatrix_sparse, float *Y,
                               float *rowsum, int n, int p, float lambda,
                               float *beta, long k, float intercept,
                               int_pair *precalc_get_num,
                               int *column_entry_cache) {
  float sumk = xmatrix_sparse.cols[k].nz;
  float sumn = xmatrix_sparse.cols[k].nz * beta[k];
  // float sumk = col.nz;
  // float sumn = col.nz * beta[k];
  int *column_entries = column_entry_cache;

  // if (k==2905) {
  //	printf("sumn: %f\n", sumn);
  //}
  long col_entry_pos = 0;
  long entry = -1;
  for (int i = 0; i < xmatrix_sparse.cols[k].nwords; i++) {
    S8bWord word = xmatrix_sparse.cols[k].compressed_indices[i];
    unsigned long values = word.values;
    for (int j = 0; j <= group_size[word.selector]; j++) {
      int diff = values & masks[word.selector];
      if (diff != 0) {
        entry += diff;
        column_entries[col_entry_pos] = entry;
        sumn += intercept - rowsum[entry];
        // if (k==4147) {
        //	printf("sumn += rowsum[%d] =  %f\n", entry, rowsum[entry]);
        //}
        col_entry_pos++;
      }
      values >>= item_width[word.selector];
    }
  }

  //if (k == interesting_col) {
  //    printf("lambda * n / 2 = %f\n", lambda * n / 2);
  //    printf("sumn: %f\n", sumn);
  //}
  // if (k==4147) {
  //	printf("sumn: %f\n", sumn);
  //	printf("sumn -bk: %f\n", sumn - sumk*beta[k]);
  //}

  // TODO: This is probably slower than necessary.
  float Bk_diff = beta[k];
  if (sumk == 0.0) {
    beta[k] = 0.0;
  } else {
    beta[k] = soft_threshold(sumn, lambda * n / 2) / sumk;
  }
  Bk_diff = beta[k] - Bk_diff;
  // update every rowsum[i] w/ effects of beta change.
  if (Bk_diff != 0) {
    if (!firstchanged) {
      firstchanged = TRUE;
      printf("first changed on col %ld (%d,%d), lambda %f ******************\n", k, precalc_get_num[k].i, precalc_get_num[k].j, lambda);
    }
    for (int e = 0; e < xmatrix_sparse.cols[k].nz; e++) {
      int i = column_entries[e];
#pragma omp atomic
      rowsum[i] += Bk_diff;
    }
  } else {
    zero_updates++;
    zero_updates_entries += xmatrix_sparse.cols[k].nz;
  }

  Changes changes;
  changes.actual_diff = Bk_diff;
  changes.pre_lambda_diff = sumn;

  return changes;
}
Changes update_beta_cyclic(S8bCol col, float *Y, float *rowsum, int n, int p,
                           float lambda, float *beta, long k,
                           float intercept, int_pair *precalc_get_num,
                           int *column_entry_cache) {
  float sumk = col.nz;
  float sumn = col.nz * beta[k];
  int *column_entries = column_entry_cache;

  long col_entry_pos = 0;
  long entry = -1;
  for (int i = 0; i < col.nwords; i++) {
    alignas(64) S8bWord word = col.compressed_indices[i];
    unsigned long values = word.values;
    for (int j = 0; j <= group_size[word.selector]; j++) {
      int diff = values & masks[word.selector];
      if (diff != 0) {
        entry += diff;
        column_entries[col_entry_pos] = entry;
        sumn += intercept - rowsum[entry];
        col_entry_pos++;
      }
      values >>= item_width[word.selector];
    }
  }

  float Bk_diff = beta[k];
  if (sumk == 0.0) {
    beta[k] = 0.0;
  } else {
    beta[k] = soft_threshold(sumn, lambda * n / 2) / sumk;
  }
  Bk_diff = beta[k] - Bk_diff;
  // update every rowsum[i] w/ effects of beta change.
  if (Bk_diff != 0) {
    for (int e = 0; e < col.nz; e++) {
      int i = column_entries[e];
#pragma omp atomic
      rowsum[i] += Bk_diff;
    }
  } else {
    zero_updates++;
    zero_updates_entries += col.nz;
  }

  Changes changes;
  changes.actual_diff = Bk_diff;
  changes.pre_lambda_diff = sumn;

  return changes;
}

float update_intercept_cyclic(float intercept, int **X, float *Y,
                               float *beta, int n, int p) {
  float new_intercept = 0.0;
  float sumn = 0.0, sumx = 0.0;

  for (int i = 0; i < n; i++) {
    sumx = 0.0;
    for (int j = 0; j < p; j++) {
      sumx += X[i][j] * beta[j];
    }
    sumn += Y[i] - sumx;
  }
  new_intercept = sumn / n;
  return new_intercept;
}
