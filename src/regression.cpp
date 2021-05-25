#include "liblasso.h"
#include <gsl/gsl_complex.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>

using namespace std;

struct timespec start_time, end_time;
long total_beta_updates = 0;
long total_beta_nz_updates = 0;

// check a particular pair of betas in the adaptive calibration scheme
int adaptive_calibration_check_beta(float c_bar, float lambda_1,
                                    Sparse_Betas *beta_1, float lambda_2,
                                    Sparse_Betas *beta_2, int beta_length,
                                    int n) {
  float max_diff = 0.0;
  float adjusted_max_diff = 0.0;

  int b1_ind = 0;
  int b2_ind = 0;

  while (b1_ind < beta_1->count && b2_ind < beta_2->count) {
    while (beta_1->indices[b1_ind] < beta_2->indices[b2_ind] &&
           b1_ind < beta_1->count)
      b1_ind++;
    while (beta_2->indices[b2_ind] < beta_1->indices[b1_ind] &&
           b2_ind < beta_2->count)
      b2_ind++;
    if (b1_ind < beta_1->count && b2_ind < beta_2->count &&
        beta_1->indices[b1_ind] == beta_2->indices[b2_ind]) {
      float diff = fabs(beta_1->betas.at(b1_ind) - beta_2->betas.at(b2_ind));
      if (diff > max_diff)
        max_diff = diff;
      b1_ind++;
    }
  }

  // adjusted_max_diff = max_diff / ((lambda_1 + lambda_2) * (n / 2));
  adjusted_max_diff = (double)max_diff / (((double)lambda_1 + (double)lambda_2));

  // printf("adjusted_max_diff: %f\n", adjusted_max_diff);
  if (adjusted_max_diff <= c_bar) {
    return 1;
  }
  return 0;
}

// checks whether the last element in the beta_sequence is the one we should
// stop at, according to Chichignoud et als 'Adaptive Calibration Scheme'
// returns TRUE if we are finished, FALSE if we should continue.
int check_adaptive_calibration(float c_bar, Beta_Sequence *beta_sequence,
                               int n) {
  // printf("\nchecking %d betas\n", beta_sequence.count);
  for (int i = 0; i < beta_sequence->count; i++) {
    int this_result = adaptive_calibration_check_beta(
        c_bar, beta_sequence->lambdas[beta_sequence->count - 1],
        &beta_sequence->betas[beta_sequence->count - 1], beta_sequence->lambdas[i],
        &beta_sequence->betas[i], beta_sequence->vec_length, n);
    // printf("result: %d\n", this_result);
    if (this_result == 0) {
      return TRUE;
    }
  }
  return FALSE;
}


float calculate_error(int n, long p_int, float *Y, int **X,
                       robin_hood::unordered_flat_map<long, float> beta, float p, float intercept,
                       float *rowsum) {
  float error = 0.0;
  for (int row = 0; row < n; row++) {
    float row_err = intercept - rowsum[row];
    error += row_err * row_err;
  }
  return error;
}

static float halt_error_diff;

int run_lambda_iters_pruned(Iter_Vars *vars, float lambda, float *rowsum,
                            float *old_rowsum, Active_Set *active_set, struct OpenCL_Setup* ocl_setup) {
  XMatrixSparse Xc = vars->Xc;
  float **last_rowsum = vars->last_rowsum;
  Thread_Cache *thread_caches = vars->thread_caches;
  int n = vars->n;
  robin_hood::unordered_flat_map<long, float> *beta = vars->beta;
  float *last_max = vars->last_max;
  bool *wont_update = vars->wont_update;
  int p = vars->p;
  int p_int = vars->p_int;
  //XMatrixSparse X2c = vars->X2c;
  float *Y = vars->Y;
  float *max_int_delta = vars->max_int_delta;
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
  //  for (int i = 0; i < p; i++) {
  //    new_active_branch[i] = FALSE;
  //  }

  float error = 0.0;
  for (int i = 0; i < n; i++) {
    error += rowsum[i] * rowsum[i];
  }

  // allocate a local copy of rowsum for each thread
  int **thread_rowsums[NumCores];
  //#pragma omp parallel for
  //  for (int i = 0; i < NumCores; i++) {
  //    int *tr = malloc(sizeof *rowsum * n + 64);
  //    memcpy(tr, rowsum, sizeof *rowsum * n);
  //    thread_rowsums[omp_get_thread_num()] = tr;
  //  }

  if (VERBOSE)
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
  for (int retests = 0; retests < 1; retests++) {
    if (VERBOSE)
      printf("test %d\n", retests + 1);
    long total_changed = 0;
    long total_unchanged = 0;
    int total_changes = 0;
    int total_present = 0;
    int total_notpresent = 0;
// memset(max_int_delta, 0, sizeof *max_int_delta * p);
// memset(last_max, 0, sizeof *last_max * p);
//#pragma omp parallel for schedule(static)
//    for (int i = 0; i < p; i++) {
//      max_int_delta[i] = 0;
//      last_max[i] = 0;
//    }

    //********** Branch Pruning       *******************
    if (VERBOSE)
      printf("branch pruning.\n");
    long active_branches = 0;
    long new_active_branches = 0;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);
    // make a local copy of rowsum for each thread
    //#pragma omp parallel for
    //    for (int i = 0; i < NumCores; i++) {
    //      memcpy(thread_rowsums[omp_get_thread_num()], rowsum, sizeof *rowsum
    //      * n);
    //    }

#pragma omp parallel for schedule(static) reduction(+ : new_active_branches)
    //#pragma omp target teams distribute parallel for schedule(static) \
//      reduction(+ : new_active_branches) map(to: Xc.cols[0:p], lambda, \
//      last_max[0:p], last_rowsum[0:p][0:n], rowsum[0:n], beta) map(from: \
//      wont_update[0:n])
    for (int j = 0; j < p; j++) {
      bool old_wont_update = wont_update[j];
      wont_update[j] =
           wont_update_effect(Xc, lambda, j, last_max[j], last_rowsum[j],
           rowsum,
                             thread_caches[omp_get_thread_num()].col_j);
          //wont_update_effect(Xc, lambda, j, last_max[j], last_rowsum[j], rowsum,
          //                   NULL, beta);
      char new_active_branch = old_wont_update && !wont_update[j];
      if (new_active_branch)
        new_active_branches++;
    }
    // this slows things down on multiple numa nodes. There must be something
    // going on with rowsum/last_rowsum?
    // #pragma omp threadprivate(local_rowsum) num_threads(NumCores)
    // #pragma omp parallel num_threads(NumCores) shared(last_rowsum)
    {
// int *local_rowsum = malloc(n * sizeof *rowsum);
// printf("local_rowsum: %x\n");
// memcpy(local_rowsum, rowsum, n * sizeof *rowsum);
// TODO: parallelising this loop slows down numa updates.
#pragma omp parallel for schedule(static) reduction(+ : active_branches, used_branches, pruned_branches)
      for (int j = 0; j < p; j++) {
        // if the branch hasn't been pruned then we'll get an accurate estimate
        // for this rowsum from update_working_set.
        if (!wont_update[j]) {
          memcpy(last_rowsum[j], rowsum,
                 sizeof *rowsum * n); // TODO: probably overkill
          active_branches++;
          used_branches++;
        } else {
          pruned_branches++;
        }
      }
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_time);
    pruning_time += ((float)(end_time.tv_nsec - start_time.tv_nsec)) / 1e9 +
                    (end_time.tv_sec - start_time.tv_sec);
    if (VERBOSE)
      printf("(%ld active branches, %ld new)\n", active_branches,
            new_active_branches);
    if (new_active_branches == 0) {
      break;
    }
    //********** Identify Working Set *******************
    // TODO: is it worth constructing a new set with no 'blank'
    // elements?
    // if (VERBOSE)
      printf("updating working set.\n");
    int count_may_update = 0;
    int* updateable_items = calloc(p, sizeof *updateable_items); //TODO: keep between iters
    for (int i = 0; i < p; i++) {
        if (!wont_update[i] && !active_set->entries[i].present) {
          updateable_items[count_may_update] = i;
          count_may_update++;
          printf("%d ", i);
        }
    }
    // if (VERBOSE)
      printf("\nthere were %d updateable items\n", count_may_update);
    clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);
    char increased_set =
        update_working_set(vars->Xu, Xc, rowsum, wont_update, p, n, lambda, beta, updateable_items, count_may_update, active_set, thread_caches, ocl_setup, last_max);
    free(updateable_items);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_time);
    working_set_update_time +=
        ((float)(end_time.tv_nsec - start_time.tv_nsec)) / 1e9 +
        (end_time.tv_sec - start_time.tv_sec);
    if (retests > 0 && !increased_set) {
      // there's no need to re-run on the same set. Nothing has changed
      // and the remaining retests will all do nothing.
      if (VERBOSE)
        printf("didn't increase set, no further iters\n");
      break;
    }
    if (VERBOSE)
      printf("active set size: %d, or %.2f \%\n", active_set->length,
             100 * (float)active_set->length / (float)p_int);
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
    if (VERBOSE)
     printf("solving subproblem.\n");
    clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);
    int iter = 0;
    for (iter = 0; iter < 100; iter++) {
      float prev_error = error;
      // update entire working set
      // TODO: we should shuffle the active set, not the matrix
      // if (active_set->length > NumCores) {
      //  parallel_shuffle(perm, permutation_split_size, final_split_size,
      //                   permutation_splits);
      //}
      // parallel_shuffle(iter_permutation, permutation_split_size,
      //                 final_split_size, permutation_splits);
// #pragma omp parallel for num_threads(NumCores) schedule(static) shared(Y, rowsum, beta, precalc_get_num, perm) reduction(+:total_unchanged, total_changed, total_present, total_notpresent, new_nz_beta, total_beta_updates, total_beta_nz_updates)
      for (auto it = active_set->entries.begin(); it != active_set->entries.end(); it++) {
        long k = it->first;
        AS_Entry entry = it->second;
        if (entry.present) {
          // TODO: apply permutation here.
          total_present++;
          int was_zero = TRUE;
          auto count = beta->count(k);
          // printf("found %d entries\n", count);
          if ( count > 0) {
            // printf("found %d entries for key %d\n", count, k);
            if (beta->at(k) != 0.0) {
              was_zero = FALSE;
            }
          }
          total_beta_updates++;
          Changes changes = update_beta_cyclic(
              active_set->entries[k].col, Y, rowsum, n, p, lambda, beta, k, 0,
              precalc_get_num, thread_caches[omp_get_thread_num()].col_i);
          if (changes.actual_diff == 0.0) {
            total_unchanged++;
          } else {
            total_beta_nz_updates++;
            total_changed++;
          }
          if (was_zero && changes.actual_diff != 0) {
            new_nz_beta++;
          }
          if (!was_zero && beta->contains(k) && beta->at(k) == 0) {
            new_nz_beta--;
          }
          } else {
            total_notpresent++;
          }
      }
      //for (int i = 0; i < p; i++) {
      //  for (int j = i; j < p; j++) {
      //    //TODO: this loop could be better. we don't need to check everything if we use a hashset of active columns.
      //    //int k = (2 * (p - 1) + 2 * (p - 1) * (i - 1) - (i - 1) * (i - 1) -
      //    //         (i - 1)) /
      //    //            2 +
      //    //        j;
      //    // int k = pair_to_val(std::make_tuple(i, j), p);
      //    int k = triplet_to_val(std::make_tuple(i, j, j), p);
      //    if (i == interesting_col && j == interesting_col) {
      //      printf("checking subproblem for interesting col %d, (%d,%d)\n", k, i, j);
      //    }
      //    if (active_set->entries.contains(k) && active_set->entries[k].present) {
      //      if (i == interesting_col && j == interesting_col) {
      //        printf("present in active set\n");
      //      }
      //      // TODO: apply permutation here.
      //      total_present++;
      //      int was_zero = TRUE;
      //      auto count = beta->count(k);
      //      // printf("found %d entries\n", count);
      //      if ( count > 0) {
      //        // printf("found %d entries for key %d\n", count, k);
      //        if (beta->at(k) != 0.0) {
      //          was_zero = FALSE;
      //        }
      //      }
      //      total_beta_updates++;
      //      Changes changes = update_beta_cyclic(
      //          active_set->entries[k].col, Y, rowsum, n, p, lambda, beta, k, 0,
      //          precalc_get_num, thread_caches[omp_get_thread_num()].col_i);
      //      if (changes.actual_diff == 0.0) {
      //        total_unchanged++;
      //      } else {
      //        total_beta_nz_updates++;
      //        total_changed++;
      //      }
      //      if (was_zero && changes.actual_diff != 0) {
      //        new_nz_beta++;
      //      }
      //      if (!was_zero && beta->contains(k) && beta->at(k) == 0) {
      //        new_nz_beta--;
      //      }
      //    } else {
      //      total_notpresent++;
      //    }
      //  }
      //}
      //printf("total beta updates: %ld\n", total_beta_updates);
      // for (int ki = 0; ki < active_set->length; ki++) {
      //  int k = active_set->entries[perm->data[ki]];
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
      for (int i = 0; i < n; i++) {
        error += rowsum[i] * rowsum[i];
      }
      error = sqrt(error);
      if (prev_error / error < halt_error_diff) {
        printf("done after %d iters\n", lambda, iter+1);
        break;
      }
    }
    // printf("active set length: %d, present: %d not: %d\n",
    // active_set->length, total_present, total_notpresent);
    // g_assert_true(total_present/iter+total_notpresent/iter ==
    // active_set->length-1);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_time);
    subproblem_time += ((float)(end_time.tv_nsec - start_time.tv_nsec)) / 1e9 +
                       (end_time.tv_sec - start_time.tv_sec);
    // printf("%.1f%% of active set updates didn't change\n",
    // (float)(total_changed*100)/(float)(total_changed+total_unchanged));
    // printf("%.1f%% of active set was blank\n",
    // (float)total_present/(float)(total_present+total_notpresent));
    if (active_set->length > 0) {
      gsl_permutation_free(perm);
    }
  }

  //#pragma omp parallel for
  //  for (int i = 0; i < NumCores; i++) {
  //    free(thread_rowsums[omp_get_thread_num()]);
  //  }

  // free(new_active_branch);
  gsl_rng_free(rng);
  return new_nz_beta;
}

robin_hood::unordered_flat_map<long, float> simple_coordinate_descent_lasso(
    XMatrix xmatrix, float *Y, int n, int p, long max_interaction_distance,
    float lambda_min, float lambda_max, int max_iter, int verbose,
    float frac_overlap_allowed, float hed,
    enum LOG_LEVEL log_level, char **job_args, int job_args_num,
    int use_adaptive_calibration, int max_nz_beta) {
  halt_error_diff = hed;
  printf("using halt_error_diff of %f\n", halt_error_diff);
  long num_nz_beta = 0;
  long became_zero = 0;
  float lambda = lambda_max;
  VERBOSE = verbose;
  int_pair *precalc_get_num;
  int **X = xmatrix.X;

  // Rprintf("using %d threads\n", NumCores);

  // XMatrixSparse X2 = sparse_X2_from_X(X, n, p, max_interaction_distance, FALSE);
  XMatrixSparse Xc = sparsify_X(X, n, p);
  struct X_uncompressed Xu = construct_host_X(&Xc);

  for (int i = 0; i < NUM_MAX_ROWSUMS; i++) {
    max_rowsums[i] = 0;
    max_cumulative_rowsums[i] = 0;
  }

  long p_int = get_p_int(p, max_interaction_distance);
  if (max_interaction_distance == -1) {
    max_interaction_distance = p_int / 2 + 1;
  }
  if (max_nz_beta < 0)
    max_nz_beta = p_int;
  robin_hood::unordered_flat_map<long, float> beta;
  //beta = malloc(p_int * sizeof(float)); // probably too big in most cases.
  //memset(beta, 0, p_int * sizeof(float));

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

  float error = 0.0;
  for (int i = 0; i < n; i++) {
    error += Y[i] * Y[i];
  }
  float intercept = 0.0;

  float *rowsum = (float*)calloc(n, sizeof *rowsum);
  for (int i = 0; i < n; i++)
    rowsum[i] = -Y[i];

  //colsum = malloc(p_int * sizeof(float));
  //memset(colsum, 0, p_int * sizeof(float));

  //col_ysum = malloc(p_int * sizeof(float));
  //memset(col_ysum, 0, p_int * sizeof(float));
  //for (int col = 0; col < p_int; col++) {
  //  int *col_entries = &Xu.host_X[Xu.host_col_offsets[col]];
  //  for (int i = 0; i < n; i++) {
  //    int entry = col_entries[i];
  //    col_ysum[col] += Y[entry];
  //  }
  //  //int entry = -1;
  //  //for (int i = 0; i < X2.cols[col].nwords; i++) {
  //  //  S8bWord word = X2.cols[col].compressed_indices[i];
  //  //  unsigned long values = word.values;
  //  //  for (int j = 0; j <= group_size[word.selector]; j++) {
  //  //    int diff = values & masks[word.selector];
  //  //    if (diff != 0) {
  //  //      entry += diff;
  //  //      col_ysum[col] += Y[entry];
  //  //    }
  //  //    values >>= item_width[word.selector];
  //  //  }
  //  //}
  //}

  // find largest number of non-zeros in any column
  int largest_col = 0;
  long total_col = 0;
  for (int i = 0; i < p; i++) {
    int col_size = Xu.host_col_nz[i];
    if (col_size > largest_col) {
      largest_col = col_size;
    }
    total_col += col_size;
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
  //gsl_rng_env_setup();
  //const gsl_rng_type *T = gsl_rng_default;
  //parallel_shuffle(iter_permutation, permutation_split_size, final_split_size,
  //                 permutation_splits);
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  float final_lambda = lambda_min;
  int max_lambda_count = max_iter;
  if (VERBOSE)
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
    //for (int col = 0; col < p_int; col++) {
    //  int *col_entries = &Xu.host_X[Xu.host_col_offsets[col]];
    //  for (int i = 0; i < n; i++) {
    //    int entry = col_entries[i];
    //    rowsum[entry] += beta[col];
    //  }
    //}
    for (int col_i = 0; col_i < p_int; col_i++) {
      int *col_i_entries = &Xu.host_X[Xu.host_col_offsets[col_i]];
      for (int i = 0; i < n; i++) {
        int row = col_i_entries[i];
        int *inter_row = &Xu.host_X_row[Xu.host_row_offsets[row]];
        int row_nz = Xu.host_row_nz[row];
        for (int col_j_ind = 0; col_j_ind < row_nz; col_j_ind++) {
          int col_j = inter_row[col_j_ind];
          int k = (2 * (p - 1) + 2 * (p - 1) * (col_i - 1) - (col_i - 1) * (col_i - 1) - (col_i - 1)) / 2 + col_j;
          rowsum[row] += beta[k];
        }
      }
    }
    //for (int col = 0; col < p_int; col++) {
    //  int entry = -1;
    //  for (int i = 0; i < X2.cols[col].nwords; i++) {
    //    S8bWord word = X2.cols[col].compressed_indices[i];
    //    unsigned long values = word.values;
    //    for (int j = 0; j <= group_size[word.selector]; j++) {
    //      int diff = values & masks[word.selector];
    //      if (diff != 0) {
    //        entry += diff;
    //        rowsum[entry] += beta[col];
    //      }
    //      values >>= item_width[word.selector];
    //    }
    //  }
    //}
  } else {
    Rprintf("no partial log for current job found\n");
  }
  if (log_level != NONE)
    log_file = init_log(log_filename, n, p, p_int, job_args, job_args_num);

  // set-up beta_sequence struct
  robin_hood::unordered_flat_map<long, float> beta_cache;
  int *index_cache = NULL;
  Beta_Sequence beta_sequence;
  if (use_adaptive_calibration) {
    Rprintf("Using Adaptive Calibration\n");
    beta_sequence.count = 0;
    // beta_sequence.vec_length = p_int;
    beta_sequence.betas = malloc(max_lambda_count * sizeof(Beta_Sequence));
    beta_sequence.lambdas = malloc(max_lambda_count * sizeof(float));
    // beta_cache = (float*)malloc(p_int * sizeof(float));
    // index_cache = (int*)malloc(p_int * sizeof(int));
  }


  float **last_rowsum = (float**)malloc(sizeof *last_rowsum * p);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < p; i++) {
    last_rowsum[i] = (float*)malloc(sizeof *last_rowsum[i] * n + PADDING);
    memset(last_rowsum[i], 0, sizeof *last_rowsum[i] * n);
  }
  Thread_Cache thread_caches[NumCores];

  for (int i = 0; i < NumCores; i++) {
    thread_caches[i].col_i = (int*)malloc(sizeof(int) * max(n,p));
    thread_caches[i].col_j = (int*)malloc(sizeof(int) * n);
    // robin_hood::unordered_flat_map<long, float> tmp;

    // thread_caches[i].lf_map = tmp; 
  }

  float *last_max = new float[n];
  bool *wont_update = new bool[p];
  memset(last_max, 0, n*sizeof(*last_max));
  float *max_int_delta = (float*)malloc(sizeof *max_int_delta * p);
  memset(max_int_delta, 0, sizeof *max_int_delta * p);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < p; i++) {
    memset(last_rowsum[i], 0, sizeof *last_rowsum[i] * n);
    last_max[i] = 0.0;
    max_int_delta[i] = 0;
  }
  for (int i = 0; i < n; i++) {
    rowsum[i] = -Y[i];
  }
  XMatrixSparse X2c_fake;
  Iter_Vars iter_vars_pruned = {
      Xc,
      last_rowsum,
      thread_caches,
      n,
      &beta,
      last_max,
      wont_update,
      p,
      p_int,
      X2c_fake,
      Y,
      max_int_delta,
      precalc_get_num,
      iter_permutation,
      Xu,
  };
  long nz_beta = 0;
  // struct OpenCL_Setup ocl_setup = setup_working_set_kernel(Xu, n, p);
  struct OpenCL_Setup ocl_setup;
  Active_Set active_set = active_set_new(p_int);
  float *old_rowsum = (float*)malloc(sizeof *old_rowsum * n);
  printf("final_lambda: %f\n", final_lambda);
  error = calculate_error(n, p_int, Y, X, beta, p, intercept, rowsum);
  printf("initial error: %f\n", error);
  // lambda = 100;
  // final_lambda = lambda - 1; //TODO: only for testing
  // for (; lambda > final_lambda && iter < max_iter; iter++) {
  while (lambda > final_lambda && iter < max_lambda_count) {
    //#pragma omp parallel for schedule(static) reduction(+:nz_beta)
    //for (int i = 0; i < p_int; i++) {
    //  if ( beta[i] != 0) {
    //    nz_beta++;
    //  }
    //}
    if (nz_beta >= max_nz_beta) {
      printf("reached max_nz_beta of %d\n", max_nz_beta);
      break;
    }
    // float lambda = lambda_sequence[lambda_ind];
    if (VERBOSE)
      printf("lambda: %f\n", lambda);
    float dBMax;
    // TODO: implement working set and update test
    int last_iter_count = 0;

    // nz_beta += run_lambda_iters_pruned(&iter_vars_pruned, lambda, rowsum);
    nz_beta += run_lambda_iters_pruned(&iter_vars_pruned, lambda, rowsum, old_rowsum,
                            &active_set, &ocl_setup);
    double prev_error = error;
    error = calculate_error(n, p_int, Y, X, beta, p, intercept, rowsum);
    printf("lambda %d = %f, error %.1f, nz_beta %ld\n", iter, lambda, error, nz_beta);
    if (use_adaptive_calibration && nz_beta > 0) {
        Sparse_Betas *sparse_betas = &beta_sequence.betas[beta_sequence.count];
        int count = 0;
        //TODO: it should be possible to do something more like memcpy here
        for (auto c = beta.begin(); c != beta.end(); c++) {
          sparse_betas->betas.insert_or_assign(c->first, c->second);
          count++;
        }
        //for (int b = 0; b < p_int; b++) {
        //  if ( beta[b] != 0) {
        //    beta_cache[count] = beta[b];
        //    index_cache[count] = b;
        //    count++;
        //  }
        //}
        //if (count != nz_beta) {
        // printf("count (%d) or nz_beta (%d) is wrong\n", count, nz_beta);
        //}
        // sparse_betas.betas = (float*)malloc(count * sizeof(float));
        // sparse_betas.indices = (int*)malloc(count * sizeof(int));
        // memcpy(sparse_betas.betas, beta_cache, count * sizeof(float));
        // memcpy(sparse_betas.indices, index_cache, count * sizeof(int));
        // sparse_betas.betas.

        sparse_betas->count = count;

        if (beta_sequence.count >= max_lambda_count) {
          printf("allocated too many beta sequences for adaptive calibration, things will now break. ***************************************\n");
        }
        beta_sequence.lambdas[beta_sequence.count] = lambda;
        // beta_sequence.betas[beta_sequence.count] = sparse_betas;
        beta_sequence.count++;

        printf("checking adaptive cal\n");
        if (check_adaptive_calibration(0.75, &beta_sequence, n)) {
         printf("Halting as reccommended by adaptive calibration\n");
         final_lambda = lambda;
        }
    }
    lambda *= 0.95;
    if (nz_beta > 0) {
      iter++;
    }
  }
  iter_count = iter;
  // int set_step_size
  //for (; lambda > final_lambda; iter++) {
  //  // save current beta values to log each iteration
  //  if (log_level == ITER)
  //    save_log(iter, lambda, lambda_count, beta, p_int, log_file);
  //  prev_error = error;
  //  float dBMax = 0.0; // largest beta diff this cycle

  //  // update intercept (don't for the moment, it should be 0 anyway)
  //  // intercept = update_intercept_cyclic(intercept, X, Y, beta, n, p);

  //  // update the predictor \Beta_k
  //  // TODO: shuffling is single-threaded and significantly-ish (40s -> 60s? on
  //  // the workstation) slows things down.
  //  // it might be possible to do something better than this.
  //  if (set_min_lambda == TRUE) {
  //    parallel_shuffle(iter_permutation, permutation_split_size,
  //                     final_split_size, permutation_splits);
  //  }
//#pragma omp parallel for num_threads(NumCores) private(max_rowsums, max_cumulative_rowsums) shared(col_ysum, xmatrix, X2, Y, rowsum, beta, precalc_get_num) reduction(+:total_updates, skipped_updates, skipped_updates_entries, total_updates_entries, error, num_nz_beta) reduction(max: dBMax) schedule(static) reduction(-:became_zero)
  //  for (long i = 0; i < p_int; i++) {
  //    long k = iter_permutation->data[i];

  //    // TODO: in principle this is a problem if beta is ever set back to zero,
  //    // but that rarely/never happens.
  //    int was_zero = FALSE;
  //    if ( beta[k] == 0.0) {
  //      was_zero = TRUE;
  //    }
  //    Changes changes = update_beta_cyclic_old(
  //        X2, Y, rowsum, n, p, lambda, beta, k, intercept, precalc_get_num,
  //        thread_column_caches[omp_get_thread_num()]);
  //    float diff = changes.actual_diff;
  //    // TODO: kills performance
  //    // if (fabs(diff) < lambda)
  //    //	diff=0.0;
  //    if (was_zero && diff != 0) {
  //      num_nz_beta++;
  //    }
  //    if (!was_zero && beta[k] == 0) {
  //      became_zero++;
  //    }
  //    float diff2 = diff * diff;
  //    if (diff2 > dBMax) {
  //      dBMax = diff2;
  //    }
  //    total_updates++;
  //    total_updates_entries += X2.cols[k].nz;
  //  }

  //  if (!set_min_lambda) {
  //    if (fabs(dBMax) > 0) {
  //      set_min_lambda = TRUE;
  //      // final_lambda = (pow(0.9,50))*lambda;
  //      Rprintf("first change at lambda %f, stopping at lambda %f\n", lambda,
  //              final_lambda);
  //    } else {
  //      Rprintf("done lambda %d (%f) after %d iteration(s) (dbmax: %f), final "
  //              "error %.1f\n",
  //              lambda_count, lambda, iter + 1, dBMax, error);
  //      lambda_count++;
  //      lambda *= 0.9;
  //      iter_count += iter;
  //      iter = -1;
  //    }
  //  } else {
  //    error = calculate_error(n, p_int, X2, Y, X, beta, p, intercept, rowsum);

  //    // Be sure to clean up anything extra we allocate
  //    if (prev_error / error < halt_beta_diff || iter == max_iter) {
  //      if (prev_error / error < halt_beta_diff) {
  //        Rprintf("largest change (%f) was less than %f, halting after %d "
  //                "iterations\n",
  //                prev_error / error, halt_beta_diff, iter + 1);
  //        Rprintf("done lambda %d (%f) after %d iteration(s) (dbmax: %f), "
  //                "final error %.1f\n",
  //                lambda_count, lambda, iter + 1, dBMax, error);
  //      } else {
  //        Rprintf("stopping after iter (%d) >= max_iter (%d) iterations\n",
  //                iter + 1, max_iter);
  //      }

  //      if (log_level == LAMBDA)
  //        save_log(iter, lambda, lambda_count, beta, p_int, log_file);

  //      Rprintf("%d nz beta\n", num_nz_beta);
  //      if (max_nz_beta > 0 && num_nz_beta - became_zero >= max_nz_beta) {
  //        Rprintf("Maximum non-zero beta count reached, stopping after this "
  //                "lambda");
  //        final_lambda = lambda;
  //      }
  //      if (use_adaptive_calibration) {
  //        if (set_min_lambda == TRUE) {
  //          Sparse_Betas sparse_betas;
  //          int count = 0;
  //          for (int b = 0; b < p_int; b++) {
  //            if ( beta[b] != 0) {
  //              beta_cache[count] = beta[b];
  //              index_cache[count] = b;
  //              count++;
  //            }
  //          }
  //          sparse_betas.betas = malloc(count * sizeof(float));
  //          sparse_betas.indices = malloc(count * sizeof(int));
  //          memcpy(sparse_betas.betas, beta_cache, count * sizeof(float));
  //          memcpy(sparse_betas.indices, index_cache, count * sizeof(int));

  //          sparse_betas.count = count;

  //          beta_sequence.lambdas[beta_sequence.count] = lambda;
  //          beta_sequence.betas[beta_sequence.count] = sparse_betas;
  //          beta_sequence.count++;

  //          if (check_adaptive_calibration(0.75, beta_sequence, n)) {
  //            printf("Halting as reccommended by adaptive calibration\n");
  //            final_lambda = lambda;
  //          }
  //        }
  //      }

  //      lambda_count++;
  //      lambda *= 0.9;
  //      iter_count += iter;
  //      iter = -1;
  //    }
  //  }
  //}
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
  float *temp_rowsum = calloc(n, sizeof *temp_rowsum);
  for (int col_i = 0; col_i < p; col_i++) {
    int *col_i_entries = &Xu.host_X[Xu.host_col_offsets[col_i]];
    for (int i = 0; i < Xu.host_col_nz[col_i]; i++) {
      int row = col_i_entries[i];
      int *inter_row = &Xu.host_X_row[Xu.host_row_offsets[row]];
      int row_nz = Xu.host_row_nz[row];
      for (int col_j_ind = 0; col_j_ind < row_nz; col_j_ind++) {
        int col_j = inter_row[col_j_ind];
        long k = (2 * (p - 1) + 2 * (p - 1) * (col_i - 1) - (col_i - 1) * (col_i - 1) - (col_i - 1)) / 2 + col_j;
        temp_rowsum[row] += beta[k];
      }
    }
  }
  //for (long col = 0; col < p_int; col++) {
  //  int entry = -1;
  //  for (int i = 0; i < X2.cols[col].nwords; i++) {
  //    S8bWord word = X2.cols[col].compressed_indices[i];
  //    unsigned long values = word.values;
  //    for (int j = 0; j <= group_size[word.selector]; j++) {
  //      int diff = values & masks[word.selector];
  //      if (diff != 0) {
  //        entry += diff;
  //        temp_rowsum[entry] += beta[col];
  //      }
  //      values >>= item_width[word.selector];
  //    }
  //  }
  //}
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
      beta_sequence.betas[i].betas.clear();
      // free(beta_sequence.betas[i].betas);
      free(beta_sequence.betas[i].indices);
    }
    free(beta_sequence.betas);
    free(beta_sequence.lambdas);

    // free(beta_cache);
    // free(index_cache);
  }

  // free beta sets
  // free X2
  free_sparse_matrix(Xc);
  // free(col_ysum);
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
    if ( (beta)[i] != 0) {
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
                               robin_hood::unordered_flat_map<long, float> *beta, long k, float intercept,
                               int_pair *precalc_get_num,
                               int *column_entry_cache) {
  float sumk = xmatrix_sparse.cols[k].nz;
  float sumn = xmatrix_sparse.cols[k].nz * beta->at(k);
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
  //	printf("sumn -bk: %f\n", sumn - sumkbeta[k]);
  //}

  // TODO: This is probably slower than necessary.
  float Bk_diff = beta->at(k);
  if (sumk == 0.0) {
    // beta[k] = 0.0;
  } else {
    beta->insert_or_assign(k, soft_threshold(sumn, lambda ) / sumk);
  }
  Bk_diff = beta->at(k) - Bk_diff;
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
                           float lambda, robin_hood::unordered_flat_map<long, float> *beta, long k,
                           float intercept, int_pair *precalc_get_num,
                           int *column_entry_cache) {
  float sumk = col.nz;
  float bk = 0.0;
  if (beta->contains(k)) {
    bk = beta->at(k);
  }
  float sumn = col.nz * bk;
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

  float new_value =  soft_threshold(sumn, lambda) / sumk;
  if (k == triplet_to_val(std::make_tuple(interesting_col, interesting_col, interesting_col), p)) {
    printf("interesting col sumn: %f, new_values: %f\n", sumn, new_value);
  }
  auto tp = val_to_triplet(k, p);
  if (sumk == 0.0) {
    // beta[k] = 0.0;
  } else {
    printf("assigning %f to: %ld,%ld,%ld\n", new_value, std::get<0>(tp), std::get<1>(tp), std::get<2>(tp));
    beta->insert_or_assign(k, new_value);
  }
  float Bk_diff = new_value - bk;
  // update every rowsum[i] w/ effects of beta change.
  if (Bk_diff != 0) {
    printf("nz beta update for: %ld,%ld,%ld\n", std::get<0>(tp), std::get<1>(tp), std::get<2>(tp));
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
                               robin_hood::unordered_flat_map<long, float> beta, int n, int p) {
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
