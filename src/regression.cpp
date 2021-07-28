#include "liblasso.h"
#include "robin_hood.h"
#include <cmath>
#include <limits>
#ifdef NOT_R
#include <glib-2.0/glib.h>
#endif
#include <gsl/gsl_complex.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <random>
#include <vector>

using namespace std;

struct timespec start_time, end_time;
long total_beta_updates = 0;
long total_beta_nz_updates = 0;

static float c_bar = 0.75;
// static float c_bar = 0.001;
// static float c_bar = 750;

void check_beta_order(robin_hood::unordered_flat_map<long, float>* beta,
    long p)
{
    for (auto it = beta->begin(); it != beta->end(); it++) {
        long value = it->first;
        float bv = it->second;
        auto tuple = val_to_triplet(value, p);
        long a = std::get<0>(tuple);
        long b = std::get<1>(tuple);
        long c = std::get<2>(tuple);

        if (a > b || b > c) {
            printf("problem! %ld,%ld,%ld: %f\n", a, b, c, bv);
        }
#ifdef NOT_R
        g_assert_true(a <= b);
        g_assert_true(b <= c);
#endif
    }
}

// check a particular pair of betas in the adaptive calibration scheme
long adaptive_calibration_check_beta(float c_bar, float lambda_1,
    Sparse_Betas* beta_1, float lambda_2,
    Sparse_Betas* beta_2, long n)
{
    float max_diff = 0.0;
    float adjusted_max_diff = 0.0;

    long b1_count = 0;
    long b2_count = 0;

    long b1_ind = 0;
    long b2_ind = 0;

    // advance whichever is smaller, accounting for overlap
    while (b1_count < beta_1->count && b2_count < beta_2->count) {
        float b1v = beta_1->values[b1_count];
        float b2v = beta_2->values[b2_count];
        long b1_ind = beta_1->indices[b1_count];
        long b2_ind = beta_2->indices[b2_count];

        if (b1_ind < b2_ind) {
            float diff = fabs(b1v);
            if (diff > max_diff)
                max_diff = diff;
            b1_count++;
        } else if (b2_ind < b1_ind) {
            float diff = fabs(b2v);
            if (diff > max_diff)
                max_diff = diff;
            b2_count++;
        } else if (b1_ind == b2_ind) {
            float diff = fabs(b1v - b2v);
            if (diff > max_diff)
                max_diff = diff;
            b1_count++;
            b2_count++;
        }
    }
    // remaining b1
    while (b1_count < beta_1->count) {
        float b1v = beta_1->values[b1_ind];
        float diff = fabs(b1v);
        if (diff > max_diff)
            max_diff = diff;
        b1_count++;
    }
    // remaining b2
    while (b2_count < beta_2->count) {
        float b2v = beta_2->values[b2_ind];
        float diff = fabs(b2v);
        if (diff > max_diff)
            max_diff = diff;
        b2_count++;
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
long check_adaptive_calibration(float c_bar, Beta_Sequence* beta_sequence,
    long n)
{
    // printf("\nchecking %ld betas\n", beta_sequence.count);
    for (long i = 0; i < beta_sequence->count; i++) {
        long this_result = adaptive_calibration_check_beta(
            c_bar, beta_sequence->lambdas[beta_sequence->count - 1],
            &beta_sequence->betas[beta_sequence->count - 1],
            beta_sequence->lambdas[i], &beta_sequence->betas[i], n);
        // printf("result: %ld\n", this_result);
        if (this_result == 0) {
            return TRUE;
        }
    }
    return FALSE;
}

float calculate_error(float* Y, float* rowsum, long n)
{
    float error = 0.0;
    for (int row = 0; row < n; row++) {
        float row_err = -rowsum[row];
        error += row_err * row_err;
    }
    return error;
}

static float halt_error_diff;
static auto rng = std::default_random_engine();

long run_lambda_iters_pruned(Iter_Vars* vars, float lambda, float* rowsum,
    float* old_rowsum, Active_Set* active_set,
    struct OpenCL_Setup* ocl_setup, long depth)
{
    XMatrixSparse Xc = vars->Xc;
    X_uncompressed Xu = vars->Xu;
    float** last_rowsum = vars->last_rowsum;
    Thread_Cache* thread_caches = vars->thread_caches;
    long n = vars->n;
    Beta_Value_Sets* beta_sets = vars->beta_sets;
    robin_hood::unordered_flat_map<long, float>* beta1 = &beta_sets->beta1;
    robin_hood::unordered_flat_map<long, float>* beta2 = &beta_sets->beta2;
    robin_hood::unordered_flat_map<long, float>* beta3 = &beta_sets->beta3;
    robin_hood::unordered_flat_map<long, float>* beta = &beta_sets->beta3; // TODO: dont
    float* last_max = vars->last_max;
    bool* wont_update = vars->wont_update;
    long p = vars->p;
    long p_int = vars->p_int;
    float* Y = vars->Y;
    float* max_int_delta = vars->max_int_delta;
    int_pair* precalc_get_num = vars->precalc_get_num;
    long new_nz_beta = 0;
    //gsl_permutation* iter_permutation = vars->iter_permutation;
    // gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
    //gsl_permutation* perm;

    float error = 0.0;
    for (long i = 0; i < n; i++) {
        error += rowsum[i] * rowsum[i];
    }

    if (VERBOSE)
        printf("\nrunning lambda %f w/ depth %ld\n", lambda, depth);
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
    for (long retests = 0; retests < 1; retests++) {
        if (VERBOSE)
            printf("test %ld\n", retests + 1);
        long total_changed = 0;
        long total_unchanged = 0;
        long total_changes = 0;
        long total_present = 0;
        long total_notpresent = 0;

        //********** Branch Pruning       *******************
        if (VERBOSE)
            printf("branch pruning.\n");
        long active_branches = 0;
        long new_active_branches = 0;

        clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);

#pragma omp parallel for schedule(static) reduction(+ \
                                                    : new_active_branches)
        for (long j = 0; j < p; j++) {
            bool old_wont_update = wont_update[j];
            wont_update[j] = wont_update_effect(Xu, lambda, j, last_max[j], last_rowsum[j], rowsum,
                thread_caches[omp_get_thread_num()].col_j);
            char new_active_branch = old_wont_update && !wont_update[j];
            if (new_active_branch)
                new_active_branches++;
        }
        // this slows things down on multiple numa nodes. There must be something
        // going on with rowsum/last_rowsum?
        // #pragma omp threadprivate(local_rowsum) num_threads(NumCores)
        // #pragma omp parallel num_threads(NumCores) shared(last_rowsum)
        {
// TODO: parallelising this loop slows down numa updates.
#pragma omp parallel for schedule(static) reduction(+ \
                                                    : active_branches, used_branches, pruned_branches)
            for (long j = 0; j < p; j++) {
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
            //// we'll also update last_rowums for the active set
            //for (auto it = active_set->entries2.begin(); it != active_set->entries2.end(); it++) {
            //    AS_Entry *entry = &it->second;
            //    memcpy(entry->last_rowsum, rowsum, sizeof *rowsum * n);
            //}
        }
        clock_gettime(CLOCK_MONOTONIC_RAW, &end_time);
        pruning_time += ((float)(end_time.tv_nsec - start_time.tv_nsec)) / 1e9 + (end_time.tv_sec - start_time.tv_sec);
        if (VERBOSE)
            printf("(%ld active branches, %ld new)\n", active_branches,
                new_active_branches);
        // if (new_active_branches == 0) {
        //  break;
        //}
        //********** Identify Working Set *******************
        // TODO: is it worth constructing a new set with no 'blank'
        // elements?
        if (VERBOSE)
            printf("updating working set.\n");
        long count_may_update = 0;
        long* updateable_items = calloc(p, sizeof *updateable_items); // TODO: keep between iters
        for (long i = 0; i < p; i++) {
            // if (!wont_update[i] && !active_set_present(active_set, i)) {
            if (!wont_update[i]) {
                updateable_items[count_may_update] = i;
                count_may_update++;
                // printf("%ld ", i);
            }
        }

        if (VERBOSE)
            printf("\nthere were %ld updateable items\n", count_may_update);
        clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);
        char increased_set = update_working_set(vars->Xu, Xc, rowsum, wont_update, p, n, lambda,
            beta, updateable_items, count_may_update, active_set,
            thread_caches, ocl_setup, last_max, depth);
        free(updateable_items);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end_time);
        working_set_update_time += ((float)(end_time.tv_nsec - start_time.tv_nsec)) / 1e9 + (end_time.tv_sec - start_time.tv_sec);
        if (retests > 0 && !increased_set) {
            // there's no need to re-run on the same set. Nothing has changed
            // and the remaining retests will all do nothing.
            if (VERBOSE)
                printf("didn't increase set, no further iters\n");
            break;
        }
        //********** Solve subproblem     *******************
        if (VERBOSE)
            printf("active set size: %ld, or %.2f \%\n", active_set->length,
                100 * (float)active_set->length / (float)p_int);
        if (VERBOSE)
            printf("solving subproblem.\n");
        clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);

        std::vector<std::pair<long, AS_Entry>> entry_vec1;
        entry_vec1.reserve(active_set->entries1.size());
        std::vector<std::pair<long, AS_Entry>> entry_vec2;
        entry_vec2.reserve(active_set->entries2.size());
        std::vector<std::pair<long, AS_Entry>> entry_vec3;
        entry_vec3.reserve(active_set->entries3.size());

        for (auto& e : active_set->entries1)
            entry_vec1.push_back(std::make_pair(e.first, e.second));
        for (auto& e : active_set->entries2)
            entry_vec2.push_back(std::make_pair(e.first, e.second));
        for (auto& e : active_set->entries3)
            entry_vec3.push_back(std::make_pair(e.first, e.second));

        std::shuffle(entry_vec1.begin(), entry_vec1.end(), rng);
        std::shuffle(entry_vec2.begin(), entry_vec2.end(), rng);
        std::shuffle(entry_vec3.begin(), entry_vec3.end(), rng);

        auto run_beta = [&](auto* current_beta_set, auto& as_entries) {
            for (const auto& it : as_entries) {
                long k = std::get<0>(it);
                AS_Entry entry = std::get<1>(it);
                if (entry.present) {
                    total_present++;
                    long was_zero = TRUE;
                    if (current_beta_set->contains(k) && fabs(current_beta_set->at(k)) != 0.0)
                        was_zero = FALSE;
                    total_beta_updates++;
                    Changes changes = update_beta_cyclic(
                        entry.col, Y, rowsum, n, p, lambda, current_beta_set, k, 0,
                        precalc_get_num, thread_caches[omp_get_thread_num()].col_i);
                    if (changes.actual_diff == 0.0) {
                        total_unchanged++;
                    } else {
                        total_beta_nz_updates++;
                        total_changed++;
                    }
                    if (was_zero && fabs(changes.actual_diff) > (double)0.0) {
                        new_nz_beta++;
                    }
                    if (!was_zero && changes.removed) {
                        new_nz_beta--;
                    }
                } else {
                    total_notpresent++;
                }
            }
        };
        long iter = 0;
        for (iter = 0; iter < 100; iter++) {
            if (VERBOSE)
                printf("iter %ld\n", iter);
            float prev_error = error;
            // update entire working set
            // #pragma omp parallel for num_threads(NumCores) schedule(static)
            // shared(Y, rowsum, beta, precalc_get_num, perm)
            // reduction(+:total_unchanged, total_changed, total_present,
            // total_notpresent, new_nz_beta, total_beta_updates,
            // total_beta_nz_updates)

            auto print_entries = [&](robin_hood::unordered_flat_map<long, AS_Entry>* entries,
                                     robin_hood::unordered_flat_map<long, float>* current_beta_set) {
                printf("set contains: { ");
                for (auto it = entries->begin(); it != entries->end(); it++) {
                    printf("%ld, ", it->first);
                }
                printf("}\n");
            };

            run_beta(&beta_sets->beta1, entry_vec1);
            run_beta(&beta_sets->beta2, entry_vec2);
            run_beta(&beta_sets->beta3, entry_vec3);

            // check whether we need another iteration
            error = 0.0;
            for (long i = 0; i < n; i++) {
                error += rowsum[i] * rowsum[i];
            }
            error = sqrt(error);
            if (VERBOSE)
                printf("error: %f\n", error);
            if (prev_error / error < halt_error_diff) {
                if (VERBOSE)
                    printf("done after %ld iters\n", lambda, iter + 1);
                break;
            }
        }
        // printf("active set length: %ld, present: %ld not: %ld\n",
        // active_set->length, total_present, total_notpresent);
        // g_assert_true(total_present/iter+total_notpresent/iter ==
        // active_set->length-1);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end_time);
        subproblem_time += ((float)(end_time.tv_nsec - start_time.tv_nsec)) / 1e9 + (end_time.tv_sec - start_time.tv_sec);
        // printf("%.1f%% of active set updates didn't change\n",
        // (float)(total_changed*100)/(float)(total_changed+total_unchanged));
        // printf("%.1f%% of active set was blank\n",
        // (float)total_present/(float)(total_present+total_notpresent));
        //if (active_set->length > 0) {
        //    gsl_permutation_free(perm);
        //}
    }

    // gsl_rng_free(rng);
    if (VERBOSE)
        printf("new nz beta: %ld\n", new_nz_beta);
    return new_nz_beta;
}

long copy_beta_sets(Beta_Value_Sets* from_set, Sparse_Betas* to_set)
{
    long count = 0;
    auto from_beta = from_set->beta1;
    long total_size = from_set->beta1.size() + from_set->beta2.size() + from_set->beta3.size();
    to_set->indices = new long[total_size];
    to_set->values = new float[total_size];
    for (auto c = from_beta.begin(); c != from_beta.end(); c++) {
        // to_set->betas.insert_or_assign(c->first, c->second);
        to_set->indices[count] = c->first;
        to_set->values[count] = c->second;
        count++;
    }
    from_beta = from_set->beta2;
    for (auto c = from_beta.begin(); c != from_beta.end(); c++) {
        // to_set->betas.insert_or_assign(c->first, c->second);
        to_set->indices[count] = c->first;
        to_set->values[count] = c->second;
        count++;
    }
    from_beta = from_set->beta3;
    for (auto c = from_beta.begin(); c != from_beta.end(); c++) {
        // to_set->betas.insert_or_assign(c->first, c->second);
        to_set->indices[count] = c->first;
        to_set->values[count] = c->second;
        count++;
    }
    to_set->count = count;
    return count;
}

double phi_inv(double x)
{
    return std::sqrt(std::abs(2.0 * std::log(sqrt(2.0 * M_PI) * x)));
}

float total_sqrt_error = 0.0;
Beta_Value_Sets simple_coordinate_descent_lasso(
    XMatrix xmatrix, float* Y, long n, long p, long max_interaction_distance,
    float lambda_min, float lambda_max, long max_iter, long verbose,
    float frac_overlap_allowed, float hed, enum LOG_LEVEL log_level,
    const char** job_args, long job_args_num, long use_adaptive_calibration,
    long mnz_beta, const char* log_filename, long depth)
{
    long max_nz_beta = mnz_beta;
    printf("n: %ld, p: %ld\n", n, p);
    halt_error_diff = hed;
    printf("using halt_error_diff of %f\n", halt_error_diff);
    printf("using depth: %ld\n", depth);
    long num_nz_beta = 0;
    long became_zero = 0;
    float lambda = lambda_max;
    VERBOSE = verbose;
    int_pair* precalc_get_num;
    long** X = xmatrix.X;

    long real_p_int = -1;
    switch (depth) {
    case 1:
        real_p_int = (long)p;
    case 2:
        real_p_int = (long)p * ((long)p + 1) / 2;
    case 3:
        real_p_int = (long)p * ((long)p + 1) * ((long)p - 1) / (2 * 3);
        break;
    }
    double tmp = 396.952547477011765511e-3;
    printf("ϕ⁻¹(~396.95e-3) = %f\n", phi_inv(tmp));
    printf("p = %ld, phi_inv(0.95/(2.0 * p)) = %f\n", real_p_int, phi_inv(0.95 / (2.0 * (double)real_p_int)));
    double final_lambda = 1.1 * std::sqrt((double)n) * phi_inv(0.95 / (2.0 * (double)real_p_int));
    final_lambda /= n; // not very well justified, but seems like it might be helping.
    // final_lambda /= std::sqrt(n); // not very well justified, but seems like it might be helping.
    printf("using final lambda: %f\n", final_lambda);

    // work out min lambda for sqrt lasso

    // Rprintf("using %ld threads\n", NumCores);

    // XMatrixSparse X2 = sparse_X2_from_X(X, n, p, max_interaction_distance,
    // FALSE);
    XMatrixSparse Xc
        = sparsify_X(X, n, p);
    struct X_uncompressed Xu = construct_host_X(&Xc);

    for (long i = 0; i < NUM_MAX_ROWSUMS; i++) {
        max_rowsums[i] = 0;
        max_cumulative_rowsums[i] = 0;
    }

    long p_int = get_p_int(p, max_interaction_distance);
    if (max_interaction_distance == -1) {
        max_interaction_distance = p_int / 2 + 1;
    }
    if (max_nz_beta < 0)
        max_nz_beta = p_int;
    robin_hood::unordered_flat_map<long, float> beta1;
    robin_hood::unordered_flat_map<long, float> beta2;
    robin_hood::unordered_flat_map<long, float> beta3;
    Beta_Value_Sets beta_sets = { beta1, beta2, beta3, p };
    // beta = malloc(p_int * sizeof(float)); // probably too big in most cases.
    // memset(beta, 0, p_int * sizeof(float));

    precalc_get_num = malloc(p_int * sizeof(int_pair));
    long offset = 0;
    for (long i = 0; i < p; i++) {
        for (long j = i; j < min((long)p, i + max_interaction_distance + 1); j++) {
            // printf("i,j: %ld,%ld\n", i, j);
            precalc_get_num[offset]
                .i
                = i;
            precalc_get_num[offset]
                .j
                = j;
            offset++;
        }
    }

    //cached_nums = get_all_nums(p, max_interaction_distance);

    float error = 0.0;
    for (long i = 0; i < n; i++) {
        error += Y[i] * Y[i];
    }
    float intercept = 0.0;

    float* rowsum = (float*)calloc(n, sizeof *rowsum);
    for (long i = 0; i < n; i++)
        rowsum[i] = -Y[i];

    // find largest number of non-zeros in any column
    long largest_col = 0;
    long total_col = 0;
    for (long i = 0; i < p; i++) {
        long col_size = Xu.host_col_nz[i];
        if (col_size > largest_col) {
            largest_col = col_size;
        }
        total_col += col_size;
    }
    long main_sum = 0;
    for (long i = 0; i < p; i++)
        for (long j = 0; j < n; j++)
            main_sum += X[i][j];

    struct timespec start, end;
    float cpu_time_used;

    long set_min_lambda = FALSE;
    //gsl_permutation* iter_permutation = gsl_permutation_alloc(p_int);
    gsl_rng* iter_rng;
    //gsl_permutation_init(iter_permutation);
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    // final_lambda = lambda_min;
    long max_lambda_count = max_iter;
    if (VERBOSE)
        Rprintf("running from lambda %.2f to lambda %.2f\n", lambda, final_lambda);
    long lambda_count = 1;
    long iter_count = 0;

    long max_num_threads = omp_get_max_threads();
    long** thread_column_caches = (long**)malloc(max_num_threads * sizeof *thread_column_caches);
    for (long i = 0; i < max_num_threads; i++) {
        thread_column_caches[i] = (long*)malloc(largest_col * sizeof *thread_column_caches[i]);
    }

    FILE* log_file;
    long iter = 0;
    if (log_level != NONE && check_can_restore_from_log(log_filename, n, p, p_int, job_args, job_args_num)) {
        Rprintf("We can restore from a partial log!\n");
        restore_from_log(log_filename, true, n, p, job_args, job_args_num, &iter,
            &lambda_count, &lambda, &beta_sets);
        // we need to recalculate the rowsums
        auto update_beta_set = [&](auto beta_set) {
            for (auto it = beta_set->begin(); it != beta_set->end(); it++) {
                long val = it->first;
                float bv = it->second;
                auto update_columns = [&](long a, long b, long c) {
                    long* colA = &Xu.host_X[Xu.host_col_offsets[a]];
                    long* colB = &Xu.host_X[Xu.host_col_offsets[b]];
                    long* colC = &Xu.host_X[Xu.host_col_offsets[c]];
                    long ib = 0, ic = 0;
                    for (long ia = 0; ia < Xu.host_col_nz[a]; ia++) {
                        long cur_row = colA[ia];
                        while (colB[ib] < cur_row && ib < Xu.host_col_nz[b] - 1)
                            ib++;
                        while (colC[ic] < cur_row && ic < Xu.host_col_nz[c] - 1)
                            ic++;
                        if (cur_row == colB[ib] && cur_row == colC[ic]) {
                            rowsum[cur_row] += bv;
                        }
                    }
                };
                if (val < p) {
                    update_columns(val, val, val);
                } else if (val < p * p) {
                    auto pair = val_to_pair(val, p);
                    long a = std::get<0>(pair);
                    long b = std::get<1>(pair);
                    update_columns(a, a, b);
                } else {
                    if (val >= p * p * p) {
                        printf("broken val: %ld\n", val);
                    }
                    auto triple = val_to_triplet(val, p);
                    long a = std::get<0>(triple);
                    long b = std::get<1>(triple);
                    long c = std::get<2>(triple);
                    update_columns(a, b, c);
                }
            }
        };

        update_beta_set(&beta_sets.beta1);
        update_beta_set(&beta_sets.beta2);
        update_beta_set(&beta_sets.beta3);
    } else {
        Rprintf("no partial log for current job found\n");
    }
    if (log_level != NONE)
        log_file = init_log(log_filename, n, p, p_int, job_args, job_args_num);

    // set-up beta_sequence struct
    robin_hood::unordered_flat_map<long, float> beta_cache;
    long* index_cache = NULL;
    Beta_Sequence beta_sequence;
    if (use_adaptive_calibration) {
        Rprintf("Using Adaptive Calibration\n");
        beta_sequence.count = 0;
        beta_sequence.betas = (Sparse_Betas*)malloc(max_lambda_count * sizeof *beta_sequence.betas);
        beta_sequence.lambdas = (float*)malloc(max_lambda_count * sizeof *beta_sequence.lambdas);
    }

    float** last_rowsum = (float**)malloc(sizeof *last_rowsum * p);
#pragma omp parallel for schedule(static)
    for (long i = 0; i < p; i++) {
        last_rowsum[i] = (float*)malloc(sizeof *last_rowsum[i] * n + PADDING);
        memset(last_rowsum[i], 0, sizeof *last_rowsum[i] * n);
    }
    Thread_Cache thread_caches[NumCores];

    for (long i = 0; i < NumCores; i++) {
        thread_caches[i].col_i = (long*)malloc(max(n, p) * sizeof *thread_caches[i].col_i);
        thread_caches[i].col_j = (long*)malloc(n * sizeof *thread_caches[i].col_j);
    }

    float* last_max = new float[p];
    bool* wont_update = new bool[p];
    memset(last_max, 0, p * sizeof(*last_max));
    float* max_int_delta = (float*)malloc(sizeof *max_int_delta * p);
    memset(max_int_delta, 0, sizeof *max_int_delta * p);
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
        &beta_sets,
        last_max,
        wont_update,
        p,
        p_int,
        X2c_fake,
        Y,
        max_int_delta,
        //precalc_get_num,
        NULL,
        NULL,
        Xu,
    };
    long nz_beta = 0;
    // struct OpenCL_Setup ocl_setup = setup_working_set_kernel(Xu, n, p);
    struct OpenCL_Setup ocl_setup;
    Active_Set active_set = active_set_new(p_int, p);
    float* old_rowsum = (float*)malloc(sizeof *old_rowsum * n);
    printf("final_lambda: %f\n", final_lambda);
    error = calculate_error(Y, rowsum, n);
    total_sqrt_error = std::sqrt(error);
    printf("initial error: %f\n", error);
    while (lambda > final_lambda && iter < max_lambda_count) {
        if (log_level == LAMBDA) {
            save_log(0, lambda, lambda_count, &beta_sets, log_file);
        }
        if (nz_beta >= max_nz_beta) {
            printf("reached max_nz_beta of %ld\n", max_nz_beta);
            break;
        }
        // float lambda = lambda_sequence[lambda_ind];
        if (VERBOSE)
            printf("lambda: %f\n", lambda);
        float dBMax;
        // TODO: implement working set and update test
        long last_iter_count = 0;

        if (VERBOSE)
            printf("nz_beta %ld\n", nz_beta);
        nz_beta += run_lambda_iters_pruned(&iter_vars_pruned, lambda, rowsum,
            old_rowsum, &active_set, &ocl_setup, depth);

        {
            long nonzero = beta_sets.beta1.size() + beta_sets.beta2.size() + beta_sets.beta3.size();
            //if (nonzero != nz_beta) { // TODO: debugging only, disable for release.
            //    printf("nonzero %ld == nz_beta %ld ?\n", nonzero, nz_beta);
            //    printf("beta 1 contains: { ");
            //    for (auto it = beta_sets.beta1.begin(); it != beta_sets.beta1.end();
            //         it++) {
            //        printf("%ld[%.2f], ", it->first, beta_sets.beta1.at(it->first));
            //    }
            //    printf("}\n");
            //    printf("beta 2 contains: { ");
            //    for (auto it = beta_sets.beta2.begin(); it != beta_sets.beta2.end();
            //         it++) {
            //        printf("%ld[%.2f], ", it->first, beta_sets.beta2.at(it->first));
            //    }
            //    printf("}\n");
            //    printf("beta 3 contains: { ");
            //    for (auto it = beta_sets.beta3.begin(); it != beta_sets.beta3.end();
            //         it++) {
            //        printf("%ld[%.2f], ", it->first, beta_sets.beta3.at(it->first));
            //    }
            //    printf("}\n");
            //}
#ifdef NOT_R
            g_assert_true(nonzero == nz_beta);
#endif
        }
        double prev_error = error;
        error = calculate_error(Y, rowsum, n);
        total_sqrt_error = std::sqrt(error);
        printf("lambda %ld = %f, error %.4e, nz_beta %ld,%ld,%ld\n", lambda_count, lambda, error,
            beta_sets.beta1.size(), beta_sets.beta2.size(), beta_sets.beta3.size());
        if (use_adaptive_calibration && nz_beta > 0) {
            Sparse_Betas* sparse_betas = &beta_sequence.betas[beta_sequence.count];
            // TODO: it should be possible to do something more like memcpy here
            copy_beta_sets(&beta_sets, sparse_betas);

            if (beta_sequence.count >= max_lambda_count) {
                printf(
                    "allocated too many beta sequences for adaptive calibration, "
                    "things will now break. ***************************************\n");
            }
            beta_sequence.lambdas[beta_sequence.count] = lambda;
            beta_sequence.count++;

            if (VERBOSE)
                printf("checking adaptive cal\n");
            if (check_adaptive_calibration(c_bar, &beta_sequence, n)) {
                printf("Halting as reccommended by adaptive calibration\n");
                final_lambda = lambda;
            }
        }
        lambda *= 0.95;
        if (nz_beta > 0) {
            iter++;
        }
        lambda_count++;
    }
    iter_count = iter;
    if (log_level != NONE)
        close_log(log_file);
    Rprintf("\nfinished at lambda = %f\n", lambda);
    Rprintf("after %ld total iters\n", iter_count);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    cpu_time_used = ((float)(end.tv_nsec - start.tv_nsec)) / 1e9 + (end.tv_sec - start.tv_sec);

    Rprintf("lasso done in %.4f seconds\n", cpu_time_used);
    //free(precalc_get_num);

    // TODO: this really should be 0. Fix things until it is.
    // Rprintf("checking how much rowsums have diverged:\n");
    // float *temp_rowsum = calloc(n, sizeof *temp_rowsum);
    // for (long col_i = 0; col_i < p; col_i++) {
    //  long *col_i_entries = &Xu.host_X[Xu.host_col_offsets[col_i]];
    //  for (long i = 0; i < Xu.host_col_nz[col_i]; i++) {
    //    long row = col_i_entries[i];
    //    long *inter_row = &Xu.host_X_row[Xu.host_row_offsets[row]];
    //    long row_nz = Xu.host_row_nz[row];
    //    for (long col_j_ind = 0; col_j_ind < row_nz; col_j_ind++) {
    //      long col_j = inter_row[col_j_ind];
    //      long k = (2 * (p - 1) + 2 * (p - 1) * (col_i - 1) - (col_i - 1) *
    //      (col_i - 1) - (col_i - 1)) / 2 + col_j; temp_rowsum[row] += beta[k];
    //    }
    //  }
    //}
    // for (long col = 0; col < p_int; col++) {
    //  long entry = -1;
    //  for (long i = 0; i < X2.cols[col].nwords; i++) {
    //    S8bWord word = X2.cols[col].compressed_indices[i];
    //    unsigned long values = word.values;
    //    for (long j = 0; j <= group_size[word.selector]; j++) {
    //      long diff = values & masks[word.selector];
    //      if (diff != 0) {
    //        entry += diff;
    //        temp_rowsum[entry] += beta[col];
    //      }
    //      values >>= item_width[word.selector];
    //    }
    //  }
    //}
    // float total_rowsum_diff = 0;
    // float frac_rowsum_diff = 0;
    // for (long i = 0; i < n; i++) {
    //  total_rowsum_diff += fabs((temp_rowsum[i] - rowsum[i]));
    //  if (fabs(rowsum[i]) > 1)
    //    frac_rowsum_diff += fabs((temp_rowsum[i] - rowsum[i]) / rowsum[i]);
    //}
    // Rprintf("mean diff: %.2f (%.2f%%)\n", total_rowsum_diff / n,
    //        (frac_rowsum_diff * 100));
    // free(temp_rowsum);

    if (use_adaptive_calibration) {
        for (long i = 0; i < beta_sequence.count; i++) {
            // beta_sequence.betas[i].betas.clear();
            delete[] beta_sequence.betas[i].indices;
            delete[] beta_sequence.betas[i].values;
        }
        free(beta_sequence.betas);
        free(beta_sequence.lambdas);
    }

    // free beta sets
    free_sparse_matrix(Xc);
    //gsl_permutation_free(iter_permutation);
    gsl_rng_free(iter_rng);
    for (long i = 0; i < max_num_threads; i++) {
        free(thread_column_caches[i]);
    }
    free(thread_column_caches);
    free(rowsum);

    printf("checking nz beta count\n");
    // for (auto it = beta.begin(); it != beta.end(); it++) {
    //  if (it->second != 0.0)
    //    nonzero++;
    //}
    long nonzero = beta_sets.beta1.size() + beta_sets.beta2.size() + beta_sets.beta3.size();
    // for debugging:
    // for (auto it = beta_sets.beta1.begin(); it != beta_sets.beta1.end(); it++)
    //  g_assert_true(it->second != 0.0);
    // for (auto it = beta_sets.beta2.begin(); it != beta_sets.beta2.end(); it++)
    //  g_assert_true(it->second != 0.0);
    // for (auto it = beta_sets.beta3.begin(); it != beta_sets.beta3.end(); it++)
    //  g_assert_true(it->second != 0.0);
    printf("%ld found\n", nonzero);
    printf("nz = %ld, became_zero = %ld\n", num_nz_beta,
        became_zero); // TODO: disable for release
    free_host_X(&Xu);

    for (int i = 0; i < NumCores; i++) {
        free(thread_caches[i].col_i);
        free(thread_caches[i].col_j);
    }
    for (int i = 0; i < p; i++) {
        free(last_rowsum[i]);
    }
    free(last_rowsum);
    delete[] wont_update;
    delete[] last_max;
    free(max_int_delta);
    active_set_free(active_set);
    free(old_rowsum);

    return beta_sets;
}

static long firstchanged = FALSE;

Changes update_beta_cyclic_old(
    XMatrixSparse xmatrix_sparse, float* Y, float* rowsum, long n, long p,
    float lambda, robin_hood::unordered_flat_map<long, float>* beta, long k,
    float intercept, int_pair* precalc_get_num, long* column_entry_cache)
{
    float sumk = xmatrix_sparse.cols[k].nz;
    // float bk = 0.0;
    // printf("checking beta for %ld\n", k);
    // if (beta->contains(k)) {
    //    printf("beta contains %ld\n", k);
    //    bk = beta->at(k);
    //}
    float sumn = xmatrix_sparse.cols[k].nz * beta->at(k);
    long* column_entries = column_entry_cache;

    long col_entry_pos = 0;
    long entry = -1;
    for (long i = 0; i < xmatrix_sparse.cols[k].nwords; i++) {
        S8bWord word = xmatrix_sparse.cols[k].compressed_indices[i];
        unsigned long values = word.values;
        for (long j = 0; j <= group_size[word.selector]; j++) {
            long diff = values & masks[word.selector];
            if (diff != 0) {
                entry += diff;
                column_entries[col_entry_pos] = entry;
                sumn += intercept - rowsum[entry];
                col_entry_pos++;
            }
            values >>= item_width[word.selector];
        }
    }

    //if (k == interesting_col) {
    //    printf("lambda: %f\n", lambda);
    //    printf("sumn: %f\n", sumn);
    //    printf("soft: %f\n", soft_threshold(sumn, lambda) / sumk);
    //}

    // TODO: This is probably slower than necessary.
    float Bk_diff = beta->at(k);
    if (sumk == 0.0) {
        // beta[k] = 0.0;
    } else {
        beta->insert_or_assign(k, soft_threshold(sumn, lambda * n) / sumk);
    }
    Bk_diff = beta->at(k) - Bk_diff;
    // update every rowsum[i] w/ effects of beta change.
    if (Bk_diff != 0) {
        if (!firstchanged) {
            firstchanged = TRUE;
            printf("first changed on col %ld (%ld,%ld), lambda %f ******************\n",
                k, precalc_get_num[k].i, precalc_get_num[k].j, lambda);
        }
        for (long e = 0; e < xmatrix_sparse.cols[k].nz; e++) {
            long i = column_entries[e];
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
Changes update_beta_cyclic(S8bCol col, float* Y, float* rowsum, long n, long p,
    float lambda,
    robin_hood::unordered_flat_map<long, float>* beta,
    long k, float intercept, int_pair* precalc_get_num,
    long* column_entry_cache)
{
    float sumk = col.nz;
    float bk = 0.0;
    if (beta->contains(k)) {
        bk = beta->at(k);
    }
    float sumn = col.nz * bk;
    // float relevant_sq_err = 0.0;
    long* column_entries = column_entry_cache;

    long col_entry_pos = 0;
    long entry = -1;
    for (long i = 0; i < col.nwords; i++) {
        alignas(64) S8bWord word = col.compressed_indices[i];
        unsigned long values = word.values;
        for (long j = 0; j <= group_size[word.selector]; j++) {
            long diff = values & masks[word.selector];
            if (diff != 0) {
                entry += diff;
                column_entries[col_entry_pos] = entry;
                sumn -= rowsum[entry];
                // relevant_sq_err += rowsum[entry] * rowsum[entry];
                col_entry_pos++;
            }
            values >>= item_width[word.selector];
        }
    }
    // relevant_sq_err = std::sqrt(relevant_sq_err);

    // float new_value = soft_threshold(sumn, lambda) / sumk;
    // float new_value = soft_threshold(sumn, lambda*n) / sumk;
    float new_value = soft_threshold(sumn, lambda * total_sqrt_error) / sumk; // square root lasso
    //if (VERBOSE && k == interesting_val) {
    //    printf("lambda: %f\n", lambda);
    //    printf("sumn: %f\n", sumn);
    //    printf("soft: %f\n", soft_threshold(sumn, lambda) / sumk);
    //}
    float Bk_diff = new_value - bk;
    Changes changes;
    changes.removed = false;
    changes.actual_diff = Bk_diff;
    changes.pre_lambda_diff = sumn;
    auto tp = val_to_triplet(k, p);
    if (new_value == 0.0) {
        beta->erase(k);
        changes.removed = true;
    } else {
        // printf("assigning %f to: (%ld) \n", new_value, k);
        beta->insert_or_assign(k, new_value);
    }
    // update every rowsum[i] w/ effects of beta change.
    if (Bk_diff != 0) {
        // printf("nz beta update for: (%ld)\n", k);
        for (long e = 0; e < col.nz; e++) {
            long i = column_entries[e];
#pragma omp atomic
            rowsum[i] += Bk_diff;
        }
    } else {
        zero_updates++;
        zero_updates_entries += col.nz;
    }

    return changes;
}

float update_intercept_cyclic(float intercept, long** X, float* Y,
    robin_hood::unordered_flat_map<long, float> beta,
    long n, long p)
{
    float new_intercept = 0.0;
    float sumn = 0.0, sumx = 0.0;

    for (long i = 0; i < n; i++) {
        sumx = 0.0;
        for (long j = 0; j < p; j++) {
            sumx += X[i][j] * beta[j];
        }
        sumn += Y[i] - sumx;
    }
    new_intercept = sumn / n;
    return new_intercept;
}
