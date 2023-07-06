#include "liblasso.h"
#include "robin_hood.h"
#include <cmath>
#include <cstdint>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef NOT_R
#include <glib-2.0/glib.h>
#endif
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <random>
#include <vector>

using namespace std;

struct timespec start_time, end_time;
int_fast64_t total_beta_updates = 0;
int_fast64_t total_beta_nz_updates = 0;

static float c_bar = 0.75;
// static float c_bar = 0.001;
// static float c_bar = 750;

void check_beta_order(robin_hood::unordered_flat_map<int_fast64_t, float>* beta,
    int_fast64_t p)
{
    for (auto it = beta->begin(); it != beta->end(); it++) {
        int_fast64_t value = it->first;
        float bv = it->second;
        auto tuple = val_to_triplet(value, p);
        int_fast64_t a = std::get<0>(tuple);
        int_fast64_t b = std::get<1>(tuple);
        int_fast64_t c = std::get<2>(tuple);

        if (a > b || b > c) {
            printf("problem! %ld,%ld,%ld: %f\n", a, b, c, bv);
        }
#ifdef NOT_R
        g_assert_true(a <= b);
        g_assert_true(b <= c);
#endif
    }
}

float update_intercept(float* rowsum, float* Y, int n, float lambda, float intercept)
{
    float sumn = intercept * n;
    for (int i = 0; i < n; i++) {
        sumn -= rowsum[i];
    }
    float new_value = sumn / n; // square root lasso
    float diff = new_value - intercept;

    for (int i = 0; i < n; i++) {
        rowsum[i] += diff;
    }
    return new_value;
}

// check a particular pair of betas in the adaptive calibration scheme
int_fast64_t adaptive_calibration_check_beta(float c_bar, float lambda_1,
    Sparse_Betas* beta_1, float lambda_2,
    Sparse_Betas* beta_2, int_fast64_t n)
{
    float max_diff = 0.0;
    float adjusted_max_diff = 0.0;

    int_fast64_t b1_count = 0;
    int_fast64_t b2_count = 0;

    int_fast64_t b1_ind = 0;
    int_fast64_t b2_ind = 0;

    // advance whichever is smaller, accounting for overlap
    while (b1_count < beta_1->count && b2_count < beta_2->count) {
        float b1v = beta_1->values[b1_count];
        float b2v = beta_2->values[b2_count];
        int_fast64_t b1_ind = beta_1->indices[b1_count];
        int_fast64_t b2_ind = beta_2->indices[b2_count];

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

    if (adjusted_max_diff <= c_bar) {
        return 1;
    }
    return 0;
}

// checks whether the last element in the beta_sequence is the one we should
// stop at, according to Chichignoud et als 'Adaptive Calibration Scheme'
// returns TRUE if we are finished, FALSE if we should continue.
int_fast64_t check_adaptive_calibration(float c_bar, Beta_Sequence* beta_sequence,
    int_fast64_t n)
{
    for (int_fast64_t i = 0; i < beta_sequence->count; i++) {
        int_fast64_t this_result = adaptive_calibration_check_beta(
            c_bar, beta_sequence->lambdas[beta_sequence->count - 1],
            &beta_sequence->betas[beta_sequence->count - 1],
            beta_sequence->lambdas[i], &beta_sequence->betas[i], n);
        if (this_result == 0) {
            return TRUE;
        }
    }
    return FALSE;
}

float calculate_error(float* Y, float* rowsum, int_fast64_t n)
{
    float error = 0.0;
    for (int row = 0; row < n; row++) {
        float row_err = -rowsum[row];
        error += row_err * row_err;
    }
    return error;
}

float halt_error_diff;
static auto rng = std::default_random_engine();

void subproblem_only(Iter_Vars* vars, float lambda, float* rowsum,
    float* old_rowsum, Active_Set* active_set,
    int_fast64_t depth, char use_intercept, struct continuous_info* cont_inf)
{
    float** last_rowsum = vars->last_rowsum;
    Thread_Cache* thread_caches = vars->thread_caches;
    int_fast64_t n = vars->n;
    Beta_Value_Sets* beta_sets = vars->beta_sets;
    robin_hood::unordered_flat_map<int_fast64_t, float>* beta1 = &beta_sets->beta1;
    robin_hood::unordered_flat_map<int_fast64_t, float>* beta2 = &beta_sets->beta2;
    robin_hood::unordered_flat_map<int_fast64_t, float>* beta3 = &beta_sets->beta3;
    robin_hood::unordered_flat_map<int_fast64_t, float>* beta = &beta_sets->beta3; // TODO: dont
    float* last_max = vars->last_max;
    bool* wont_update = vars->wont_update;
    int_fast64_t p = vars->p;
    int_fast64_t p_int = vars->p_int;
    float* Y = vars->Y;
    float* max_int_delta = vars->max_int_delta;
    int_fast64_t new_nz_beta = 0;

    float error = 0.0;
    for (int_fast64_t i = 0; i < n; i++) {
        error += rowsum[i] * rowsum[i];
    }
    std::vector<std::pair<int_fast64_t, AS_Entry>> entry_vec1;
    entry_vec1.reserve(active_set->entries1.size());
    std::vector<std::pair<int_fast64_t, AS_Entry>> entry_vec2;
    entry_vec2.reserve(active_set->entries2.size());
    std::vector<std::pair<int_fast64_t, AS_Entry>> entry_vec3;
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
            int_fast64_t k = std::get<0>(it);
            AS_Entry entry = std::get<1>(it);
            if (entry.present) {
                if (current_beta_set->contains(k) && fabs(current_beta_set->at(k)) != 0.0) {
                    update_beta_cyclic(
                        &entry, Y, rowsum, n, p, lambda, current_beta_set, k, vars->intercept,
                        thread_caches[omp_get_thread_num()].col_i, cont_inf);
                }
            }
        }
    };
    int_fast64_t iter = 0;
    for (iter = 0; iter < 100; iter++) {
        float prev_error = error;

        if (use_intercept) {
            vars->intercept = update_intercept(rowsum, Y, n, lambda, vars->intercept);
        }
        // update entire working set
        // #pragma omp parallel for num_threads(NumCores) schedule(static)
        // shared(Y, rowsum, beta, precalc_get_num, perm)
        // reduction(+:total_unchanged, total_changed, total_present,
        // total_notpresent, new_nz_beta, total_beta_updates,
        // total_beta_nz_updates)

        run_beta(&beta_sets->beta1, entry_vec1);
        run_beta(&beta_sets->beta2, entry_vec2);
        run_beta(&beta_sets->beta3, entry_vec3);

        // check whether we need another iteration
        error = 0.0;
        for (int_fast64_t i = 0; i < n; i++) {
            error += rowsum[i] * rowsum[i];
        }
        error = sqrt(error);
        if (prev_error / error < halt_error_diff) {
            break;
        }
    }
}

int_fast64_t run_lambda_iters_pruned(Iter_Vars* vars, float lambda, float* rowsum,
    float* old_rowsum, Active_Set* active_set,
    int_fast64_t depth, const bool use_intercept, IndiCols* indi, const bool check_duplicates,
    struct continuous_info* cont_inf, const bool use_hierarchy)
{
    XMatrixSparse Xc = vars->Xc;
    X_uncompressed Xu = vars->Xu;
    float** last_rowsum = vars->last_rowsum;
    Thread_Cache* thread_caches = vars->thread_caches;
    int_fast64_t n = vars->n;
    Beta_Value_Sets* beta_sets = vars->beta_sets;
    float* last_max = vars->last_max;
    bool* wont_update = vars->wont_update;
    int_fast64_t p = vars->p;
    int_fast64_t p_int = vars->p_int;
    float* Y = vars->Y;
    float* max_int_delta = vars->max_int_delta;
    int_fast64_t new_nz_beta = 0;
    int_fast64_t max_interaction_distance = vars->max_interaction_distance;

    float error = 0.0;
    for (int_fast64_t i = 0; i < n; i++) {
        error += rowsum[i] * rowsum[i];
    }
    error = sqrt(error);

    // run several iterations of will_update to make sure we catch any new
    // columns
    /*TODO: in principle we should allow more than one, but it seems to
     * only slow things down. maybe this is because any effects not chosen
     * for the first iter will be small, and therefore not really worth
     * it? (i.e. slow and unreliable).
     * TODO: suffers quite badly when numa updates are allowed
     *        - check if this is only true for small p (openmp tests seem to only
     * improve with p >= 5k)
     */
    // TODO: with multiple iters, many branches are added on the second iter. This
    // doesn't seem right.
    for (int_fast64_t retests = 0; retests < 1; retests++) {
        int_fast64_t total_changed = 0;
        int_fast64_t total_unchanged = 0;
        int_fast64_t total_changes = 0;
        int_fast64_t total_present = 0;
        int_fast64_t total_notpresent = 0;

        //********** Branch Pruning       *******************
        int_fast64_t active_branches = 0;

        clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);
        robin_hood::unordered_flat_set<int_fast64_t> thread_new_cols[NumCores];
        for (int i = 0; i < NumCores; i++) {
            thread_new_cols[i].clear();
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int_fast64_t j = 0; j < p; j++) {
            bool prev_wont_update = wont_update[j];
            wont_update[j] = wont_update_effect(Xu, lambda, j, last_max[j], last_rowsum[j], rowsum,
                thread_caches[omp_get_thread_num()].col_j, cont_inf);
            if (!wont_update[j] && !(*vars->seen_before)[j]) {
            // if (!wont_update[j] && !prev_wont_update) {
            // if (true) {
                thread_new_cols[omp_get_thread_num()].insert(j);
            }
        }
        robin_hood::unordered_flat_set<int_fast64_t> new_cols;
        for (int i = 0; i < NumCores; i++) {
            for (auto col : thread_new_cols[i]) {
                (*vars->seen_before)[col] = true;
                new_cols.insert(col);
            }
        }
        // this slows things down on multiple numa nodes. There must be something
        // going on with rowsum/last_rowsum?
        {
// TODO: parallelising this loop slows down numa updates.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ \
                                                    : active_branches, used_branches, pruned_branches)
#endif
            for (int_fast64_t j = 0; j < p; j++) {
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
            // for (auto it = active_set->entries2.begin(); it != active_set->entries2.end(); it++) {
            //     AS_Entry *entry = &it->second;
            //     memcpy(entry->last_rowsum, rowsum, sizeof *rowsum * n);
            // }
        }
        clock_gettime(CLOCK_MONOTONIC_RAW, &end_time);
        pruning_time += ((float)(end_time.tv_nsec - start_time.tv_nsec)) / 1e9 + (end_time.tv_sec - start_time.tv_sec);
        //********** Identify Working Set *******************
        // TODO: is it worth constructing a new set with no 'blank'
        // elements?
        int_fast64_t count_may_update = 0;
        int_fast64_t* updateable_items = (int_fast64_t*)calloc(p, sizeof *updateable_items); // TODO: keep between iters
        for (int_fast64_t i = 0; i < p; i++) {
            if (!wont_update[i]) {
                updateable_items[count_may_update] = i;
                count_may_update++;
            }
        }

        clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);
        auto working_set_results = update_working_set(vars->Xu, Xc, rowsum, wont_update, p, n, lambda,
            updateable_items, count_may_update, active_set,
            thread_caches, last_max, depth, indi, &new_cols, max_interaction_distance, check_duplicates, cont_inf,
            use_hierarchy, beta_sets);
        bool increased_set = working_set_results.first;
        auto vals_to_remove = working_set_results.second;
        for (auto val : vals_to_remove) {
            active_set_remove(active_set, val);
            if (val < p)
                beta_sets->beta1[val] = 0;
            else if (val < p * p)
                beta_sets->beta2[val] = 0;
            else
                beta_sets->beta3[val] = 0;
        }
        free(updateable_items);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end_time);
        working_set_update_time += ((float)(end_time.tv_nsec - start_time.tv_nsec)) / 1e9 + (end_time.tv_sec - start_time.tv_sec);
        if (retests > 0 && !increased_set) {
            // there's no need to re-run on the same set. Nothing has changed
            // and the remaining retests will all do nothing.
            break;
        }
        //********** Solve subproblem     *******************
        clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);

        std::vector<std::pair<int_fast64_t, AS_Entry>> entry_vec1;
        entry_vec1.reserve(active_set->entries1.size());
        std::vector<std::pair<int_fast64_t, AS_Entry>> entry_vec2;
        entry_vec2.reserve(active_set->entries2.size());
        std::vector<std::pair<int_fast64_t, AS_Entry>> entry_vec3;
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
                int_fast64_t k = std::get<0>(it);
                AS_Entry entry = std::get<1>(it);
                if (entry.present) {
                    total_present++;
                    int_fast64_t was_zero = TRUE;
                    if (current_beta_set->contains(k) && fabs(current_beta_set->at(k)) != 0.0)
                        was_zero = FALSE;
                    total_beta_updates++;
                    Changes changes = update_beta_cyclic(
                        &entry, Y, rowsum, n, p, lambda, current_beta_set, k, vars->intercept,
                        thread_caches[omp_get_thread_num()].col_i, cont_inf);
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
        int_fast64_t iter = 0;
        for (iter = 0; iter < 100; iter++) {
            float prev_error = error;
            // update entire working set
            if (use_intercept) {
                vars->intercept = update_intercept(rowsum, Y, n, lambda, vars->intercept);
            }
            run_beta(&beta_sets->beta1, entry_vec1);
            run_beta(&beta_sets->beta2, entry_vec2);
            run_beta(&beta_sets->beta3, entry_vec3);

            // check whether we need another iteration
            error = 0.0;
            for (int_fast64_t i = 0; i < n; i++) {
                error += rowsum[i] * rowsum[i];
            }
            error = sqrt(error);
            if (prev_error / error < halt_error_diff) {
                break;
            }
        }
        clock_gettime(CLOCK_MONOTONIC_RAW, &end_time);
        subproblem_time += ((float)(end_time.tv_nsec - start_time.tv_nsec)) / 1e9 + (end_time.tv_sec - start_time.tv_sec);
    }

    return new_nz_beta;
}

int_fast64_t copy_beta_sets(Beta_Value_Sets* from_set, Sparse_Betas* to_set)
{
    int_fast64_t count = 0;
    auto from_beta = from_set->beta1;
    int_fast64_t total_size = from_set->beta1.size() + from_set->beta2.size() + from_set->beta3.size();
    to_set->indices = new int_fast64_t[total_size];
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
    return sqrt(abs(2.0 * log(sqrt(2.0 * M_PI) * x)));
}

float total_sqrt_error = 0.0;
Lasso_Result simple_coordinate_descent_lasso(
    XMatrix xmatrix, float* Y, int_fast64_t n, int_fast64_t p, int_fast64_t max_interaction_distance,
    float lambda_min, float lambda_max, int_fast64_t max_iter, const bool verbose,
    float hed, enum LOG_LEVEL log_level,
    const char** job_args, int_fast64_t job_args_num,
    int_fast64_t mnz_beta, const char* log_filename, int_fast64_t depth,
    const bool estimate_unbiased, const bool use_intercept, const bool check_duplicates, struct continuous_info* cont_inf, const bool use_hierarchy)
{
    const bool continuous_X = cont_inf->use_cont;
    int_fast64_t max_nz_beta = mnz_beta;
    if (verbose)
        printf("n: %ld, p: %ld\n", n, p);
    halt_error_diff = hed;
    if (verbose) {
        printf("using halt_error_diff of %f\n", halt_error_diff);
        printf("using depth: %ld\n", depth);
    }
    int_fast64_t num_nz_beta = 0;
    int_fast64_t became_zero = 0;
    float lambda = lambda_max;
    VERBOSE = verbose;
    int_pair* precalc_get_num;
    int_fast64_t** X = xmatrix.X;
    IndiCols indi = get_empty_indicols(p);

    int_fast64_t real_p_int = -1;
    switch (depth) {
    case 1:
        real_p_int = (int_fast64_t)p;
    case 2:
        real_p_int = (int_fast64_t)p * ((int_fast64_t)p + 1) / 2;
    case 3:
        real_p_int = (int_fast64_t)p * ((int_fast64_t)p + 1) * ((int_fast64_t)p - 1) / (2 * 3);
        break;
    }
    double final_lambda = lambda_min;
    if (lambda_min <= 0) {
        if (verbose)
            printf("p = %ld, phi_inv(0.95/(2.0 * p)) = %f\n", real_p_int, phi_inv(0.95 / (2.0 * (double)real_p_int)));
        final_lambda = 1.1 * std::sqrt(1.0 / (double)n) * phi_inv(0.95 / (2.0 * (double)real_p_int));
    }
    if (verbose)
        printf("using final lambda: %f\n", final_lambda);

    // work out min lambda for sqrt lasso
    XMatrixSparse Xc
        = sparsify_X(X, n, p);
    X_uncompressed Xu = construct_host_X(&Xc);

    for (int_fast64_t i = 0; i < NUM_MAX_ROWSUMS; i++) {
        max_rowsums[i] = 0;
        max_cumulative_rowsums[i] = 0;
    }

    int_fast64_t p_int = get_p_int(p, max_interaction_distance);
    if (max_interaction_distance == -1) {
        max_interaction_distance = p;
    }
    if (max_nz_beta < 0)
        max_nz_beta = p_int;
    Beta_Value_Sets beta_sets;
    beta_sets.p = p;

    float error = 0.0;
    for (int_fast64_t i = 0; i < n; i++) {
        error += Y[i] * Y[i];
    }
    float intercept = 0.0;

    float* rowsum = (float*)calloc(n, sizeof *rowsum);
    for (int_fast64_t i = 0; i < n; i++)
        rowsum[i] = -Y[i];

    // find largest number of non-zeros in any column
    int_fast64_t largest_col = 0;
    int_fast64_t total_col = 0;
    for (int_fast64_t i = 0; i < p; i++) {
        int_fast64_t col_size = Xu.host_col_nz[i];
        if (col_size > largest_col) {
            largest_col = col_size;
        }
        total_col += col_size;
    }
    int_fast64_t main_sum = 0;
    for (int_fast64_t i = 0; i < p; i++)
        for (int_fast64_t j = 0; j < n; j++)
            main_sum += X[i][j];

    struct timespec start, end;
    float cpu_time_used;

    int_fast64_t set_min_lambda = FALSE;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    int_fast64_t max_lambda_count = max_iter;
    int_fast64_t lambda_count = 1;
    int_fast64_t iter_count = 0;

    int_fast64_t max_num_threads = omp_get_max_threads();
    int_fast64_t** thread_column_caches = (int_fast64_t**)malloc(max_num_threads * sizeof *thread_column_caches);
    for (int_fast64_t i = 0; i < max_num_threads; i++) {
        thread_column_caches[i] = (int_fast64_t*)malloc(largest_col * sizeof *thread_column_caches[i]);
    }

    FILE* log_file = NULL;
    int_fast64_t iter = 0;
    if (log_level != NONE && check_can_restore_from_log(log_filename, n, p, p_int, job_args, job_args_num)) {
        if (verbose)
            Rprintf("We can restore from a partial log!\n");
        restore_from_log(log_filename, true, n, p, job_args, job_args_num, &iter,
            &lambda_count, &lambda, &beta_sets);
        // we need to recalculate the rowsums
        auto update_beta_set = [&](auto beta_set) {
            for (auto it = beta_set->begin(); it != beta_set->end(); it++) {
                int_fast64_t val = it->first;
                float bv = it->second;
                auto update_columns = [&](int_fast64_t a, int_fast64_t b, int_fast64_t c) {
                    int_fast64_t* colA = &Xu.host_X[Xu.host_col_offsets[a]];
                    int_fast64_t* colB = &Xu.host_X[Xu.host_col_offsets[b]];
                    int_fast64_t* colC = &Xu.host_X[Xu.host_col_offsets[c]];
                    int_fast64_t ib = 0, ic = 0;
                    for (int_fast64_t ia = 0; ia < Xu.host_col_nz[a]; ia++) {
                        int_fast64_t cur_row = colA[ia];
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
                    int_fast64_t a = std::get<0>(pair);
                    int_fast64_t b = std::get<1>(pair);
                    update_columns(a, a, b);
                } else {
                    if (val >= p * p * p) {
                        fprintf(stderr, "broken val: %ld\n", val);
                    }
                    auto triple = val_to_triplet(val, p);
                    int_fast64_t a = std::get<0>(triple);
                    int_fast64_t b = std::get<1>(triple);
                    int_fast64_t c = std::get<2>(triple);
                    update_columns(a, b, c);
                }
            }
        };

        update_beta_set(&beta_sets.beta1);
        update_beta_set(&beta_sets.beta2);
        update_beta_set(&beta_sets.beta3);
    } else {
        if (verbose)
            Rprintf("no partial log for current job found\n");
    }
    if (log_level != NONE)
        log_file = init_log(log_filename, n, p, p_int, job_args, job_args_num);

    // set-up beta_sequence struct
    robin_hood::unordered_flat_map<int_fast64_t, float> beta_cache;
    int_fast64_t* index_cache = NULL;
    Beta_Sequence beta_sequence;

    float** last_rowsum = (float**)malloc(sizeof *last_rowsum * p);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int_fast64_t i = 0; i < p; i++) {
        last_rowsum[i] = (float*)malloc(sizeof *last_rowsum[i] * n + PADDING);
        memset(last_rowsum[i], 0, sizeof *last_rowsum[i] * n);
    }
    Thread_Cache thread_caches[NumCores];

    for (int_fast64_t i = 0; i < NumCores; i++) {
        thread_caches[i].col_i = (int_fast64_t*)malloc(max(n, p) * sizeof *thread_caches[i].col_i);
        thread_caches[i].col_j = (int_fast64_t*)malloc(n * sizeof *thread_caches[i].col_j);
    }

    float* last_max = new float[p];
    bool* wont_update = new bool[p];
    memset(last_max, 0, p * sizeof(*last_max));
    float* max_int_delta = (float*)malloc(sizeof *max_int_delta * p);
    memset(max_int_delta, 0, sizeof *max_int_delta * p);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int_fast64_t i = 0; i < p; i++) {
        memset(last_rowsum[i], 0, sizeof *last_rowsum[i] * n);
        last_max[i] = 0.0;
        max_int_delta[i] = 0;
    }
    for (int_fast64_t i = 0; i < n; i++) {
        rowsum[i] = -Y[i];
    }
    XMatrixSparse X2c_fake;
    intercept = 0.0;
    std::vector<bool> seen_before(p, false);
    Iter_Vars iter_vars_pruned = {
        Xc,
        last_rowsum,
        thread_caches,
        n,
        &beta_sets,
        last_max,
        wont_update,
        &seen_before,
        p,
        p_int,
        X2c_fake,
        Y,
        max_int_delta,
        Xu,
        intercept,
        max_interaction_distance
    };
    int_fast64_t nz_beta = 0;
    Active_Set active_set = active_set_new(p_int, p);
    float* old_rowsum = (float*)malloc(sizeof *old_rowsum * n);
    if (verbose)
        printf("final_lambda: %f\n", final_lambda);
    error = calculate_error(Y, rowsum, n);
    total_sqrt_error = std::sqrt(error);
    if (verbose)
        printf("initial error: %f\n", error);
    while (lambda > final_lambda && iter < max_lambda_count) {
        if (log_level == LAMBDA) {
            save_log(0, lambda, lambda_count, &beta_sets, log_file);
        }
        if (nz_beta >= max_nz_beta) {
            if (verbose)
                printf("reached max_nz_beta of %ld\n", max_nz_beta);
            break;
        }
        float dBMax;
        // TODO: implement working set and update test
        int_fast64_t last_iter_count = 0;

        nz_beta += run_lambda_iters_pruned(&iter_vars_pruned, lambda, rowsum,
            old_rowsum, &active_set, depth, use_intercept, &indi, check_duplicates, cont_inf, use_hierarchy);

        {
            int_fast64_t nonzero = beta_sets.beta1.size() + beta_sets.beta2.size() + beta_sets.beta3.size();
#ifdef NOT_R
            g_assert_true(nonzero == nz_beta);
#endif
        }
        double prev_error = error;
        error = calculate_error(Y, rowsum, n);
        total_sqrt_error = std::sqrt(error);
        if (verbose)
            printf("lambda %ld = %f, error %.4e, nz_beta %ld,%ld,%ld\n", lambda_count, lambda, error,
                beta_sets.beta1.size(), beta_sets.beta2.size(), beta_sets.beta3.size());
        lambda *= 0.95;
        if (nz_beta > 0) {
            iter++;
        }
        lambda_count++;
    }
    // Interactions may be been added out of order, resulting in pairwise effects
    // that are duplicate of later main columns, or three-way effects that are
    // duplicates of later pairs.
    // We clean this up here.
    // TODO: ideally we wouldn't have to do this, but adding things out of order currently
    // breaks some stuff.
    auto beta_set_remove_dups = [&](auto* sets, robin_hood::unordered_flat_map<int_fast64_t, std::vector<int_fast64_t>>* indist_from_val) {
        std::vector<int_fast64_t> remove_3;
        for (auto beta : sets->beta3) {
            int_fast64_t val = beta.first;
            float effect = beta.second;
            auto full_col = get_col_by_id(Xu, val);
            XXH128_hash_t col_hash = XXH3_128bits(&full_col[0], full_col.size() * sizeof(int_fast64_t));
            for (auto col : indi.cols_for_hash[col_hash.high64][col_hash.low64]) {
                if (NULL != indist_from_val)
                    (*indist_from_val)[val].push_back(col);
                if (col < p) {
                    sets->beta1[col] += effect;
                    remove_3.push_back(val);
                    break;
                } else if (col < p * p) {
                    sets->beta2[col] += effect;
                    remove_3.push_back(val);
                    break;
                }
            }
        }
        std::vector<int_fast64_t> remove_2;
        for (auto beta : sets->beta2) {
            int_fast64_t val = beta.first;
            float effect = beta.second;
            auto full_col = get_col_by_id(Xu, val);
            XXH128_hash_t col_hash = XXH3_128bits(&full_col[0], full_col.size() * sizeof(int_fast64_t));
            for (auto col : indi.cols_for_hash[col_hash.high64][col_hash.low64]) {
                if (NULL != indist_from_val)
                    (*indist_from_val)[val].push_back(col);
                if (col < p) {
                    sets->beta1[col] += effect;
                    remove_2.push_back(val);
                    break;
                }
            }
        }
        if (indist_from_val != NULL) {
            for (auto beta : sets->beta1) {
                int_fast64_t val = beta.first;
                auto full_col = get_col_by_id(Xu, val);
                XXH128_hash_t col_hash = XXH3_128bits(&full_col[0], full_col.size() * sizeof(int_fast64_t));
                for (auto col : indi.cols_for_hash[col_hash.high64][col_hash.low64]) {
                    (*indist_from_val)[val].push_back(col);
                }
            }
        }
        for (auto val2 : remove_2)
            sets->beta2.erase(val2);
        for (auto val3 : remove_3)
            sets->beta3.erase(val3);
    };
    robin_hood::unordered_flat_map<int_fast64_t, std::vector<int_fast64_t>> indist_from_val;
    if (check_duplicates) {
        beta_set_remove_dups(&beta_sets, &indist_from_val);
    }
    intercept = iter_vars_pruned.intercept;
    iter_count = iter;
    if (log_level != NONE && log_file != NULL)
        close_log(log_file);
    if (verbose) {
        Rprintf("\nfinished at lambda = %f\n", lambda);
        Rprintf("after %ld total iters\n", iter_count);
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    cpu_time_used = ((float)(end.tv_nsec - start.tv_nsec)) / 1e9 + (end.tv_sec - start.tv_sec);

    if (verbose)
        Rprintf("lasso done in %.4f seconds\n", cpu_time_used);

    /*
     * Optionally attempt to provide an un-regularised estimate of the
     * beta values for the current working set.
     */
    float unbiased_intercept = intercept;
    Beta_Value_Sets unbiased_beta_sets;
    unbiased_beta_sets.p = p;
    if (estimate_unbiased) {
        if (verbose)
            printf("original_error: %f\n", calculate_error(Y, rowsum, n));
        for (auto it = beta_sets.beta1.begin(); it != beta_sets.beta1.end(); it++)
            unbiased_beta_sets.beta1[it->first] = it->second;
        for (auto it = beta_sets.beta2.begin(); it != beta_sets.beta2.end(); it++)
            unbiased_beta_sets.beta2[it->first] = it->second;
        for (auto it = beta_sets.beta3.begin(); it != beta_sets.beta3.end(); it++)
            unbiased_beta_sets.beta3[it->first] = it->second;
        Iter_Vars iter_vars_pruned = {
            Xc,
            last_rowsum,
            thread_caches,
            n,
            &unbiased_beta_sets,
            last_max,
            wont_update,
            &seen_before,
            p,
            p_int,
            X2c_fake,
            Y,
            max_int_delta,
            Xu,
            unbiased_intercept,
        };
        // run_lambda_iters_pruned(&iter_vars_pruned, 0.0, rowsum,
        subproblem_only(&iter_vars_pruned, 0.0, rowsum,
            old_rowsum, &active_set, depth, use_intercept, cont_inf);
        if (verbose)
            printf("un-regularized error: %f\n", calculate_error(Y, rowsum, n));
        unbiased_intercept = iter_vars_pruned.intercept;
    }
    beta_set_remove_dups(&unbiased_beta_sets, NULL);

    // free beta sets
    free_sparse_matrix(Xc);
    for (int_fast64_t i = 0; i < max_num_threads; i++) {
        free(thread_column_caches[i]);
    }
    free(thread_column_caches);
    free(rowsum);

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
    free_inter_cache(p);

    Lasso_Result result;
    result.regularized_result = beta_sets;
    result.unbiased_result = unbiased_beta_sets;
    result.final_lambda = lambda;
    result.regularized_intercept = intercept;
    result.unbiased_intercept = unbiased_intercept;
    result.indi = indi;
    result.indist_from_val = indist_from_val;
#ifdef _OPENMP
#pragma omp barrier
#endif
    return result;
}

void free_lasso_result(Lasso_Result lr) {
    free_indicols(lr.indi);
}

static int_fast64_t firstchanged = FALSE;

// Exclusively for testing
Changes update_beta_cyclic_old(
    XMatrixSparse xmatrix_sparse, float* Y, float* rowsum, int_fast64_t n, int_fast64_t p,
    float lambda, robin_hood::unordered_flat_map<int_fast64_t, float>* beta, int_fast64_t k,
    float intercept, int_pair* precalc_get_num, int_fast64_t* column_entry_cache)
{
    float sumk = xmatrix_sparse.cols[k].nz;
    float sumn = xmatrix_sparse.cols[k].nz * beta->at(k);
    int_fast64_t* column_entries = column_entry_cache;

    int_fast64_t col_entry_pos = 0;
    int_fast64_t entry = -1;
    for (int_fast64_t i = 0; i < xmatrix_sparse.cols[k].nwords; i++) {
        S8bWord word = xmatrix_sparse.cols[k].compressed_indices[i];
        int_fast64_t values = word.values;
        for (int_fast64_t j = 0; j <= group_size[word.selector]; j++) {
            int_fast64_t diff = values & masks[word.selector];
            if (diff != 0) {
                entry += diff;
                column_entries[col_entry_pos] = entry;
                sumn += intercept - rowsum[entry];
                col_entry_pos++;
            }
            values >>= item_width[word.selector];
        }
    }

    // TODO: This is probably slower than necessary.
    float Bk_diff = beta->at(k);
    if (sumk == 0.0) {
    } else {
        beta->insert_or_assign(k, soft_threshold(sumn, lambda * n) / sumk);
    }
    Bk_diff = beta->at(k) - Bk_diff;
    // update every rowsum[i] w/ effects of beta change.
    if (Bk_diff != 0) {
        if (!firstchanged) {
            firstchanged = TRUE;
            if (VERBOSE)
                printf("first changed on col %ld (%ld,%ld), lambda %f ******************\n",
                    k, precalc_get_num[k].i, precalc_get_num[k].j, lambda);
        }
        for (int_fast64_t e = 0; e < xmatrix_sparse.cols[k].nz; e++) {
            int_fast64_t i = column_entries[e];
#ifdef _OPENMP
#pragma omp atomic
#endif
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
Changes update_beta_cyclic(AS_Entry* as_entry, float* Y, float* rowsum, int_fast64_t n, int_fast64_t p,
    float lambda,
    robin_hood::unordered_flat_map<int_fast64_t, float>* beta,
    int_fast64_t k, float intercept, int_fast64_t* column_entry_cache, struct continuous_info* ci)
{
    S8bCol col = as_entry->col;
    float sumk = col.nz;
    if (ci->use_cont)
        sumk = 0.0;
    float bk = 0.0;
    if (beta->contains(k)) {
        bk = beta->at(k);
    }
    float sumn = 0.0;
    int_fast64_t* column_entries = column_entry_cache;
    std::vector<float> col_real_vals = as_entry->real_vals;

    int_fast64_t col_entry_pos = 0;
    int_fast64_t entry = -1;
    for (int_fast64_t i = 0; i < col.nwords; i++) {
        alignas(64) S8bWord word = col.compressed_indices[i];
        int_fast64_t values = word.values;
        for (int_fast64_t j = 0; j <= group_size[word.selector]; j++) {
            int_fast64_t diff = values & masks[word.selector];
            if (diff != 0) {
                entry += diff;
                column_entries[col_entry_pos] = entry;
                float rs = rowsum[entry];
                if (ci->use_cont) {
                    float cv = col_real_vals[col_entry_pos];
                    rs *= cv;
                    sumk += cv*cv;
                    // sumn += cv*bk;
                }
                sumn -= rs;
                col_entry_pos++;
            }
            values >>= item_width[word.selector];
        }
    }
    sumn += sumk*bk;

    float new_value = soft_threshold(sumn, lambda * total_sqrt_error) / sumk; // square root lasso
    float Bk_diff = new_value - bk;
    Changes changes;
    changes.removed = false;
    changes.actual_diff = Bk_diff;
    changes.pre_lambda_diff = sumn;
    if (new_value == 0.0) {
        beta->erase(k);
        changes.removed = true;
    } else {
        beta->insert_or_assign(k, new_value);
    }
    // update every rowsum[i] w/ effects of beta change.
    if (Bk_diff != 0) {
        for (int_fast64_t e = 0; e < col.nz; e++) {
            int_fast64_t i = column_entries[e];
            float offset = Bk_diff;
            if (ci->use_cont)
                offset *= col_real_vals[e];
#ifdef _OPENMP
#pragma omp atomic
#endif
            rowsum[i] += offset;
        }
    } else {
        zero_updates++;
        zero_updates_entries += col.nz;
    }

    return changes;
}
