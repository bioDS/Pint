#include "robin_hood.h"
#include <omp.h>
#include <stdlib.h>

#include "flat_hash_map.hpp"
#include "liblasso.h"
#ifdef NOT_R
#include <glib-2.0/glib.h>
#endif

#define TRUE 1
#define FALSE 0

Active_Set active_set_new(int_fast64_t max_length, int_fast64_t p)
{
    Active_Set as;
    as.length = 0;
    as.max_length = max_length;
    as.p = p;
    return as;
}

void active_set_free(Active_Set as)
{
    for (auto c = as.entries1.begin(); c != as.entries1.end(); c++) {
        struct AS_Entry e = c->second;
        if (NULL != e.col.compressed_indices) {
            free(e.col.compressed_indices);
        }
    }
    for (auto c = as.entries2.begin(); c != as.entries2.end(); c++) {
        struct AS_Entry e = c->second;
        if (NULL != e.col.compressed_indices) {
            free(e.col.compressed_indices);
        }
    }
    for (auto c = as.entries3.begin(); c != as.entries3.end(); c++) {
        struct AS_Entry e = c->second;
        if (NULL != e.col.compressed_indices) {
            free(e.col.compressed_indices);
        }
    }
    as.entries1.clear();
    as.entries2.clear();
    as.entries3.clear();
}

bool active_set_present(Active_Set* as, int_fast64_t value)
{
    robin_hood::unordered_flat_map<int_fast64_t, AS_Entry>* entries;
    int_fast64_t p = as->p;
    if (value < p) {
        entries = &as->entries1;
    } else if (value < p * p) {
        entries = &as->entries2;
    } else {
        entries = &as->entries3;
    }

    return (entries->contains(value) && entries->at(value).present);
}

void active_set_append(Active_Set* as, int_fast64_t value, int_fast64_t* col, int_fast64_t len)
{
    // if (value == pair_to_val(std::make_tuple(interesting_col, interesting_col),
    // 100)) {
    //  printf("appending interesting col %ld to as\n", value);
    //}
    // printf("as, adding val %ld as ", value);
    robin_hood::unordered_flat_map<int_fast64_t, AS_Entry>* entries;
    int_fast64_t p = as->p;
    if (value < p) {
        // if (VERBOSE && value == interesting_col)
        //    printf("[%ld < %ld]: main\n", value, p);
        entries = &as->entries1;
    } else if (value < p * p) {
        // printf("[%ld < %ld]: pair\n", value, p*p);
        entries = &as->entries2;
    } else {
        // printf("triple\n");
        entries = &as->entries3;
    }
    if (entries->contains(value)) {
        struct AS_Entry e = entries->at(value);
        if (e.present)
            return;
        if (e.was_present) {
            e.present = TRUE;
        }
        entries->insert_or_assign(value, e);
    } else {
        struct AS_Entry e;
        e.val = value;
        e.present = TRUE;
        e.was_present = TRUE;
        e.col = col_to_s8b_col(len, col);
        int_fast64_t i = as->length;
        entries->insert_or_assign(value, e);
    }
    as->length++;
}

void active_set_remove(Active_Set* as, int_fast64_t value)
{
    robin_hood::unordered_flat_map<int_fast64_t, AS_Entry>* entries;
    int_fast64_t p = as->p;
    if (value < p) {
        entries = &as->entries1;
    } else if (value < p * p) {
        entries = &as->entries2;
    } else {
        entries = &as->entries3;
    }
    entries->at(value).present = FALSE; // TODO does this work?
    as->length--;
}

// int active_set_get_index(Active_Set* as, int_fast64_t index)
//{
//    struct AS_Entry* e = &as->entries[index];
//    if (e->present) {
//        return e->val;
//    } else {
//        return -INT_MAX;
//    }
//}

typedef struct IC_Entry {
    bool skipped_this_iter;
    bool checked_this_iter;
    bool present;
    bool was_present;
    float last_max;
    float* last_rowsum;
    // robin_hood::unordered_flat_map<int_fast64_t, float> last_rowsum;
    S8bCol col;
} IC_Entry;
// static robin_hood::unordered_flat_map<int_fast64_t, IC_Entry> inter_cache;
static IC_Entry* inter_cache = NULL;
// static bool inter_cache_init_done = false;
void free_inter_cache(int_fast64_t p)
{
    if (NULL == inter_cache)
        return;

    int_fast64_t p_int = p * (p + 1) / 2;
    for (int_fast64_t i = 0; i < p_int; i++) {
        // if (NULL != inter_cache[i].last_rowsum) {
        if (inter_cache[i].was_present) {
            free(inter_cache[i].last_rowsum);
            free(inter_cache[i].col.compressed_indices);
        }
    }
    free(inter_cache);
    inter_cache = NULL;
}

void update_inter_cache(int_fast64_t k, int_fast64_t n, float* rowsum, float last_max,
    int_fast64_t* col, int_fast64_t col_len)
{
    // if (col_len < 100) {
    //    inter_cache[k].skip = true;
    //    return;
    //}
    if (!inter_cache[k].was_present) {
        S8bCol comp_col = col_to_s8b_col(col_len, col);
        inter_cache[k].last_rowsum = (float*)malloc(col_len * sizeof *rowsum);
        inter_cache[k].col = comp_col;
        inter_cache[k].present = true;
        inter_cache[k].was_present = true;
        inter_cache[k].skipped_this_iter = false;
    } else if (!inter_cache[k].present) {
        inter_cache[k].present = true;
        inter_cache[k].skipped_this_iter = false;
    }
    inter_cache[k].last_max = last_max;
    // memcpy(inter_cache[k].last_rowsum, rowsum, n * sizeof *rowsum);
    for (int_fast64_t i = 0; i < col_len; i++) {
        int_fast64_t entry = col[i];
        if (entry > n) {
            fprintf(stderr, "broken entry %ld in inter %ld\n", entry, k);
            exit(EXIT_FAILURE);
        }
        inter_cache[k].last_rowsum[i] = rowsum[entry];
    }
}

#define unlikely(x) __builtin_expect(!!(x), 0)

char update_working_set_cpu(struct XMatrixSparse Xc,
    struct row_set relevant_row_set,
    Thread_Cache* thread_caches, Active_Set* as,
    X_uncompressed Xu, float* rowsum,
    bool* wont_update, int_fast64_t p, int_fast64_t n, float lambda,
    robin_hood::unordered_flat_map<int_fast64_t, float>* beta,
    int_fast64_t* updateable_items, int_fast64_t count_may_update,
    float* last_max, int_fast64_t depth)
{
    int_fast64_t* host_X = Xu.host_X;
    int_fast64_t* host_col_nz = Xu.host_col_nz;
    int_fast64_t* host_col_offsets = Xu.host_col_offsets;
    char increased_set = FALSE;
    int_fast64_t length_increase = 0;
    int_fast64_t total = 0, skipped = 0;
    int_fast64_t p_int = p * (p + 1) / 2;
    int_fast64_t int2_used = 0, int2_skipped = 0;

    if (depth > 2) {
        // TODO: quite a hack
        if (unlikely(NULL == inter_cache)) {
            // init inter cache
            inter_cache = (IC_Entry*)calloc(p_int, sizeof(IC_Entry)); // TODO: free
            for (int_fast64_t i = 0; i < p_int; i++) {
                inter_cache[i].present = false;
                inter_cache[i].was_present = false;
            }
        }
        for (int_fast64_t i = 0; i < p_int; i++) {
            inter_cache[i].present = false;
        }
    }

    int_fast64_t total_inter_cols = 0;
    int_fast64_t correct_k = 0;
#pragma omp parallel for reduction(+ \
                                   : total_inter_cols, total, skipped, int2_used, int2_skipped)
    for (int_fast64_t main_i = 0; main_i < count_may_update; main_i++) {
        // use Xc to read main effect
        Thread_Cache thread_cache = thread_caches[omp_get_thread_num()];
        int_fast64_t* col_i_cache = thread_cache.col_i;
        int_fast64_t* col_j_cache = thread_cache.col_j;
        int_fast64_t main = updateable_items[main_i];
        float max_inter_val = 0;
        int_fast64_t inter_cols = 0;
        robin_hood::unordered_flat_map<int_fast64_t, float> sum_with_col;
        // robin_hood::unordered_flat_map<int_fast64_t, float> sum_with_col =
        // thread_cache.lf_map; robin_hood::unordered_flat_map<int_fast64_t, >

        int_fast64_t main_col_len = Xu.host_col_nz[main];
        int_fast64_t* column_entries = &Xu.host_X[Xu.host_col_offsets[main]];

        // bool checked_interesting_cols = false;
        // if (main == interesting_col1) {
        //    printf("checking main col %ld\n", main);
        //}

        for (int_fast64_t entry_i = 0; entry_i < main_col_len; entry_i++) {
            int_fast64_t row_main = column_entries[entry_i];
            float rowsum_diff = rowsum[row_main];
            sum_with_col[main] += rowsum_diff;
            if (depth > 1) {
                int_fast64_t ri = 0;
                while (ri < relevant_row_set.row_lengths[row_main] && relevant_row_set.rows[row_main][ri] <= main)
                    ri++;
                for (; ri < relevant_row_set.row_lengths[row_main]; ri++) {
                    int_fast64_t inter = relevant_row_set.rows[row_main][ri];
                    // printf("checking pairwise %ld,%ld\n", main, inter); //TOOD:
                    // maintain separate lists so we can solve them in order
                    sum_with_col[inter] += rowsum_diff;
                    // if (!checked_interesting_cols && main == interesting_col1 && inter
                    // == interesting_col2) {
                    //    // printf(" adding %f to sum %ld,%ld. new total: %f\n",
                    //    rowsum_diff, main, inter, sum_with_col[inter]);
                    //    checked_interesting_cols = true;
                    //}
                    if (depth > 2) {
                        int_fast64_t k = (2 * (p - 1) + 2 * (p - 1) * (main - 1) - (main - 1) * (main - 1) - (main - 1)) / 2 + inter;
                        auto check_inter_cache = [&](int_fast64_t k) -> bool {
                            // return true;
                            // bool res = as_wont_update(Xu, lambda, inter_cache[k].last_max,
                            // inter_cache[k].last_rowsum, rowsum, inter_cache[k].col,
                            // thread_caches[omp_get_thread_num()].col_j); return res;
                            if (!inter_cache[k].present)
                                return true;
                            if (inter_cache[k].skipped_this_iter)
                                return false;
                            if (inter_cache[k].checked_this_iter)
                                return true;
                            // if (relevant_row_set.row_lengths[row_main] - ri <
                            // inter_cache[k].col.nz)
                            //    return true;
                            bool res = !as_wont_update(
                                Xu, lambda, inter_cache[k].last_max,
                                inter_cache[k].last_rowsum, rowsum, inter_cache[k].col,
                                thread_caches[omp_get_thread_num()].col_j);
                            inter_cache[k].checked_this_iter = true;
                            if (!res) {
                                inter_cache[k].skipped_this_iter = true;
                            }
                            return res;
                        };
                        // int_fast64_t k = pair_to_val(std::make_tuple(main, inter), p);
                        // if (!inter_cache.contains(k) || !as_wont_update(Xu, lambda,
                        // inter_cache[k].last_max, inter_cache[k].last_rowsum, rowsum,
                        // inter_cache[k].col, thread_caches[omp_get_thread_num()].col_j)) {
                        // if (!inter_cache.contains(k) || !as_pessimistic_est(lambda,
                        // rowsum, inter_cache[k].col)) { if (!inter_cache[k].present ||
                        // !as_pessimistic_est(lambda, rowsum, inter_cache[k].col)) { if
                        // (inter_cache[k].skip || !inter_cache[k].present ||
                        // relevant_row_set.row_lengths[row_main] - ri <
                        // inter_cache[k].col.nz || !as_wont_update(Xu, lambda,
                        // inter_cache[k].last_max, inter_cache[k].last_rowsum, rowsum,
                        // inter_cache[k].col, thread_caches[omp_get_thread_num()].col_j)) {
                        // if (!inter_cache[k].present || !inter_cache[k].skipped_this_iter
                        // && (relevant_row_set.row_lengths[row_main] - ri <
                        // inter_cache[k].col.nz || !as_wont_update(Xu, lambda,
                        // inter_cache[k].last_max, inter_cache[k].last_rowsum, rowsum,
                        // inter_cache[k].col, thread_caches[omp_get_thread_num()].col_j)))
                        // { if (true) { if (true) {
                        if (check_inter_cache(k)) {
                            int2_used++;
                            for (int_fast64_t ri2 = ri + 1;
                                 ri2 < relevant_row_set.row_lengths[row_main]; ri2++) {
                                int_fast64_t inter2 = relevant_row_set.rows[row_main][ri2];
                                int_fast64_t inter_ind = pair_to_val(std::make_tuple(inter, inter2), p);
                                // printf("checking triple %ld,%ld,%ld: diff %f\n", main, inter,
                                // inter2, rowsum_diff);
                                if (row_main == 0 && inter == 1 && inter2 == 2) {
                                    // printf("interesting col ind == %ld", inter_ind);
                                }
                                sum_with_col[inter_ind] += rowsum_diff;
                                // if (main == interesting_col && inter == interesting_col) {
                                //    // printf("appending %f to interesting col (%ld,%ld)\n",
                                //    rowsum_diff, main, inter);
                                //}
                            }
                        } else {
                            if (inter_cache[k].skipped_this_iter == true) {
                                int2_skipped++;
                            }
                            // inter_cache[k].skipped_this_iter = true;
                            int2_skipped++;
                        }
                        // inter_cache[k].skipped_this_iter = true; // interestingly enough
                        // this doesn't seem to break the results
                    }
                }
            }
        }

        robin_hood::unordered_flat_map<int_fast64_t, float> last_inter_max;
        // if (VERBOSE && main == interesting_col) {
        //    printf("interesting column sum %ld: %f\n", main, sum_with_col[main]);
        //}
        // if (main == interesting_col1)
        //    printf(" %ld sum with col %ld: %f\n", main, interesting_col2,
        //    sum_with_col[interesting_col2]);
        inter_cols = sum_with_col.size();
        total_inter_cols += inter_cols;
        auto curr_inter = sum_with_col.cbegin();
        auto last_inter = sum_with_col.cend();
        while (curr_inter != last_inter) {
            int_fast64_t tuple_val = curr_inter->first;
            float sum = std::abs(curr_inter->second);
            //            if (VERBOSE && tuple_val == main && main == interesting_col)
            //            {
            //                printf("%ld,sum: %f > %f (lambda)?\n", main, sum,
            //                lambda);
            //            } else if (tuple_val < p) {
            //                // printf("%ld,%ld, sum: %f > %f (lambda)?\n", main,
            //                tuple_val, sum, lambda);
            //            } else {
            //#ifdef NOT_R
            //                g_assert_true(tuple_val < p * p);
            //#endif
            //                // std::tuple<int_fast64_t,long> inter_pair_tmp =
            //                val_to_pair(tuple_val, p);
            //                // printf("%ld,%ld,%ld sum: %f > %f (lambda)?\n", main,
            //                std::get<0>(inter_pair_tmp),
            //                std::get<1>(inter_pair_tmp), sum, lambda);
            //            }
            max_inter_val = std::max(max_inter_val, sum);
            // printf("testing inter %ld, sum is %ld\n", inter, sum_with_col[inter]);
            if (sum > lambda * total_sqrt_error) {
                int_fast64_t a, b, c;
                int_fast64_t k;
                std::tuple<int_fast64_t, long> inter_pair = val_to_pair(tuple_val, p);
                if (tuple_val == main) {
                    a = main;
                    b = main; // TODO: unnecessary
                    c = main;
                    // k = pair_to_val(std::make_tuple(a, b), p);
                    k = main;
                } else if (tuple_val < p) {
                    a = main;
                    b = tuple_val;
                    c = main; // TODO: unnecessary
                    k = pair_to_val(std::make_tuple(a, b), p);
                    // if (a == interesting_col1 && b == interesting_col2)
                    //    printf("%ld, %ld: sum %f\n", a, b, sum);
                    // if (k < p) {
                    //    printf("(%ld,%ld|%ld): k = %ld\n", a, b, p, k);
                    //}
#ifdef NOT_R
                    g_assert_true(k >= p || k < p * p);
#endif
                } else {
#ifdef NOT_R
                    g_assert_true(tuple_val <= p * p);
#endif
                    // this is a three way interaction, update the last_inter_max of the
                    // relevant pair as well
                    a = main;
                    b = std::get<0>(inter_pair);
                    c = std::get<1>(inter_pair);
                    last_inter_max[b] = std::max(last_inter_max[b],
                        sum); // TODO: assumes that non-entries return 0.0
                    k = triplet_to_val(std::make_tuple(a, b, c), p);
                }
                int_fast64_t inter = std::get<0>(inter_pair);

                int_fast64_t* colA = &Xu.host_X[Xu.host_col_offsets[a]];
                int_fast64_t* colB = &Xu.host_X[Xu.host_col_offsets[b]];
                int_fast64_t* colC = &Xu.host_X[Xu.host_col_offsets[c]];
                int_fast64_t ib = 0, ic = 0;
                int_fast64_t inter_len = 0;
                for (int_fast64_t ia = 0; ia < Xu.host_col_nz[a]; ia++) {
                    int_fast64_t cur_row = colA[ia];
                    // if (a == b && a == c) {
                    //  printf("%ld: %ld ", ia, cur_row);
                    //}
                    while (colB[ib] < cur_row && ib < Xu.host_col_nz[b] - 1)
                        ib++;
                    while (colC[ic] < cur_row && ic < Xu.host_col_nz[c] - 1)
                        ic++;
                    if (cur_row == colB[ib] && cur_row == colC[ic]) {
                        // if (a == b && a == c) {
                        //  printf("\n%ld,%ld,%ld\n", ia, ib, ic);
                        //}
                        col_j_cache[inter_len] = cur_row;
                        inter_len++;
                    }
                }
#pragma omp critical
                active_set_append(as, k, col_j_cache, inter_len);
                increased_set = TRUE;
                total++;
            }
            if (depth > 2 && tuple_val != main && tuple_val < p) {
                int_fast64_t ik = (2 * (p - 1) + 2 * (p - 1) * (main - 1) - (main - 1) * (main - 1) - (main - 1)) / 2 + tuple_val;
                if (!inter_cache[ik].present) {
                    int_fast64_t a = main;
                    int_fast64_t b = tuple_val;
                    int_fast64_t* colA = &Xu.host_X[Xu.host_col_offsets[a]];
                    int_fast64_t* colB = &Xu.host_X[Xu.host_col_offsets[b]];
                    int_fast64_t ib = 0, ic = 0;
                    int_fast64_t inter_len = 0;
                    for (int_fast64_t ia = 0; ia < Xu.host_col_nz[a]; ia++) {
                        int_fast64_t cur_row = colA[ia];
                        // if (a == b && a == c) {
                        //  printf("%ld: %ld ", ia, cur_row);
                        //}
                        while (colB[ib] < cur_row && ib < Xu.host_col_nz[b] - 1)
                            ib++;
                        if (cur_row == colB[ib]) {
                            // if (a == b && a == c) {
                            //  printf("\n%ld,%ld,%ld\n", ia, ib, ic);
                            //}
                            col_j_cache[inter_len] = cur_row;
                            inter_len++;
                        }
                    }
                    // #pragma omp critical
                    update_inter_cache(ik, n, rowsum, last_inter_max[tuple_val],
                        col_j_cache, inter_len);
                } else {
                    update_inter_cache(ik, n, rowsum, last_inter_max[tuple_val],
                        col_j_cache, 0);
                }
            }
            curr_inter++;
        }
        // if (VERBOSE && main == interesting_col)
        //    printf("largest inter found for effect %ld was %f\n", main,
        //    max_inter_val);
        last_max[main] = max_inter_val;
        sum_with_col.clear();
    }

    // printf("total: %ld, skipped %ld, inter_cols %ld\n", total, skipped,
    // total_inter_cols); printf("int2 used: %ld, skipped %ld (%.0f\%)\n",
    // int2_used, int2_skipped, 100.0 * (double)int2_skipped /
    // (double)(int2_skipped + int2_used)); printf("as size: %ld,%ld,%ld\n",
    // as->entries1.size(), as->entries2.size(), as->entries3.size());
    return increased_set;
}

char update_working_set(X_uncompressed Xu, XMatrixSparse Xc,
    float* rowsum, bool* wont_update, int_fast64_t p, int_fast64_t n,
    float lambda,
    robin_hood::unordered_flat_map<int_fast64_t, float>* beta,
    int_fast64_t* updateable_items, int_fast64_t count_may_update,
    Active_Set* as, Thread_Cache* thread_caches,
    struct OpenCL_Setup* setup, float* last_max,
    int_fast64_t depth)
{
    struct row_set new_row_set = row_list_without_columns(Xc, Xu, wont_update, thread_caches);
    char increased_set = update_working_set_cpu(
        Xc, new_row_set, thread_caches, as, Xu, rowsum, wont_update, p, n, lambda,
        beta, updateable_items, count_may_update, last_max, depth);

    free_row_set(new_row_set);
    return increased_set;
}

static struct timespec start, end;
static float gpu_time = 0.0;
