#include "robin_hood.h"
#include <cstdint>
#include <omp.h>
#include <stdlib.h>
#include <xxhash.h>

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
    robin_hood::unordered_flat_map<int_fast64_t, AS_Entry>* entries;
    int_fast64_t p = as->p;
    if (value < p) {
        entries = &as->entries1;
    } else if (value < p * p) {
        entries = &as->entries2;
    } else {
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

typedef struct IC_Entry {
    bool skipped_this_iter;
    bool checked_this_iter;
    bool present;
    bool was_present;
    float last_max;
    float* last_rowsum;
    S8bCol col;
} IC_Entry;
static IC_Entry* inter_cache = NULL;
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
}

#define unlikely(x) __builtin_expect(!!(x), 0)

char update_working_set_cpu(struct XMatrixSparse Xc,
    struct row_set relevant_row_set,
    Thread_Cache* thread_caches, Active_Set* as,
    X_uncompressed Xu, float* rowsum,
    bool* wont_update, int_fast64_t p, int_fast64_t n, float lambda,
    int_fast64_t* updateable_items, int_fast64_t count_may_update,
    float* last_max, int_fast64_t depth, IndiCols* indicols, robin_hood::unordered_flat_set<int_fast64_t>* new_cols, int_fast64_t max_interaction_distance)
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
    int_fast64_t main_cols_used = 0;
    int_fast64_t skipped_pair_cols = 0, used_pair_cols = 0;
    robin_hood::unordered_flat_map<XXH64_hash_t, std::vector<int_fast64_t>> thread_new_cols_for_hash[NumCores];
    robin_hood::unordered_flat_set<int64_t> thread_new_skip_pair_ids[NumCores];
#pragma omp parallel for reduction(+ \
                                   : total_inter_cols, total, skipped, int2_used, int2_skipped, main_cols_used, skipped_pair_cols, used_pair_cols)
    for (int_fast64_t main_i = 0; main_i < count_may_update; main_i++) {
        // use Xc to read main effect
        int_fast64_t main = updateable_items[main_i];
        if (!indicols->defining_main_col_ids.contains(main)) {
            continue;
        }
        main_cols_used++;
        Thread_Cache thread_cache = thread_caches[omp_get_thread_num()];
        int_fast64_t* col_i_cache = thread_cache.col_i;
        int_fast64_t* col_j_cache = thread_cache.col_j;
        float max_inter_val = 0;
        int_fast64_t inter_cols = 0;
        robin_hood::unordered_flat_map<int_fast64_t, float> sum_with_col;
        robin_hood::unordered_flat_map<int_fast64_t, XXH3_state_t*> hash_with_col;

        int_fast64_t main_col_len = Xu.host_col_nz[main];
        int_fast64_t* column_entries = &Xu.host_X[Xu.host_col_offsets[main]];

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
                    if (inter - main > max_interaction_distance)
                        break;
                    auto inter_id = pair_to_val(std::tuple<int_fast64_t, int_fast64_t>(main, inter), p);
                    if (indicols->skip_pair_ids.contains(inter_id)) {
                        skipped_pair_cols++;
                        continue;
                    } else {
                        used_pair_cols++;
                    }
                    // printf("checking pairwise %ld,%ld\n", main, inter); //TOOD:
                    // maintain separate lists so we can solve them in order
                    sum_with_col[inter] += rowsum_diff;
                    if (!hash_with_col.contains(inter)) {
                        hash_with_col[inter] = XXH3_createState();
                        XXH3_64bits_reset(hash_with_col[inter]);
                    }
                    XXH3_64bits_update(hash_with_col[inter], &row_main, sizeof(int_fast64_t));
                    if (depth > 2) {
                        int_fast64_t k = (2 * (p - 1) + 2 * (p - 1) * (main - 1) - (main - 1) * (main - 1) - (main - 1)) / 2 + inter;
                        auto check_inter_cache = [&](int_fast64_t k) -> bool {
                            if (!inter_cache[k].present)
                                return true;
                            if (inter_cache[k].skipped_this_iter)
                                return false;
                            if (inter_cache[k].checked_this_iter)
                                return true;
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
                        if (check_inter_cache(k)) {
                            int2_used++;
                            for (int_fast64_t ri2 = ri + 1;
                                 ri2 < relevant_row_set.row_lengths[row_main]; ri2++) {
                                int_fast64_t inter2 = relevant_row_set.rows[row_main][ri2];
                                    if (inter2 - main > max_interaction_distance)
                                        break;
                                int_fast64_t inter_ind = pair_to_val(std::make_tuple(inter, inter2), p);
                                sum_with_col[inter_ind] += rowsum_diff;
                            }
                        } else {
                            if (inter_cache[k].skipped_this_iter == true) {
                                int2_skipped++;
                            }
                            int2_skipped++;
                        }
                    }
                }
            }
        }

        robin_hood::unordered_flat_map<int_fast64_t, float> last_inter_max;
        inter_cols = sum_with_col.size();
        total_inter_cols += inter_cols;
        for (auto inter_hash: hash_with_col) {
            int_fast64_t val = inter_hash.first;
            XXH3_state_t* hash_state = inter_hash.second;
            if (!new_cols->contains(main) && !new_cols->contains(val)) {
                XXH3_freeState(hash_state);
                continue;
            }
            // XXH64_state_t* hash_state = inter_hash.second;
            XXH64_hash_t hash_value = XXH3_64bits_digest(hash_state);
            XXH3_freeState(hash_state);
            auto pair_id = pair_to_val(std::tuple<int_fast64_t, int_fast64_t>(main, val), p);
            if (indicols->cols_for_hash.contains(hash_value) || thread_new_cols_for_hash[omp_get_thread_num()].contains(hash_value)) {
                thread_new_skip_pair_ids[omp_get_thread_num()].insert(pair_id);
            } else {
                thread_new_cols_for_hash[omp_get_thread_num()][hash_value].push_back(pair_id);
            }
        }
        for (auto curr_inter : sum_with_col) {
            int_fast64_t tuple_val = curr_inter.first;
            auto pair_id = pair_to_val(std::tuple<int_fast64_t, int_fast64_t>(main, tuple_val), p);
            if (indicols->skip_pair_ids.contains(pair_id) || thread_new_skip_pair_ids[omp_get_thread_num()].contains(pair_id)) {
                continue;
            }
            float sum = std::abs(curr_inter.second);
            max_inter_val = std::max(max_inter_val, sum);
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
                } else {
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
                    while (colB[ib] < cur_row && ib < Xu.host_col_nz[b] - 1)
                        ib++;
                    while (colC[ic] < cur_row && ic < Xu.host_col_nz[c] - 1)
                        ic++;
                    if (cur_row == colB[ib] && cur_row == colC[ic]) {
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
                        while (colB[ib] < cur_row && ib < Xu.host_col_nz[b] - 1)
                            ib++;
                        if (cur_row == colB[ib]) {
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
        }
        last_max[main] = max_inter_val;
        sum_with_col.clear();
    }
    for (int thread_id = 0; thread_id < NumCores; thread_id++) {
        for (auto val_key : thread_new_cols_for_hash[thread_id]) {
            // indicols.cols_for_hash
            int_fast64_t hash = val_key.first;
            std::vector<int_fast64_t> new_cols = val_key.second;
            for (auto col : new_cols) {
                indicols->cols_for_hash[hash].insert(col);
            }
        }
        for (auto val: thread_new_skip_pair_ids[thread_id]) {
            indicols->skip_pair_ids.insert(val);
        }
    }
    // printf("%.2f%% (%ld) main columns used\n", 100.0*(double)main_cols_used/(double)count_may_update, main_cols_used);
    // printf("%.2f%% (%ld) pair columns skipped\n", 100.0*(double)skipped_pair_cols/(double)(skipped_pair_cols+used_pair_cols), skipped_pair_cols);
    // printf("current unique cols: %ld\n", indicols->cols_for_hash.size());
    return increased_set;
}

char update_working_set(X_uncompressed Xu, XMatrixSparse Xc,
    float* rowsum, bool* wont_update, int_fast64_t p, int_fast64_t n,
    float lambda,
    int_fast64_t* updateable_items, int_fast64_t count_may_update,
    Active_Set* as, Thread_Cache* thread_caches,
    struct OpenCL_Setup* setup, float* last_max,
    int_fast64_t depth, IndiCols *indicols, robin_hood::unordered_flat_set<int_fast64_t>* new_cols,
    int_fast64_t max_interaction_distance)
{
    // const std::vector<int_fast64_t> new_cols(updateable_items, &updateable_items[count_may_update]);
    struct row_set new_row_set = row_list_without_columns(Xc, Xu, wont_update, thread_caches);
    update_main_indistinguishable_cols(Xu, wont_update, new_row_set, indicols, new_cols);
    char increased_set = update_working_set_cpu(
        Xc, new_row_set, thread_caches, as, Xu, rowsum, wont_update, p, n, lambda,
        updateable_items, count_may_update, last_max, depth, indicols, new_cols, max_interaction_distance);

    free_row_set(new_row_set);
    return increased_set;
}

static struct timespec start, end;
static float gpu_time = 0.0;
