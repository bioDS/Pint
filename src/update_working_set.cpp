#include "robin_hood.h"
#include <omp.h>
#include <stdlib.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "flat_hash_map.hpp"
#include "liblasso.h"
#ifdef NOT_R
#include <glib-2.0/glib.h>
#endif

#define TRUE 1
#define FALSE 0

struct CL_Source {
    char* buffer;
    size_t len;
};

#define MAX_FILE_SIZE 1e6
/// Reads the entire file *filename into a new buffer and returns it.
struct CL_Source read_file(char* filename)
{
    char* big_buf = malloc(MAX_FILE_SIZE);
    char* line_buf;
    char* actual_buf;

    FILE* fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr,
            "error reading file %s, the program will probably now crash\n",
            filename);
    }

    // read entire file
    size_t pos = 0;
    size_t line_size = 0;
    long bytes_read = 0;
    while ((bytes_read = getline(&line_buf, &line_size, fp)) > 0) {
        memcpy(&big_buf[pos], line_buf, bytes_read);
        pos += bytes_read;
    }
    actual_buf = malloc(pos);
    memcpy(actual_buf, big_buf, pos);

    free(line_buf);
    free(big_buf);
    struct CL_Source src = { actual_buf, pos };
    return src;
}

Active_Set active_set_new(long max_length, long p)
{
    Active_Set as;
    as.length = 0;
    as.max_length = max_length;
    as.permutation = NULL;
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
    if (NULL != as.permutation)
        gsl_permutation_free(as.permutation);
}

bool active_set_present(Active_Set* as, long value)
{
    robin_hood::unordered_flat_map<long, AS_Entry>* entries;
    long p = as->p;
    if (value < p) {
        entries = &as->entries1;
    } else if (value < p * p) {
        entries = &as->entries2;
    } else {
        entries = &as->entries3;
    }

    return (entries->contains(value) && entries->at(value).present);
}

void active_set_append(Active_Set* as, long value, long* col, long len)
{
    //if (value == pair_to_val(std::make_tuple(interesting_col, interesting_col), 100)) {
    //  printf("appending interesting col %ld to as\n", value);
    //}
    // printf("as, adding val %ld as ", value);
    robin_hood::unordered_flat_map<long, AS_Entry>* entries;
    long p = as->p;
    if (value < p) {
        //if (VERBOSE && value == interesting_col)
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
        long i = as->length;
        entries->insert_or_assign(value, e);
    }
    as->length++;
}

void active_set_remove(Active_Set* as, long value)
{
    robin_hood::unordered_flat_map<long, AS_Entry>* entries;
    long p = as->p;
    if (value < p) {
        entries = &as->entries1;
    } else if (value < p * p) {
        entries = &as->entries2;
    } else {
        entries = &as->entries3;
    }
    entries->at(value).present = FALSE; //TODO does this work?
    as->length--;
}

//int active_set_get_index(Active_Set* as, long index)
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
    // robin_hood::unordered_flat_map<long, float> last_rowsum;
    S8bCol col;
} IC_Entry;
// static robin_hood::unordered_flat_map<long, IC_Entry> inter_cache;
static IC_Entry* inter_cache = NULL;
// static bool inter_cache_init_done = false;
void free_inter_cache(long p) {
    if (NULL == inter_cache)
        return;

    long p_int = p * (p + 1) / 2;
    for (long i = 0; i < p_int; i++) {
        // if (NULL != inter_cache[i].last_rowsum) {
        if (inter_cache[i].was_present) {
            free(inter_cache[i].last_rowsum);
            free(inter_cache[i].col.compressed_indices);
        }
    }
    free(inter_cache);
}

void update_inter_cache(long k, long n, float* rowsum, float last_max, long* col, long col_len)
{
    //if (col_len < 100) {
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
    for (long i = 0; i < col_len; i++) {
        long entry = col[i];
        inter_cache[k].last_rowsum[i] = rowsum[entry];
    }
}

#define unlikely(x) __builtin_expect(!!(x), 0)

char update_working_set_cpu(
    struct XMatrixSparse Xc, struct row_set relevant_row_set, Thread_Cache* thread_caches, Active_Set* as,
    struct X_uncompressed Xu, float* rowsum, bool* wont_update, long p, long n,
    float lambda, robin_hood::unordered_flat_map<long, float>* beta, long* updateable_items, long count_may_update,
    float* last_max, long depth)
{
    long* host_X = Xu.host_X;
    long* host_col_nz = Xu.host_col_nz;
    long* host_col_offsets = Xu.host_col_offsets;
    char increased_set = FALSE;
    long length_increase = 0;
    long total = 0, skipped = 0;
    long p_int = p * (p + 1) / 2;
    long int2_used = 0, int2_skipped = 0;

    if (depth > 2) {
        //TODO: quite a hack
        if (unlikely(NULL == inter_cache)) {
            // init inter cache
            inter_cache = (IC_Entry*)calloc(p_int, sizeof(IC_Entry)); //TODO: free
            for (long i = 0; i < p_int; i++) {
                inter_cache[i].present = false;
            }
        }
        for (long i = 0; i < p_int; i++) {
            inter_cache[i].present = false;
        }
    }

    long total_inter_cols = 0;
    long correct_k = 0;
#pragma omp parallel for reduction(+ \
                                   : total_inter_cols, total, skipped, int2_used, int2_skipped)
    for (long main_i = 0; main_i < count_may_update; main_i++) {
        // use Xc to read main effect
        Thread_Cache thread_cache = thread_caches[omp_get_thread_num()];
        long* col_i_cache = thread_cache.col_i;
        long* col_j_cache = thread_cache.col_j;
        long main = updateable_items[main_i];
        float max_inter_val = 0;
        long inter_cols = 0;
        robin_hood::unordered_flat_map<long, float> sum_with_col;
        // robin_hood::unordered_flat_map<long, float> sum_with_col = thread_cache.lf_map;
        // robin_hood::unordered_flat_map<long, >

        long main_col_len = Xu.host_col_nz[main];
        long* column_entries = &Xu.host_X[Xu.host_col_offsets[main]];

        // bool checked_interesting_cols = false;
        //if (main == interesting_col1) {
        //    printf("checking main col %ld\n", main);
        //}

        for (long entry_i = 0; entry_i < main_col_len; entry_i++) {
            long row_main = column_entries[entry_i];
            float rowsum_diff = rowsum[row_main];
            sum_with_col[main] += rowsum_diff;
            if (depth > 1) {
                long ri = 0;
                while (ri < relevant_row_set.row_lengths[row_main] && relevant_row_set.rows[row_main][ri] <= main)
                    ri++;
                for (; ri < relevant_row_set.row_lengths[row_main]; ri++) {
                    long inter = relevant_row_set.rows[row_main][ri];
                    // printf("checking pairwise %ld,%ld\n", main, inter); //TOOD: maintain separate lists so we can solve them in order
                    sum_with_col[inter] += rowsum_diff;
                    //if (!checked_interesting_cols && main == interesting_col1 && inter == interesting_col2) {
                    //    // printf(" adding %f to sum %ld,%ld. new total: %f\n", rowsum_diff, main, inter, sum_with_col[inter]);
                    //    checked_interesting_cols = true;
                    //}
                    if (depth > 2) {
                        long k = (2 * (p - 1) + 2 * (p - 1) * (main - 1) - (main - 1) * (main - 1) - (main - 1)) / 2 + inter;
                        auto check_inter_cache = [&](long k) -> bool {
                            // return true;
                            //bool res = as_wont_update(Xu, lambda, inter_cache[k].last_max, inter_cache[k].last_rowsum, rowsum, inter_cache[k].col, thread_caches[omp_get_thread_num()].col_j);
                            //return res;
                            if (!inter_cache[k].present)
                                return true;
                            if (inter_cache[k].skipped_this_iter)
                                return false;
                            if (inter_cache[k].checked_this_iter)
                                return true;
                            //if (relevant_row_set.row_lengths[row_main] - ri < inter_cache[k].col.nz)
                            //    return true;
                            bool res = !as_wont_update(Xu, lambda, inter_cache[k].last_max, inter_cache[k].last_rowsum, rowsum, inter_cache[k].col, thread_caches[omp_get_thread_num()].col_j);
                            inter_cache[k].checked_this_iter = true;
                            if (!res) {
                                inter_cache[k].skipped_this_iter = true;
                            }
                            return res;
                        };
                        // long k = pair_to_val(std::make_tuple(main, inter), p);
                        // if (!inter_cache.contains(k) || !as_wont_update(Xu, lambda, inter_cache[k].last_max, inter_cache[k].last_rowsum, rowsum, inter_cache[k].col, thread_caches[omp_get_thread_num()].col_j)) {
                        // if (!inter_cache.contains(k) || !as_pessimistic_est(lambda, rowsum, inter_cache[k].col)) {
                        // if (!inter_cache[k].present || !as_pessimistic_est(lambda, rowsum, inter_cache[k].col)) {
                        // if (inter_cache[k].skip || !inter_cache[k].present || relevant_row_set.row_lengths[row_main] - ri < inter_cache[k].col.nz || !as_wont_update(Xu, lambda, inter_cache[k].last_max, inter_cache[k].last_rowsum, rowsum, inter_cache[k].col, thread_caches[omp_get_thread_num()].col_j)) {
                        // if (!inter_cache[k].present || !inter_cache[k].skipped_this_iter && (relevant_row_set.row_lengths[row_main] - ri < inter_cache[k].col.nz || !as_wont_update(Xu, lambda, inter_cache[k].last_max, inter_cache[k].last_rowsum, rowsum, inter_cache[k].col, thread_caches[omp_get_thread_num()].col_j))) {
                        // if (true) {
                        // if (true) {
                        if (check_inter_cache(k)) {
                            int2_used++;
                            for (long ri2 = ri + 1; ri2 < relevant_row_set.row_lengths[row_main]; ri2++) {
                                long inter2 = relevant_row_set.rows[row_main][ri2];
                                long inter_ind = pair_to_val(std::make_tuple(inter, inter2), p);
                                // printf("checking triple %ld,%ld,%ld: diff %f\n", main, inter, inter2, rowsum_diff);
                                if (row_main == 0 && inter == 1 && inter2 == 2) {
                                    // printf("interesting col ind == %ld", inter_ind);
                                }
                                sum_with_col[inter_ind] += rowsum_diff;
                                //if (main == interesting_col && inter == interesting_col) {
                                //    // printf("appending %f to interesting col (%ld,%ld)\n", rowsum_diff, main, inter);
                                //}
                            }
                        } else {
                            if (inter_cache[k].skipped_this_iter == true) {
                                int2_skipped++;
                            }
                            // inter_cache[k].skipped_this_iter = true;
                            int2_skipped++;
                        }
                        // inter_cache[k].skipped_this_iter = true; // interestingly enough this doesn't seem to break the results
                    }
                }
            }
        }

        robin_hood::unordered_flat_map<long, float> last_inter_max;
        //if (VERBOSE && main == interesting_col) {
        //    printf("interesting column sum %ld: %f\n", main, sum_with_col[main]);
        //}
        //if (main == interesting_col1)
        //    printf(" %ld sum with col %ld: %f\n", main, interesting_col2, sum_with_col[interesting_col2]);
        inter_cols = sum_with_col.size();
        total_inter_cols += inter_cols;
        auto curr_inter = sum_with_col.cbegin();
        auto last_inter = sum_with_col.cend();
        while (curr_inter != last_inter) {
            long tuple_val = curr_inter->first;
            float sum = std::abs(curr_inter->second);
//            if (VERBOSE && tuple_val == main && main == interesting_col) {
//                printf("%ld,sum: %f > %f (lambda)?\n", main, sum, lambda);
//            } else if (tuple_val < p) {
//                // printf("%ld,%ld, sum: %f > %f (lambda)?\n", main, tuple_val, sum, lambda);
//            } else {
//#ifdef NOT_R
//                g_assert_true(tuple_val < p * p);
//#endif
//                // std::tuple<long,long> inter_pair_tmp = val_to_pair(tuple_val, p);
//                // printf("%ld,%ld,%ld sum: %f > %f (lambda)?\n", main, std::get<0>(inter_pair_tmp), std::get<1>(inter_pair_tmp), sum, lambda);
//            }
            max_inter_val = std::max(max_inter_val, sum);
            // printf("testing inter %ld, sum is %ld\n", inter, sum_with_col[inter]);
            if (sum > lambda*total_sqrt_error) {
                long a, b, c;
                long k;
                std::tuple<long, long> inter_pair = val_to_pair(tuple_val, p);
                if (tuple_val == main) {
                    a = main;
                    b = main; //TODO: unnecessary
                    c = main;
                    // k = pair_to_val(std::make_tuple(a, b), p);
                    k = main;
                } else if (tuple_val < p) {
                    a = main;
                    b = tuple_val;
                    c = main; //TODO: unnecessary
                    k = pair_to_val(std::make_tuple(a, b), p);
                    //if (a == interesting_col1 && b == interesting_col2)
                    //    printf("%ld, %ld: sum %f\n", a, b, sum);
                    //if (k < p) {
                    //    printf("(%ld,%ld|%ld): k = %ld\n", a, b, p, k);
                    //}
#ifdef NOT_R
                    g_assert_true(k >= p || k < p * p);
#endif
                } else {
#ifdef NOT_R
                    g_assert_true(tuple_val <= p * p);
#endif
                    // this is a three way interaction, update the last_inter_max of the relevant pair as well
                    a = main;
                    b = std::get<0>(inter_pair);
                    c = std::get<1>(inter_pair);
                    last_inter_max[b] = std::max(last_inter_max[b], sum); //TODO: assumes that non-entries return 0.0
                    k = triplet_to_val(std::make_tuple(a, b, c), p);
                }
                long inter = std::get<0>(inter_pair);

                long* colA = &Xu.host_X[Xu.host_col_offsets[a]];
                long* colB = &Xu.host_X[Xu.host_col_offsets[b]];
                long* colC = &Xu.host_X[Xu.host_col_offsets[c]];
                long ib = 0, ic = 0;
                long inter_len = 0;
                for (long ia = 0; ia < Xu.host_col_nz[a]; ia++) {
                    long cur_row = colA[ia];
                    //if (a == b && a == c) {
                    //  printf("%ld: %ld ", ia, cur_row);
                    //}
                    while (colB[ib] < cur_row && ib < Xu.host_col_nz[b] - 1)
                        ib++;
                    while (colC[ic] < cur_row && ic < Xu.host_col_nz[c] - 1)
                        ic++;
                    if (cur_row == colB[ib] && cur_row == colC[ic]) {
                        //if (a == b && a == c) {
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
                long ik = (2 * (p - 1) + 2 * (p - 1) * (main - 1) - (main - 1) * (main - 1) - (main - 1)) / 2 + tuple_val;
                if (!inter_cache[ik].present) {
                    long a = main;
                    long b = tuple_val;
                    long* colA = &Xu.host_X[Xu.host_col_offsets[a]];
                    long* colB = &Xu.host_X[Xu.host_col_offsets[b]];
                    long ib = 0, ic = 0;
                    long inter_len = 0;
                    for (long ia = 0; ia < Xu.host_col_nz[a]; ia++) {
                        long cur_row = colA[ia];
                        //if (a == b && a == c) {
                        //  printf("%ld: %ld ", ia, cur_row);
                        //}
                        while (colB[ib] < cur_row && ib < Xu.host_col_nz[b] - 1)
                            ib++;
                        if (cur_row == colB[ib]) {
                            //if (a == b && a == c) {
                            //  printf("\n%ld,%ld,%ld\n", ia, ib, ic);
                            //}
                            col_j_cache[inter_len] = cur_row;
                            inter_len++;
                        }
                    }
                    // #pragma omp critical
                    update_inter_cache(ik, n, rowsum, last_inter_max[tuple_val], col_j_cache, inter_len);
                } else {
                    update_inter_cache(ik, n, rowsum, last_inter_max[tuple_val], col_j_cache, 0);
                }
            }
            curr_inter++;
        }
        //if (VERBOSE && main == interesting_col)
        //    printf("largest inter found for effect %ld was %f\n", main, max_inter_val);
        last_max[main] = max_inter_val;
        sum_with_col.clear();
    }

    // printf("total: %ld, skipped %ld, inter_cols %ld\n", total, skipped, total_inter_cols);
    // printf("int2 used: %ld, skipped %ld (%.0f\%)\n", int2_used, int2_skipped, 100.0 * (double)int2_skipped / (double)(int2_skipped + int2_used));
    // printf("as size: %ld,%ld,%ld\n", as->entries1.size(), as->entries2.size(), as->entries3.size());
    return increased_set;
}

char update_working_set(
    struct X_uncompressed Xu, XMatrixSparse Xc, float* rowsum, bool* wont_update, long p, long n,
    float lambda, robin_hood::unordered_flat_map<long, float>* beta, long* updateable_items, long count_may_update, Active_Set* as,
    Thread_Cache* thread_caches, struct OpenCL_Setup* setup, float* last_max, long depth)
{
    long p_int = p * (p + 1) / 2;

    // construct small Xc containing only the relevant columns.
    // in particular, we want the row index with no columns outside the updateable_items set.

    // printf("wont_update:\n");
    // for (long i = 0; i < p; i++) {
    //   if (wont_update[i])
    //     printf("%ld ", i);
    // }
    // printf("\n");
    //int count = 0;
    //for (long i = 0; i < p; i++)
    //    if (!wont_update[i])
    //        count++;
    //if (count_may_update != count) {
    //    printf("count_may_update was %ld, should have been %ld\n", count_may_update, count);
    //}
    //g_assert_true(count == count_may_update);
    //if (!wont_update[interesting_col1] && !wont_update[interesting_col2])
    //    printf(" both cols %ld,%ld may update\n", interesting_col1, interesting_col2);
    //bool f1 = false, f2 = false;
    //for (long i = 0; i < count_may_update; i++) {
    //    if (updateable_items[count_may_update] == interesting_col1)
    //        f1 = true;
    //    if (updateable_items[count_may_update] == interesting_col2)
    //        f2 = true;
    //}
    //printf("found: ");
    //if (f1)
    //    printf("f1, ");
    //if (f2)
    //    printf("f2, ");
    //printf("\n");
    struct row_set new_row_set = row_list_without_columns(Xc, Xu, wont_update, thread_caches);
    // quick test:
    //bool found_both = false;
    //for (long row = 0; row < n; row++) {
    //    long found = 0;
    //    bool found1 = false, found2 = false;
    //    long f1pos = -1, f2pos = -2;
    //    for (long inter_i = 0; inter_i < Xu.host_row_nz[row]; inter_i++) {
    //        long col = Xu.host_X_row[Xu.host_row_offsets[row] + inter_i];
    //        if (!wont_update[col]) {
    //            if (!found_both) {
    //                if (!found_both && new_row_set.rows[row][found] == interesting_col1) {
    //                    found1 = true;
    //                    f1pos = found;
    //                }
    //                if (!found_both && new_row_set.rows[row][found] == interesting_col2) {
    //                    found2 = true;
    //                    f2pos = found;
    //                }
    //                if (found1 && found2) {
    //                    // printf("found both in row %ld at positions %ld,%ld\n", row, f1pos, f2pos);
    //                    found_both = true;
    //                }
    //            }
    //            g_assert_true(new_row_set.rows[row][found] == col);
    //            found++;
    //        }
    //    }
    //    g_assert_true(found == new_row_set.row_lengths[row]);
    //}
    char increased_set = update_working_set_cpu(Xc, new_row_set, thread_caches, as, Xu, rowsum, wont_update, p, n, lambda, beta, updateable_items, count_may_update, last_max, depth);
    for (long i = 0; i < n; i++) {
        if(NULL != new_row_set.rows[i])
            free(new_row_set.rows[i]);
    }
    free(new_row_set.rows);
    free(new_row_set.row_lengths);

    return increased_set;
}

static struct timespec start, end;
static float gpu_time = 0.0;
