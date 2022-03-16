#include "liblasso.h"
#include "robin_hood.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <omp.h>
#include <xxhash.h>
// #include<glib-2.0/glib.h>

using namespace std;

XMatrixSparse sparsify_X(int_fast64_t** X, int_fast64_t n, int_fast64_t p)
{
    return sparse_X2_from_X(X, n, p, 1, FALSE);
}

void free_sparse_matrix(XMatrixSparse X)
{
    for (int_fast64_t i = 0; i < X.p; i++) {
        free(X.cols[i].compressed_indices);
    }
    free(X.cols);
}

void free_row_set(struct row_set rs)
{
    for (int i = 0; i < rs.num_rows; i++) {
        if (NULL != rs.rows[i])
            free(rs.rows[i]);
    }
    free(rs.row_lengths);
    free(rs.rows);
}

struct row_set row_list_without_columns(XMatrixSparse Xc, X_uncompressed Xu, bool* remove, Thread_Cache* thread_caches)
{
    int_fast64_t p = Xc.p;
    int_fast64_t n = Xc.n;
    struct row_set rs;
    rs.num_rows = n;
    int_fast64_t** new_rows = (int_fast64_t**)calloc(n, sizeof *new_rows);
    int_fast64_t* row_lengths = (int_fast64_t*)calloc(n, sizeof *row_lengths);

    // #pragma omp parallel for
    for (int_fast64_t row = 0; row < n; row++) {
        Thread_Cache thread_cache = thread_caches[omp_get_thread_num()];
        int_fast64_t* row_cache = thread_cache.col_i; // N.B col_i cache must be at least size p
        // check inverted list for interactions aint row_main
        // int_fast64_t *X_row = &Xu.host_X_row[Xu.host_row_offsets[row]];
        int_fast64_t row_pos = 0;
        for (int_fast64_t i = 0; i < Xu.host_row_nz[row]; i++) {
            int_fast64_t col = Xu.host_X_row[Xu.host_row_offsets[row] + i];
            // int_fast64_t col = X_row[i];
            if (!remove[col]) {
                row_cache[row_pos] = col;
                row_pos++;
            }
        }

        row_lengths[row] = row_pos;
        if (row_pos > 0)
            new_rows[row] = (int_fast64_t*)malloc(row_pos * sizeof *new_rows);
        memcpy(new_rows[row], row_cache, row_pos * sizeof *new_rows);
    }

    rs.rows = new_rows;
    rs.row_lengths = row_lengths;
    return rs;
}

std::vector<int_fast64_t> get_col_by_id(X_uncompressed Xu, int_fast64_t id) {
    std::vector<int_fast64_t> new_col;
    if (id < Xu.p) {
        int_fast64_t col_len = Xu.host_col_nz[id];
        int_fast64_t* entries = &Xu.host_X[Xu.host_col_offsets[id]];
        for (int i = 0; i < col_len; i++) {
            new_col.push_back(entries[i]);
        }
    } else if (id < Xu.p*Xu.p) {
        auto pair = val_to_pair(id, Xu.p);
        auto ida = std::get<0>(pair);
        auto idb = std::get<1>(pair);
        int_fast64_t ca_len = Xu.host_col_nz[ida];
        int_fast64_t* ca_entries = &Xu.host_X[Xu.host_col_offsets[ida]];
        int_fast64_t cb_len = Xu.host_col_nz[idb];
        int_fast64_t* cb_entries = &Xu.host_X[Xu.host_col_offsets[idb]];
        int_fast64_t ca_ind = 0, cb_ind = 0;
        while(ca_ind < ca_len && cb_ind < cb_len) {
            if (ca_entries[ca_ind] == cb_entries[cb_ind]) {
                new_col.push_back(ca_entries[ca_ind]);
                ca_ind++;
                cb_ind++;
                continue;
            }
            while(ca_ind < ca_len && ca_entries[ca_ind] < cb_entries[cb_ind])
                ca_ind++;
            while(cb_ind < cb_len && ca_entries[ca_ind] > cb_entries[cb_ind])
                cb_ind++;
        }
    } else {
        auto triple = val_to_triplet(id, Xu.p);
        auto ida = std::get<0>(triple);
        auto idb = std::get<1>(triple);
        auto idc = std::get<2>(triple);
        int_fast64_t ca_len = Xu.host_col_nz[ida];
        int_fast64_t* ca_entries = &Xu.host_X[Xu.host_col_offsets[ida]];
        int_fast64_t cb_len = Xu.host_col_nz[idb];
        int_fast64_t* cb_entries = &Xu.host_X[Xu.host_col_offsets[idb]];
        int_fast64_t cc_len = Xu.host_col_nz[idc];
        int_fast64_t* cc_entries = &Xu.host_X[Xu.host_col_offsets[idc]];
        int_fast64_t ca_ind = 0, cb_ind = 0, cc_ind = 0;
        while(ca_ind < ca_len && cb_ind < cb_len && cc_ind < cc_len) {
            if (ca_entries[ca_ind] == cb_entries[cb_ind] && ca_entries[ca_ind] == cc_entries[cc_ind]) {
                new_col.push_back(ca_entries[ca_ind]);
                ca_ind++;
                cb_ind++;
                cc_ind++;
                continue;
            }
            while(ca_ind < ca_len &&
                ca_entries[ca_ind] < std::max(cb_entries[cb_ind], cc_entries[cc_ind]))
                ca_ind++;
            while(cb_ind < cb_len &&
                cb_entries[cb_ind] < std::max(ca_entries[ca_ind], cc_entries[cc_ind]))
                cb_ind++;
            while(cc_ind < cc_len &&
                cc_entries[cc_ind] < std::max(ca_entries[ca_ind], cb_entries[cb_ind]))
                cc_ind++;
        }

    }
    return new_col;
}

SingleCol get_inter_col(X_uncompressed Xu, int_fast64_t cola, int_fast64_t colb) {
        int_fast64_t cola_len = Xu.host_col_nz[cola];
        int_fast64_t* cola_vals = &Xu.host_X[Xu.host_col_offsets[cola]];
        int_fast64_t colb_len = Xu.host_col_nz[colb];
        int_fast64_t* colb_vals = &Xu.host_X[Xu.host_col_offsets[colb]];
}

// bool check_cols_match(X_uncompressed Xu, int_fast64_t cola, int_fast64_t colb) {
    
// }

bool check_cols_match(std::vector<int_fast64_t> cola, std::vector<int_fast64_t> colb) {
    if (cola.size() != colb.size())
        return false;
    for (int i = 0; i < cola.size(); i++) {
        if (cola[i] != colb[i])
            return false;
    }
    return true;
}

IndiCols get_empty_indicols() {
    IndiCols id; // empty hash map is valid, this should be fine.
    return id;
}

void update_main_indistinguishable_cols(X_uncompressed Xu, bool* wont_update, struct row_set relevant_row_set, IndiCols *indi, robin_hood::unordered_flat_set<int_fast64_t>* new_cols)
{
    // auto cols_matching_defining_id = indi.cols_matching_defining_id;
    // int_fast64_t initial_unique_col_count = indi.defining_main_col_ids.size() + indi.defining_pair_ids.size();
    int_fast64_t total_cols_checked = 0;
    robin_hood::unordered_flat_map<int64_t, std::vector<int64_t>> new_col_ids_for_hashvalue;
    for (auto main : *new_cols) {
        total_cols_checked++;
        int_fast64_t main_col_len = Xu.host_col_nz[main];
        int_fast64_t* column_entries = &Xu.host_X[Xu.host_col_offsets[main]];
        XXH64_hash_t main_hash = XXH3_64bits(column_entries, main_col_len*sizeof(int_fast64_t));
        
        // uint64_t main_hash = XXHash64::hash(column_entries, main_col_len*sizeof(int_fast64_t));
        // main_hash = 5; //TODO: testing only
        // printf("col %ld hash %ld\n", main, main_hash);
        // new_col_ids_for_hashvalue[main_hash].push_back(main);
        if (!indi->cols_for_hash.contains(main_hash))
            indi->defining_main_col_ids.insert(main);
        indi->cols_for_hash[main_hash].insert(main);
        // indi->found_hashes.insert(main_hash);
    }
    
    // use rolling hash for interactions
    //for (auto main_col : new_cols) {
    //    int_fast64_t main_col_len = Xu.host_col_nz[main_col];
    //    int_fast64_t* column_entries = &Xu.host_X[Xu.host_col_offsets[main_col]];
    //    robin_hood::unordered_flat_map<int_fast64_t, XXH64_state_t*> int_col_hash;

    //    for (int_fast64_t col_pos_ind = 0; col_pos_ind < main_col_len; col_pos_ind++) {
    //        int_fast64_t col_pos = column_entries[col_pos_ind];

    //        int_fast64_t col_pos_row_len = relevant_row_set.row_lengths[col_pos];
    //        int_fast64_t* row_entries = relevant_row_set.rows[col_pos];
    //        int_fast64_t int_row_ind = 0;
    //        int_fast64_t temp_new_col_ind = 0;
    //        for (; int_row_ind < col_pos_row_len; int_row_ind++) {
    //            int_fast64_t int_row = row_entries[int_row_ind];
    //            // rule out new cols less than col_pos (to avoid duplicates)
    //            if (int_row < main_col) {
    //                while (temp_new_col_ind < new_cols.size()-1 && new_cols[temp_new_col_ind] < int_row)
    //                    temp_new_col_ind++;
    //                int_fast64_t tmp_new_col = new_cols[temp_new_col_ind];
    //                total_cols_checked++;
    //                if (int_row != tmp_new_col) { //TODO: broken?
    //                    if (!int_col_hash.contains(int_row)) {
    //                        int_col_hash[int_row] = XXH64_createState();
    //                        XXH64_reset(int_col_hash[int_row], 0);
    //                    }
    //                    // int_col_hash[int_row].add(&col_pos, sizeof(col_pos));
    //                    XXH64_update(int_col_hash[int_row], &col_pos, sizeof(int_fast64_t));
    //                }
    //            } else {
    //                total_cols_checked++;
    //                if (int_row != main_col) {
    //                    if (!int_col_hash.contains(int_row)) {
    //                        int_col_hash[int_row] = XXH64_createState();
    //                        XXH64_reset(int_col_hash[int_row], 0);
    //                    }
    //                    // int_col_hash[int_row].add(&col_pos, sizeof(col_pos));
    //                    XXH64_update(int_col_hash[int_row], &col_pos, sizeof(int_fast64_t));
    //                }
    //            }
    //        }
    //    }
    //    for (auto pair : int_col_hash) {
    //        int_fast64_t inter_col = pair.first;
    //        XXH64_state_t* hash_state = pair.second;
    //        uint64_t hash_result = XXH64_digest(hash_state);
    //        // hash_result = 5; //TODO: for testing only

    //        uint64_t inter_id = pair_to_val(std::tuple<int_fast64_t, int_fast64_t>(main_col, inter_col), Xu.p);
    //        if (!indi.cols_for_hash.contains(hash_result))
    //            indi.defining_pair_ids.insert(inter_id);
    //        indi.cols_for_hash[hash_result].insert(inter_id);
    //        // new_col_ids_for_hashvalue[hash_result].push_back(inter_id);
    //        // printf("inter %ld-%ld, hash: %ld\n", main_col, inter_col, hash_result);
    //    }
    //}

    // auto new_col_count = indi.defining_main_col_ids.size() + indi.defining_pair_ids.size() - initial_unique_col_count;
    // printf("new unique pairwise columns: %ld (%.1f%%)\n", new_col_count, double(100.0*new_col_count/(double)(total_cols_checked)));

    // IndiCols id = {cols_matching_defining_id, defining_col_ids_for_hashvalue};
}

/**
 * max_interaction_distance: interactions will be up to, but not including, this
 * distance from each other. set to 1 for no interactions.
 */
XMatrixSparse sparse_X2_from_X(int_fast64_t** X, int_fast64_t n, int_fast64_t p,
    int_fast64_t max_interaction_distance, int_fast64_t shuffle)
{
    XMatrixSparse X2;
    int_fast64_t colno, length;

    int_fast64_t iter_done = 0;
    int_fast64_t p_int = p * (p + 1) / 2;
    // TODO: for the moment we use the maximum possible p_int for allocation,
    // because things assume it.
    // TODO: this is wrong for dist == 1! (i.e. the main only case). Or at least,
    // so we hope.
    p_int = get_p_int(p, max_interaction_distance);
    if (max_interaction_distance < 0)
        max_interaction_distance = p;
    printf("p_int: %ld\n", p_int);

    X2.cols = (S8bCol*)malloc(sizeof *X2.cols * p_int);

    int_fast64_t done_percent = 0;
    int_fast64_t total_count = 0;
    int_fast64_t total_sum = 0;
    // size_t testcol = -INT_MAX;
    colno = 0;
    int_fast64_t d = max_interaction_distance;
    int_fast64_t limit_instead = ((p - d) * p - (p - d) * (p - d - 1) / 2 - (p - d));
// TODO: iter_done isn't exactly being updated safely
#pragma omp parallel for shared(X2, X, iter_done) private(length, colno) num_threads(NumCores) reduction(+ \
                                                                                                         : total_count, total_sum) schedule(static)
    for (int_fast64_t i = 0; i < p; i++) {
        for (int_fast64_t j = i; j < min(i + max_interaction_distance, (int_fast64_t)p); j++) {
            int_fast64_t val;
            // GQueue *current_col = g_queue_new();
            // GQueue *current_col_actual = g_queue_new();
            Queue* current_col = queue_new();
            // worked out by hand as being equivalent to the offset we would have
            // reached.
            int_fast64_t a = min(i, p - d); // number of iters limited by d.
            int_fast64_t b = max(i - (p - d), (int_fast64_t)0); // number of iters of i limited by p rather than d.
            // int_fast64_t tmp = j + b*(d) + a*p - a*(a-1)/2 - i;
            int_fast64_t suma = a * (d - 1);
            int_fast64_t k = max(p - d + b, (int_fast64_t)0);
            // sumb is the amount we would have reached w/o the limit - the amount
            // that was actually covered by the limit.
            int_fast64_t sumb = (k * p - k * (k - 1) / 2 - k) - limit_instead;
            colno = j + suma + sumb;
            // Read through the the current column entries, and append them to X2 as
            // an s8b-encoded list of offsets
            int_fast64_t* col_entries = (int_fast64_t*)malloc(60 * sizeof *col_entries);
            int_fast64_t count = 0;
            int_fast64_t largest_entry = 0;
            // int_fast64_t max_bits = max_size_given_entries[0];
            int_fast64_t diff = 0;
            int_fast64_t prev_row = -1;
            int_fast64_t total_nz_entries = 0;
            for (int_fast64_t row = 0; row < n; row++) {
                val = X[i][row] * X[j][row];
                if (val == 1) {
                    total_nz_entries++;
                    diff = row - prev_row;
                    total_sum += diff;
                    int_fast64_t used = 0;
                    int_fast64_t tdiff = diff;
                    while (tdiff > 0) {
                        used++;
                        tdiff >>= 1;
                    }
                    // max_bits = max_size_given_entries[count + 1];
                    // if the current diff won't fit in the s8b word, push the word and
                    // start a new one
                    if (max(used, largest_entry) > max_size_given_entries[count + 1]) {
                        S8bWord* word = (S8bWord*)malloc(sizeof(
                            S8bWord)); // we (maybe?) can't rely on this being the size of a
                        // pointer, so we'll add by reference
                        S8bWord tempword = to_s8b(count, col_entries);
                        total_count += count;
                        memcpy(word, &tempword, sizeof(S8bWord));
                        queue_push_tail(current_col, word);
                        count = 0;
                        largest_entry = 0;
                        // max_bits = max_size_given_entries[1];
                    }
                    // things for the next iter
                    // g_assert_true(count < 60);
                    col_entries[count] = diff;
                    count++;
                    if (used > largest_entry)
                        largest_entry = used;
                    prev_row = row;
                } else if (val != 0)
                    fprintf(stderr, "Attempted to convert a non-binary matrix, values "
                                    "will be missing!\n");
            }
            // push the last (non-full) word
            S8bWord* word = (S8bWord*)malloc(sizeof(S8bWord));
            S8bWord tempword = to_s8b(count, col_entries);
            memcpy(word, &tempword, sizeof(S8bWord));
            queue_push_tail(current_col, word);
            free(col_entries);
            length = queue_get_length(current_col);

            S8bWord* indices = (S8bWord*)malloc(sizeof *indices * length);
            count = 0;
            while (!queue_is_empty(current_col)) {
                S8bWord* current_word = (S8bWord*)queue_pop_head(current_col);
                indices[count] = *current_word;
                free(current_word);
                count++;
            }

            S8bCol new_col = { indices, total_nz_entries, length };
            X2.cols[colno] = new_col;

            queue_free(current_col);
            current_col = NULL;
        }
        iter_done++;
        if (p >= 100 && iter_done % (p / 100) == 0) {
            if (VERBOSE)
                printf("create interaction matrix, %ld%%\n", done_percent);
            done_percent++;
        }
    }
    int_fast64_t total_words = 0;
    int_fast64_t total_entries = 0;
    for (int_fast64_t i = 0; i < p_int; i++) {
        total_words += X2.cols[i].nwords;
        total_entries += X2.cols[i].nz;
    }
    printf("mean nz entries: %f\n", (float)total_entries / (float)p_int);
    printf("mean words: %f\n", (float)total_count / (float)total_words);
    printf("mean size: %f\n", (float)total_sum / (float)total_entries);
    X2.total_words = total_words;
    X2.total_entries = total_entries;

    S8bWord* compressed_indices;
    int_fast64_t* col_start;
    int_fast64_t* col_nz;
    int_fast64_t offset = 0;

    permutation_splits = NumCores;
    permutation_split_size = p_int / permutation_splits;
    final_split_size = p_int % permutation_splits;
    printf("%ld splits of size %ld\n", permutation_splits, permutation_split_size);
    printf("final split size: %ld\n", final_split_size);

    X2.n = n;
    X2.p = p_int;

    return X2;
}

void free_host_X(X_uncompressed* Xu)
{
    free(Xu->host_X);
    free(Xu->host_col_nz);
    free(Xu->host_col_offsets);
    free(Xu->host_X_row);
    free(Xu->host_row_nz);
    free(Xu->host_row_offsets);
}

X_uncompressed construct_host_X(XMatrixSparse* Xc)
{
    int_fast64_t* host_X = (int_fast64_t*)calloc(Xc->total_entries, sizeof(int_fast64_t));
    int_fast64_t* host_col_nz = (int_fast64_t*)calloc(Xc->p, sizeof(int_fast64_t));
    int_fast64_t* host_col_offsets = (int_fast64_t*)calloc(Xc->p, sizeof(int_fast64_t));
    int_fast64_t* host_X_row = (int_fast64_t*)calloc(Xc->total_entries, sizeof(int_fast64_t));
    int_fast64_t* host_row_nz = (int_fast64_t*)calloc(Xc->n, sizeof(int_fast64_t));
    int_fast64_t* host_row_offsets = (int_fast64_t*)calloc(Xc->n, sizeof(int_fast64_t));
    int_fast64_t p = Xc->p;
    int_fast64_t n = Xc->n;

    char *full_X = (char*)calloc(n * p, sizeof(char));

    // read through compressed matrix and construct continuous
    // uncompressed matrix
    size_t offset = 0;
    for (int k = 0; k < p; k++) {
        host_col_offsets[k] = offset;
        host_col_nz[k] = Xc->cols[k].nz;
        int_fast64_t* col = &host_X[offset];
        // read column
        {
            int_fast64_t col_entry_pos = 0;
            int_fast64_t entry = -1;
            for (int_fast64_t i = 0; i < Xc->cols[k].nwords; i++) {
                S8bWord word = Xc->cols[k].compressed_indices[i];
                uint_fast64_t values = word.values;
                for (int_fast64_t j = 0; j <= group_size[word.selector]; j++) {
                    int_fast64_t diff = values & masks[word.selector];
                    if (diff != 0) {
                        entry += diff;
                        col[col_entry_pos] = entry;
                        col_entry_pos++;
                        offset++;
                        full_X[entry*p+k] = 1;
                    }
                    values >>= item_width[word.selector];
                }
            }
        }
    }
    
    // for (int ri = 0; ri < n; ri++) {
    //     printf("%d: ", ri);
    //     for (int ci = 0; ci < p; ci++) {
    //         printf("%d ", full_X[ri*p+ci]);
    //     }
    //     printf("\n");
    // }

    // construct row-major indices.
    offset = 0;
    for (int_fast64_t row = 0; row < n; row++) {
        host_row_offsets[row] = offset;
        int_fast64_t row_nz = 0;
        for (int_fast64_t col = 0; col < p; col++) {
            if (full_X[row*p+col] == 1) {
                host_X_row[offset] = col;
                row_nz++;
                offset++;
            }
        }
        host_row_nz[row] = row_nz;
        // printf("row %d len: %d\n", row, row_nz);
        // printf("row %d offset %d\n", row, host_row_offsets[row]);
        // printf("row %d inds: ", row);
        // int to = host_row_offsets[row];
        // for (int i = 0; i < host_row_nz[row]; i++) {
        //     // printf("%d ", host_X_row[to + i]);
        //     printf(" %p,", &host_X_row[to + i]);
        // }
        // printf("\n");
    }

    free(full_X);
    X_uncompressed Xu;
    Xu.host_col_nz = host_col_nz;
    Xu.host_col_offsets = host_col_offsets;
    Xu.host_X = host_X;
    Xu.total_size = offset;
    Xu.host_row_nz = host_row_nz;
    Xu.host_row_offsets = host_row_offsets;
    Xu.host_X_row = host_X_row;
    Xu.n = n;
    Xu.p = p;

    return Xu;
}
