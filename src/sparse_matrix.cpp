#include "liblasso.h"
#include <algorithm>
#include <cstdlib>
#include <omp.h>

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
            // if (tmp != colno) {
            // segfault for debugger
            // printf("%ld != %ld\ni,j a,b sa,sb = %ld,%ld %ld,%ld %ld,%ld", tmp,
            // colno, i, j, a, b, suma, sumb);
            // (*(int_fast64_t*)NULL)++;
            // }

            // Read through the the current column entries, and append them to X2 as
            // an s8b-encoded list of offsets
            int_fast64_t* col_entries = (int_fast64_t*)malloc(60 * sizeof *col_entries);
            int_fast64_t count = 0;
            int_fast64_t largest_entry = 0;
            int_fast64_t max_bits = max_size_given_entries[0];
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
                    max_bits = max_size_given_entries[count + 1];
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
                        max_bits = max_size_given_entries[1];
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

struct X_uncompressed construct_host_X(XMatrixSparse* Xc)
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

    // construct row-major indices.
    offset = 0;
    for (int_fast64_t row = 0; row < n; row++) {
        host_row_offsets[row] = offset;
        int_fast64_t* col = &host_X[offset];
        int_fast64_t row_nz = 0;
        for (int_fast64_t col = 0; col < p; col++) {
            if (full_X[row*p+col] == 1) {
                host_X_row[offset] = col;
                row_nz++;
                offset++;
            }
        }
        host_row_nz[row] = row_nz;
    }

    free(full_X);
    struct X_uncompressed Xu;
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
