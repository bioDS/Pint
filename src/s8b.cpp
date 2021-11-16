#include "liblasso.h"
#include <algorithm>

using namespace std;

S8bWord to_s8b(int_fast64_t count, int_fast64_t* vals)
{
    S8bWord word;
    word.values = 0;
    word.selector = 0;
    int_fast64_t t = 0;
    word.selector = selector_given_count[count];
    int_fast64_t test = 0;
    for (int_fast64_t i = 0; i < count; i++) {
        test |= vals[count - i - 1];
        if (i < count - 1)
            test <<= item_width[word.selector];
    }
    word.values = test;
    return word;
}

S8bCol col_to_s8b_col(int_fast64_t size, int_fast64_t* col)
{
    // Read through the the current column entries, and append them as an
    // s8b-encoded list of offsets
    // printf("writing new s8b col of length %ld\n", size);
    int_fast64_t col_entries[60];
    int_fast64_t count = 0;
    int_fast64_t largest_entry = 0;
    int_fast64_t max_bits = max_size_given_entries[0];
    int_fast64_t diff = 0;
    int_fast64_t total_nz_entries = 0;
    Queue* current_col = queue_new();
    Queue* current_col_actual = queue_new();
    int_fast64_t prev_entry = -1;
    for (int_fast64_t entry_ind = 0; entry_ind < size; entry_ind++) {
        int_fast64_t entry = col[entry_ind];
        total_nz_entries++;
        diff = entry - prev_entry;
        int_fast64_t used = 0;
        int_fast64_t tdiff = diff;
        while (tdiff > 0) {
            used++;
            tdiff >>= 1;
        }
        max_bits = max_size_given_entries[count + 1];
        // if the current diff won't fit in the s8b word, push the word and start
        // a new one
        if (max(used, largest_entry) > max_size_given_entries[count + 1]) {
            S8bWord* word = (S8bWord*)malloc(sizeof *word);
            S8bWord tempword = to_s8b(count, col_entries);
            memcpy(word, &tempword, sizeof *word);
            queue_push_tail(current_col, word);
            count = 0;
            largest_entry = 0;
            max_bits = max_size_given_entries[1];
        }
        // things for the next iter
        col_entries[count] = diff;
        count++;
        if (used > largest_entry)
            largest_entry = used;
        prev_entry = entry;
    }
    // push the last (non-full) word
    S8bWord* word = (S8bWord*)malloc(sizeof(S8bWord));
    S8bWord tempword = to_s8b(count, col_entries);
    memcpy(word, &tempword, sizeof(S8bWord));
    queue_push_tail(current_col, word);
    int_fast64_t length = queue_get_length(current_col);

    // push all our words to an array in the new col
    S8bCol s8bCol;
    s8bCol.compressed_indices = (S8bWord*)malloc(length * sizeof(S8bWord));
    s8bCol.nz = total_nz_entries;
    s8bCol.nwords = length;
    count = 0;
    while (!queue_is_empty(current_col)) {
        S8bWord* current_word = (S8bWord*)queue_pop_head(current_col);
        s8bCol.compressed_indices[count] = *current_word;
        free(current_word);
        count++;
    }
    queue_free(current_col);
    queue_free(current_col_actual);
    return s8bCol;
}