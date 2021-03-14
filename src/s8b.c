#include <liblasso.h>

S8bWord to_s8b(int count, int *vals) {
  S8bWord word;
  word.values = 0;
  word.selector = 0;
  int t = 0;
  // TODO: improve on this
  while (group_size[t] >= count && t < 16)
    t++;
  word.selector = t - 1;
  unsigned long test = 0;
  for (int i = 0; i < count; i++) {
    test |= vals[count - i - 1];
    if (i < count - 1)
      test <<= item_width[word.selector];
  }
  word.values = test;
  return word;
}

S8bCol to_s8b_col(int size, int *col) {
  // Read through the the current column entries, and append them as an
  // s8b-encoded list of offsets
  int col_entries[60];
  int count = 0;
  int largest_entry = 0;
  int max_bits = max_size_given_entries[0];
  int diff = 0;
  int prev_row = -1;
  int total_nz_entries = 0;
  Queue *current_col = queue_new();
  Queue *current_col_actual = queue_new();
  for (int row = 0; row < size; row++) {
    int val = col[row] * col[row];
    if (val == 1) {
      total_nz_entries++;
      diff = row - prev_row;
      int used = 0;
      int tdiff = diff;
      while (tdiff > 0) {
        used++;
        tdiff >>= 1;
      }
      max_bits = max_size_given_entries[count + 1];
      // if the current diff won't fit in the s8b word, push the word and start
      // a new one
      if (max(used, largest_entry) > max_size_given_entries[count + 1]) {
        S8bWord *word = malloc(
            sizeof(S8bWord)); // we (maybe?) can't rely on this being the size
                              // of a pointer, so we'll add by reference
        S8bWord tempword = to_s8b(count, col_entries);
        memcpy(word, &tempword, sizeof(S8bWord));
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
      prev_row = row;
    } else if (val != 0)
      fprintf(stderr, "Attempted to convert a non-binary matrix, values will "
                      "be missing!\n");
  }
  // push the last (non-full) word
  S8bWord *word = malloc(sizeof(S8bWord));
  S8bWord tempword = to_s8b(count, col_entries);
  memcpy(word, &tempword, sizeof(S8bWord));
  queue_push_tail(current_col, word);
  int length = queue_get_length(current_col);

  // push all our words to an array in the new col
  S8bCol s8bCol;
  s8bCol.compressed_indices = malloc(length * sizeof(S8bWord));
  s8bCol.col_nz = total_nz_entries;
  s8bCol.col_nwords = length;
  count = 0;
  while (!queue_is_empty(current_col)) {
    S8bWord *current_word = queue_pop_head(current_col);
    s8bCol.compressed_indices[count] = *current_word;
    free(current_word);
    count++;
  }
}