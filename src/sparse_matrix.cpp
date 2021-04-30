#include "liblasso.h"
#include <algorithm>

using namespace std;

XMatrixSparse sparsify_X(int **X, int n, int p) {
  return sparse_X2_from_X(X, n, p, 1, FALSE);
}

void free_sparse_matrix(XMatrixSparse X) {
  for (int i = 0; i < X.p; i++) {
    free(X.cols[i].compressed_indices);
  }
  free(X.cols);
  gsl_permutation_free(X.permutation);
}

/**
 * max_interaction_distance: interactions will be up to, but not including, this
 * distance from each other. set to 1 for no interactions.
 */
XMatrixSparse sparse_X2_from_X(int **X, int n, int p,
                               long max_interaction_distance, int shuffle) {
  XMatrixSparse X2;
  long colno, val, length;

  int iter_done = 0;
  long p_int = p * (p + 1) / 2;
  // TODO: for the moment we use the maximum possible p_int for allocation,
  // because things assume it.
  // TODO: this is wrong for dist == 1! (i.e. the main only case). Or at least,
  // so we hope.
  p_int = get_p_int(p, max_interaction_distance);
  if (max_interaction_distance < 0)
    max_interaction_distance = p;
  printf("p_int: %d\n", p_int);

  // TODO: granted all these pointers are the same size, but it's messy
  // X2.compressed_indices = malloc(p_int * sizeof(int *));
  // X2.col_nz = malloc(p_int * sizeof(*X2.col_nz));
  // memset(X2.col_nz, 0, p_int * sizeof(*X2.col_nz));
  // X2.col_nwords = malloc(p_int * sizeof(int));
  // memset(X2.col_nwords, 0, p_int * sizeof(int));

  X2.cols = malloc(sizeof *X2.cols * p_int);

  int done_percent = 0;
  long total_count = 0;
  long total_sum = 0;
  // size_t testcol = -INT_MAX;
  colno = 0;
  long d = max_interaction_distance;
  long limit_instead = ((p - d) * p - (p - d) * (p - d - 1) / 2 - (p - d));
// TODO: iter_done isn't exactly being updated safely
#pragma omp parallel for shared(X2, X, iter_done) private(length, val, colno) num_threads(NumCores) reduction(+:total_count, total_sum) schedule(static)
  for (long i = 0; i < p; i++) {
    for (long j = i; j < min(i + max_interaction_distance, (long)p); j++) {
      // GQueue *current_col = g_queue_new();
      // GQueue *current_col_actual = g_queue_new();
      Queue *current_col = queue_new();
      // worked out by hand as being equivalent to the offset we would have
      // reached.
      long a = min(i, p - d); // number of iters limited by d.
      long b = max(i - (p - d),
                   0l); // number of iters of i limited by p rather than d.
      // long tmp = j + b*(d) + a*p - a*(a-1)/2 - i;
      long suma = a * (d - 1);
      long k = max(p - d + b, 0l);
      // sumb is the amount we would have reached w/o the limit - the amount
      // that was actually covered by the limit.
      long sumb = (k * p - k * (k - 1) / 2 - k) - limit_instead;
      colno = j + suma + sumb;
      // if (tmp != colno) {
      // segfault for debugger
      // printf("%ld != %ld\ni,j a,b sa,sb = %ld,%ld %ld,%ld %ld,%ld", tmp,
      // colno, i, j, a, b, suma, sumb);
      // (*(int*)NULL)++;
      // }

      // Read through the the current column entries, and append them to X2 as
      // an s8b-encoded list of offsets
      int *col_entries = malloc(60 * sizeof(int));
      int count = 0;
      int largest_entry = 0;
      int max_bits = max_size_given_entries[0];
      int diff = 0;
      int prev_row = -1;
      int total_nz_entries = 0;
      for (int row = 0; row < n; row++) {
        val = X[i][row] * X[j][row];
        if (val == 1) {
          total_nz_entries++;
          diff = row - prev_row;
          total_sum += diff;
          int used = 0;
          int tdiff = diff;
          while (tdiff > 0) {
            used++;
            tdiff >>= 1;
          }
          max_bits = max_size_given_entries[count + 1];
          // if the current diff won't fit in the s8b word, push the word and
          // start a new one
          if (max(used, largest_entry) > max_size_given_entries[count + 1]) {
            S8bWord *word = malloc(sizeof(
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
      S8bWord *word = malloc(sizeof(S8bWord));
      S8bWord tempword = to_s8b(count, col_entries);
      memcpy(word, &tempword, sizeof(S8bWord));
      queue_push_tail(current_col, word);
      free(col_entries);
      length = queue_get_length(current_col);

      S8bWord *indices = malloc(sizeof *indices * length);
      count = 0;
      while (!queue_is_empty(current_col)) {
        S8bWord *current_word = queue_pop_head(current_col);
        indices[count] = *current_word;
        free(current_word);
        count++;
      }

      S8bCol new_col = {indices, total_nz_entries, length};
      X2.cols[colno] = new_col;

      queue_free(current_col);
      current_col = NULL;
    }
    iter_done++;
    if (p >= 100 && iter_done % (p / 100) == 0) {
      printf("create interaction matrix, %d\%\n", done_percent);
      done_percent++;
    }
  }

  long total_words = 0;
  long total_entries = 0;
  for (int i = 0; i < p_int; i++) {
    total_words += X2.cols[i].nwords;
    total_entries += X2.cols[i].nz;
  }
  printf("mean nz entries: %f\n", (float)total_entries / (float)p_int);
  printf("mean words: %f\n", (float)total_count / (float)total_words);
  printf("mean size: %f\n", (float)total_sum / (float)total_entries);
  X2.total_words = total_words;
  X2.total_entries = total_entries;

  S8bWord *compressed_indices;
  unsigned long *col_start;
  int *col_nz;
  unsigned long offset = 0;
  if (p_int == p) {
    compressed_indices = malloc(total_words * sizeof *compressed_indices);
    col_start = malloc(p_int * sizeof *col_start);
    col_nz = malloc(p_int * sizeof *col_nz);
    offset = 0;
    for (int i = 0; i < p_int; i++) {
      col_start[i] = offset;
      col_nz[i] = X2.cols[i].nz;
      for (int w = 0; w < X2.cols[i].nwords; w++) {
        compressed_indices[col_start[i] + w] = X2.cols[i].compressed_indices[w];
        if (compressed_indices[col_start[i] + w].values !=
            X2.cols[i].compressed_indices[w].values) {
          printf("error!\n");
        }
        offset++;
      }
      total_words += X2.cols[i].nwords;
      total_entries += X2.cols[i].nz;
    }
  }
  X2.col_start = col_start;
  X2.compressed_indices = compressed_indices;
  // X2.col_nz = col_nz;

  gsl_rng *r;
  gsl_permutation *permutation = gsl_permutation_alloc(p_int);
  gsl_permutation_init(permutation);
  gsl_rng_env_setup();
  // permutation_splits is the number of splits excluding the final (smaller)
  // split
  permutation_splits = NumCores;
  permutation_split_size = p_int / permutation_splits;
  const gsl_rng_type *T = gsl_rng_default;
  // if (permutation_split_size > T->max) {
  //	permutation_split_size = T->max;
  //	permutation_splits = actual_p_int/permutation_split_size;
  //	if (actual_p_int % permutation_split_size != 0) {
  //		permutation_splits++;
  //	}
  //}
  final_split_size = p_int % permutation_splits;
  printf("%d splits of size %d\n", permutation_splits, permutation_split_size);
  printf("final split size: %d\n", final_split_size);
  r = gsl_rng_alloc(T);
  gsl_rng *thread_r[NumCores];
#pragma omp parallel for
  for (int i = 0; i < NumCores; i++)
    thread_r[i] = gsl_rng_alloc(T);

  if (shuffle == TRUE) {
    parallel_shuffle(permutation, permutation_split_size, final_split_size,
                     permutation_splits);
  }
  X2.permutation = permutation;
  global_permutation = permutation;
  global_permutation_inverse = gsl_permutation_alloc(permutation->size);
  gsl_permutation_inverse(global_permutation_inverse, permutation);

  X2.n = n;
  X2.p = p_int;

  return X2;
}