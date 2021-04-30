typedef struct XMatrix {
  int **X;
  int actual_cols;
} XMatrix;

typedef struct column_set_entry {
  int value;
  int nextEntry;
} ColEntry;

typedef struct Column_Set {
  int size;
  ColEntry *cols;
} Column_Set;

typedef struct XMatrixSparse {
  pad_int *col_nz;
  int *col_nwords;
  unsigned long *col_start;
  // unsigned short **col_nz_indices;
  gsl_permutation *permutation;
  // S8bWord **compressed_indices;
  S8bWord *compressed_indices;
  long n;
  long p;
  long total_words;
  long total_entries;
  S8bCol *cols;
  S8bCol *rows;
  // Also include row entries for update_working_set
  //pad_int *row_nz;
  //int *row_nwords;
  //S8bWord *row_compressed_indices;
} XMatrixSparse;

void free_sparse_matrix(XMatrixSparse X);
typedef struct XMatrix_sparse_row {
  unsigned short **row_nz_indices;
  int *row_nz;
} XMatrix_sparse_row;

XMatrixSparse sparse_X2_from_X(int **X, int n, int p,
                               long max_interaction_distance, int shuffle);
XMatrixSparse sparsify_X(int **X, int n, int p);