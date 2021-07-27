typedef struct XMatrix {
    long** X;
    long actual_cols;
} XMatrix;

typedef struct column_set_entry {
    long value;
    long nextEntry;
} ColEntry;

typedef struct Column_Set {
    long size;
    ColEntry* cols;
} Column_Set;

struct row_set {
    long** rows;
    long* row_lengths;
    // S8bCol* s8b_rows;
};

typedef struct XMatrixSparse {
    pad_int* col_nz;
    long* col_nwords;
    unsigned long* col_start;
    // unsigned short **col_nz_indices;
    gsl_permutation* permutation;
    // S8bWord **compressed_indices;
    S8bWord* compressed_indices;
    long n;
    long p;
    long total_words;
    long total_entries;
    S8bCol* cols;
    S8bCol* rows;
    // Also include row entries for update_working_set
    //pad_long *row_nz;
    //long *row_nwords;
    //S8bWord *row_compressed_indices;
} XMatrixSparse;

void free_sparse_matrix(XMatrixSparse X);
typedef struct XMatrix_sparse_row {
    unsigned short** row_nz_indices;
    long* row_nz;
} XMatrix_sparse_row;

XMatrixSparse sparse_X2_from_X(long** X, long n, long p,
    long max_interaction_distance, long shuffle);
XMatrixSparse sparsify_X(long** X, long n, long p);

struct row_set row_list_without_columns(XMatrixSparse Xc, X_uncompressed Xu, bool* remove, Thread_Cache* thread_caches);
struct X_uncompressed construct_host_X(XMatrixSparse* Xc);