#include "robin_hood.h"
#include <cstdint>
#include <vector>
typedef struct XMatrix {
    int_fast64_t** X;
    int_fast64_t actual_cols;
} XMatrix;

typedef struct column_set_entry {
    int_fast64_t value;
    int_fast64_t nextEntry;
} ColEntry;

typedef struct Column_Set {
    int_fast64_t size;
    ColEntry* cols;
} Column_Set;

struct row_set {
    int_fast64_t** rows;
    int_fast64_t* row_lengths;
    int_fast64_t num_rows;
    // S8bCol* s8b_rows;
};

typedef struct XMatrixSparse {
    pad_int* col_nz;
    int_fast64_t* col_nwords;
    int_fast64_t* col_start;
    // unsigned short **col_nz_indices;
    // S8bWord **compressed_indices;
    S8bWord* compressed_indices;
    int_fast64_t n;
    int_fast64_t p;
    int_fast64_t total_words;
    int_fast64_t total_entries;
    S8bCol* cols;
    S8bCol* rows;
    // Also include row entries for update_working_set
    //pad_int_fast64_t *row_nz;
    //int_fast64_t *row_nwords;
    //S8bWord *row_compressed_indices;
} XMatrixSparse;

typedef struct {
    int_fast64_t len;
    int_fast64_t *vals;
} SingleCol;

void free_sparse_matrix(XMatrixSparse X);
typedef struct XMatrix_sparse_row {
    unsigned short** row_nz_indices;
    int_fast64_t* row_nz;
} XMatrix_sparse_row;

XMatrixSparse sparse_X2_from_X(int_fast64_t** X, int_fast64_t n, int_fast64_t p,
    int_fast64_t max_interaction_distance, int_fast64_t shuffle);
XMatrixSparse sparsify_X(int_fast64_t** X, int_fast64_t n, int_fast64_t p);

struct row_set row_list_without_columns(XMatrixSparse Xc, X_uncompressed Xu, bool* remove, Thread_Cache* thread_caches);
void free_row_set(struct row_set rs);
X_uncompressed construct_host_X(XMatrixSparse* Xc);
void free_host_X(X_uncompressed *Xu);
std::vector<int_fast64_t> update_main_indistinguishable_cols(X_uncompressed Xu, bool* wont_update, struct row_set relevant_row_set, IndiCols* last_result, robin_hood::unordered_flat_set<int_fast64_t>* new_cols);
std::vector<int_fast64_t> get_col_by_id(X_uncompressed Xu, int_fast64_t id);
IndiCols get_empty_indicols(int_fast64_t p);
bool check_cols_match(std::vector<int_fast64_t> cola, std::vector<int_fast64_t> colb);