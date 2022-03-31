#include "robin_hood.h"
#include <cstdint>
Active_Set active_set_new(int_fast64_t max_length, int_fast64_t p);
void active_set_free(Active_Set as);
// void active_set_append(Active_Set* as, int_fast64_t value, int_fast64_t* col, int_fast64_t len, int_fast64_t n);
void active_set_remove(Active_Set* as, int_fast64_t value);
bool active_set_present(Active_Set* as, int_fast64_t value);

std::pair<bool, std::vector<int_fast64_t>> update_working_set(
    X_uncompressed Xu, XMatrixSparse Xc,
    float* rowsum, bool* wont_update, int_fast64_t p, int_fast64_t n,
    float lambda, int_fast64_t* updateable_items, int_fast64_t count_may_update, Active_Set* as,
    Thread_Cache* thread_caches,
    float* last_max, int_fast64_t depth, IndiCols* indicols, robin_hood::unordered_flat_set<int_fast64_t>* new_cols, int_fast64_t max_interaction_distance, const bool check_duplicates);

void free_inter_cache(int_fast64_t p);
//struct OpenCL_Setup setup_working_set_kernel(
//  X_uncompressed Xu, int_fast64_t n, int_fast64_t p);
//void opencl_cleanup(struct OpenCL_Setup setup);