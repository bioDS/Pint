Active_Set active_set_new(int max_length);
void active_set_free(Active_Set as);
void active_set_append(Active_Set *as, int value, int *col, int len);
void active_set_remove(Active_Set *as, int index);
int active_set_get_index(Active_Set *as, int index);

char update_working_set(
    struct X_uncompressed Xu, XMatrixSparse Xc,
    float* rowsum, bool* wont_update, int p, int n,
    float lambda, robin_hood::unordered_flat_map<long, float> *beta, int* updateable_items, int count_may_update, Active_Set* as,
    Thread_Cache *thread_caches, struct OpenCL_Setup *setup, float* last_max);

//struct OpenCL_Setup setup_working_set_kernel(
//  struct X_uncompressed Xu, int n, int p);
//void opencl_cleanup(struct OpenCL_Setup setup);