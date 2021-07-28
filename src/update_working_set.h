Active_Set active_set_new(long max_length, long p);
void active_set_free(Active_Set as);
// void active_set_append(Active_Set* as, long value, long* col, long len, long n);
void active_set_remove(Active_Set* as, long value);
bool active_set_present(Active_Set* as, long value);

char update_working_set(
    struct X_uncompressed Xu, XMatrixSparse Xc,
    float* rowsum, bool* wont_update, long p, long n,
    float lambda, robin_hood::unordered_flat_map<long, float>* beta, long* updateable_items, long count_may_update, Active_Set* as,
    Thread_Cache* thread_caches, struct OpenCL_Setup* setup, float* last_max, long depth);

void free_inter_cache(long p);
//struct OpenCL_Setup setup_working_set_kernel(
//  struct X_uncompressed Xu, long n, long p);
//void opencl_cleanup(struct OpenCL_Setup setup);