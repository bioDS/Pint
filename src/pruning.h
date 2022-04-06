bool wont_update_effect(X_uncompressed X, float lambda, int_fast64_t k, float last_max,
    float* last_rowsum, float* rowsum, int_fast64_t* column_cache, struct continuous_info* ci);
bool as_wont_update(X_uncompressed Xu, float lambda, float last_max, float* last_rowsum, float* rowsum, S8bCol col, int_fast64_t* column_cache, float col_max, bool use_cont);
bool as_pessimistic_est(float lambda, float* rowsum, S8bCol col);