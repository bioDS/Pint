bool wont_update_effect(X_uncompressed X, float lambda, long k, float last_max,
    float* last_rowsum, float* rowsum, long* column_cache);
bool as_wont_update(X_uncompressed Xu, float lambda, float last_max, float* last_rowsum, float* rowsum, S8bCol col, long* column_cache);
bool as_pessimistic_est(float lambda, float* rowsum, S8bCol col);