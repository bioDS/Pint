bool wont_update_effect(X_uncompressed X, float lambda, int k, float last_max,
    float* last_rowsum, float* rowsum, int* column_cache);
bool as_wont_update(X_uncompressed Xu, float lambda, float last_max, float* last_rowsum, float* rowsum, S8bCol col, int* column_cache);