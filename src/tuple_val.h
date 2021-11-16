std::tuple<int_fast64_t, long> val_to_pair(int_fast64_t val, int_fast64_t range);
std::tuple<int_fast64_t, int_fast64_t, long> val_to_triplet(int_fast64_t val, int_fast64_t range);
int_fast64_t pair_to_val(std::tuple<int_fast64_t, long> tp, int_fast64_t range);
int_fast64_t triplet_to_val(std::tuple<int_fast64_t, int_fast64_t, long> tp, int_fast64_t range);