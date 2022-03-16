#include "liblasso.h"
#include <cstdint>
#include <type_traits>
//TODO: test these

std::tuple<int_fast64_t, int_fast64_t> val_to_pair(int_fast64_t val, int_fast64_t range)
{
    int_fast64_t a = val / range;
    int_fast64_t b = val % range;
    a -= 1;
    return std::make_tuple(a, b);
}

std::tuple<int_fast64_t, int_fast64_t, long> val_to_triplet(int_fast64_t val, int_fast64_t range)
{
    int_fast64_t a = val / (range * range);
    int_fast64_t b = (val - (a * range * range)) / (range);
    int_fast64_t c = val % range;
    a -= 1;
    return std::make_tuple(a, b, c);
}

int_fast64_t pair_to_val(std::tuple<int_fast64_t, long> tp, int_fast64_t range)
{
    int_fast64_t a = std::get<0>(tp);
    int_fast64_t b = std::get<1>(tp);
    if (a > b)
        std::swap(a,b);
    return (a + 1) * range + b;
}

int_fast64_t triplet_to_val(std::tuple<int_fast64_t, int_fast64_t, long> tp, int_fast64_t range)
{
    int_fast64_t a = std::get<0>(tp);
    int_fast64_t b = std::get<1>(tp);
    int_fast64_t c = std::get<2>(tp);
    return (a + 1) * range * range + b * range + c;
}