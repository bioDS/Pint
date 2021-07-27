#include "liblasso.h"
//TODO: test these

std::tuple<long, long> val_to_pair(long val, long range)
{
    long a = val / range;
    long b = val % range;
    a -= 1;
    return std::make_tuple(a, b);
}

std::tuple<long, long, long> val_to_triplet(long val, long range)
{
    long a = val / (range * range);
    long b = (val - (a * range * range)) / (range);
    long c = val % range;
    a -= 1;
    return std::make_tuple(a, b, c);
}

long pair_to_val(std::tuple<long, long> tp, long range)
{
    long a = std::get<0>(tp);
    long b = std::get<1>(tp);
    return (a + 1) * range + b;
}

long triplet_to_val(std::tuple<long, long, long> tp, long range)
{
    long a = std::get<0>(tp);
    long b = std::get<1>(tp);
    long c = std::get<2>(tp);
    return (a + 1) * range * range + b * range + c;
}