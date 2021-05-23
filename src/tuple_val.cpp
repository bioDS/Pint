#include "liblasso.h"
//TODO: test these

std::tuple<long, long> val_to_pair(long val, long range) {
    int a = val / range;
    int b = val % range;
    return std::make_tuple(a, b);
}

std::tuple<long, long, long> val_to_triplet(long val, long range) {
    int a = val / (range*range);
    int b = (val-(a*range*range)) / (range);
    int c = val % range;
    return std::make_tuple(a, b, c);
}

long pair_to_val(std::tuple<long,long> tp, long range) {
    int a = std::get<0>(tp);
    int b = std::get<1>(tp);
    return a*range + b;
}

long triplet_to_val(std::tuple<long,long,long> tp, long range) {
    int a = std::get<0>(tp);
    int b = std::get<1>(tp);
    int c = std::get<2>(tp);
    return a*range*range + b*range + c;
}