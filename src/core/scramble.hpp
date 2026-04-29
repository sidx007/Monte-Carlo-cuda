#pragma once
#include <cstdint>
#include <vector>
#include <random>

inline std::vector<uint32_t> make_digital_shift(int num_dimensions, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> dist;
    std::vector<uint32_t> shift(num_dimensions);
    for (auto& s : shift) s = dist(rng);
    return shift;
}

void owen_scramble(std::vector<double>& points, int N, int d, uint32_t seed);
