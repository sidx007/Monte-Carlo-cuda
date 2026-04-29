#include "scramble.hpp"
#include <algorithm>

void owen_scramble(std::vector<double>& points, int N, int d, uint32_t seed) {
    static constexpr int BITS = 32;
    static constexpr double SCALE = 1.0 / (1ULL << BITS);

    std::mt19937 rng(seed);

    std::vector<uint32_t> fixed(static_cast<size_t>(N) * d);
    for (int i = 0; i < N * d; ++i)
        fixed[i] = static_cast<uint32_t>(points[i] / SCALE);

    for (int dim = 0; dim < d; ++dim) {
        for (int b = 0; b < BITS; ++b) {
            uint32_t mask_high = (b == 0) ? 0u : (~0u << (BITS - b));
            uint32_t flip_bit  = 1u << (BITS - 1 - b);

            for (int n = 0; n < N; ++n) {
                uint32_t& val    = fixed[n * d + dim];
                uint32_t  prefix = val & mask_high;
                uint32_t coin_seed = seed ^ (dim * 1000003u)
                                          ^ (b   * 10007u)
                                          ^ prefix;
                std::mt19937 coin_rng(coin_seed);
                if (coin_rng() & 1u) val ^= flip_bit;
            }
        }
    }

    for (int i = 0; i < N * d; ++i)
        points[i] = static_cast<double>(fixed[i]) * SCALE;
}
