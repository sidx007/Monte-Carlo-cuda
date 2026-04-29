#pragma once
#include <cstdint>
#include <vector>
#include <array>

inline constexpr int HALTON_PRIMES[20] = {
     2,  3,  5,  7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71
};

inline double radical_inverse(uint64_t n, int base) {
    double result = 0.0;
    double f      = 1.0 / static_cast<double>(base);
    uint64_t i    = n;
    while (i > 0) {
        result += f * static_cast<double>(i % base);
        i      /= base;
        f      /= static_cast<double>(base);
    }
    return result;
}

double scrambled_radical_inverse(uint64_t n, int base,
                                  const std::vector<int>& perm);

struct HaltonSampler {
    int      d_max;   
    uint64_t offset;  

    double sample(uint64_t n, int dim) const {
        return radical_inverse(n + offset, HALTON_PRIMES[dim]);
    }

    void fill(uint64_t n, double* out) const {
        for (int d = 0; d < d_max; ++d)
            out[d] = sample(n, d);
    }
};
