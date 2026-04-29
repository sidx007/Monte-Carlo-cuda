#include "halton.hpp"
#include <cassert>

double scrambled_radical_inverse(uint64_t n, int base,
                                  const std::vector<int>& perm) {
    assert(static_cast<int>(perm.size()) >= base);
    double result = 0.0;
    double f      = 1.0 / static_cast<double>(base);
    uint64_t i    = n;
    while (i > 0) {
        int digit  = static_cast<int>(i % base);
        result    += f * static_cast<double>(perm[digit]);
        i         /= base;
        f         /= static_cast<double>(base);
    }
    return result;
}
