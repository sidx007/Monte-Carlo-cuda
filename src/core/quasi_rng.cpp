#include "quasi_rng.hpp"

double lcg_next_uniform(uint32_t& state) {
    state = 1664525u * state + 1013904223u;
    // divide by 2^32
    return static_cast<double>(state) * 2.3283064365386963e-10;
}