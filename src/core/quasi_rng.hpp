#pragma once
#include <cstdint>

// advances state in-place and returns a uniform double in [0, 1)
double lcg_next_uniform(uint32_t& state);