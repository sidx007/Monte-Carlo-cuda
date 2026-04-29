#pragma once
#include "sobol_joe_kuo.hpp"
#include <cstdint>
#include <vector>
#include <array>

class SobolGenerator {
public:
    explicit SobolGenerator(int num_dimensions);

    void generate(int N, std::vector<double>& points) const;

    void point(uint32_t n, double* out) const;

    void set_digital_shift(const std::vector<uint32_t>& shifts);

private:
    int                              d_;
    std::vector<std::array<uint32_t, 32>> V_;  
    std::vector<uint32_t>            shift_;    

    void init_direction_numbers();
};
