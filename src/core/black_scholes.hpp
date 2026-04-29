#pragma once
#include "math_utils.hpp"

inline double cnd(double d) {
    return 0.5 * std::erfc(-d * M_SQRT1_2);
}

double bs_call(double S, double X, double t, double v, double r);