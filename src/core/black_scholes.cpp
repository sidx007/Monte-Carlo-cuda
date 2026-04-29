#include "black_scholes.hpp"
#include <cmath>
#include <algorithm>

double bs_call(double S, double X, double t, double v, double r) {
    if (t <= 0.0) return std::max(S - X, 0.0);
    double sqrt_t = std::sqrt(t);
    double d1 = (std::log(S / X) + (r + 0.5 * v * v) * t) / (v * sqrt_t);
    double d2 = d1 - v * sqrt_t;
    return S * cnd(d1) - X * std::exp(-r * t) * cnd(d2);
}