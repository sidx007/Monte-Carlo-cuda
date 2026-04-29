#include "../core/black_scholes.hpp"
#include "../core/quasi_rng.hpp"
#include "../core/moro_inv_cnd.hpp"
#include "../core/math_utils.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

double price_american_call_serial(const OptionParams& p) {
    const double dt    = p.T / static_cast<double>(p.m + 1);
    const double sqdt  = std::sqrt(dt);
    const double drift = (p.r - 0.5 * p.v * p.v) * dt;
    const double discount = std::exp(-p.r * dt);

    double sum = 0.0;
    std::vector<double> S(p.m + 1);

    for (int path = 0; path < p.N; ++path) {
        uint32_t seed = static_cast<uint32_t>(path + 1) * 1234567u;

        S[0] = p.S0;
        for (int i = 1; i <= p.m; ++i) {
            double u = lcg_next_uniform(seed);
            double z = moro_inv_cnd(u);
            S[i] = S[i-1] * std::exp(drift + p.v * sqdt * z);
        }

        double c = bs_call(S[p.m - 1], p.X, dt, p.v, p.r);
        for (int i = p.m - 1; i >= 1; --i) {
            double continuation = c * discount;
            double intrinsic    = S[i] - p.X;
            c = std::max(intrinsic, continuation);
        }

        sum += c;
    }

    return (sum / static_cast<double>(p.N)) * std::exp(-p.r * p.T);
}