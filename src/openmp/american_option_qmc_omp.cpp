#include "core/sobol.hpp"
#include "core/scramble.hpp"
#include "core/moro_inv_cnd.hpp"
#include "core/black_scholes.hpp"
#include "core/brownian_bridge.hpp"
#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

double price_american_call_qmc_omp(const OptionParams& p,
                                    int num_threads = 0,
                                    uint32_t seed   = 42)
{
    if (num_threads > 0) omp_set_num_threads(num_threads);

    const int    m        = p.m;
    const double dt       = p.T / static_cast<double>(m + 1);
    const double discount = std::exp(-p.r * dt);

    SobolGenerator gen(m);
    gen.set_digital_shift(make_digital_shift(m, seed));
    std::vector<double> u_flat;
    gen.generate(p.N, u_flat);
    // NOTE: gen.generate() stores in DIMENSION-MAJOR order:
    // u_flat[d * p.N + n] = Sobol value for dimension d, path n

    std::vector<double> z_flat(u_flat.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(u_flat.size()); ++i) {
        double u = std::max(1e-10, std::min(u_flat[i], 1.0 - 1e-10));
        z_flat[i] = moro_inv_cnd(u);
    }
    // z_flat is also dimension-major: z_flat[d * p.N + n]

    auto bridge = build_brownian_bridge(m, dt);

    double total = 0.0;

    #pragma omp parallel reduction(+:total)
    {
        std::vector<double> z_path(m);   // FIX: per-thread path-ordered z buffer
        std::vector<double> S_path(m + 1);

        #pragma omp for schedule(static)
        for (int n = 0; n < p.N; ++n) {

            // FIX: transpose dimension-major → path-major for this path
            // BUG WAS: const double* z = &z_flat[n * m];
            //   This reads z_flat[n*m], z_flat[n*m+1], ..., z_flat[n*m+m-1]
            //   which are NOT the m dimensions of path n — they are
            //   all from the same dimension block, giving a wrong price (~7.19)
            // CORRECT: for path n, dimension d lives at index d*N+n
            for (int d = 0; d < m; ++d) {
                z_path[d] = z_flat[d * p.N + n];
            }

            simulate_path_bb(z_path.data(), bridge, p.S0, p.r, p.v, dt, m,
                             S_path.data());

            double c = bs_call(S_path[m - 1], p.X, dt, p.v, p.r);
            for (int i = m - 1; i >= 1; --i) {
                double continuation = c * discount;
                double intrinsic    = S_path[i] - p.X;
                c = std::max(intrinsic, continuation);
            }
            total += c;
        }
    }

    return (total / static_cast<double>(p.N)) * std::exp(-p.r * p.T);
}