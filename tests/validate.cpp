#include "../src/core/math_utils.hpp"
#include "../src/core/black_scholes.hpp"
#include "../src/core/moro_inv_cnd.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>

double price_american_call_serial(const OptionParams&);

static void test_bs_formula() {
    // Hull textbook: S=42, X=40, T=0.5, r=0.1, v=0.2 -> ~4.76
    double c = bs_call(42.0, 40.0, 0.5, 0.2, 0.1);
    printf("[test_bs_formula] c=%.4f (expected ~4.76)\n", c);
    assert(std::fabs(c - 4.76) < 0.01);
}

static void test_moro() {
    // Phi^-1(0.975) ~ 1.96
    double z = moro_inv_cnd(0.975);
    printf("[test_moro] moro(0.975)=%.4f (expected ~1.96)\n", z);
    assert(std::fabs(z - 1.96) < 0.01);
}

static void test_american_ge_european() {
    OptionParams p;
    p.S0=100; p.X=100; p.T=1.0; p.r=0.05; p.v=0.2; p.m=10; p.N=100000;
    double american = price_american_call_serial(p);
    double european = bs_call(p.S0, p.X, p.T, p.v, p.r);
    printf("[test_american_ge_european] american=%.4f european=%.4f\n", american, european);
    assert(american >= european - 0.5);
}

static void test_convergence_vs_exercise_points() {
    double prev = 0.0;
    for (int m = 1; m <= 10; ++m) {
        OptionParams p;
        p.S0=100; p.X=90; p.T=1.0; p.r=0.05; p.v=0.3; p.m=m; p.N=50000;
        double price = price_american_call_serial(p);
        printf("[test_convergence] m=%d price=%.4f\n", m, price);
        assert(price >= prev - 0.5);
        prev = price;
    }
}

int main() {
    test_bs_formula();
    test_moro();
    test_american_ge_european();
    test_convergence_vs_exercise_points();
    printf("\nAll validation tests passed.\n");
    return 0;
}