#include "kernels.cuh"
#include "reduction.cuh"
#include "../core/math_utils.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>

// ---------------- Moro inverse CND (device, single precision) ----------------
__device__ float moro_inv_cnd_device(float u) {
    const float a0 =  2.50662823884f;
    const float a1 = -18.61500062529f;
    const float a2 =  41.39119773534f;
    const float a3 = -25.44106049637f;
    const float b0 = -8.47351093090f;
    const float b1 =  23.08336743743f;
    const float b2 = -21.06224101826f;
    const float b3 =   3.13082909833f;
    const float c0 = 0.3374754822726147f;
    const float c1 = 0.9761690190917186f;
    const float c2 = 0.1607979714918209f;
    const float c3 = 0.0276438810333863f;
    const float c4 = 0.0038405729373609f;
    const float c5 = 0.0003951896511349f;
    const float c6 = 0.0000321767881768f;
    const float c7 = 0.0000002888167364f;
    const float c8 = 0.0000003960315187f;

    float x = u - 0.5f;
    float r;
    if (fabsf(x) < 0.42f) {
        r = x * x;
        r = x * (((a3*r + a2)*r + a1)*r + a0) /
               ((((b3*r + b2)*r + b1)*r + b0)*r + 1.0f);
    } else {
        r = (x > 0.0f) ? logf(-logf(1.0f - u)) : logf(-logf(u));
        r = c0 + r*(c1 + r*(c2 + r*(c3 + r*(c4 +
            r*(c5 + r*(c6 + r*(c7 + r*c8)))))));
        if (x < 0.0f) r = -r;
    }
    return r;
}

// ---------------- Black-Scholes call (device) ----------------
__device__ double bs_call_device(double S, double X, double t, double v, double r) {
    if (t <= 0.0) return fmax(S - X, 0.0);
    double sqt = sqrt(t);
    double d1  = (log(S / X) + (r + 0.5 * v * v) * t) / (v * sqt);
    double d2  = d1 - v * sqt;
    return S * cnd_device(d1) - X * exp(-r * t) * cnd_device(d2);
}

// ---------------- Main pricing kernel ----------------
__global__ void american_option_kernel(
    double* __restrict__ d_partial,
    double S0, double X, double T,
    double r,  double v,
    int m, int N)
{
    extern __shared__ double sdata[];

    int path = blockIdx.x * blockDim.x + threadIdx.x;

    const double dt       = T / static_cast<double>(m + 1);
    const double sqdt     = sqrt(dt);
    const double drift    = (r - 0.5 * v * v) * dt;
    const double discount = exp(-r * dt);

    double payoff = 0.0;

    if (path < N) {
        uint32_t seed = static_cast<uint32_t>(path + 1) * 1234567u;

        // Per-thread path buffer. Safe for m <= 63.
        double S_path[64];
        S_path[0] = S0;

        #pragma unroll 1
        for (int i = 1; i <= m; ++i) {
            float  u = lcg_next(seed);
            double z = static_cast<double>(moro_inv_cnd_device(u));
            S_path[i] = S_path[i-1] * exp(drift + v * sqdt * z);
        }

        double c = bs_call_device(S_path[m - 1], X, dt, v, r);
        for (int i = m - 1; i >= 1; --i) {
            double continuation = c * discount;
            double intrinsic    = S_path[i] - X;
            c = fmax(intrinsic, continuation);
        }

        payoff = c;
    }

    double block_sum = block_reduce_sum(payoff, sdata);

    if (threadIdx.x == 0) {
        d_partial[blockIdx.x] = block_sum;
    }
}

// ---------------- Host launcher ----------------
double price_american_call_cuda(const OptionParams& p, int threads_per_block = 512) {
    int blocks = (p.N + threads_per_block - 1) / threads_per_block;

    double* d_partial = nullptr;
    cudaMalloc(&d_partial, blocks * sizeof(double));
    cudaMemset(d_partial, 0, blocks * sizeof(double));

    int num_warps = (threads_per_block + 31) / 32;
    int shared_mem_bytes = num_warps * sizeof(double);

    american_option_kernel<<<blocks, threads_per_block, shared_mem_bytes>>>(
        d_partial,
        p.S0, p.X, p.T,
        p.r, p.v,
        p.m, p.N
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_partial);
        return 0.0;
    }
    cudaDeviceSynchronize();

    std::vector<double> h_partial(blocks);
    cudaMemcpy(h_partial.data(), d_partial, blocks * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_partial);

    double total = 0.0;
    for (double x : h_partial) total += x;

    return (total / static_cast<double>(p.N)) * std::exp(-p.r * p.T);
}