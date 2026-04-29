# American Options Pricing — CUDA & OpenMP Reimplementation Guide

> **Target audience:** GitHub Copilot / AI coding assistant  
> **Paper:** *"Using High Performance Computing and Monte Carlo Simulation for Pricing American Options"* — Cvetanoska & Stojanovski, European University Skopje  
> **Goal:** Full reimplement of the paper's algorithm in two parallel back-ends — NVIDIA CUDA and OpenMP — with a shared C++ core, a validation harness, and a benchmark runner.

---

## Table of Contents

1. [Project Overview](#1-project-overview)  
2. [Repository Layout](#2-repository-layout)  
3. [Mathematical Foundation](#3-mathematical-foundation)  
4. [Shared C++ Core (CPU Reference)](#4-shared-c-core-cpu-reference)  
5. [OpenMP Implementation](#5-openmp-implementation)  
6. [CUDA Implementation](#6-cuda-implementation)  
7. [CUDA Optimisation Techniques](#7-cuda-optimisation-techniques)  
8. [Build System](#8-build-system)  
9. [Validation & Testing](#9-validation--testing)  
10. [Benchmarking](#10-benchmarking)  
11. [Expected Results](#11-expected-results)  
12. [Common Pitfalls](#12-common-pitfalls)

---

## 1. Project Overview

### What the paper does

The paper prices **American call options** using a **Quasi-Monte Carlo (QMC)** simulation on both CPU (serial) and GPU (CUDA parallel). The core algorithm is:

1. Treat the American option as a **Bermudan option** discretised into `m` equally-spaced time steps.
2. Use **backward induction** (dynamic programming) starting from maturity `T`.
3. At each step, use the **Black-Scholes formula** to compute the exercise boundary.
4. Use **Quasi-random number generation** (linear congruential generator + Moro inverse CND) for better convergence than plain Monte Carlo.
5. Average all path payoffs and discount back to get the option price.

### What you will build

| Component | File(s) |
|---|---|
| Shared math primitives | `src/core/math_utils.hpp` |
| Black-Scholes pricer | `src/core/black_scholes.hpp / .cpp` |
| Quasi-random generator | `src/core/quasi_rng.hpp / .cpp` |
| Moro inverse CND | `src/core/moro_inv_cnd.hpp / .cpp` |
| Serial CPU reference | `src/cpu/american_option_serial.cpp` |
| OpenMP parallel CPU | `src/openmp/american_option_omp.cpp` |
| CUDA GPU implementation | `src/cuda/american_option.cu` |
| CUDA kernels | `src/cuda/kernels.cuh` |
| CUDA reduction utilities | `src/cuda/reduction.cuh` |
| Benchmark runner | `src/benchmark/main.cpp` |
| Validation harness | `tests/validate.cpp` |
| CMakeLists | `CMakeLists.txt` |

---

## 2. Repository Layout

```
american_options/
├── CMakeLists.txt
├── README.md
├── src/
│   ├── core/
│   │   ├── math_utils.hpp        # Constants, exp, log helpers
│   │   ├── black_scholes.hpp
│   │   ├── black_scholes.cpp
│   │   ├── quasi_rng.hpp
│   │   ├── quasi_rng.cpp
│   │   ├── moro_inv_cnd.hpp
│   │   └── moro_inv_cnd.cpp
│   ├── cpu/
│   │   └── american_option_serial.cpp
│   ├── openmp/
│   │   └── american_option_omp.cpp
│   ├── cuda/
│   │   ├── kernels.cuh
│   │   ├── reduction.cuh
│   │   └── american_option.cu
│   └── benchmark/
│       └── main.cpp
└── tests/
    └── validate.cpp
```

---

## 3. Mathematical Foundation

Implement **every formula below exactly**. Copilot: do not approximate or substitute; the paper's algorithm depends on each formula precisely.

### 3.1 Option Parameters Struct

```cpp
// src/core/math_utils.hpp
#pragma once
#include <cmath>

struct OptionParams {
    double S0;       // Current underlying price
    double X;        // Exercise (strike) price
    double T;        // Time to maturity (years)
    double r;        // Risk-free rate (continuously compounded)
    double v;        // Implied volatility
    int    m;        // Number of discrete exercise points (Bermudan steps)
    int    N;        // Number of Monte Carlo paths
};
```

### 3.2 Black-Scholes Call Formula (Equations 1, 3, 4, 5)

```
d1 = [log(S0/X) + (r + v²/2) * T] / (v * sqrt(T))
d2 = d1 - v * sqrt(T)
c  = S0 * CND(d1) - X * exp(-r*T) * CND(d2)
```

Where `CND(d)` is the standard cumulative normal distribution.

**Implementation note:** The paper uses `CND(d) = 1 - CND(-d)` (equation 5) — this is just the symmetry property; use a proper `erfc`-based implementation, not a polynomial table.

```cpp
// src/core/black_scholes.hpp
#pragma once
#include "math_utils.hpp"

// CND via erfc for full double precision
inline double cnd(double d) {
    return 0.5 * erfc(-d * M_SQRT1_2);
}

// Black-Scholes call price: c(S, X, t, v, r)
// t here is the sub-interval length, NOT total T
double bs_call(double S, double X, double t, double v, double r);
```

```cpp
// src/core/black_scholes.cpp
#include "black_scholes.hpp"
#include <cmath>

double bs_call(double S, double X, double t, double v, double r) {
    if (t <= 0.0) return std::max(S - X, 0.0);
    double sqrt_t = std::sqrt(t);
    double d1 = (std::log(S / X) + (r + 0.5 * v * v) * t) / (v * sqrt_t);
    double d2 = d1 - v * sqrt_t;
    return S * cnd(d1) - X * std::exp(-r * t) * cnd(d2);
}
```

### 3.3 Stock Price Path Simulation (Equation 9)

```
S_{t_{i-1}} = S_{t_{(i-1)-1}} * exp( (r - 0.5*v²) * Δt + v * sqrt(Δt) * z_i )
```

Where:
- `Δt = T / (m + 1)` — time step size
- `z_i` — quasi-random sample drawn from N(0,1)

### 3.4 Backward Induction — Call Value at Each Node

**At the boundary (equation 10):**
```
c_{t_{i-1}} = max( S_{t_{i-1}} - X,  c(S_{t_{i-1}}, X, t, v, r) )
```

**Continuing backward (equation 11):**
```
c_{t_{(i-1)-1}} = max( S_{t_{(i-1)-1}} - X,  c_{t_{i-1}} * exp(-r * Δt) )
```

**Final price (equations 6, 8):**
```
fair_value = mean( c_{t_0} over all N paths ) * exp(-r * T)
```

### 3.5 Quasi-Random Number Generation

The paper uses a **linear congruential generator (LCG)** to permute quasi-random arrays, followed by **Moro's inverse CND** to map uniform [0,1] samples to N(0,1).

#### LCG Parameters (standard choice)
```
a = 1664525
c = 1013904223
m = 2^32
x_{n+1} = (a * x_n + c) mod m
u_n = x_n / 2^32   ∈ [0, 1)
```

#### Moro Inverse CND

Moro's approximation maps `u ∈ (0,1)` to `z = Φ⁻¹(u)`. Implement using the rational polynomial from Moro (1995). The key coefficients are:

```cpp
// src/core/moro_inv_cnd.cpp
// Moro's inverse CND — rational polynomial approximation
// Valid for u in (0, 1); returns z ~ N(0,1)
double moro_inv_cnd(double u) {
    // Central region coefficients
    static const double a[] = {
        2.50662823884, -18.61500062529,  41.39119773534, -25.44106049637
    };
    static const double b[] = {
        -8.47351093090,  23.08336743743, -21.06224101826,   3.13082909833
    };
    // Tail region coefficients
    static const double c[] = {
        0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
        0.0276438810333863, 0.0038405729373609, 0.0003951896511349,
        0.0000321767881768, 0.0000002888167364, 0.0000003960315187
    };

    double x = u - 0.5;
    double r;
    if (std::fabs(x) < 0.42) {
        r = x * x;
        r = x * (((a[3]*r + a[2])*r + a[1])*r + a[0]) /
               ((((b[3]*r + b[2])*r + b[1])*r + b[0])*r + 1.0);
    } else {
        r = (x > 0.0) ? std::log(-std::log(1.0 - u))
                      : std::log(-std::log(u));
        r = c[0] + r*(c[1] + r*(c[2] + r*(c[3] + r*(c[4] +
            r*(c[5] + r*(c[6] + r*(c[7] + r*c[8])))))));
        if (x < 0.0) r = -r;
    }
    return r;
}
```

---

## 4. Shared C++ Core (CPU Reference)

Implement the **serial** version first. It is the correctness baseline for all parallel versions.

```cpp
// src/cpu/american_option_serial.cpp
#include "core/black_scholes.hpp"
#include "core/quasi_rng.hpp"
#include "core/moro_inv_cnd.hpp"
#include <vector>
#include <cmath>
#include <numeric>

double price_american_call_serial(const OptionParams& p) {
    const double dt   = p.T / static_cast<double>(p.m + 1);
    const double sqdt = std::sqrt(dt);
    const double drift = (p.r - 0.5 * p.v * p.v) * dt;

    double sum = 0.0;

    for (int path = 0; path < p.N; ++path) {
        // Seed the LCG differently per path for independence
        uint32_t seed = static_cast<uint32_t>(path + 1) * 1234567u;

        // --- Forward pass: simulate stock prices along this path ---
        std::vector<double> S(p.m + 1);
        S[0] = p.S0;
        for (int i = 1; i <= p.m; ++i) {
            double u  = lcg_next_uniform(seed);     // LCG uniform sample
            double z  = moro_inv_cnd(u);            // N(0,1) via Moro
            S[i] = S[i-1] * std::exp(drift + p.v * sqdt * z);
        }

        // --- Backward induction: compute option value ---
        // At maturity (step m), use Black-Scholes for sub-interval [t_{m-1}, t_m]
        double c = bs_call(S[p.m - 1], p.X, dt, p.v, p.r);

        for (int i = p.m - 1; i >= 1; --i) {
            // Equation 10 at the boundary step; equation 11 for all earlier steps
            double continuation = c * std::exp(-p.r * dt);
            double intrinsic    = S[i] - p.X;
            c = std::max(intrinsic, continuation);
        }

        // Add final discounted payoff to sum
        sum += c;
    }

    return (sum / static_cast<double>(p.N)) * std::exp(-p.r * p.T);
}
```

**Copilot note:** Implement `lcg_next_uniform(seed)` in `quasi_rng.cpp` as a standard 32-bit LCG returning `double` in `[0, 1)`. The `seed` argument must be passed by reference so state advances.

---

## 5. OpenMP Implementation

The OpenMP version parallelises the outer `path` loop. Each thread maintains its own LCG state so there are no race conditions on the RNG.

```cpp
// src/openmp/american_option_omp.cpp
#include "core/black_scholes.hpp"
#include "core/quasi_rng.hpp"
#include "core/moro_inv_cnd.hpp"
#include <omp.h>
#include <vector>
#include <cmath>

double price_american_call_omp(const OptionParams& p, int num_threads = 0) {
    if (num_threads > 0) omp_set_num_threads(num_threads);

    const double dt    = p.T / static_cast<double>(p.m + 1);
    const double sqdt  = std::sqrt(dt);
    const double drift = (p.r - 0.5 * p.v * p.v) * dt;

    double total_sum = 0.0;

    // PARALLEL REGION
    // Each thread gets its own stack-allocated path buffer — no shared state.
    #pragma omp parallel reduction(+:total_sum)
    {
        std::vector<double> S(p.m + 1);

        #pragma omp for schedule(dynamic, 64)
        for (int path = 0; path < p.N; ++path) {
            uint32_t seed = static_cast<uint32_t>(path + 1) * 1234567u;

            // Forward pass
            S[0] = p.S0;
            for (int i = 1; i <= p.m; ++i) {
                double u = lcg_next_uniform(seed);
                double z = moro_inv_cnd(u);
                S[i] = S[i-1] * std::exp(drift + p.v * sqdt * z);
            }

            // Backward induction
            double c = bs_call(S[p.m - 1], p.X, dt, p.v, p.r);
            for (int i = p.m - 1; i >= 1; --i) {
                double continuation = c * std::exp(-p.r * dt);
                double intrinsic    = S[i] - p.X;
                c = std::max(intrinsic, continuation);
            }

            total_sum += c;
        }
    }

    return (total_sum / static_cast<double>(p.N)) * std::exp(-p.r * p.T);
}
```

### OpenMP Compilation Flags

```cmake
target_compile_options(american_omp PRIVATE -fopenmp -O3 -march=native)
target_link_libraries(american_omp PRIVATE OpenMP::OpenMP_CXX)
```

### OpenMP Tuning Notes for Copilot

- Use `schedule(dynamic, 64)` — chunk size 64 balances overhead vs load imbalance.
- Do **not** share the `S` vector between threads. Declare it inside the parallel block.
- The `reduction(+:total_sum)` clause handles the final summation correctly.
- For NUMA systems, consider per-thread `first_touch` initialisation.

---

## 6. CUDA Implementation

### 6.1 Overview of the GPU Strategy

The paper uses the following GPU mapping (Table 2 in the paper):

| Level | Maps to |
|---|---|
| 1 option | 1 CUDA kernel launch |
| N paths | N threads total |
| Blocks | `ceil(N / threads_per_block)` |
| Threads per block | 256 or 512 (tunable) |

Each thread simulates **one path** end-to-end (forward + backward), then writes its payoff to shared memory. A parallel reduction sums payoffs within each block, and `atomicAdd` accumulates across blocks into a single global sum.

### 6.2 Device-Side Math Helpers

```cuda
// src/cuda/kernels.cuh
#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>  // CUDART_PI_F, etc.

// ---- LCG on device ----
__device__ __forceinline__
float lcg_next(uint32_t& state) {
    state = 1664525u * state + 1013904223u;
    return __uint2float_rn(state) * 2.3283064365386963e-10f;  // / 2^32
}

// ---- Moro Inverse CND on device (single precision) ----
__device__ float moro_inv_cnd_device(float u);

// ---- Black-Scholes call (device, double precision) ----
__device__ double bs_call_device(double S, double X, double t, double v, double r);

// ---- CND via erfc ----
__device__ __forceinline__
double cnd_device(double d) {
    return 0.5 * erfc(-d * M_SQRT1_2);
}
```

```cuda
// Device implementation of bs_call_device in american_option.cu
__device__ double bs_call_device(double S, double X, double t, double v, double r) {
    if (t <= 0.0) return fmax(S - X, 0.0);
    double sqt = sqrt(t);
    double d1  = (log(S / X) + (r + 0.5 * v * v) * t) / (v * sqt);
    double d2  = d1 - v * sqt;
    return S * cnd_device(d1) - X * exp(-r * t) * cnd_device(d2);
}
```

### 6.3 Reduction Utility

Copilot: implement the **tree-reduction** described in Section VI of the paper. This avoids warp divergence by using the half-section stride pattern.

```cuda
// src/cuda/reduction.cuh
#pragma once
#include <cuda_runtime.h>

// Warp-level reduction using shuffle intrinsics (no shared mem needed for warp)
__device__ __forceinline__
double warp_reduce_sum(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// Block-level reduction: assumes blockDim.x is power of 2, max 512
__device__ double block_reduce_sum(double val, double* sdata) {
    int tid  = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;

    // Warp reduction
    val = warp_reduce_sum(val);

    // Write warp results to shared memory
    if (lane == 0) sdata[wid] = val;
    __syncthreads();

    // Final warp reduces warp results
    val = (tid < (blockDim.x >> 5)) ? sdata[tid] : 0.0;
    if (wid == 0) val = warp_reduce_sum(val);

    return val;
}
```

### 6.4 Main CUDA Kernel

```cuda
// src/cuda/american_option.cu
#include "kernels.cuh"
#include "reduction.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

/*
 * Each thread = one Monte Carlo path.
 * Shared memory holds intermediate reduction results.
 *
 * Kernel parameters:
 *   d_result   — output: array of per-block partial sums (size = gridDim.x)
 *   S0, X, T   — option parameters
 *   r, v       — risk-free rate, volatility
 *   m          — number of discrete time steps
 *   N          — total number of paths
 */
__global__ void american_option_kernel(
    double* __restrict__ d_result,
    double S0, double X, double T,
    double r,  double v,
    int m, int N)
{
    // Shared memory for block reduction
    extern __shared__ double sdata[];

    int path = blockIdx.x * blockDim.x + threadIdx.x;

    const double dt    = T / static_cast<double>(m + 1);
    const double sqdt  = sqrt(dt);
    const double drift = (r - 0.5 * v * v) * dt;

    double payoff = 0.0;

    if (path < N) {
        // Per-path LCG seed — must differ per thread
        uint32_t seed = static_cast<uint32_t>(path + 1) * 1234567u;

        // ---- Forward pass: simulate stock path ----
        // We do NOT store the entire path to avoid high register/local mem use.
        // Instead we exploit the backward induction structure:
        // store only the prices at each step in a local array.
        // For large m, consider two-pass or register-spill-aware approaches.

        // NOTE: For m <= 64, this local array fits in registers/L1.
        // For larger m, allocate on heap via cudaMalloc per block, or
        // reduce m (the paper uses m=10 in Figure 1).
        double S_path[64];   // Adjust size if m > 63; see Copilot note below
        S_path[0] = S0;
        for (int i = 1; i <= m; ++i) {
            float  u = lcg_next(seed);
            double z = static_cast<double>(moro_inv_cnd_device(u));
            S_path[i] = S_path[i-1] * exp(drift + v * sqdt * z);
        }

        // ---- Backward induction ----
        double c = bs_call_device(S_path[m - 1], X, dt, v, r);
        for (int i = m - 1; i >= 1; --i) {
            double continuation = c * exp(-r * dt);
            double intrinsic    = S_path[i] - X;
            c = fmax(intrinsic, continuation);
        }

        payoff = c;
    }

    // ---- Block reduction ----
    double block_sum = block_reduce_sum(payoff, sdata);

    // Thread 0 of each block writes partial sum to global memory
    if (threadIdx.x == 0) {
        d_result[blockIdx.x] = block_sum;
    }
}

/*
 * Host-side launcher.
 * Returns the American call option price.
 */
double price_american_call_cuda(const OptionParams& p,
                                 int threads_per_block = 512)
{
    int blocks = (p.N + threads_per_block - 1) / threads_per_block;

    // Allocate device memory for partial sums
    double* d_result = nullptr;
    cudaMalloc(&d_result, blocks * sizeof(double));
    cudaMemset(d_result, 0, blocks * sizeof(double));

    // Shared memory size: one double per warp
    int shared_mem_bytes = (threads_per_block / 32) * sizeof(double);

    // Launch kernel
    american_option_kernel<<<blocks, threads_per_block, shared_mem_bytes>>>(
        d_result,
        p.S0, p.X, p.T,
        p.r, p.v,
        p.m, p.N
    );
    cudaDeviceSynchronize();

    // Copy partial sums back and do final reduction on host
    std::vector<double> h_result(blocks);
    cudaMemcpy(h_result.data(), d_result, blocks * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    double total = 0.0;
    for (double v : h_result) total += v;

    return (total / static_cast<double>(p.N)) * std::exp(-p.r * p.T);
}
```

> **Copilot note on `S_path` array size:** The paper uses `m = 10` exercise points (Figure 1 shows 1–10 points). The fixed-size array `double S_path[64]` is safe for `m ≤ 63`. If you need `m > 63`, replace with a device-side dynamic allocation using `extern __shared__` with a stride of `blockDim.x`, or split into two kernel passes.

---

## 7. CUDA Optimisation Techniques

Implement **all five** techniques described in Section VI of the paper.

### 7.1 Reduction Technique (Paper: "Reduction technique")

Already shown above. Key rule: **use half-section stride** to avoid warp divergence.

```cuda
// BAD (divergent): if (tid % 2 == 0) ...
// GOOD (non-divergent): stride = blockDim.x / 2; if (tid < stride) ...

// Additionally: unroll the last 5 iterations (warp is SIMD synchronous)
if (blockDim.x >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
if (blockDim.x >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
if (blockDim.x >= 128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }
// Warp-synchronous tail (no __syncthreads needed for warp)
if (tid < 32) {
    volatile double* vs = sdata;
    vs[tid] += vs[tid + 32];
    vs[tid] += vs[tid + 16];
    vs[tid] += vs[tid +  8];
    vs[tid] += vs[tid +  4];
    vs[tid] += vs[tid +  2];
    vs[tid] += vs[tid +  1];
}
```

### 7.2 Global Memory Coalescing (Paper: "Global memory bandwidth")

Ensure all global memory reads/writes are **coalesced** — threads in a warp access consecutive memory addresses.

```cuda
// GOOD: thread 0 accesses index 0, thread 1 accesses index 1, ...
d_result[blockIdx.x * blockDim.x + threadIdx.x] = value;

// BAD (strided): thread 0 accesses index 0, thread 1 accesses index 32, ...
d_result[threadIdx.x * blockDim.x + blockIdx.x] = value;
```

For this algorithm the main coalescing opportunity is writing per-thread payoffs before reduction. If you store intermediate per-path stock prices in global memory (for very large `m`), always store column-major: `d_S[step * N + path]` not `d_S[path * m + step]`.

### 7.3 Dynamic SM Partitioning (Paper: "Dynamic partitioning of the SM resources")

The paper detects at runtime whether to use fewer, larger blocks or more, smaller ones. Implement this via a **occupancy query**:

```cuda
// Host side: query max active blocks for our kernel
int min_grid, block_size;
cudaOccupancyMaxPotentialBlockSize(
    &min_grid, &block_size,
    american_option_kernel,
    shared_mem_bytes,
    0
);
// Use block_size as threads_per_block
```

Alternatively, replicate the paper's explicit logic:

```cpp
int threads_per_block;
if      (p.N >= 500000) threads_per_block = 512;
else if (p.N >= 10000)  threads_per_block = 512;
else if (p.N >= 1000)   threads_per_block = 256;
else                    threads_per_block = 128;
```

### 7.4 Data Prefetching

Inside the kernel's forward loop, issue the next LCG value before the current `exp` completes:

```cuda
// Prefetch next random number while waiting for exp()
uint32_t seed_next = 1664525u * seed + 1013904223u;
float u_current = __uint2float_rn(seed) * 2.3283064365386963e-10f;
seed = seed_next;  // advance state
double z = moro_inv_cnd_device(u_current);
S_path[i] = S_path[i-1] * exp(drift + v * sqdt * z);
// By the time exp() returns, next seed is already computed
```

### 7.5 Instruction Mix (Paper: "Instruction mix")

- **Avoid branches in the hot loop.** The `fmax(intrinsic, continuation)` call is branchless — keep it that way. Do NOT write `if (intrinsic > continuation) c = intrinsic;`.
- **Use `__expf` / `__logf` for single-precision** where full double precision is not needed (e.g., the LCG path). Use `exp` / `log` (double) for the Black-Scholes and discounting calculations.
- **Precompute constants** outside loops: `drift`, `sqdt`, `exp(-r*dt)` are loop-invariant.

```cuda
// Precompute before the backward loop
const double discount = exp(-r * dt);   // computed ONCE, reused m times
for (int i = m - 1; i >= 1; --i) {
    double continuation = c * discount;   // multiply, not exp(), each step
    ...
}
```

---

## 8. Build System

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(AmericanOptions LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# --- Find packages ---
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# --- Shared core library ---
add_library(option_core STATIC
    src/core/black_scholes.cpp
    src/core/quasi_rng.cpp
    src/core/moro_inv_cnd.cpp
)
target_include_directories(option_core PUBLIC src/core)
target_compile_options(option_core PRIVATE -O3 -march=native)

# --- Serial CPU target ---
add_executable(american_serial
    src/cpu/american_option_serial.cpp
    src/benchmark/main.cpp
)
target_link_libraries(american_serial PRIVATE option_core)
target_compile_definitions(american_serial PRIVATE BACKEND_SERIAL)

# --- OpenMP target ---
add_executable(american_omp
    src/openmp/american_option_omp.cpp
    src/benchmark/main.cpp
)
target_compile_options(american_omp PRIVATE -fopenmp -O3 -march=native)
target_link_libraries(american_omp PRIVATE option_core OpenMP::OpenMP_CXX)
target_compile_definitions(american_omp PRIVATE BACKEND_OMP)

# --- CUDA target ---
add_executable(american_cuda
    src/cuda/american_option.cu
    src/benchmark/main.cpp
)
target_compile_options(american_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math          # Enables __expf, __logf etc. — test accuracy first
        -arch=sm_86              # Adjust for your GPU (sm_75 for Turing, sm_86 for Ampere)
        --ptxas-options=-v       # Print register usage — aim for ≤ 32 registers/thread
        -maxrregcount=32         # Cap registers to improve occupancy
    >
)
target_link_libraries(american_cuda PRIVATE option_core CUDA::cudart)
target_compile_definitions(american_cuda PRIVATE BACKEND_CUDA)

# --- Tests ---
enable_testing()
add_executable(validate tests/validate.cpp)
target_link_libraries(validate PRIVATE option_core)
add_test(NAME validation COMMAND validate)
```

### Important CUDA Compilation Notes for Copilot

- Always specify `-arch=sm_XX` matching your physical GPU. The paper tested on GeForce GT 540M (`sm_21`). For modern GPUs use `sm_75` (Turing), `sm_86` (Ampere), `sm_89` (Ada).
- `--use_fast_math` trades ~1 ULP accuracy for ~2x speed on transcendentals. Test that option prices are still within tolerance before enabling in production.
- `-maxrregcount=32` forces higher occupancy by limiting register use per thread. If the kernel uses register spilling, test with and without.

---

## 9. Validation & Testing

### 9.1 Ground Truth

For European options, Black-Scholes gives an exact closed-form price — use it to validate your CND, BS formula, and simulation framework.

For American call options on **non-dividend-paying stocks**, the American call price equals the European call price (it is never optimal to exercise early). Use this as a basic sanity check.

### 9.2 Validation Test Cases

```cpp
// tests/validate.cpp

// Test 1: Black-Scholes formula correctness
// Parameters from Hull "Options, Futures, and Other Derivatives" (standard textbook example)
// S0=42, X=40, T=0.5, r=0.1, v=0.2 → c ≈ 4.76
void test_bs_formula() {
    double c = bs_call(42.0, 40.0, 0.5, 0.2, 0.1);
    assert(std::fabs(c - 4.76) < 0.01);
}

// Test 2: American option >= European option
void test_american_ge_european() {
    OptionParams p = {100.0, 100.0, 1.0, 0.05, 0.2, 10, 100000};
    double american = price_american_call_serial(p);
    double european = bs_call(p.S0, p.X, p.T, p.v, p.r);
    assert(american >= european - 1e-6);  // American ≥ European always
}

// Test 3: GPU and CPU produce the same price (within statistical tolerance)
void test_gpu_cpu_agreement() {
    OptionParams p = {100.0, 100.0, 1.0, 0.05, 0.2, 10, 100000};
    double cpu_price  = price_american_call_serial(p);
    double cuda_price = price_american_call_cuda(p);
    assert(std::fabs(cpu_price - cuda_price) < 0.5);  // within 50 cents
}

// Test 4: Reproduction of Figure 1 — American option value vs exercise points
// As m increases (1→10), the American option price should converge upward
void test_convergence_vs_exercise_points() {
    double prev = 0.0;
    for (int m = 1; m <= 10; ++m) {
        OptionParams p = {100.0, 90.0, 1.0, 0.05, 0.3, m, 50000};
        double price = price_american_call_serial(p);
        // Price should be non-decreasing as m increases
        assert(price >= prev - 0.5);  // allow small MC noise
        prev = price;
    }
}
```

### 9.3 Statistical Convergence Test

MC error decreases as `O(1/√N)`. Verify this:

```cpp
void test_mc_convergence() {
    OptionParams p = {100.0, 100.0, 1.0, 0.05, 0.2, 10, 0};
    double ref = 0.0;
    for (int logN = 10; logN <= 20; logN += 2) {
        p.N = 1 << logN;
        double price = price_american_call_serial(p);
        if (logN == 20) ref = price;
        printf("N=%-8d  price=%.4f\n", p.N, price);
    }
    // At N=1M, price should be within 0.1 of reference
    assert(std::fabs(price_american_call_serial({100,100,1,0.05,0.2,10,1000000}) - ref) < 0.1);
}
```

---

## 10. Benchmarking

Reproduce **Table 2** from the paper (GPU vs CPU timing for different path counts).

```cpp
// src/benchmark/main.cpp
#include <chrono>
#include <cstdio>
#include "core/math_utils.hpp"

// Declare whichever backend is compiled in
double price_american_call_serial(const OptionParams&);
double price_american_call_omp(const OptionParams&, int);
double price_american_call_cuda(const OptionParams&, int);

int main() {
    OptionParams base = {
        .S0 = 100.0, .X = 100.0, .T = 1.0,
        .r  = 0.05,  .v = 0.20,  .m = 10
    };

    const int path_counts[] = {10, 100, 1000, 10000, 100000,
                                200000, 300000, 500000, 1000000};

    printf("%-12s  %-12s  %-12s  %-12s\n",
           "Paths", "Serial(s)", "OMP(s)", "CUDA(ms)");

    for (int N : path_counts) {
        base.N = N;

        auto t0 = std::chrono::high_resolution_clock::now();
        double p_serial = price_american_call_serial(base);
        auto t1 = std::chrono::high_resolution_clock::now();
        double serial_s = std::chrono::duration<double>(t1-t0).count();

        auto t2 = std::chrono::high_resolution_clock::now();
        double p_omp = price_american_call_omp(base, 0);   // 0 = use all cores
        auto t3 = std::chrono::high_resolution_clock::now();
        double omp_s = std::chrono::duration<double>(t3-t2).count();

        auto t4 = std::chrono::high_resolution_clock::now();
        double p_cuda = price_american_call_cuda(base, 512);
        auto t5 = std::chrono::high_resolution_clock::now();
        double cuda_ms = std::chrono::duration<double,std::milli>(t5-t4).count();

        printf("%-12d  %-12.4f  %-12.4f  %-12.4f  "
               "| prices: S=%.3f O=%.3f C=%.3f\n",
               N, serial_s, omp_s, cuda_ms,
               p_serial, p_omp, p_cuda);
    }
    return 0;
}
```

### Expected Table 2 Reproduction

The paper's results (GT 540M, i7 2.2GHz) should be directionally reproduced. On modern hardware expect **much larger speedups**. The key pattern to verify is:

| N paths | Expected CPU behaviour | Expected GPU behaviour |
|---|---|---|
| 10 | ~0.08–0.09 s | ~0.04–0.06 s (2× faster) |
| 1,000 | ~1.2–1.3 s | ~0.04–0.09 s (15–30× faster) |
| 10,000 | ~11.5 s | ~0.07–0.09 s (130× faster) |
| 1,000,000 | ~1100 s | ~0.15–0.25 s (**>6500× faster**) |

---

## 11. Expected Results

### Figure 1 Reproduction

Plot American option value vs. number of exercise points (m = 1 to 10). The American option price should always be ≥ the European price (same parameters). Both curves should converge as m increases.

**Typical values** (S0=100, X=90, T=1, r=5%, v=30%, N=100,000):

| m | American Call | European Call |
|---|---|---|
| 1 | ~18.5 | ~17.2 |
| 5 | ~19.8 | ~17.2 |
| 10 | ~20.1 | ~17.2 |

### Convergence of QMC vs MC

With the same N, Quasi-MC (LCG + Moro) achieves lower variance than plain MC with `std::mt19937`. The error should decrease roughly as `O(log(N)^d / N)` rather than `O(1/√N)`.

---

## 12. Common Pitfalls

### Mathematical Pitfalls

| Pitfall | Fix |
|---|---|
| Using `CND(d) = 1 - CND(-d)` instead of a real implementation | Implement via `erfc(-d/sqrt(2)) / 2` |
| Wrong sign in `drift = (r - 0.5*v²)*dt` | The `- 0.5*v²` term is the Itô correction; never omit it |
| Forgetting final `exp(-r*T)` discounting | Applied once at the end after averaging, not inside the loop |
| Using `m` instead of `m+1` in `dt = T/(m+1)` | The paper is explicit: `dt = T/(m+1)` |
| Moro coefficients copied incorrectly | Cross-check: `moro_inv_cnd(0.975)` ≈ `1.96` |

### CUDA Pitfalls

| Pitfall | Fix |
|---|---|
| All threads using the same LCG seed | Seed with `(path_id + 1) * 1234567u` so each thread diverges |
| Using `__syncthreads()` after the warp size threshold (< 32 threads) | Switch to shuffle-based reduction; never call `__syncthreads()` for intra-warp ops |
| `S_path` array too large for register file | Either reduce `m`, or use shared memory with stride `blockDim.x` |
| Using `atomicAdd` for every path | Use shared-memory block reduction first; only one `atomicAdd` per block |
| Not checking `cudaGetLastError()` | Always check after kernel launch and after `cudaDeviceSynchronize()` |

### OpenMP Pitfalls

| Pitfall | Fix |
|---|---|
| Shared `S` vector across threads | Move `std::vector<double> S(m+1)` inside the `#pragma omp parallel` block |
| False sharing on `total_sum` | Use `reduction(+:total_sum)` not a plain shared variable |
| Non-reproducible results due to floating-point reduction order | This is expected; allow ±0.5 tolerance in tests across implementations |

---

## Appendix A: Quick Reference — Key Equations

```
dt   = T / (m + 1)
S_i  = S_{i-1} * exp( (r - v²/2)*dt + v*sqrt(dt)*z_i )
d1   = [ln(S/X) + (r + v²/2)*t] / (v*sqrt(t))
d2   = d1 - v*sqrt(t)
c_BS = S*CND(d1) - X*exp(-r*t)*CND(d2)

Backward step:
  c_{i-1} = max( S_{i-1} - X,  c_i * exp(-r*dt) )

Final price:
  V = mean(c_0 over N paths) * exp(-r*T)
```

## Appendix B: GPU Architecture Quick Reference

The paper tested on NVidia GeForce GT 540M (Fermi, `sm_21`, 96 CUDA cores). For modern targets:

| GPU Generation | Compute Cap. | Warp Size | Max Threads/Block | Shared Mem/SM |
|---|---|---|---|---|
| Fermi (paper) | sm_21 | 32 | 1024 | 48 KB |
| Maxwell | sm_52 | 32 | 1024 | 64 KB |
| Pascal | sm_61 | 32 | 1024 | 64 KB |
| Turing | sm_75 | 32 | 1024 | 64 KB |
| Ampere | sm_86 | 32 | 1024 | 100 KB |

Adjust `-arch=sm_XX` in CMakeLists accordingly.

---

*End of implementation guide. Copilot: implement files in the order they appear in Section 2 (Repository Layout). Build and run `validate` before running benchmarks.*