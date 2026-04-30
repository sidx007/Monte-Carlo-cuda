#pragma once
#include <cuda_runtime.h>

static constexpr int GPU_SOBOL_DIM  = 21;  
static constexpr int GPU_SOBOL_BITS = 32;  

void sobol_gpu_init(const unsigned int V_host[GPU_SOBOL_DIM][GPU_SOBOL_BITS]);

__device__ __forceinline__ void sobol_point_device(
    unsigned int n, int dim,
    const unsigned int* __restrict__ d_shift,
    const unsigned int* __restrict__ d_directions,
    float* out)
{
    unsigned int gray = n ^ (n >> 1);

    for (int d = 0; d < dim; ++d) {
        unsigned int x = d_shift[d];
        for (int bit = 0; bit < GPU_SOBOL_BITS; ++bit) {
            if (gray & (1u << bit)) {
                x ^= d_directions[d * GPU_SOBOL_BITS + bit];
            }
        }
        out[d] = __uint2float_rn(x) * 2.3283064365386963e-10f; 
    }
}
