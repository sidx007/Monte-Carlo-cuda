#pragma once
#include <cuda_runtime.h>

static constexpr int GPU_SOBOL_DIM  = 21;  
static constexpr int GPU_SOBOL_BITS = 32;  

void sobol_gpu_init(const unsigned int V_host[GPU_SOBOL_DIM][GPU_SOBOL_BITS]);

__device__ void sobol_point_device(
    unsigned int n,
    int          dim,
    const unsigned int* __restrict__ d_shift,
    float*       out);
