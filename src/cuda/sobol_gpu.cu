#include "sobol_gpu.cuh"

__constant__ unsigned int d_V[GPU_SOBOL_DIM][GPU_SOBOL_BITS];

