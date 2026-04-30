#include "sobol_gpu.cuh"

__device__ unsigned int d_V[GPU_SOBOL_DIM * GPU_SOBOL_BITS];

void sobol_gpu_init(const unsigned int V_host[GPU_SOBOL_DIM][GPU_SOBOL_BITS]) {
	cudaMemcpyToSymbol(d_V, V_host,
		GPU_SOBOL_DIM * GPU_SOBOL_BITS * sizeof(unsigned int));
}



