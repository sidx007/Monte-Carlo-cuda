#include "sobol_gpu.cuh"

static __device__ unsigned int d_V[GPU_SOBOL_DIM * GPU_SOBOL_BITS];

void sobol_gpu_init(const unsigned int V_host[GPU_SOBOL_DIM][GPU_SOBOL_BITS]) {
	cudaMemcpyToSymbol(d_V, V_host,
		GPU_SOBOL_DIM * GPU_SOBOL_BITS * sizeof(unsigned int));
}

unsigned int* get_d_V_ptr() {
	unsigned int* ptr = nullptr;
	cudaGetSymbolAddress(reinterpret_cast<void**>(&ptr), d_V);
	return ptr;
}



