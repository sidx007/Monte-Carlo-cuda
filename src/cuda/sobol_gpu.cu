#include "sobol_gpu.cuh"

__constant__ unsigned int d_V[GPU_SOBOL_DIM][GPU_SOBOL_BITS];

void sobol_gpu_init(const unsigned int V_host[GPU_SOBOL_DIM][GPU_SOBOL_BITS]) {
    cudaMemcpyToSymbol(d_V, V_host,
        GPU_SOBOL_DIM * GPU_SOBOL_BITS * sizeof(unsigned int));
}

__device__ void sobol_point_device(
    unsigned int n, int dim,
    const unsigned int* __restrict__ d_shift,
    float* out)
{
    unsigned int gray = n ^ (n >> 1);

    for (int d = 0; d < dim; ++d) {
        unsigned int x = d_shift[d];
        unsigned int g = gray;
        int bit = 0;
        while (g) {
            if (g & 1u) x ^= d_V[d][GPU_SOBOL_BITS - 1 - bit];
            g >>= 1;
            ++bit;
        }
        out[d] = __uint2float_rn(x) * 2.3283064365386963e-10f; 
    }
}
