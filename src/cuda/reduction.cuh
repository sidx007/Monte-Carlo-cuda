#pragma once
#include <cuda_runtime.h>

__device__ __forceinline__
double warp_reduce_sum(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__
double block_reduce_sum(double val, double* sdata) {
    int tid  = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;

    val = warp_reduce_sum(val);

    if (lane == 0) sdata[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + 31) >> 5;
    val = (tid < num_warps) ? sdata[tid] : 0.0;
    if (wid == 0) val = warp_reduce_sum(val);

    return val;
}