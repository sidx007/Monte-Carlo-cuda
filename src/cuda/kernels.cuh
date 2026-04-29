#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "moro_device.cuh"

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

__device__ __forceinline__
float lcg_next(uint32_t& state) {
    state = 1664525u * state + 1013904223u;
    return __uint2float_rn(state) * 2.3283064365386963e-10f;
}

__device__ __forceinline__
double cnd_device(double d);

__device__ __forceinline__
double bs_call_device(double S, double X, double t, double v, double r) {
    if (t <= 0.0) return fmax(S - X, 0.0);
    double sqt = sqrt(t);
    double d1  = (log(S / X) + (r + 0.5 * v * v) * t) / (v * sqt);
    double d2  = d1 - v * sqt;
    return S * cnd_device(d1) - X * exp(-r * t) * cnd_device(d2);
}

__device__ __forceinline__
double cnd_device(double d) {
    return 0.5 * erfc(-d * M_SQRT1_2);
}