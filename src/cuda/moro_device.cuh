#pragma once
#include <cuda_runtime.h>

__device__ __forceinline__ float moro_inv_cnd_device(float u) {
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
