#pragma once

namespace VshipColorConvert{

enum chromaSubType{
    chroma_444,
    chroma_420,
    chroma_422,
    chroma_440,
    chroma_411,
    chroma_410,
};

//launch in 16*16 blocks
template<AVChromaLocation location, chromaSubType chromatype>
__global__ void upsample_kernel(float* dst, float* src, int64_t width, int64_t height);

template<AVChromaLocation location, chromaSubType chromatype>
__host__ void inline upsample(float* dst, float* src, int64_t width, int64_t height, hipStream_t stream){
    int64_t thx = 16;
    int64_t thy = 16;
    int64_t blx = (width+thx-1)/thx;
    int64_t bly = (height+thy-1)/thy;
    upsample_kernel<location, chromatype><<<dim3(blx, bly), dim3(thx, thy), 0, stream>>>(dst, src, width, height);
}

class CubicHermitSplineInterpolator{
    float v1; float v2; float v3; float v4;
public:
    __device__ __host__ CubicHermitSplineInterpolator(float p0, float m0, float p1, float m1){
        v1 = 2*p0 + m0 - 2*p1 + m1;
        v2 = -3*p0 + 3*p1 - 2*m0 - m1;
        v3 = m0;
        v4 = p0;
    }
    __device__ __host__ float get(float t){ //cubic uses t between 0 and 1
        float res = v1;
        res *= t;
        res += v2;
        res *= t;
        res += v3;
        res *= t;
        res += v4;
        return res;
    }
};

}