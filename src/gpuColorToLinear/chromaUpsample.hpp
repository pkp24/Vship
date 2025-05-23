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

class CubicHermitSplineInterpolator{
    float v1; float v2; float v3; float v4;
public:
    __device__ __host__ CubicHermitSplineInterpolator(const float p0, const float m0, const float p1, const float m1){
        v1 = 2*p0 + m0 - 2*p1 + m1;
        v2 = -3*p0 + 3*p1 - 2*m0 - m1;
        v3 = m0;
        v4 = p0;
    }
    __device__ __host__ float get(const float t){ //cubic uses t between 0 and 1
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

__device__ CubicHermitSplineInterpolator getHorizontalInterpolator_device(float* src, int64_t x, int64_t y, int64_t width, int64_t height){ //width and height must be the one of source!!!!
    const float el0 = src[std::min(y, height-1)*width + std::min(x, width-1)];
    const float elm1 = src[std::min(y, height-1)*width + std::min(std::max(x-1, (int64_t)0), width-1)]; //left element
    const float el1 = src[std::min(y, height-1)*width + std::min(x+1, width-1)];
    const float el2 = src[std::min(y, height-1)*width + std::min(x+2, width-1)];
    return CubicHermitSplineInterpolator(el0, (el1 - elm1)/2, el1, (el2 - el0)/2);
}

__device__ CubicHermitSplineInterpolator getVerticalInterpolator_device(float* src, int64_t x, int64_t y, int64_t width, int64_t height){ //width and height must be the one of source!!!!
    const float el0 = src[std::min(y, height-1)*width + std::min(x, width-1)];
    const float elm1 = src[std::min(std::max(y-1, (int64_t)0), height-1)*width + std::min(x, width-1)]; //left element
    const float el1 = src[std::min(y+1, height-1)*width + std::min(x, width-1)];
    const float el2 = src[std::min(y+2, height-1)*width + std::min(x, width-1)];
    return CubicHermitSplineInterpolator(el0, (el1 - elm1)/2, el1, (el2 - el0)/2);
}

//dst of size (2*width)*height while src is of size width*height
__global__ void bicubicHorizontalCenterUpscaleX2_Kernel(float* dst, float* src, int64_t width, int64_t height){
    int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t y = threadIdx.y + blockIdx.y * blockDim.y;
    CubicHermitSplineInterpolator interpolator = getHorizontalInterpolator_device(src, x, y, width, height);
    //this interpolator is valid on interval [0, 1] representing [x, x+1]
    //we are Center so we are interested in values: 0.25 and 0.75
    if (y < height && x < width){
        dst[y*2*width + 2*x] = interpolator.get(0.25);
        dst[y*2*width + 2*x+1] = interpolator.get(0.75);
    }
}

//dst of size (2*width)*height while src is of size width*height
__global__ void bicubicHorizontalLeftUpscaleX2_Kernel(float* dst, float* src, int64_t width, int64_t height){
    int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t y = threadIdx.y + blockIdx.y * blockDim.y;
    CubicHermitSplineInterpolator interpolator = getHorizontalInterpolator_device(src, x, y, width, height);
    //this interpolator is valid on interval [0, 1] representing [x, x+1]
    //we are Left so we are interested in values: 0 and 0.5 (0 is directly our value)
    if (y < height && x < width){
        dst[y*2*width + 2*x] = src[y*width + x];
        dst[y*2*width + 2*x+1] = interpolator.get(0.5);
    }
}

//dst of size (4*width)*height while src is of size width*height
__global__ void bicubicHorizontalCenterUpscaleX4_Kernel(float* dst, float* src, int64_t width, int64_t height){
    int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t y = threadIdx.y + blockIdx.y * blockDim.y;
    CubicHermitSplineInterpolator interpolator = getHorizontalInterpolator_device(src, x, y, width, height);
    //this interpolator is valid on interval [0, 1] representing [x, x+1]
    //we are Center so we are interested in values: 0.125, 0.375, 0.625 and 0.875
    if (y < height && x < width){
        dst[y*4*width + 4*x] = interpolator.get(0.125);
        dst[y*4*width + 4*x+1] = interpolator.get(0.375);
        dst[y*4*width + 4*x+2] = interpolator.get(0.625);
        dst[y*4*width + 4*x+3] = interpolator.get(0.875);
    }
}

//dst of size (4*width)*height while src is of size width*height
__global__ void bicubicHorizontalLeftUpscaleX4_Kernel(float* dst, float* src, int64_t width, int64_t height){
    int64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t y = threadIdx.y + blockIdx.y * blockDim.y;
    CubicHermitSplineInterpolator interpolator = getHorizontalInterpolator_device(src, x, y, width, height);
    //this interpolator is valid on interval [0, 1] representing [x, x+1]
    //we are Left so we are interested in values: 0, 0.25, 0.5 and 0.75 (0 is directly our value)
    if (y < height && x < width){
        dst[y*4*width + 4*x] = src[y*width + x];
        dst[y*4*width + 4*x+1] = interpolator.get(0.25);
        dst[y*4*width + 4*x+2] = interpolator.get(0.5);
        dst[y*4*width + 4*x+3] = interpolator.get(0.75);
    }
}

template<AVChromaLocation location, chromaSubType chromatype>
__host__ void inline upsample(float* dst, float* src, int64_t width, int64_t height, hipStream_t stream){
    int64_t thx = 16;
    int64_t thy = 16;
    int64_t blx = (width+thx-1)/thx;
    int64_t bly = (height+thy-1)/thy;
    //upsample_kernel<location, chromatype><<<dim3(blx, bly), dim3(thx, thy), 0, stream>>>(dst, src, width, height);
}

}