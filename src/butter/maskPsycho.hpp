namespace butter{

__launch_bounds__(256)
__global__ void diffPrecompute_Kernel(float* mem1, float* mem2, float* dst, int width, int height){
    size_t thx = threadIdx.x + blockIdx.x*blockDim.x;
    int x = thx%width;
    int y = thx/width;

    if (y >= height) return;

    int x2, y2;

    if (y + 1 < height) {
        y2 = y + 1;
    } else if (y > 0) {
        y2 = y - 1;
    } else {
        y2 = y;
    }

    if (x + 1 < width) {
        x2 = x + 1;
    } else if (x > 0) {
        x2 = x - 1;
    } else {
        x2 = x;
    }

    float sup0 = abs(mem1[y*width + x] - mem1[y*width + x2]) + abs(mem1[y*width + x] - mem1[y2*width + x]);
    float sup1 = abs(mem2[y*width + x] - mem2[y*width + x2]) + abs(mem2[y*width + x] - mem2[y2*width + x]);
    dst[y*width + x] = 0.918416534734 * min(sup0, sup1);
    if (dst[y*width + x] >= 55.0184555849) dst[y*width + x] = 55.0184555849;
    
}

void diffPrecompute(float* src1, float* src2, float* dst, int width, int height, hipStream_t stream){
    int th_x = std::min(256, width*height);
    int bl_x = (width*height-1)/th_x + 1;
    diffPrecompute_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(src1, src2, dst, width, height);
    GPU_CHECK(hipGetLastError());
}

__launch_bounds__(256)
__global__ void maskmuladd_Kernel(float* mem1, float* mem2, float* dst, int width, float a, float b){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;

    if (x >= width) return;

    dst[x] = a*mem1[x] + b*mem2[x];
}

void maskmuladd(float* src1, float* src2, float* dst, int width, float a, float b, hipStream_t stream){
    int th_x = std::min(256, width);
    int bl_x = (width-1)/th_x + 1;
    maskmuladd_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(src1, src2, dst, width, a, b);
    GPU_CHECK(hipGetLastError());
}

__device__ float makeMask(float extmul, float extoff, float mul, float offset, float scaler, int index){
    float c = mul / ((0.01 * scaler * index) + offset);
    float res = (1.0 + extmul * (c + extoff))/20.35;
    if (res < 1e-5) {
      res = 1e-5;
    }
    return res * res;
}

__device__ float negativeClampInterpolateMakeMask(float extmul, float extoff, float mul, float offset, float scaler, float index){
    if (index < 0) index = 0;
    int lower = (int)index;
    int upper = lower + 1;
    if (lower >= 512) lower = 511;
    if (upper >= 512) upper = 511;
    float val1 = makeMask(extmul, extoff, mul, offset, scaler, lower);
    float val2 = makeMask(extmul, extoff, mul, offset, scaler, upper);
    return val1*(index - lower) + val2*(1 - (index - lower));
}

__device__ float maskX(float p){
    return negativeClampInterpolateMakeMask(2.59885507073, 3.08805636789, 5.62939030582, 0.315424196682, 16.2770141832, p);
}

__device__ float maskY(float p){
    return negativeClampInterpolateMakeMask(0.9613705131, -0.581933100068, 6.64307621174, 1.00846207765, 2.2342321176, p);
}

__device__ float maskDcX(float p){
    return negativeClampInterpolateMakeMask(10.0470705878, 3.18472654033, 0.373092999662, 0.0551512255218, 70.0, p);
}

__device__ float maskDcY(float p){
    return negativeClampInterpolateMakeMask(0.0115640939227, 45.9483175519, 2.52611324247, 0.0142290066313, 5.0, p);
}

__launch_bounds__(256)
__global__ void bcomponentmasking_Kernel(float* mask0, float* mask1, float* mask2, float* mask_dc0, float* mask_dc1, float* mask_dc2, int width, int height){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;

    if (x >= width) return;

    const float p1 = mask1[x] * 2.1887170895 * 2.1364621982;
    const float p0 = mask0[x] * 36.4671237619 * 16.6963293877 + p1 * 0.0513061271723;

    mask0[x] = maskX(p0);
    mask1[x] = maskY(p1);
    mask2[x] = maskY(p1) * 0.086624184478;
    mask_dc0[x] = maskDcX(p0);
    mask_dc1[x] = maskDcY(p1);
    mask_dc2[x] = maskDcY(p1) * 21.6804277046;
}

void bcomponentmasking(float* mask0, float* mask1, float* mask2, float* mask_dc0, float* mask_dc1, float* mask_dc2, int width, int height, hipStream_t stream){
    int th_x = std::min(256, width*height);
    int bl_x = (width*height-1)/th_x + 1;
    bcomponentmasking_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(mask0, mask1, mask2, mask_dc0, mask_dc1, mask_dc2, width, height);
    GPU_CHECK(hipGetLastError());
}

void MaskPsychoImage(Plane_d* hf1, Plane_d* uhf1, Plane_d* hf2, Plane_d* uhf2, Plane_d* mask_xyb0, Plane_d* mask_xyb1, Plane_d* mask, Plane_d* mask_dc, float* gaussiankernel) {
    const hipStream_t stream = hf1[0].stream;
    const int width = hf1[0].width;
    const int height = hf1[0].height;

    const float muls[4] = {
        0,
        1.64178305129,
        0.831081703362,
        3.23680933546,
    };

    for (int i = 0; i < 2; i++){
        maskmuladd(hf1[i].mem_d, uhf1[i].mem_d, mask_xyb0[i].mem_d, width*height, muls[2*i], muls[2*i+1], stream);
        maskmuladd(hf2[i].mem_d, uhf2[i].mem_d, mask_xyb1[i].mem_d, width*height, muls[2*i], muls[2*i+1], stream);
    }
    //hf and uhf are not used anymore and can serve as temporary planes
    Plane_d temp1 = hf1[0];
    Plane_d temp2 = hf1[1];
    Plane_d temp3 = uhf1[0];
    Plane_d temp4 = uhf1[1];

    //X component
    diffPrecompute(mask_xyb0[0].mem_d, mask_xyb1[0].mem_d, temp1.mem_d, width, height, stream);
    temp1.blur(mask[0], temp2, 9.24456601467, -0.0724948220913, gaussiankernel);

    //Y component
    diffPrecompute(mask_xyb0[1].mem_d, mask_xyb1[1].mem_d, temp1.mem_d, width, height, stream);
    temp1.blur(temp2, temp4, 2.3770330432, -0.0724948220913, gaussiankernel);
    temp1.blur(temp3, temp4, 9.04353323561, -0.0724948220913, gaussiankernel);
    maskmuladd(temp2.mem_d, temp3.mem_d, mask[1].mem_d, width*height, 0.43660192108469253, 0.5633980789153075, stream);

    //B component
    bcomponentmasking(mask[0].mem_d, mask[1].mem_d, mask[2].mem_d, mask_dc[0].mem_d, mask_dc[1].mem_d, mask_dc[2].mem_d, width, height, stream);
}

}