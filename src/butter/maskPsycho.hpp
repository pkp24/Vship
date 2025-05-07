namespace butter{

__launch_bounds__(256)
__global__ void diffPrecompute_Kernel(float* mem1, float* dst, int width, int height, float mul, float bias_arg){
    size_t thx = threadIdx.x + blockIdx.x*blockDim.x;
    //int x = thx%width;
    int y = thx/width;

    if (y >= height) return;

    float bias = mul*bias_arg;
    dst[thx] = sqrtf(mul * abs(mem1[thx]) + bias) - sqrtf(bias);
    //if (thx == 10000) printf("diffprecompute : %f from %f\n", dst[thx], mem1[thx]);
}

void diffPrecompute(float* src, float* dst, int width, int height, float mul, float bias_arg, hipStream_t stream){
    int th_x = std::min(256, width*height);
    int bl_x = (width*height-1)/th_x + 1;
    diffPrecompute_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(src, dst, width, height, mul, bias_arg);
    GPU_CHECK(hipGetLastError());
}

__launch_bounds__(256)
__global__ void maskinit_Kernel(float* hfx, float* uhfx, float* hfy, float* uhfy, float* dst, int width, float a, float b, float c){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;

    if (x >= width) return;

    float xdiff = (uhfx[x]+hfx[x])*a;
    float ydiff = uhfy[x]*b + hfy[x]*c;
    dst[x] = sqrtf(xdiff*xdiff + ydiff*ydiff);
    //if (x == 10000) printf("maskxyb : %f from hfx %f, uhfx %f, hfy %f, uhfy %f\n", dst[x], hfx[x], uhfx[x], hfy[x], uhfy[x]);
}

void maskinit(float* hfx, float* uhfx, float* hfy, float* uhfy, float* dst, int width, float a, float b, float c, hipStream_t stream){
    int th_x = std::min(256, width);
    int bl_x = (width-1)/th_x + 1;
    maskinit_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(hfx, uhfx, hfy, uhfy, dst, width, a, b, c);
    GPU_CHECK(hipGetLastError());
}

__device__ void StoreMin3(const float v, float& min0, float& min1, float& min2) {
  if (v < min2) {
    if (v < min0) {
      min2 = min1;
      min1 = min0;
      min0 = v;
    } else if (v < min1) {
      min2 = min1;
      min1 = v;
    } else {
      min2 = v;
    }
  }
}

__launch_bounds__(256)
__global__ void fuzzyerrosion_Kernel(float* src, float* dst, int width, int height){
    size_t ux = threadIdx.x + blockIdx.x*blockDim.x;

    int x = ux%width;
    int y = ux/width;

    if (y >= height) return;

    const int kStep = 3;
    float min0 = src[ux];
    float min1 = 2.0f * min0;
    float min2 = min1;
    if (x >= kStep) {
        float v = src[y*width + x - kStep];
        StoreMin3(v, min0, min1, min2);
        if (y >= kStep) {
            float v = src[(y-kStep)*width + x - kStep];
            StoreMin3(v, min0, min1, min2);
        }
        if (y < height - kStep) {
            float v = src[(y+kStep)*width + x - kStep];
            StoreMin3(v, min0, min1, min2);
        }
    }
    if (x < width - kStep) {
        float v = src[y*width + x + kStep];
        StoreMin3(v, min0, min1, min2);
        if (y >= kStep) {
            float v = src[(y-kStep)*width + x + kStep];
            StoreMin3(v, min0, min1, min2);
        }
        if (y < height - kStep) {
            float v = src[(y+kStep)*width + x + kStep];
            StoreMin3(v, min0, min1, min2);
        }
    }
    if (y >= kStep) {
        float v = src[(y-kStep)*width + x];
        StoreMin3(v, min0, min1, min2);
    }
    if (y < height - kStep) {
        float v = src[(y+kStep)*width + x];
        StoreMin3(v, min0, min1, min2);
    }
    dst[ux] = (0.45f * min0 + 0.3f * min1 + 0.25f * min2);
    //if (ux == 10000) printf("Erosion : %f from %f\n", dst[ux], src[ux]);
}

void fuzzyerrosion(float* src, float* dst, int width, int height, hipStream_t stream){
    int th_x = std::min(256, width*height);
    int bl_x = (width*height-1)/th_x + 1;
    fuzzyerrosion_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(src, dst, width, height);
    GPU_CHECK(hipGetLastError());
}

void MaskPsychoImage(float* hf1[3], float* uhf1[3], float* hf2[3], float* uhf2[3], float* mask_xyb0, float* mask_xyb1, float* mask, float* block_diff_ac[3], int width, int height, GaussianHandle& gaussianHandle, hipStream_t stream) {
    const float muls[3] = {
      2.5f,
      0.4f,
      0.4f,
    };

    maskinit(hf1[0], uhf1[0], hf1[1], uhf1[1], mask_xyb0, width*height, muls[0], muls[1], muls[2], stream);
    maskinit(hf2[0], uhf2[0], hf2[1], uhf2[1], mask_xyb1, width*height, muls[0], muls[1], muls[2], stream);
    
    //hf and uhf are not used anymore and can serve as temporary planes
    float* diff0 = hf1[0];
    float* diff1 = hf1[1];

    diffPrecompute(mask_xyb0, diff0, width, height, 6.19424080439f, 12.61050594197f, stream);
    diffPrecompute(mask_xyb1, diff1, width, height, 6.19424080439f, 12.61050594197f, stream);
    float* blurred0 = mask_xyb0;
    float* blurred1 = mask_xyb1;
    blurDstNoTemp(blurred0, diff0, width, height, gaussianHandle, 2, stream);
    blurDstNoTemp(blurred1, diff1, width, height, gaussianHandle, 2, stream);
    fuzzyerrosion(blurred0, mask, width, height, stream);
    L2diff(blurred0, blurred1, block_diff_ac[1], width*height, 10.0f, stream);
}

}