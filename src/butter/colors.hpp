namespace butter{

__device__ float gamma(float v) {
    const float kGamma = 0.372322653176;
    const float limit = 37.8000499603;
    float bright = v - limit;
    if (bright >= 0) {
        const float mul = 0.0950819040934;
        v -= bright * mul;
    }
    {
        const float limit2 = 74.6154406429;
        float bright2 = v - limit2;
        if (bright2 >= 0) {
        const float mul = 0.01;
        v -= bright2 * mul;
        }
    }
    {
        const float limit2 = 82.8505938033;
        float bright2 = v - limit2;
        if (bright2 >= 0) {
        const float mul = 0.0316722592629;
        v -= bright2 * mul;
        }
    }
    {
        const float limit2 = 92.8505938033;
        float bright2 = v - limit2;
        if (bright2 >= 0) {
        const float mul = 0.221249885752;
        v -= bright2 * mul;
        }
    }
    {
        const float limit2 = 102.8505938033;
        float bright2 = v - limit2;
        if (bright2 >= 0) {
        const float mul = 0.0402547853939;
        v -= bright2 * mul;
        }
    }
    {
        const float limit2 = 112.8505938033;
        float bright2 = v - limit2;
        if (bright2 >= 0) {
        const float mul = 0.021471798711500003;
        v -= bright2 * mul;
        }
    }
    const float offset = 0.106544447664;
    const float scale = 10.7950943969;
    float retval = scale * (offset + pow(v, kGamma));
    return retval;
}

__global__ void linearrgb_kernel(float* src1, float* src2, float* src3, int width, int height){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    if (x >= width*height) return;
    rgb_to_linrgbfunc(src1[x]);
    rgb_to_linrgbfunc(src2[x]);
    rgb_to_linrgbfunc(src3[x]);
}

__device__ inline void butterOpsinAbsorbance(float3& a){
    float3 out;
    out.x = fmaf(0.254462330846, a.x,
    fmaf(0.488238255095, a.y,
    fmaf(0.0635278003854, a.z,
    1.01681026909)));

    out.y = fmaf(0.195214015766, a.x,
    fmaf(0.568019861857, a.y,
    fmaf(0.0860755536007, a.z,
    1.1510118369)));

    out.z = fmaf(0.07374607900105684, a.x,
    fmaf(0.06142425304154509, a.y,
    fmaf(0.24416850520714256, a.z,
    1.20481945273)));

    a = out;
}

__global__ void opsinDynamicsImage_kernel(float* src1, float* src2, float* src3, float* blurred1, float* blurred2, float* blurred3, int width, int height, float intensity_multiplier){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    if (x >= width*height) return;

    float3 sensitivity;
    float3 src = {src1[x], src2[x], src3[x]};
    float3 blurred = {blurred1[x], blurred2[x], blurred3[x]};
    //float3 oldsrc = src; float3 oldblurred = blurred;
    src *= intensity_multiplier;
    blurred *= intensity_multiplier;
    butterOpsinAbsorbance(blurred);
    blurred = max(blurred, 1e-4f);
    sensitivity.x = gamma(blurred.x) / blurred.x;
    sensitivity.y = gamma(blurred.y) / blurred.y;
    sensitivity.z = gamma(blurred.z) / blurred.z;
    butterOpsinAbsorbance(src);
    src *= sensitivity;
    src.x = max(src.x, 1.7557483643287353f);
    src.y = max(src.y, 1.7557483643287353f);
    src.z = max(src.z, 12.226454707163354f);

    //make positive
    src.x = src.x - src.y; // x - y
    src.y = src.x + 2*src.y; // x + y = (x-y)+2y
    //if (x == 0) printf("%f, %f, %f and %f, %f, %f to %f, %f, %f with %f, %f, %f sens\n", oldsrc.x, oldsrc.y, oldsrc.z, oldblurred.x, oldblurred.y, oldblurred.z, src.x, src.y, src.z, sensitivity.x, sensitivity.y, sensitivity.z);
    src1[x] = src.x; src2[x] = src.y; src3[x] = src.z;
}

void opsinDynamicsImage(Plane_d src[3], Plane_d temp[3], Plane_d temp2, float* gaussiankernel, float intensity_multiplier){
    //change src from SRGB to opsin dynamic XYB
    int width = src[0].width; int height = src[0].height;
    int th_x = std::min(256, width*height);
    int bl_x = (width*height-1)/th_x + 1;
    linearrgb_kernel<<<dim3(bl_x), dim3(th_x), 0, src[0].stream>>>(src[0].mem_d, src[1].mem_d, src[2].mem_d, width, height);
    //printf("initial adress: %llu\n", (unsigned long long)src[0].mem_d);
    for (int i = 0; i < 3; i++){
        src[i].blur(temp[i], temp2, 1.2f, 0.0, gaussiankernel);
    }
    opsinDynamicsImage_kernel<<<dim3(bl_x), dim3(th_x), 0, src[0].stream>>>(src[0].mem_d, src[1].mem_d, src[2].mem_d, temp[0].mem_d, temp[1].mem_d, temp[2].mem_d, width, height, intensity_multiplier);
    GPU_CHECK(hipGetLastError());
}

}