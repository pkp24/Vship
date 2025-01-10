namespace butter{

__device__ float gamma(float v) {
    return fmaf(19.245013259874995f, logf(v + 9.9710635769299145), -23.16046239805755);
}

__global__ void linearrgb_kernel(float* src1, float* src2, float* src3, int width, int height){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    if (x >= width*height) return;
    rgb_to_linrgbfunc(src1[x]);
    rgb_to_linrgbfunc(src2[x]);
    rgb_to_linrgbfunc(src3[x]);
}

__device__ inline void butterOpsinAbsorbance(float3& a, bool clamp = false){
    float3 out;
    out.x = fmaf(0.29956550340058319, a.x,
    fmaf(0.63373087833825936, a.y,
    fmaf(0.077705617820981968, a.z,
    1.7557483643287353)));

    out.y = fmaf(0.22158691104574774, a.x,
    fmaf(0.69391388044116142, a.y,
    fmaf(0.0987313588422, a.z,
    1.7557483643287353)));

    out.z = fmaf(0.02, a.x,
    fmaf(0.2, a.y,
    fmaf(0.20480129041026129, a.z,
    12.226454707163354)));

    if (clamp){
        a.x = max(a.x, 1.7557483643287353);
        a.y = max(a.y, 1.7557483643287353);
        a.z = max(a.z, 12.226454707163354);
    }

    a = out;
}

__global__ void opsinDynamicsImage_kernel(float* src1, float* src2, float* src3, float* blurred1, float* blurred2, float* blurred3, int width, int height, float intensity_multiplier){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    if (x >= width*height) return;
    

    float3 sensitivity;
    float3 src = {src1[x], src2[x], src3[x]};
    float3 blurred = {blurred1[x], blurred2[x], blurred3[x]};
    src *= intensity_multiplier;
    blurred *= intensity_multiplier;
    butterOpsinAbsorbance(blurred, true);
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
}

void opsinDynamicsImage(Plane_d src[3], Plane_d temp[3], Plane_d temp2, float* gaussiankernel, float intensity_multiplier){
    //change src from SRGB to opsin dynamic XYB
    int width = src[0].width; int height = src[0].height;
    int th_x = std::min(256, width*height);
    int bl_x = (width*height-1)/th_x + 1;
    linearrgb_kernel<<<dim3(bl_x), dim3(th_x), 0, src[0].stream>>>(src[0].mem_d, src[1].mem_d, src[2].mem_d, width, height);
    for (int i = 0; i < 3; i++){
        src[i].blur(temp[i], temp2, 1.2f, 0.0, gaussiankernel);
    }
    opsinDynamicsImage_kernel<<<dim3(bl_x), dim3(th_x), 0, src[0].stream>>>(src[0].mem_d, src[1].mem_d, src[2].mem_d, temp[0].mem_d, temp[1].mem_d, temp[2].mem_d, width, height, intensity_multiplier);
}

}