const float K_D0 = 0.0037930734;
const float K_D1 = std::cbrt(K_D0);

const float K_M02 = 0.078;
const float K_M00 = 0.30;
const float K_M01 = 1.0 - K_M02 - K_M00;

const float K_M12 = 0.078;
const float K_M10 = 0.23;
const float K_M11 = 1.0 - K_M12 - K_M10;

const float K_M20 = 0.24342269;
const float K_M21 = 0.20476745;
const float K_M22 = 1.0 - K_M20 - K_M21;

const float OPSIN_ABSORBANCE_MATRIX[9] = {K_M00, K_M01, K_M02, K_M10, K_M11, K_M12, K_M20, K_M21, K_M22};
const float OPSIN_ABSORBANCE_BIAS = K_D0;
const float ABSORBANCE_BIAS = -K_D1;

__device__ inline void opsin_absorbance(float3& a, const float matrix[9], const float opsin_bias){
    float3 out;
    out.x = fmaf(matrix[0], a.x,
    fmaf(matrix[1], a.y,
    fmaf(matrix[2], a.z,
    opsin_bias)));

    out.y = fmaf(matrix[3], a.x,
    fmaf(matrix[4], a.y,
    fmaf(matrix[5], a.z,
    opsin_bias)));

    out.z = fmaf(matrix[6], a.x,
    fmaf(matrix[7], a.y,
    fmaf(matrix[8], a.z,
    opsin_bias)));

    a = out;
}

__device__ inline void mixed_to_xyb(float3& a){
    a.x = 0.5 * (a.x - a.y);
    a.y = a.x + a.y;
}

__device__ inline void linear_rgb_to_xyb(float3& a, const float matrix[9], const float opsin_bias, const float abs_bias){
    opsin_absorbance(a, matrix, opsin_bias);
    //printf("from %f to %f\n", a.x, cbrtf(a.x*((int)(a.x >= 0))));
    a.x = cbrtf(a.x * ((int)(a.x >= 0))) + abs_bias;
    a.y = cbrtf(a.y * ((int)(a.y >= 0))) + abs_bias;
    a.z = cbrtf(a.z * ((int)(a.z >= 0))) + abs_bias;
    //printf("got %f, %f, %f\n", a.x, a.y, a.z);
    mixed_to_xyb(a);
}

__device__ inline void make_positive_xyb(float3& a){
    a.z = (a.z - a.y) + 0.55;
    a.x = a.x * 14 + 0.42;
    a.y += 0.01;
}

__device__ inline void rgb_to_positive_xyb_d(float3& a, const float matrix[9], const float opsin_bias, const float abs_bias){
    linear_rgb_to_xyb(a, matrix, opsin_bias, abs_bias);
    make_positive_xyb(a);
}

__device__ inline void rgb_to_linrgbfunc(float& a){
    if (a > 0.04045){
        a = powf(((a+0.055)/(1+0.055)), 2.4f);
    } else {
        a = a/12.92;
    }
}

__device__ inline void rgb_to_linrgb(float3& a){
    rgb_to_linrgbfunc(a.x);
    rgb_to_linrgbfunc(a.y);
    rgb_to_linrgbfunc(a.z);
}

__launch_bounds__(256)
__global__ void rgb_to_positive_xyb_Kernel(float3* array, int width, const float M_00, const float M_01, const float M_02, const float M_10, const float M_11, const float M_12, const float M_20, const float M_21, const float M_22, const float opsin_bias, const float abs_bias){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    if (x >= width) return;
    const float matrix[9] = {M_00, M_01, M_02, M_10, M_11, M_12, M_20, M_21, M_22};
    //float3 old = array[x];
    rgb_to_linrgb(array[x]);
    rgb_to_positive_xyb_d(array[x], matrix, opsin_bias, abs_bias);
    //printf("from %f, %f, %f to %f, %f, %f\n", old.x, old.y, old.z, array[x].x, array[x].y, array[x].z);
}

__host__ inline void rgb_to_positive_xyb(float3* array, int width, hipStream_t stream){
    int th_x = std::min(256, width);
    int bl_x = (width-1)/th_x + 1;
    rgb_to_positive_xyb_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(array, width, K_M00, K_M01, K_M02, K_M10, K_M11, K_M12, K_M20, K_M21, K_M22, OPSIN_ABSORBANCE_BIAS, ABSORBANCE_BIAS);
    GPU_CHECK(hipGetLastError());
}