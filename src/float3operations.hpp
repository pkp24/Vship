__device__ __host__ void inline operator+=(float3& a, float3& b){
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__device__ __host__ void inline operator/=(float3& a, float3& b){
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}

__device__ __host__ void inline operator/=(float3& a, const float b){
    a.x /= b;
    a.y /= b;
    a.z /= b;
}

#ifndef __HIPCC__
__device__ __host__ float3 inline operator*(const float3& a, const float3& b){
    float3 out;
    out.x = a.x*b.x;
    out.y = a.y*b.y;
    out.z = a.z*b.z;
    return out;
}
#endif

__device__ __host__ float3 inline operator*(const float3& a, const float& b){
    float3 out;
    out.x = a.x*b;
    out.y = a.y*b;
    out.z = a.z*b;
    return out;
}

__global__ void multarray_Kernel(float3* src1, float3* src2, float3* dst, int width){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;

    if (x >= width) return;

    dst[x] = src1[x]*src2[x];
}

void multarray(float3* src1, float3* src2, float3* dst, int width, hipStream_t stream){
    int th_x = std::min(256, width);
    int bl_x = (width-1)/th_x + 1;
    multarray_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(src1, src2, dst, width);
}