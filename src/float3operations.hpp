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