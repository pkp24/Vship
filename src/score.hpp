__device__ __host__ float tothe4th(float x){
    float y = x*x;
    return y*y;
}

__device__ __host__ float3 tothe4th(float3 x){
    float3 y = x*x;
    return y*y;
}

__global__ void ssim_map_Kernel(float3* dst, float3* s11, float3* s22, float3* s12, float3* mu1, float3* mu2, int width, int height){
    //dst must be of size 2*sizeof(float3)*blocknum at least
    //shared memory needed is 2*sizeof(float3)*threadnum at least
    const int x = threadIdx.x + blockIdx.x*blockDim.x;
    const int thx = threadIdx.x;
    const int threadnum = blockDim.x;
    
    extern __shared__ float3 sharedmem[];
    float3* sum1 = sharedmem; //size sizeof(float3)*threadnum
    float3* sum4 = sharedmem+sizeof(float3)*blockDim.x; //size sizeof(float3)*threadnum

    float3 d1;
    if (x < width*height){
        const float3 m1 = mu1[x]; const float3 m2 = mu2[x];
        const float3 m11 = m1*m1;
        const float3 m22 = m2*m2;
        const float3 m12 = m1*m2;
        const float3 m_diff = m1-m2;
        const float3 num_m = fmaf(m_diff, m_diff*-1., 1.0);
        const float3 num_s = fmaf(s12[x] - m12, 2.0, 0.0009);
        const float3 denom_s = (s11[x] - m11) + (s22[x] - m22) + 0.0009;
        d1 = max(1.0 - ((num_m * num_s)/denom_s), 0.);
    } else {
        d1.x = 0;
        d1.y = 0;
        d1.z = 0;
    }

    sum1[thx] = d1;
    sum4[thx] = tothe4th(d1);
    __syncthreads();
    //now we need to do some pointer jumping to regroup every block sums;
    int next = 1;
    while (next < threadnum){
        if (thx + next < threadnum && (thx%(next*2) == 0)){
            sum1[thx] += sum1[thx+next];
            sum4[thx] += sum4[thx+next];
        }
        next *= 2;
        __syncthreads();
    }
    if (thx == 0){
        dst[blockIdx.x] = sum1[0];
        dst[gridDim.x + blockIdx.x] = sum4[0];
    }
}

std::vector<float3> ssim_map(float3* s11, float3* s22, float3* s12, float3* mu1, float3* mu2, float3* temp, int basewidth, int baseheight, hipStream_t stream){
    //output is {norm1scale1, norm4scale1, norm1scale2, ...}
    std::vector<float3> result(2*6);

    int w = basewidth;
    int h = baseheight;
    int th_x;
    int bl_x;
    int index = 0;
    for (int scale = 0; scale < 6; scale++){
        th_x = std::min(1024, w*h);
        bl_x = (w*h-1)/th_x + 1;
        ssim_map_Kernel<<<dim3(bl_x), dim3(th_x), 2*sizeof(float3)*th_x, stream>>>(temp, s11, s22, s12, mu1, mu2, w, h);

        index += w*h;
        w = (w-1)/2+1;
        h = (h-1)/2+1;
    }

    return result;
} //to do: manage hipEvent and get it inside the function to ensure kernel is done before reducing blocks on CPU to then return the result
