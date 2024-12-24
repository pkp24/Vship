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
    float3* sum4 = sharedmem+blockDim.x; //size sizeof(float3)*threadnum

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
        dst[blockIdx.x] = sum1[0]/(width*height);
        dst[gridDim.x + blockIdx.x] = sum4[0]/(width*height);
    }
}

std::vector<float3> ssim_map(float3* s11, float3* s22, float3* s12, float3* mu1, float3* mu2, float3* temp, int basewidth, int baseheight, hipEvent_t event_d, hipStream_t stream){
    //output is {norm1scale1, norm4scale1, norm1scale2, ...}
    std::vector<float3> result(2*6);
    for (int i = 0; i < 2*6; i++) {result[i].x = 0; result[i].y = 0; result[i].z = 0;}

    int w = basewidth;
    int h = baseheight;
    int th_x;
    int bl_x;
    int index = 0;
    std::vector<int> scaleoutdone(7);
    scaleoutdone[0] = 0;
    for (int scale = 0; scale < 6; scale++){
        th_x = std::min(1024, w*h);
        bl_x = (w*h-1)/th_x + 1;
        ssim_map_Kernel<<<dim3(bl_x), dim3(th_x), 2*sizeof(float3)*th_x, stream>>>(temp+scaleoutdone[scale], s11, s22, s12, mu1, mu2, w, h);
        GPU_CHECK(hipGetLastError());
        
        scaleoutdone[scale+1] = scaleoutdone[scale]+2*bl_x;
        index += w*h;
        w = (w-1)/2+1;
        h = (h-1)/2+1;
    }
    float3* hostback = (float3*)malloc(sizeof(float3)*scaleoutdone[6]);
    hipMemcpyDtoHAsync(hostback, (hipDeviceptr_t)temp, sizeof(float3)*scaleoutdone[6], stream);
    //the data as already been reduced by a factor of 512 which can now be reasonably retrieved from GPU

    hipEventRecord(event_d, stream); //place an event in the stream at the end of all our operations
    hipEventSynchronize(event_d);

    //let s reduce!
    for (int scale = 0; scale < 6; scale++){
        bl_x = (scaleoutdone[scale+1] - scaleoutdone[scale])/2;
        for (int i = 0; i < 2*bl_x; i++){
            if (i < bl_x){
                result[2*scale] += hostback[scaleoutdone[scale] + i];
            } else {
                result[2*scale+1] += hostback[scaleoutdone[scale] + i];
            }
        }
    }

    free(hostback);

    for (int i = 0; i < 6; i++){
        result[2*i+1].x = std::sqrt(std::sqrt(result[2*i+1].x));
        result[2*i+1].y = std::sqrt(std::sqrt(result[2*i+1].y));
        result[2*i+1].z = std::sqrt(std::sqrt(result[2*i+1].z));
    } //completing 4th norm

    return result;
}

__global__ void edgediff_map_Kernel(float3* dst, float3* im1, float3* im2, float3* mu1, float3* mu2, int width, int height){
    //dst must be of size 4*sizeof(float3)*blocknum at least
    //shared memory needed is 4*sizeof(float3)*threadnum at least
    const int x = threadIdx.x + blockIdx.x*blockDim.x;
    const int thx = threadIdx.x;
    const int threadnum = blockDim.x;
    
    extern __shared__ float3 sharedmem[];
    float3* suma1 = sharedmem; //size sizeof(float3)*threadnum
    float3* suma4 = sharedmem+blockDim.x; //size sizeof(float3)*threadnum
    float3* sumd1 = sharedmem+2*blockDim.x; //size sizeof(float3)*threadnum
    float3* sumd4 = sharedmem+3*blockDim.x; //size sizeof(float3)*threadnum
    

    float3 d1, d2;
    if (x < width*height){
        const float3 v1 = (abs(im2[x] - mu2[x])+1.) / (abs(im1[x] - mu1[x])+1.) - 1.;
        const float3 artifact = max(v1, 0);
        const float3 detailloss = max(v1*-1, 0);
        d1 = artifact; d2 = detailloss;
    } else {
        d1.x = 0;
        d1.y = 0;
        d1.z = 0;
        d2.x = 0;
        d2.y = 0;
        d2.z = 0;
    }

    suma1[thx] = d1;
    suma4[thx] = tothe4th(d1);
    sumd1[thx] = d2;
    sumd4[thx] = tothe4th(d2);
    __syncthreads();
    //now we need to do some pointer jumping to regroup every block sums;
    int next = 1;
    while (next < threadnum){
        if (thx + next < threadnum && (thx%(next*2) == 0)){
            suma1[thx] += suma1[thx+next];
            suma4[thx] += suma4[thx+next];
            sumd1[thx] += sumd1[thx+next];
            sumd4[thx] += sumd4[thx+next];
        }
        next *= 2;
        __syncthreads();
    }
    if (thx == 0){
        dst[blockIdx.x] = suma1[0]/(width*height);
        dst[gridDim.x + blockIdx.x] = suma4[0]/(width*height);
        dst[2*gridDim.x + blockIdx.x] = sumd1[0]/(width*height);
        dst[3*gridDim.x + blockIdx.x] = sumd4[0]/(width*height);
    }
}

std::vector<float3> edgediff_map(float3* im1, float3* im2, float3* mu1, float3* mu2, float3* temp, int basewidth, int baseheight, hipEvent_t event_d, hipStream_t stream){
    //output is {norma1scale1, norma4scale1, normad1scale1, normd4scale1, norm1scale2, ...}
    std::vector<float3> result(2*6*2);
    for (int i = 0; i < 2*6*2; i++) {result[i].x = 0; result[i].y = 0; result[i].z = 0;}

    int w = basewidth;
    int h = baseheight;
    int th_x;
    int bl_x;
    int index = 0;
    std::vector<int> scaleoutdone(7);
    scaleoutdone[0] = 0;
    for (int scale = 0; scale < 6; scale++){
        th_x = std::min(1024, w*h);
        bl_x = (w*h-1)/th_x + 1;
        edgediff_map_Kernel<<<dim3(bl_x), dim3(th_x), 4*sizeof(float3)*th_x, stream>>>(temp+scaleoutdone[scale], im1, im2, mu1, mu2, w, h);
        GPU_CHECK(hipGetLastError());
        
        scaleoutdone[scale+1] = scaleoutdone[scale]+4*bl_x;
        index += w*h;
        w = (w-1)/2+1;
        h = (h-1)/2+1;
    }
    float3* hostback = (float3*)malloc(sizeof(float3)*scaleoutdone[6]);
    hipMemcpyDtoHAsync(hostback, (hipDeviceptr_t)temp, sizeof(float3)*scaleoutdone[6], stream);
    //the data as already been reduced by a factor of 512 which can now be reasonably retrieved from GPU

    hipEventRecord(event_d, stream); //place an event in the stream at the end of all our operations
    hipEventSynchronize(event_d);

    //let s reduce!
    for (int scale = 0; scale < 6; scale++){
        bl_x = (scaleoutdone[scale+1] - scaleoutdone[scale])/4;
        for (int i = 0; i < 4*bl_x; i++){
            if (i < bl_x){
                result[4*scale] += hostback[scaleoutdone[scale] + i];
            } else if (i < 2*bl_x) {
                result[4*scale+1] += hostback[scaleoutdone[scale] + i];
            } else if (i < 3*bl_x) {
                result[4*scale+2] += hostback[scaleoutdone[scale] + i];
            } else {
                result[4*scale+3] += hostback[scaleoutdone[scale] + i];
            }
        }
    }

    free(hostback);

    for (int i = 0; i < 12; i++){
        result[2*i+1].x = std::sqrt(std::sqrt(result[2*i+1].x));
        result[2*i+1].y = std::sqrt(std::sqrt(result[2*i+1].y));
        result[2*i+1].z = std::sqrt(std::sqrt(result[2*i+1].z));
    } //completing 4th norm

    return result;
}

//TO NOT USE, this version uses too much shared cache so it cannot work
__global__ void allscore_map_Kernel(float3* dst, float3* im1, float3* im2, float3* mu1, float3* mu2, float3* s11, float3* s22, float3* s12, int width, int height){
    //dst must be of size 6*sizeof(float3)*blocknum at least
    //shared memory needed is 6*sizeof(float3)*threadnum at least
    const int x = threadIdx.x + blockIdx.x*blockDim.x;
    const int thx = threadIdx.x;
    const int threadnum = blockDim.x;
    
    extern __shared__ float3 sharedmem[];
    float3* sumssim1 = sharedmem; //size sizeof(float3)*threadnum
    float3* sumssim4 = sharedmem+blockDim.x; //size sizeof(float3)*threadnum
    float3* suma1 = sharedmem+2*blockDim.x; //size sizeof(float3)*threadnum
    float3* suma4 = sharedmem+3*blockDim.x; //size sizeof(float3)*threadnum
    float3* sumd1 = sharedmem+4*blockDim.x; //size sizeof(float3)*threadnum
    float3* sumd4 = sharedmem+5*blockDim.x; //size sizeof(float3)*threadnum
    

    float3 d0, d1, d2;
    if (x < width*height){
        //ssim
        const float3 m1 = mu1[x]; const float3 m2 = mu2[x];
        const float3 m11 = m1*m1;
        const float3 m22 = m2*m2;
        const float3 m12 = m1*m2;
        const float3 m_diff = m1-m2;
        const float3 num_m = fmaf(m_diff, m_diff*-1., 1.0);
        const float3 num_s = fmaf(s12[x] - m12, 2.0, 0.0009);
        const float3 denom_s = (s11[x] - m11) + (s22[x] - m22) + 0.0009;
        d0 = max(1.0 - ((num_m * num_s)/denom_s), 0.);

        //edge diff
        const float3 v1 = (abs(im2[x] - mu2[x])+1.) / (abs(im1[x] - mu1[x])+1.) - 1.;
        const float3 artifact = max(v1, 0);
        const float3 detailloss = max(v1*-1, 0);
        d1 = artifact; d2 = detailloss;
    } else {
        d0.x = 0;
        d0.y = 0;
        d0.z = 0;
        d1.x = 0;
        d1.y = 0;
        d1.z = 0;
        d2.x = 0;
        d2.y = 0;
        d2.z = 0;
    }

    sumssim1[thx] = d0;
    sumssim4[thx] = tothe4th(d0);
    suma1[thx] = d1;
    suma4[thx] = tothe4th(d1);
    sumd1[thx] = d2;
    sumd4[thx] = tothe4th(d2);
    __syncthreads();
    //now we need to do some pointer jumping to regroup every block sums;
    int next = 1;
    while (next < threadnum){
        if (thx + next < threadnum && (thx%(next*2) == 0)){
            sumssim1[thx] += sumssim1[thx+next];
            sumssim4[thx] += sumssim4[thx+next];
            suma1[thx] += suma1[thx+next];
            suma4[thx] += suma4[thx+next];
            sumd1[thx] += sumd1[thx+next];
            sumd4[thx] += sumd4[thx+next];
        }
        next *= 2;
        __syncthreads();
    }
    if (thx == 0){
        dst[blockIdx.x] = sumssim1[0]/(width*height);
        dst[gridDim.x + blockIdx.x] = sumssim4[0]/(width*height);
        dst[2*gridDim.x + blockIdx.x] = suma1[0]/(width*height);
        dst[3*gridDim.x + blockIdx.x] = suma4[0]/(width*height);
        dst[4*gridDim.x + blockIdx.x] = sumd1[0]/(width*height);
        dst[5*gridDim.x + blockIdx.x] = sumd4[0]/(width*height);
    }
}

std::vector<float3> allscore_map(float3* im1, float3* im2, float3* mu1, float3* mu2, float3* s11, float3* s22, float3* s12, float3* temp, int basewidth, int baseheight, int maxshared, hipEvent_t event_d, hipStream_t stream){
    //output is {normssim1scale1, normssim4scale1, norma1scale1, norma4scale1, normad1scale1, normd4scale1, norm1scale2, ...}
    std::vector<float3> result(2*6*3);
    for (int i = 0; i < 2*6*3; i++) {result[i].x = 0; result[i].y = 0; result[i].z = 0;}

    int w = basewidth;
    int h = baseheight;
    int th_x;
    int bl_x;
    int index = 0;
    std::vector<int> scaleoutdone(7);
    scaleoutdone[0] = 0;
    for (int scale = 0; scale < 6; scale++){
        th_x = std::min((const int)(maxshared/(6*sizeof(float3)))/32*32, std::min(1024, w*h));
        bl_x = (w*h-1)/th_x + 1;
        allscore_map_Kernel<<<dim3(bl_x), dim3(th_x), 6*sizeof(float3)*th_x, stream>>>(temp+scaleoutdone[scale], im1, im2, mu1, mu2, s11, s22, s12, w, h);
        //printf("I got %s with %d\n", hipGetErrorString(hipGetLastError()), 6*sizeof(float3)*th_x);
        GPU_CHECK(hipGetLastError());

        scaleoutdone[scale+1] = scaleoutdone[scale]+6*bl_x;
        index += w*h;
        w = (w-1)/2+1;
        h = (h-1)/2+1;
    }
    float3* hostback = (float3*)malloc(sizeof(float3)*scaleoutdone[6]);
    hipMemcpyDtoHAsync(hostback, (hipDeviceptr_t)temp, sizeof(float3)*scaleoutdone[6], stream);
    //the data as already been reduced by a factor of 512 which can now be reasonably retrieved from GPU

    hipEventRecord(event_d, stream); //place an event in the stream at the end of all our operations
    hipEventSynchronize(event_d);

    //let s reduce!
    for (int scale = 0; scale < 6; scale++){
        bl_x = (scaleoutdone[scale+1] - scaleoutdone[scale])/6;
        for (int i = 0; i < 6*bl_x; i++){
            if (i < bl_x){
                result[6*scale] += hostback[scaleoutdone[scale] + i];
            } else if (i < 2*bl_x) {
                result[6*scale+1] += hostback[scaleoutdone[scale] + i];
            } else if (i < 3*bl_x) {
                result[6*scale+2] += hostback[scaleoutdone[scale] + i];
            } else if (i < 4*bl_x) {
                result[6*scale+3] += hostback[scaleoutdone[scale] + i];
            } else if (i < 5*bl_x) {
                result[6*scale+4] += hostback[scaleoutdone[scale] + i];
            } else {
                result[6*scale+5] += hostback[scaleoutdone[scale] + i];
            }
        }
    }

    free(hostback);

    for (int i = 0; i < 18; i++){
        result[2*i+1].x = std::sqrt(std::sqrt(result[2*i+1].x));
        result[2*i+1].y = std::sqrt(std::sqrt(result[2*i+1].y));
        result[2*i+1].z = std::sqrt(std::sqrt(result[2*i+1].z));
    } //completing 4th norm

    return result;
}