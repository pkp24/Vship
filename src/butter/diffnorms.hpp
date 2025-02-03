namespace butter{

//first norm usage should be the desired norm, then it should be 1 for further reduction
__global__ void sumreduce(float* dst, float* src, int width, int norm){
    //dst must be of size sizeof(float)*blocknum at least
    //shared memory needed is sizeof(float)*threadnum at least
    const int x = threadIdx.x + blockIdx.x*blockDim.x;
    const int thx = threadIdx.x;
    const int threadnum = blockDim.x;
    
    __shared__ float sharedmem[1024];

    if (x >= width){
        sharedmem[thx] = 0;
    } else {
        sharedmem[thx] = powf(abs(src[x]), norm);
    }
    __syncthreads();
    //now we need to do some pointer jumping to regroup every block sums;
    int next = 1;
    while (next < threadnum){
        if (thx + next < threadnum && (thx%(next*2) == 0)){
            sharedmem[thx] += sharedmem[thx+next];
        }
        next *= 2;
        __syncthreads();
    }
    if (thx == 0){
        dst[blockIdx.x] = sharedmem[0];
    }
}

float diffmapnorm(float* diffmap, float* temp, float* temp2, int width, int norm, hipEvent_t event_d, hipStream_t stream){
    int basenorm = norm;
    int basewidth = width;
    float* src = diffmap;
    float* temps[2] = {temp, temp2};
    int oscillate = 0;
    int th_x, bl_x;
    while (width > 1024){
        th_x = 1024;
        bl_x = (width - 1)/th_x + 1;
        sumreduce<<<dim3(bl_x), dim3(th_x), 0, stream>>>(temps[oscillate], src, width, norm);
        src = temps[oscillate];
        oscillate ^= 1;
        norm = 1;
        width = bl_x;
    }
    float* back_to_cpu = (float*)malloc(sizeof(float)*width);
    float res = 0;
    hipMemcpyDtoHAsync(back_to_cpu, src, sizeof(float)*width, stream);

    hipEventRecord(event_d, stream); //place an event in the stream at the end of all our operations
    hipEventSynchronize(event_d); //when the event is complete, we know our gpu result is ready!

    for (int i = 0; i < width; i++){
        res += std::powf(std::abs(back_to_cpu[i]), norm);
    }

    free(back_to_cpu);
    res = std::powf(res/basewidth, 1./(float)basenorm);
    return res;
}

__global__ void maxreduce(float* dst, float* src, int width){
    //dst must be of size sizeof(float)*blocknum at least
    //shared memory needed is sizeof(float)*threadnum at least
    const int x = threadIdx.x + blockIdx.x*blockDim.x;
    const int thx = threadIdx.x;
    const int threadnum = blockDim.x;
    
    __shared__ float sharedmem[1024];

    if (x >= width){
        sharedmem[thx] = 0;
    } else {
        sharedmem[thx] = abs(src[x]);
    }
    __syncthreads();
    //now we need to do some pointer jumping to regroup every block sums;
    int next = 1;
    while (next < threadnum){
        if (thx + next < threadnum && (thx%(next*2) == 0)){
            sharedmem[thx] = max(sharedmem[thx], sharedmem[thx+next]);
        }
        next *= 2;
        __syncthreads();
    }
    if (thx == 0){
        dst[blockIdx.x] = sharedmem[0];
    }
}

float diffmapnorminf(float* diffmap, float* temp, float* temp2, int width, hipEvent_t event_d, hipStream_t stream){
    float* src = diffmap;
    float* temps[2] = {temp, temp2};
    int oscillate = 0;
    int th_x, bl_x;
    while (width > 1024){
        th_x = 1024;
        bl_x = (width - 1)/th_x + 1;
        maxreduce<<<dim3(bl_x), dim3(th_x), 0, stream>>>(temps[oscillate], src, width);
        src = temps[oscillate];
        oscillate ^= 1;
        width = bl_x;
    }
    float* back_to_cpu = (float*)malloc(sizeof(float)*width);
    float res = 0;
    hipMemcpyDtoHAsync(back_to_cpu, src, sizeof(float)*width, stream);

    hipEventRecord(event_d, stream); //place an event in the stream at the end of all our operations
    hipEventSynchronize(event_d); //when the event is complete, we know our gpu result is ready!

    for (int i = 0; i < width; i++){
        res = max(res, std::abs(back_to_cpu[i]));
    }

    free(back_to_cpu);
    return res;
}

}