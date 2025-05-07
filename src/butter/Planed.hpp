namespace butter {

template <InputMemType T>
__launch_bounds__(256)
__global__ void strideEliminator_kernel(float* mem_d, const uint8_t* src, int stride, int width, int height){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    if (x >= width*height) return;
    int j = x%width;
    int i = x/width;
    mem_d[i*width+j] = convertPointer<T>(src, i, j, stride);
}

__launch_bounds__(256)
__global__ void strideAdder_kernel(const uint8_t* dst, float* mem_d, int stride, int width, int height){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    if (x >= width*height) return;
    int j = x%width;
    int i = x/width;
    ((float*)(dst + i*stride))[j] = mem_d[i*width+j];
}

template <InputMemType T>
void strideEliminator(float* mem_d, float* strided, int stride, int width, int height, hipStream_t stream){
    int wh = width*height;
    int th_x = std::min(256, wh);
    int bl_x = (wh-1)/th_x + 1;
    strideEliminator_kernel<T><<<dim3(bl_x), dim3(th_x), 0, stream>>>(mem_d, (const uint8_t*)strided, stride, width, height);
}
void strideAdder(float* mem_d, float* strided, int stride, int width, int height, hipStream_t stream){
    int wh = width*height;
    int th_x = std::min(256, wh);
    int bl_x = (wh-1)/th_x + 1;
    strideAdder_kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>((const uint8_t*)strided, mem_d, stride, width, height);
}

}