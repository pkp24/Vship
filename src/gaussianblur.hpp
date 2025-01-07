__launch_bounds__(256)
__global__ void horizontalBlur_Kernel(float3* src, float3* dst, int width, int height, float* gaussiankernel){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int size = width*height;
    int w = width; int h = height;
    for (int i = 0; i < 5; i++){
        if (x < size) break; //we have the right scale
        w = (w-1)/2+1; h = (h-1)/2+1;
        src += size; dst += size; x -= size;
        size = h*w;
    }
    if (x >= size) return;

    int current_line = x/w;
    int current_column = x%w;

    float3 out; out.x = 0; out.y = 0; out.z = 0;
    for (int i = max(x-GAUSSIANSIZE, current_line*w); i <= min(x+GAUSSIANSIZE, (current_line+1)*w-1); i++){
        out += src[i]*gaussiankernel[GAUSSIANSIZE+i-x];
        //printf("%f at %d\n", gaussiankernel[GAUSSIANSIZE+i-x], GAUSSIANSIZE+i-x);
    }
    dst[current_column*h + current_line] = out;
}

void gaussianBlur(float3* src, float3* dst, float3* temp, int totalscalesize, int basewidth, int baseheight, float* gaussiankernel_d, hipStream_t stream){
    int w = basewidth;
    int h = baseheight;
    int th_x;
    int bl_x;

    th_x = std::min(256, totalscalesize);
    bl_x = (totalscalesize-1)/th_x + 1;
    horizontalBlur_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(src, temp, w, h, gaussiankernel_d);
    GPU_CHECK(hipGetLastError());
    horizontalBlur_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(temp, dst, h, w, gaussiankernel_d);
    GPU_CHECK(hipGetLastError());
}