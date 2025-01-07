namespace butter{

//transpose the result at the end
__launch_bounds__(256)
__global__ void butterHorizontalBlur_Kernel(float3* src, float3* dst, int w, int h, float border_ratio, float weight_no_border, float* gaussiankernel, int gaussiansize){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int size = w*h;
    if (x >= size) return;

    int current_line = x/w;
    int current_column = x%w;

    float weight = 0;

    float3 out; out.x = 0; out.y = 0; out.z = 0;
    for (int i = max(x-gaussiansize, current_line*w); i <= min(x+gaussiansize, (current_line+1)*w-1); i++){
        out += src[i]*gaussiankernel[gaussiansize+i-x];
        weight += gaussiankernel[gaussiansize+i-x];
        //printf("%f at %d\n", gaussiankernel[GAUSSIANSIZE+i-x], GAUSSIANSIZE+i-x);
    }
    dst[current_column*h + current_line] = out*(1.f/((1.f-border_ratio)*weight + border_ratio*weight_no_border));
}

void butterGaussianBlur(float3* src, float3* dst, float3* temp, int w, int h, float sigma, float border_ratio, float* gaussiankernel_mem, float* gaussiankernel_dmem, hipStream_t stream){
    int th_x;
    int bl_x;
    int wh = w*h;

    int gaussiansize = (std::min(BUTTERMAXGAUSSIANSIZE, (int)(sigma*4))-1)/2;
    float weight_no_border = 0;
    for (int i = 0; i < 2*gaussiansize+1; i++){
        gaussiankernel_mem[i] = std::exp(-(gaussiansize-i)*(gaussiansize-i)/(2*sigma*sigma))/(std::sqrt(2*PI*sigma*sigma));
        weight_no_border += gaussiankernel_mem[i];
    }
    hipMemcpyHtoDAsync((hipDeviceptr_t)gaussiankernel_dmem, (void*)gaussiankernel_mem, sizeof(float)*gaussiansize, stream);

    th_x = std::min(256, wh);
    bl_x = (wh-1)/th_x + 1;
    butterHorizontalBlur_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(src, temp, w, h, border_ratio, weight_no_border, gaussiankernel_dmem, gaussiansize);
    GPU_CHECK(hipGetLastError());
    butterHorizontalBlur_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(temp, dst, h, w, border_ratio, weight_no_border, gaussiankernel_dmem, gaussiansize);
    GPU_CHECK(hipGetLastError());
}

}