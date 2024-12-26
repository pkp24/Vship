__global__ void horizontalBlur_Kernel(float3* src, float3* dst, int width, int height, float* gaussiankernel){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int size = width*height;
    if (x >= size) return;
    int current_line = x/width;

    float3 out; out.x = 0; out.y = 0; out.z = 0;
    for (int i = max(x-GAUSSIANSIZE, current_line*width); i <= min(x+GAUSSIANSIZE, (current_line+1)*width-1); i++){
        out += src[i]*gaussiankernel[GAUSSIANSIZE+i-x];
        //printf("%f at %d\n", gaussiankernel[GAUSSIANSIZE+i-x], GAUSSIANSIZE+i-x);
    }
    dst[x] = out;
}

__global__ void verticalBlur_Kernel(float3* src, float3* dst, int width, int height, float* gaussiankernel){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int size = width*height;
    if (x >= size) return;
    int current_line = x/width;
    int current_column = x%width;

    float3 out; out.x = 0; out.y = 0; out.z = 0;
    for (int i = max(current_line-GAUSSIANSIZE, 0); i <= min(current_line+GAUSSIANSIZE, height-1); i++){
        out += src[i * width + current_column]*gaussiankernel[GAUSSIANSIZE+i-current_line];
        //printf("%f\n", gaussiankernel[GAUSSIANSIZE+i-current_line]);
    }
    dst[x] = out;
    //printf("from %f, %f, %f to %f, %f, %f\n", src[x].x, src[x].y, src[x].z, dst[x].x, dst[x].y, dst[x].z);
}

void gaussianBlur(float3* src, float3* dst, float3* temp, int basewidth, int baseheight, float* gaussiankernel_d, hipStream_t stream){
    int w = basewidth;
    int h = baseheight;
    int th_x;
    int bl_x;
    int index = 0;
    for (int scale = 0; scale < 6; scale++){
        th_x = std::min(256, w*h);
        bl_x = (w*h-1)/th_x + 1;
        horizontalBlur_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(src+index, temp, w, h, gaussiankernel_d);
        GPU_CHECK(hipGetLastError());
        verticalBlur_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(temp, dst+index, w, h, gaussiankernel_d);
        GPU_CHECK(hipGetLastError());

        index += w*h;
        w = (w-1)/2+1;
        h = (h-1)/2+1;
    }
}