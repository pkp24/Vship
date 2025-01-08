namespace butter{

__global__ void loadGaussianKernel(float* gaussiankernel, int gaussiansize, float sigma){
    int x = threadIdx.x;
    if (x > 2*gaussiansize) return;
    gaussiankernel[x] = exp(-(gaussiansize-x)*(gaussiansize-x)/(2*sigma*sigma))/(sqrt(2*PI*sigma*sigma));
}

//transpose the result at the end
__launch_bounds__(256)
__global__ void horizontalBlur_Kernel(float* src, float* dst, int w, int h, float border_ratio, float weight_no_border, float* gaussiankernel, int gaussiansize){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int size = w*h;
    if (x >= size) return;

    int current_line = x/w;
    int current_column = x%w;

    float weight = 0;

    float out = 0;
    for (int i = max(x-gaussiansize, current_line*w); i <= min(x+gaussiansize, (current_line+1)*w-1); i++){
        out += src[i]*gaussiankernel[gaussiansize+i-x];
        weight += gaussiankernel[gaussiansize+i-x];
        //printf("%f at %d\n", gaussiankernel[GAUSSIANSIZE+i-x], GAUSSIANSIZE+i-x);
    }
    dst[current_column*h + current_line] = out*(1.f/((1.f-border_ratio)*weight + border_ratio*weight_no_border));
}

}