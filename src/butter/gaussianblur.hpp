namespace butter{

__global__ void loadGaussianKernel(float* gaussiankernel, int gaussiansize, float sigma){
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x > 2*gaussiansize) return;
    gaussiankernel[x] = expf(-(gaussiansize-x)*(gaussiansize-x)/(2*sigma*sigma))/(sqrt(TAU*sigma*sigma));
}

//alows to precompute and reuse easily the same kernel array in gpu
class GaussianHandle{
    float* gaussiankernel_d;
    int* indices;
public:
    void init(){
        hipError_t erralloc = hipMalloc(&gaussiankernel_d, sizeof(float)*1024);
        if (erralloc != hipSuccess){
            throw VshipError(OutOfVRAM, __FILE__, __LINE__);
        }
        indices = (int*)malloc(sizeof(int)*6);
        if (indices == NULL){
            throw VshipError(OutOfRAM, __FILE__, __LINE__);
        }
        indices[0] = 0;
        const float sigmas[5] = {1.2f, 1.56416327805f, 2.7f, 3.22489901262f, 7.5593339443f};

        for (int i = 0; i < 5; i++){
            const float sigma = sigmas[i];
            const int windowsize = (int)(sigma*5);
            indices[i+1] = indices[i]+2*windowsize+1;
            const int th_x = std::min(windowsize*2+1, 256);
            const int bl_x = (windowsize*2)/th_x+1;
            loadGaussianKernel<<<dim3(bl_x), dim3(th_x), 0, 0>>>(gaussiankernel_d+indices[i], windowsize, sigma);
        }
        hipDeviceSynchronize();
    }
    float* get(int i){
        return gaussiankernel_d+indices[i];
    }
    int getWindow(int i){
        return ((indices[i+1]-indices[i])-1)/2;
    }
    void destroy(){
        hipFree(gaussiankernel_d);
        free(indices);
    }
};

//transpose the result at the end
__launch_bounds__(256)
__global__ void horizontalBlur_Kernel(float* dst, float* src, int w, int h, float* gaussiankernel, int gaussiansize){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int size = w*h;
    if (x >= size) return;

    int current_line = x/w;

    float weight = 0.0f;

    float out = 0.0f;
    for (int i = max(x-gaussiansize, current_line*w); i <= min(x+gaussiansize, (current_line+1)*w-1); i++){
        const float gauss = gaussiankernel[gaussiansize+i-x];
        out += src[i]*gauss;
        weight += gauss;
    }
    dst[x] = out/weight;
}

__launch_bounds__(256)
__global__ void verticalBlur_Kernel(float* dst, float* src, int w, int h, float* gaussiankernel, int gaussiansize){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int size = w*h;
    if (x >= size) return;

    int current_line = x/w;
    int current_column = x%w;

    float weight = 0.0f;

    float out = 0.0f;
    for (int i = max(current_line-gaussiansize, 0); i <= min(current_line+gaussiansize, h-1); i++){
        out += src[i * w + current_column]*gaussiankernel[gaussiansize+i-current_line];
        weight += gaussiankernel[gaussiansize+i-current_line];
        //if (threadIdx.x == 0) printf("%f at %d\n", gaussiankernel[gaussiansize+i-current_line], gaussiansize+i-current_line);
    }
    dst[x] = out/weight;
}

//all gaussian sigmas:
/*
1.2
1.56416327805
2.7
3.22489901262
7.15593339443
*/

}