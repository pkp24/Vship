namespace butter{

__global__ void loadGaussianKernel(float* gaussiankernel, int gaussiansize, double sigma){
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
        const double sigmas[5] = {1.2, 1.56416327805, 2.7, 3.22489901262, 7.15593339443};

        for (int i = 0; i < 5; i++){
            const double sigma = sigmas[i];
            const int windowsize = std::max((int)(sigma*3), (int)8); //we use 8 to trigger the special gaussian blur when possible
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

//best is to use 8x32 rectangle 
__launch_bounds__(256)
__global__ void verticalBlur_Kernel(float* dst, float* src, int w, int h, float* gaussiankernel, int gaussiansize){
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    if (x >= w) return;
    if (y >= h) return;

    float weight = 0.0f;
    float out = 0.0f;
    for (int i = max(y-gaussiansize, 0); i <= min(y+gaussiansize, h-1); i++){
        //if (x == 423 && y == 323) printf("%f at %d for %f\n", gaussiankernel[gaussiansize+i-y], gaussiansize+i-y, src[i*w+x]);
        out += src[i * w + x]*gaussiankernel[gaussiansize+i-y];
        weight += gaussiankernel[gaussiansize+i-y];
    }
    dst[y*w+x] = out/weight;
}

//blur for windowsize == 8
//launch in 16*16
//manage a block of 32*32
__launch_bounds__(256)
__global__ void GaussianBlur_Kernel(float* src, float* dst, int width, int height, float* gaussiankernel){
    int originalBl_X = blockIdx.x;
    //let's determine which scale our block is in and adjust our input parameters accordingly

    const int blockwidth = (width-1)/32+1;

    const int x = threadIdx.x + 32*(originalBl_X%blockwidth);
    const int y = threadIdx.y + 32*(originalBl_X/blockwidth);
    const int thx = threadIdx.x;
    const int thy = threadIdx.y;

    __shared__ float tampon[48*48]; //we import into tampon, compute onto tampon and then put into dst
    //tampon has 8 of border on each side with no thread
    const int tampon_base_x = x - thx - 8;
    const int tampon_base_y = y - thy - 8;

    //fill tampon
    #pragma unroll
    for (int region_x = 0; region_x < 48; region_x += 16){
        #pragma unroll
        for (int region_y = 0; region_y < 48; region_y += 16){
            tampon[(thy+region_y)*48+thx+region_x] = (tampon_base_x + thx +region_x >= 0 && tampon_base_x + thx +region_x < width && tampon_base_y + thy + region_y >= 0 && tampon_base_y + thy + region_y < height) ? src[(tampon_base_y+thy+region_y)*width + tampon_base_x+thx+region_x] : 0.f;
        }
    }
    __syncthreads();

    //horizontalBlur on tampon restraint into rectangle [8 - 40][0 - 48] -> 6 pass per thread
    float tot[2] = {0.f, 0.f};
    float out[2][3] = {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}};

    #pragma unroll
    for (int region_x = 0; region_x < 2; region_x++){

        //border handling precompute
        for (int i = 0; i < 17; i++){
            if (tampon_base_x+thx+i+region_x*16 >= 0 && tampon_base_x+thx+i+region_x*16 < width) tot[region_x] += gaussiankernel[i];
        }

        #pragma unroll
        for (int region_y = 0; region_y < 3; region_y++){
            for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
                out[region_x][region_y] += tampon[(thy+region_y*16)*48 + thx+i+region_x*16]*gaussiankernel[i];
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for (int region_x = 0; region_x < 2; region_x++){
        #pragma unroll
        for (int region_y = 0; region_y < 3; region_y++){
            tampon[(thy+region_y*16)*48 + thx+8+region_x*16] = out[region_x][region_y]/tot[region_x];
        }
    }
    __syncthreads();

    //verticalBlur on tampon restraint into rectangle [8 - 40][8 - 40] -> 4 pass per thread
    #pragma unroll
    for (int region_y = 0; region_y < 2; region_y++){
        
        tot[region_y] = 0;
        //border handling precompute
        for (int i = 0; i < 17; i++){
            if (tampon_base_y+thy+i+region_y*16 >= 0 && tampon_base_y+thy+i+region_y*16 < height) tot[region_y] += gaussiankernel[i];
        }

        #pragma unroll
        for (int region_x = 0; region_x < 2; region_x++){
            out[region_y][region_x] = 0;
            for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
                out[region_y][region_x] += tampon[(thy+i+region_y*16)*48 + thx+8+region_x*16]*gaussiankernel[i];
            }
        }
    }

    __syncthreads();
    #pragma unroll
    for (int region_y = 0; region_y < 2; region_y++){
        #pragma unroll
        for (int region_x = 0; region_x < 2; region_x++){
            tampon[(thy+8+region_y*16)*48 + thx+8+region_x*16] = out[region_y][region_x]/tot[region_y];
        }
    }
    __syncthreads();

    #pragma unroll
    for (int region_x = 0; region_x < 2; region_x++){
        #pragma unroll
        for (int region_y = 0; region_y < 2; region_y++){
            if (tampon_base_x + thx +8+region_x*16 >= 0 && tampon_base_x + thx +8+region_x*16 < width && tampon_base_y + thy +8+region_y*16 >= 0 && tampon_base_y + thy +8+region_y*16 < height) dst[(tampon_base_y+thy+8+region_y*16)*width + tampon_base_x+thx+8+region_x*16] = tampon[(thy+8+region_y*16)*48+thx+8+region_x*16];
        }
    }
}

void blur(float* mem_d, float* temp, int width, int height, GaussianHandle& gaussianHandle, int i, hipStream_t stream){
    const int gaussiansize = gaussianHandle.getWindow(i);
    float* gaussianKernel = gaussianHandle.get(i);

    int wh = width*height;
    int th_x = std::min(256, wh);
    int bl_x = (wh-1)/th_x + 1;

    int verticalth_x = 8;
    int verticalth_y = 32;
    int verticalbl_x = (width-1)/verticalth_x+1;
    int verticalbl_y = (height-1)/verticalth_y+1;

    horizontalBlur_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(temp, mem_d, width, height, gaussianKernel, gaussiansize);
    verticalBlur_Kernel<<<dim3(verticalbl_x, verticalbl_y), dim3(verticalth_x, verticalth_y), 0, stream>>>(mem_d, temp, width, height, gaussianKernel, gaussiansize);
}

void blur(float* dst, float* mem_d, float* temp, int width, int height, GaussianHandle& gaussianHandle, int i, hipStream_t stream){
    const int gaussiansize = gaussianHandle.getWindow(i);
    float* gaussianKernel = gaussianHandle.get(i);

    int wh = width*height;

    if (gaussiansize == 8){ //special gaussian blur! It doesnt even use temp
        int th_x = 16;
        int th_y = 16;
        int bl_x = (width-1)/(2*th_x)+1;
        int bl_y = (height-1)/(2*th_y)+1;
        GaussianBlur_Kernel<<<dim3(bl_x*bl_y), dim3(th_x, th_y), 0, stream>>>(mem_d, dst, width, height, gaussianKernel);
    } else {
        int th_x = std::min(256, wh);
        int bl_x = (wh-1)/th_x + 1;

        int verticalth_x = 8;
        int verticalth_y = 32;
        int verticalbl_x = (width-1)/verticalth_x+1;
        int verticalbl_y = (height-1)/verticalth_y+1;
        horizontalBlur_Kernel<<<dim3(bl_x), dim3(th_x), 0, stream>>>(temp, mem_d, width, height, gaussianKernel, gaussiansize);
        verticalBlur_Kernel<<<dim3(verticalbl_x, verticalbl_y), dim3(verticalth_x, verticalth_y), 0, stream>>>(dst, temp, width, height, gaussianKernel, gaussiansize);
    }
}
void blurDstNoTemp(float* dst, float* mem_d, int width, int height, GaussianHandle& gaussianHandle, int i, hipStream_t stream){
    const int gaussiansize = gaussianHandle.getWindow(i);
    float* gaussianKernel = gaussianHandle.get(i);

    assert(gaussiansize == 8);
    //special gaussian blur! It doesnt even use temp
    int th_x = 16;
    int th_y = 16;
    int bl_x = (width-1)/(2*th_x)+1;
    int bl_y = (height-1)/(2*th_y)+1;
    GaussianBlur_Kernel<<<dim3(bl_x*bl_y), dim3(th_x, th_y), 0, stream>>>(mem_d, dst, width, height, gaussianKernel);
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