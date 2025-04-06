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
    tampon[thy*48+thx] = (tampon_base_x + thx >= 0 && tampon_base_x + thx < width && tampon_base_y + thy >= 0 && tampon_base_y + thy < height) ? src[(tampon_base_y+thy)*width + tampon_base_x+thx] : 0.f;
    tampon[(thy+16)*48+thx] = (tampon_base_x + thx >= 0 && tampon_base_x + thx < width && tampon_base_y + thy + 16 >= 0 && tampon_base_y + thy + 16 < height) ? src[(tampon_base_y+thy+16)*width + tampon_base_x+thx] : 0.f;
    tampon[(thy+32)*48+thx] = (tampon_base_x + thx >= 0 && tampon_base_x + thx < width && tampon_base_y + thy + 32 >= 0 && tampon_base_y + thy + 32 < height) ? src[(tampon_base_y+thy+32)*width + tampon_base_x+thx] : 0.f;
    tampon[thy*48+thx+16] = (tampon_base_x + thx +16 >= 0 && tampon_base_x + thx +16 < width && tampon_base_y + thy >= 0 && tampon_base_y + thy < height) ? src[(tampon_base_y+thy)*width + tampon_base_x+thx+16] : 0.f;
    tampon[(thy+16)*48+thx+16] = (tampon_base_x + thx +16 >= 0 && tampon_base_x + thx +16 < width && tampon_base_y + thy + 16 >= 0 && tampon_base_y + thy + 16 < height) ? src[(tampon_base_y+thy+16)*width + tampon_base_x+thx+16] : 0.f;
    tampon[(thy+32)*48+thx+16] = (tampon_base_x + thx +16 >= 0 && tampon_base_x + thx +16 < width && tampon_base_y + thy + 32 >= 0 && tampon_base_y + thy + 32 < height) ? src[(tampon_base_y+thy+32)*width + tampon_base_x+thx+16] : 0.f;
    tampon[thy*48+thx+32] = (tampon_base_x + thx +32 >= 0 && tampon_base_x + thx +32 < width && tampon_base_y + thy >= 0 && tampon_base_y + thy < height) ? src[(tampon_base_y+thy)*width + tampon_base_x+thx+32] : 0.f;
    tampon[(thy+16)*48+thx+32] = (tampon_base_x + thx +32 >= 0 && tampon_base_x + thx +32 < width && tampon_base_y + thy + 16 >= 0 && tampon_base_y + thy + 16 < height) ? src[(tampon_base_y+thy+16)*width + tampon_base_x+thx+32] : 0.f;
    tampon[(thy+32)*48+thx+32] = (tampon_base_x + thx +32 >= 0 && tampon_base_x + thx +32 < width && tampon_base_y + thy + 32 >= 0 && tampon_base_y + thy + 32 < height) ? src[(tampon_base_y+thy+32)*width + tampon_base_x+thx+32] : 0.f;
    
    __syncthreads();

    //horizontalBlur on tampon restraint into rectangle [8 - 40][0 - 48] -> 6 pass per thread

    //1st pass in [8 - 24][0 - 16]
    float tot = 0.f;
    float tot2 = 0.f;
    float out = 0.f;
    float out2 = 0.f;
    float out3 = 0.f;
    float out4 = 0.f;
    float out5 = 0.f;
    float out6 = 0.f;
    for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
        out += tampon[thy*48 + thx+i]*gaussiankernel[i];

        //border handling precompute
        if (tampon_base_x+thx+i >= 0 && tampon_base_x+thx+i < width) tot += gaussiankernel[i];
    }

    //2nd pass in [8 - 24][16 - 32]
    for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
        out2 += tampon[(thy+16)*48 + thx+i]*gaussiankernel[i];
    }

    //3rd pass in [8 - 24][32 - 48]
    for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
        out3 += tampon[(thy+32)*48 + thx+i]*gaussiankernel[i];
    }

    //second column
    for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
        out4 += tampon[thy*48 + thx+i+16]*gaussiankernel[i];

        //border handling precompute
        if (tampon_base_x+thx+i+16 >= 0 && tampon_base_x+thx+i+16 < width) tot2 += gaussiankernel[i];
    }

    //2nd pass in [24 - 40][16 - 32]
    for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
        out5 += tampon[(thy+16)*48 + thx+i+16]*gaussiankernel[i];
    }

    //3rd pass in [24 - 40][32 - 48]
    for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
        out6 += tampon[(thy+32)*48 + thx+i+16]*gaussiankernel[i];
    }

    __syncthreads();
    tampon[thy*48 + thx+8] = out/tot;
    tampon[(thy+16)*48 + thx+8] = out2/tot;
    tampon[(thy+32)*48 + thx+8] = out3/tot;

    tampon[thy*48 + thx+8+16] = out4/tot2;
    tampon[(thy+16)*48 + thx+8+16] = out5/tot2;
    tampon[(thy+32)*48 + thx+8+16] = out6/tot2;
    __syncthreads();

    //verticalBlur on tampon restraint into rectangle [8 - 40][8 - 40] -> 4 pass per thread
    tot = 0.f;
    tot2 = 0.f;
    out = 0.f;
    out2 = 0.f;
    out3 = 0.f;
    out4 = 0.f;

    for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
        out += tampon[(thy+i)*48 + thx+8]*gaussiankernel[i];

        //border handling precompute
        if (tampon_base_y+thy+i >= 0 && tampon_base_y+thy+i < height) tot += gaussiankernel[i];
    }

    //2nd pass [24 - 40][8 - 24]
    for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
        out2 += tampon[(thy+i)*48 + thx+8+16]*gaussiankernel[i];
    }

    //second row
    for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
        out3 += tampon[(thy+i+16)*48 + thx+8]*gaussiankernel[i];

        //border handling precompute
        if (tampon_base_y+thy+i+16 >= 0 && tampon_base_y+thy+i+16 < height) tot2 += gaussiankernel[i];
    }

    //2nd pass [24 - 40][24 - 40]
    for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
        out4 += tampon[(thy+i+16)*48 + thx+8+16]*gaussiankernel[i];
    }

    __syncthreads();
    tampon[(thy+8)*48 + thx+8] = out/tot;
    tampon[(thy+8)*48 + thx+8+16] = out2/tot;
    tampon[(thy+8+16)*48 + thx+8] = out3/tot2;
    tampon[(thy+8+16)*48 + thx+8+16] = out4/tot2;
    __syncthreads();

    //tampon [8 - 24][8 - 24] -> dst
    if (tampon_base_x + thx +8 >= 0 && tampon_base_x + thx +8 < width && tampon_base_y + thy +8 >= 0 && tampon_base_y + thy +8 < height) dst[(tampon_base_y+thy+8)*width + tampon_base_x+thx+8] = tampon[(thy+8)*48+thx+8];
    //tampon [24 - 40][8 - 24] -> dst
    if (tampon_base_x + thx +8+16 >= 0 && tampon_base_x + thx +8+16 < width && tampon_base_y + thy +8 >= 0 && tampon_base_y + thy +8 < height) dst[(tampon_base_y+thy+8)*width + tampon_base_x+thx+8+16] = tampon[(thy+8)*48+thx+8+16];
    //tampon [8 - 24][24 - 40] -> dst
    if (tampon_base_x + thx +8 >= 0 && tampon_base_x + thx +8 < width && tampon_base_y + thy +8+16 >= 0 && tampon_base_y + thy +8+16 < height) dst[(tampon_base_y+thy+8+16)*width + tampon_base_x+thx+8] = tampon[(thy+8+16)*48+thx+8];
    //tampon [24 - 40][24 - 40] -> dst
    if (tampon_base_x + thx +8+16 >= 0 && tampon_base_x + thx +8+16 < width && tampon_base_y + thy +8+16 >= 0 && tampon_base_y + thy +8+16 < height) dst[(tampon_base_y+thy+8+16)*width + tampon_base_x+thx+8+16] = tampon[(thy+8+16)*48+thx+8+16];
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