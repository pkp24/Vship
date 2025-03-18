//launch in 16*16
__launch_bounds__(256)
__global__ void GaussianBlur_Kernel(float3* src, float3* dst, int width, int height, float* gaussiankernel){
    const int x = threadIdx.x + blockIdx.x*blockDim.x;
    const int y = threadIdx.y + blockIdx.y*blockDim.y;
    const int thx = threadIdx.x;
    const int thy = threadIdx.y;

    __shared__ float3 tampon[32*32]; //we import into tampon, compute onto tampon and then put into dst
    //tampon has 8 of border on each side with no thread
    const int tampon_base_x = x - thx - 8;
    const int tampon_base_y = y - thy - 8;

    //fill tampon
    tampon[thy*32+thx] = (tampon_base_x + thx >= 0 && tampon_base_x + thx < width && tampon_base_y + thy >= 0 && tampon_base_y + thy < height) ? src[(tampon_base_y+thy)*width + tampon_base_x+thx] : makeFloat3(0., 0., 0.);
    tampon[(thy+16)*32+thx] = (tampon_base_x + thx >= 0 && tampon_base_x + thx < width && tampon_base_y + thy + 16 >= 0 && tampon_base_y + thy + 16 < height) ? src[(tampon_base_y+thy+16)*width + tampon_base_x+thx] : makeFloat3(0., 0., 0.);
    tampon[thy*32+thx+16] = (tampon_base_x + thx +16 >= 0 && tampon_base_x + thx +16 < width && tampon_base_y + thy >= 0 && tampon_base_y + thy < height) ? src[(tampon_base_y+thy)*width + tampon_base_x+thx+16] : makeFloat3(0., 0., 0.);
    tampon[(thy+16)*32+thx+16] = (tampon_base_x + thx +16 >= 0 && tampon_base_x + thx +16 < width && tampon_base_y + thy + 16 >= 0 && tampon_base_y + thy + 16 < height) ? src[(tampon_base_y+thy+16)*width + tampon_base_x+thx+16] : makeFloat3(0., 0., 0.);

    //horizontalBlur on tampon restraint into rectangle [8 - 24][0 - 32] -> 2 pass per thread

    //1st pass in [8 - 24][0 - 16]
    float tot = 0.;
    float3 out = makeFloat3(0., 0., 0.);
    float3 out2 = makeFloat3(0., 0., 0.);
    for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
        out += tampon[thy*32 + thx+i]*gaussiankernel[i];

        //border handling precompute
        if (tampon_base_x+thx+i >= 0 && tampon_base_x+thx+i < width) tot += gaussiankernel[i];
    }

    //2nd pass in [8 - 24][16 - 32]
    for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
        out2 += tampon[(thy+16)*32 + thx+i]*gaussiankernel[i];
    }

    __syncthreads();
    tampon[thy*32 + thx+8] = out/tot;
    tampon[(thy+16)*32 + thx+8] = out2/tot;
    __syncthreads();

    //verticalBlur on tampon restraint into rectangle [8 - 24][8 - 24] -> 1 pass per thread
    out = makeFloat3(0., 0., 0.);
    tot = 0.;
    for (int i = 0; i < 17; i++){ //starting 8 to the left and going 8 to the right
        out += tampon[(thy+i)*32 + thx+8]*gaussiankernel[i];

        //border handling precompute
        if (tampon_base_y+thy+i >= 0 && tampon_base_y+thy+i < height) tot += gaussiankernel[i];
    }

    __syncthreads();
    tampon[(thy+8)*32 + thx+8] = out/tot;
    __syncthreads();

    //tampon [8 - 24][8 - 24] -> dst
    if (tampon_base_x + thx +8 >= 0 && tampon_base_x + thx +8 < width && tampon_base_y + thy +8 >= 0 && tampon_base_y + thy +8 < height) dst[(tampon_base_y+thy+8)*width + tampon_base_x+thx+8] = tampon[(thy+8)*32+thx+8];
}

void gaussianBlur(float3* src, float3* dst, int totalscalesize, int basewidth, int baseheight, float* gaussiankernel_d, hipStream_t stream){
    int w = basewidth;
    int h = baseheight;
    int bl_x, bl_y;

    int cumulate = 0;
    for (int scale = 0; scale <= 5; scale++){
        bl_x = (w-1)/16 + 1;
        bl_y = (h-1)/16 + 1;
        GaussianBlur_Kernel<<<dim3(bl_x, bl_y), dim3(16, 16), 0, stream>>>(src+cumulate, dst+cumulate, w, h, gaussiankernel_d);
        GPU_CHECK(hipGetLastError());
        cumulate += w*h;
        w = (w-1)/2+1;
        h = (h-1)/2+1;
    }
}