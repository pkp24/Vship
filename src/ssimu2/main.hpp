#pragma once

#include "../util/preprocessor.hpp"
#include "../util/VshipExceptions.hpp"
#include "../util/gpuhelper.hpp"
#include "../util/float3operations.hpp"
#include "../util/threadsafeset.hpp"
#include "makeXYB.hpp"
#include "downsample.hpp"
#include "gaussianblur.hpp"
#include "score.hpp"

namespace ssimu2{

template <InputMemType T>
__launch_bounds__(256)
__global__ void memoryorganizer_kernel(float3* out, const uint8_t *srcp0, const uint8_t *srcp1, const uint8_t *srcp2, int stride, int width, int height){
    size_t x = threadIdx.x + blockIdx.x*blockDim.x;
    if (x >= width*height) return;
    int j = x%width;
    int i = x/width;
    out[i*width + j].x = convertPointer<T>(srcp0, i, j, stride);
    out[i*width + j].y = convertPointer<T>(srcp1, i, j, stride);
    out[i*width + j].z = convertPointer<T>(srcp2, i, j, stride);
}

template <InputMemType T>
void memoryorganizer(float3* out, const uint8_t *srcp0, const uint8_t *srcp1, const uint8_t *srcp2, int stride, int width, int height, hipStream_t stream){
    int th_x = std::min(256, width*height);
    int bl_x = (width*height-1)/th_x + 1;
    memoryorganizer_kernel<T><<<dim3(bl_x), dim3(th_x), 0, stream>>>(out, srcp0, srcp1, srcp2, stride, width, height);
}

template <InputMemType T>
double ssimu2process(const uint8_t *srcp1[3], const uint8_t *srcp2[3], float3* pinned, int stride, int width, int height, float* gaussiankernel, int maxshared, hipStream_t stream){

    int wh = width*height;
    int whs[6] = {wh, ((height-1)/2 + 1)*((width-1)/2 + 1), ((height-1)/4 + 1)*((width-1)/4 + 1), ((height-1)/8 + 1)*((width-1)/8 + 1), ((height-1)/16 + 1)*((width-1)/16 + 1), ((height-1)/32 + 1)*((width-1)/32 + 1)};
    int whs_integral[7];
    whs_integral[0] = 0;
    for (int i = 0; i < 7; i++){
        whs_integral[i+1] = whs_integral[i] + whs[i];
    }
    int totalscalesize = whs_integral[6];

    //big memory allocation, we will try it multiple time if failed to save when too much threads are used
    hipError_t erralloc;

    float3* mem_d;
    erralloc = hipMallocAsync(&mem_d, sizeof(float3)*totalscalesize*(2 + 6), stream); //2 base image and 6 working buffers
    if (erralloc != hipSuccess){
        throw VshipError(OutOfVRAM, __FILE__, __LINE__);
    }

    float3* src1_d = mem_d; //length totalscalesize
    float3* src2_d = mem_d + totalscalesize;

    float3* temp_d = mem_d + 2*totalscalesize;
    float3* temps11_d = mem_d + 3*totalscalesize;
    float3* temps22_d = mem_d + 4*totalscalesize;
    float3* temps12_d = mem_d + 5*totalscalesize;
    float3* tempb1_d = mem_d + 6*totalscalesize;
    float3* tempb2_d = mem_d + 7*totalscalesize;

    uint8_t *memory_placeholder[3] = {(uint8_t*)temp_d, (uint8_t*)temp_d+stride*height, (uint8_t*)temp_d+2*stride*height};
    GPU_CHECK(hipMemcpyHtoDAsync(memory_placeholder[0], (void*)srcp1[0], stride * height, stream));
    GPU_CHECK(hipMemcpyHtoDAsync(memory_placeholder[1], (void*)srcp1[1], stride * height, stream));
    GPU_CHECK(hipMemcpyHtoDAsync(memory_placeholder[2], (void*)srcp1[2], stride * height, stream));
    memoryorganizer<T>(src1_d, memory_placeholder[0], memory_placeholder[1], memory_placeholder[2], stride, width, height, stream);

    GPU_CHECK(hipMemcpyHtoDAsync(memory_placeholder[0], (void*)srcp2[0], stride * height, stream));
    GPU_CHECK(hipMemcpyHtoDAsync(memory_placeholder[1], (void*)srcp2[1], stride * height, stream));
    GPU_CHECK(hipMemcpyHtoDAsync(memory_placeholder[2], (void*)srcp2[2], stride * height, stream));
    memoryorganizer<T>(src2_d, memory_placeholder[0], memory_placeholder[1], memory_placeholder[2], stride, width, height, stream);

    rgb_to_linear(src1_d, totalscalesize, stream);
    rgb_to_linear(src2_d, totalscalesize, stream);

    //step 1 : fill the downsample part
    int nw = width;
    int nh = height;
    for (int scale = 1; scale <= 5; scale++){
        downsample(src1_d+whs_integral[scale-1], src1_d+whs_integral[scale], nw, nh, stream);
        nw = (nw -1)/2 + 1;
        nh = (nh - 1)/2 + 1;
    }
    nw = width;
    nh = height;
    for (int scale = 1; scale <= 5; scale++){
        downsample(src2_d+whs_integral[scale-1], src2_d+whs_integral[scale], nw, nh, stream);
        nw = (nw -1)/2 + 1;
        nh = (nh - 1)/2 + 1;
    }

    //step 2 : positive XYB transition
    rgb_to_positive_xyb(src1_d, totalscalesize, stream);
    rgb_to_positive_xyb(src2_d, totalscalesize, stream);

    //step 3 : fill buffer s11 s22 and s12 and blur everything
    multarray(src1_d, src1_d, temp_d, totalscalesize, stream);
    gaussianBlur(temp_d, temps11_d, totalscalesize, width, height, gaussiankernel, stream);

    multarray(src1_d, src2_d, temp_d, totalscalesize, stream);
    gaussianBlur(temp_d, temps12_d, totalscalesize, width, height, gaussiankernel, stream);

    multarray(src2_d, src2_d, temp_d, totalscalesize, stream);
    gaussianBlur(temp_d, temps22_d, totalscalesize, width, height, gaussiankernel, stream);

    gaussianBlur(src1_d, tempb1_d, totalscalesize, width, height, gaussiankernel, stream);
    gaussianBlur(src2_d, tempb2_d, totalscalesize, width, height, gaussiankernel, stream);

    //step 4 : ssim map
    
    //step 5 : edge diff map    
    std::vector<float3> allscore_res;
    try{
        allscore_res = allscore_map(src1_d, src2_d, tempb1_d, tempb2_d, temps11_d, temps22_d, temps12_d, temp_d, pinned, width, height, maxshared, stream);
    } catch (const VshipError& e){
        hipFree(mem_d);
        throw e;
    }

    //we are done with the gpu at that point and the synchronization has already been done in allscore_map
    hipFreeAsync(mem_d, stream);

    //step 6 : format the vector
    std::vector<float> measure_vec(108);

    for (int plane = 0; plane < 3; plane++){
        for (int scale = 0; scale < 6; scale++){
            for (int n = 0; n < 2; n++){
                for (int i = 0; i < 3; i++){
                    if (plane == 0) measure_vec[plane*6*2*3 + scale*2*3 + n*3 + i] = allscore_res[scale*2*3 + i*2 + n].x;
                    if (plane == 1) measure_vec[plane*6*2*3 + scale*2*3 + n*3 + i] = allscore_res[scale*2*3 + i*2 + n].y;
                    if (plane == 2) measure_vec[plane*6*2*3 + scale*2*3 + n*3 + i] = allscore_res[scale*2*3 + i*2 + n].z;
                }
            }
        }
    }

    //step 7 : enjoy !
    const float ssim = final_score(measure_vec);

    //hipEventRecord(event_d, stream); //place an event in the stream at the end of all our operations
    //hipEventSynchronize(event_d); //when the event is complete, we know our gpu result is ready!

    return ssim;
}

}