#pragma once

#include <string>

#include "../util/preprocessor.hpp"
#include "../util/VshipExceptions.hpp"
#include "../util/gpuhelper.hpp"
#include "../util/float3operations.hpp"
#include "../util/threadsafeset.hpp"
#include "gaussianblur.hpp" 
#include "downupsample.hpp"
#include "Planed.hpp" //Plane_d class
#include "colors.hpp" //OpsinDynamicsImage
#include "separatefrequencies.hpp"
#include "maltaDiff.hpp"
#include "simplerdiff.hpp" //L2 +asym diff + same noise diff
#include "maskPsycho.hpp"
#include "combineMasks.hpp"
#include "diffnorms.hpp" //takes diffmap and returns norm2, norm3 and norminf

namespace butter{

Plane_d getdiffmap(Plane_d* src1_d, Plane_d* src2_d, float* mem_d, int width, int height, float intensity_multiplier, int maxshared, GaussianHandle& gaussianHandle, hipStream_t stream){
    //temporary planes
    Plane_d temp[3] = {Plane_d(mem_d, width, height, stream), Plane_d(mem_d+1*width*height, width, height, stream), Plane_d(mem_d+2*width*height, width, height, stream)};

    //Psycho Image planes
    Plane_d lf1[3] = {Plane_d(mem_d+3*width*height, width, height, stream), Plane_d(mem_d+4*width*height, width, height, stream), Plane_d(mem_d+5*width*height, width, height, stream)};
    Plane_d mf1[3] = {Plane_d(mem_d+6*width*height, width, height, stream), Plane_d(mem_d+7*width*height, width, height, stream), Plane_d(mem_d+8*width*height, width, height, stream)};
    Plane_d hf1[2] = {Plane_d(mem_d+9*width*height, width, height, stream), Plane_d(mem_d+10*width*height, width, height, stream)};
    Plane_d uhf1[2] = {Plane_d(mem_d+11*width*height, width, height, stream), Plane_d(mem_d+12*width*height, width, height, stream)};

    Plane_d lf2[3] = {Plane_d(mem_d+13*width*height, width, height, stream), Plane_d(mem_d+14*width*height, width, height, stream), Plane_d(mem_d+15*width*height, width, height, stream)};
    Plane_d mf2[3] = {Plane_d(mem_d+16*width*height, width, height, stream), Plane_d(mem_d+17*width*height, width, height, stream), Plane_d(mem_d+18*width*height, width, height, stream)};
    Plane_d hf2[2] = {Plane_d(mem_d+19*width*height, width, height, stream), Plane_d(mem_d+20*width*height, width, height, stream)};
    Plane_d uhf2[2] = {Plane_d(mem_d+21*width*height, width, height, stream), Plane_d(mem_d+22*width*height, width, height, stream)};

    //to XYB
    opsinDynamicsImage(src1_d, temp, gaussianHandle, intensity_multiplier);
    opsinDynamicsImage(src2_d, temp, gaussianHandle, intensity_multiplier);
    GPU_CHECK(hipGetLastError());

    separateFrequencies(src1_d, temp, lf1, mf1, hf1, uhf1, gaussianHandle);
    separateFrequencies(src2_d, temp, lf2, mf2, hf2, uhf2, gaussianHandle);

    //no more needs for src1_d and src2_d so we reuse them as masks for butter
    Plane_d* block_diff_dc = src1_d; //size 3
    Plane_d* block_diff_ac = src2_d; //size 3

    //set the accumulators to 0
    for (int c = 0; c < 3; c++){
        block_diff_ac[c].fill0();
        block_diff_dc[c].fill0();
    }

    const float hf_asymmetry_ = 0.8f;

    const float wUhfMalta = 1.10039032555f;
    const float norm1Uhf = 71.7800275169f;
    MaltaDiffMap(uhf1[1].mem_d, uhf2[1].mem_d, block_diff_ac[1].mem_d, width, height, wUhfMalta * hf_asymmetry_, wUhfMalta / hf_asymmetry_, norm1Uhf, stream);

    const float wUhfMaltaX = 173.5f;
    const float norm1UhfX = 5.0f;
    MaltaDiffMap(uhf1[0].mem_d, uhf2[0].mem_d, block_diff_ac[0].mem_d, width, height, wUhfMaltaX * hf_asymmetry_, wUhfMaltaX / hf_asymmetry_, norm1UhfX, stream);

    const float wHfMalta = 18.7237414387f;
    const float norm1Hf = 4498534.45232f;
    MaltaDiffMapLF(hf1[1].mem_d, hf2[1].mem_d, block_diff_ac[1].mem_d, width, height, wHfMalta * std::sqrt(hf_asymmetry_), wHfMalta / std::sqrt(hf_asymmetry_), norm1Hf, stream);

    const float wHfMaltaX = 6923.99476109f;
    const float norm1HfX = 8051.15833247f;
    MaltaDiffMapLF(hf1[0].mem_d, hf2[0].mem_d, block_diff_ac[0].mem_d, width, height, wHfMaltaX * std::sqrt(hf_asymmetry_), wHfMaltaX / std::sqrt(hf_asymmetry_), norm1HfX, stream);

    const float wMfMalta = 37.0819870399f;
    const float norm1Mf = 130262059.556f;
    MaltaDiffMapLF(mf1[1].mem_d, mf2[1].mem_d, block_diff_ac[1].mem_d, width, height, wMfMalta, wMfMalta, norm1Mf, stream);

    const float wMfMaltaX = 8246.75321353f;
    const float norm1MfX = 1009002.70582f;
    MaltaDiffMapLF(mf1[0].mem_d, mf2[0].mem_d, block_diff_ac[0].mem_d, width, height, wMfMaltaX, wMfMaltaX, norm1MfX, stream);

    const float wmul[9] = {
      400.0f,         1.50815703118f,  0.0f,
      2150.0f,        10.6195433239f,  16.2176043152f,
      29.2353797994f, 0.844626970982f, 0.703646627719f,
    };

    //const float maxclamp = 85.7047444518;
    //const float kSigmaHfX = 10.6666499623;
    //const float w = 884.809801415;
    //sameNoiseLevels(hf1[1], hf2[1], block_diff_ac[1], temp[0], temp[1], kSigmaHfX, w, maxclamp, gaussiankernel_dmem);

    for (int c = 0; c < 3; c++){
        if (c < 2){
            L2AsymDiff(hf1[c].mem_d, hf2[c].mem_d, block_diff_ac[c].mem_d, width*height, wmul[c] * hf_asymmetry_, wmul[c] / hf_asymmetry_, stream);
        }
        L2diff(mf1[c].mem_d, mf2[c].mem_d, block_diff_ac[c].mem_d, width*height, wmul[3 + c], stream);
        L2diff(lf1[c].mem_d, lf2[c].mem_d, block_diff_dc[c].mem_d, width*height, wmul[6 + c], stream);
    }

    //from now on, lf and mf are not used so we will reuse the memory
    Plane_d mask = temp[1];
    Plane_d* temp3 = lf2;
    Plane_d* temp4 = mf2;

    MaskPsychoImage(hf1, uhf1, hf2, uhf2, temp3[0], temp4[0], mask, block_diff_ac, gaussianHandle);
    //at this point hf and uhf cannot be used anymore (they have been invalidated by the function)

    Plane_d diffmap = temp[0]; //we only need one plane
    computeDiffmap(mask.mem_d, block_diff_dc[0].mem_d, block_diff_dc[1].mem_d, block_diff_dc[2].mem_d, block_diff_ac[0].mem_d, block_diff_ac[1].mem_d, block_diff_ac[2].mem_d, diffmap.mem_d, width*height, stream);

    //hipEventRecord(event_d, stream); //place an event in the stream at the end of all our operations
    //hipEventSynchronize(event_d); //when the event is complete, we know our gpu result is ready!
    //GPU_CHECK(hipGetLastError());

    //printf("End result: %f, %f and %f\n", norm2, norm3, norminf);
    
    return diffmap;
}

template <InputMemType T>
std::tuple<float, float, float> butterprocess(const uint8_t *dstp, int dststride, const uint8_t *srcp1[3], const uint8_t *srcp2[3], float* pinned, GaussianHandle& gaussianHandle, int stride, int width, int height, float intensity_multiplier, int maxshared, hipStream_t stream){
    int wh = width*height;
    const int totalscalesize = wh;

    //big memory allocation, we will try it multiple time if failed to save when too much threads are used
    hipError_t erralloc;

    const int totalplane = 31;
    float* mem_d;
    erralloc = hipMallocAsync(&mem_d, sizeof(float)*totalscalesize*(totalplane), stream); //2 base image and 6 working buffers
    if (erralloc != hipSuccess){
        throw VshipError(OutOfVRAM, __FILE__, __LINE__);
    }
    //initial color planes
    Plane_d src1_d[3] = {Plane_d(mem_d, width, height, stream), Plane_d(mem_d+width*height, width, height, stream), Plane_d(mem_d+2*width*height, width, height, stream)};
    Plane_d src2_d[3] = {Plane_d(mem_d+3*width*height, width, height, stream), Plane_d(mem_d+4*width*height, width, height, stream), Plane_d(mem_d+5*width*height, width, height, stream)};

    //we put the frame's planes on GPU
    GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)(srcp1[0]), stride * height, stream));
    src1_d[0].strideEliminator<T>(mem_d+6*width*height, stride);
    GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)(srcp1[1]), stride * height, stream));
    src1_d[1].strideEliminator<T>(mem_d+6*width*height, stride);
    GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)(srcp1[2]), stride * height, stream));
    src1_d[2].strideEliminator<T>(mem_d+6*width*height, stride);

    GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)(srcp2[0]), stride * height, stream));
    src2_d[0].strideEliminator<T>(mem_d+6*width*height, stride);
    GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)(srcp2[1]), stride * height, stream));
    src2_d[1].strideEliminator<T>(mem_d+6*width*height, stride);
    GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)(srcp2[2]), stride * height, stream));
    src2_d[2].strideEliminator<T>(mem_d+6*width*height, stride);

    //computing downscaled before we overwrite src in getdiffmap (it s better for memory)
    int nwidth = (width-1)/2+1;
    int nheight = (height-1)/2+1;
    float* nmem_d = mem_d+6*width*height; //allow usage up to mem_d+8*width*height;
    Plane_d nsrc1_d[3] = {Plane_d(nmem_d, nwidth, nheight, stream), Plane_d(nmem_d+nwidth*nheight, nwidth, nheight, stream), Plane_d(nmem_d+2*nwidth*nheight, nwidth, nheight, stream)};
    Plane_d nsrc2_d[3] = {Plane_d(nmem_d+3*nwidth*nheight, nwidth, nheight, stream), Plane_d(nmem_d+4*nwidth*nheight, nwidth, nheight, stream), Plane_d(nmem_d+5*nwidth*nheight, nwidth, nheight, stream)};

    //we need to convert to linear rgb before downsampling
    linearRGB(src1_d);
    linearRGB(src2_d);

    //using 6 smaller planes is equivalent to 1.5 standard planes, so it fits within the 2 planes given here!)
    for (int i = 0; i < 3; i++){
        downsample(src1_d[i].mem_d, nsrc1_d[i].mem_d, width, height, stream);
        downsample(src2_d[i].mem_d, nsrc2_d[i].mem_d, width, height, stream);
    }

    Plane_d diffmap = getdiffmap(src1_d, src2_d, mem_d+8*width*height, width, height, intensity_multiplier, maxshared, gaussianHandle, stream);
    //diffmap is stored at mem_d+8*width*height so we can build after that the second smaller scale
    //smaller scale now
    nmem_d = mem_d+9*width*height;
    Plane_d diffmapsmall = getdiffmap(nsrc1_d, nsrc2_d, nmem_d+6*nwidth*nheight, nwidth, nheight, intensity_multiplier, maxshared, gaussianHandle, stream);

    addsupersample2X(diffmap.mem_d, diffmapsmall.mem_d, width, height, 0.5f, stream);

    //diffmap is in its final form
    if (dstp != NULL){
        diffmap.strideAdder(nmem_d, dststride);
        GPU_CHECK(hipMemcpyDtoHAsync((void*)(dstp), nmem_d, dststride * height, stream));
    }

    std::tuple<float, float, float> finalres;
    try{
        finalres = diffmapscore(diffmap.mem_d, mem_d+9*width*height, mem_d+10*width*height, pinned, width*height, stream);
    } catch (const VshipError& e){
        hipFree(mem_d);
        throw e;
    }

    hipFreeAsync(mem_d, stream);

    return finalres;
}

}
