#include "../util/preprocessor.hpp"
#include "../util/float3operations.hpp"
#include "../util/makeXYB.hpp"
#include "gaussianblur.hpp"
#include "Planed.hpp"
#include "colors.hpp"
#include "separatefrequencies.hpp"
#include "maltaDiff.hpp"

namespace butter{

double butterprocess(const uint8_t *srcp1[3], const uint8_t *srcp2[3], int stride, int width, int height, float intensity_multiplier, int maxshared, hipStream_t stream){
    int wh = width*height;
    const int totalscalesize = wh;

    //big memory allocation, we will try it multiple time if failed to save when too much threads are used
    hipError_t erralloc;
    int tries = 10;

    const int gaussiantotal = 1024;
    const int totalplane = 32;
    float* mem_d;
    erralloc = hipMalloc(&mem_d, sizeof(float)*totalscalesize*(totalplane) + sizeof(float)*gaussiantotal); //2 base image and 6 working buffers
    while (erralloc != hipSuccess){
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); //0.5s with 10 tries -> shut down after 5 seconds of failing
        erralloc = hipMalloc(&mem_d, sizeof(float)*totalscalesize*(totalplane) + sizeof(float)*gaussiantotal); //2 base image and 6 working buffers
        tries--;
        if (tries <= 0){
            printf("ERROR, could not allocate VRAM for a frame, try lowering the number of vapoursynth threads\n");
            return -10000.;
        }
    }
    //GPU_CHECK(hipGetLastError());

    //initial color planes
    Plane_d src1_d[3] = {Plane_d(mem_d, width, height, stream), Plane_d(mem_d+width*height, width, height, stream), Plane_d(mem_d+2*width*height, width, height, stream)};
    Plane_d src2_d[3] = {Plane_d(mem_d+3*width*height, width, height, stream), Plane_d(mem_d+4*width*height, width, height, stream), Plane_d(mem_d+5*width*height, width, height, stream)};
    
    //temporary planes
    Plane_d temp[3] = {Plane_d(mem_d+6*width*height, width, height, stream), Plane_d(mem_d+7*width*height, width, height, stream), Plane_d(mem_d+8*width*height, width, height, stream)};
    Plane_d temp2[3] = {Plane_d(mem_d+9*width*height, width, height, stream), Plane_d(mem_d+10*width*height, width, height, stream), Plane_d(mem_d+11*width*height, width, height, stream)};

    //Psycho Image planes
    Plane_d lf1[3] = {Plane_d(mem_d+12*width*height, width, height, stream), Plane_d(mem_d+13*width*height, width, height, stream), Plane_d(mem_d+14*width*height, width, height, stream)};
    Plane_d mf1[3] = {Plane_d(mem_d+15*width*height, width, height, stream), Plane_d(mem_d+16*width*height, width, height, stream), Plane_d(mem_d+17*width*height, width, height, stream)};
    Plane_d hf1[2] = {Plane_d(mem_d+18*width*height, width, height, stream), Plane_d(mem_d+19*width*height, width, height, stream)};
    Plane_d uhf1[2] = {Plane_d(mem_d+20*width*height, width, height, stream), Plane_d(mem_d+21*width*height, width, height, stream)};

    Plane_d lf2[3] = {Plane_d(mem_d+22*width*height, width, height, stream), Plane_d(mem_d+23*width*height, width, height, stream), Plane_d(mem_d+24*width*height, width, height, stream)};
    Plane_d mf2[3] = {Plane_d(mem_d+25*width*height, width, height, stream), Plane_d(mem_d+26*width*height, width, height, stream), Plane_d(mem_d+27*width*height, width, height, stream)};
    Plane_d hf2[2] = {Plane_d(mem_d+28*width*height, width, height, stream), Plane_d(mem_d+29*width*height, width, height, stream)};
    Plane_d uhf2[2] = {Plane_d(mem_d+30*width*height, width, height, stream), Plane_d(mem_d+31*width*height, width, height, stream)};

    float* gaussiankernel_dmem = (mem_d + totalplane*totalscalesize);

    hipEvent_t event_d;
    hipEventCreate(&event_d);

    //we put the frame's planes on GPU
    GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)(srcp1[0]), stride * height, stream));
    src1_d[0].strideEliminator(mem_d+6*width*height, stride);
    GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)(srcp1[1]), stride * height, stream));
    src1_d[1].strideEliminator(mem_d+6*width*height, stride);
    GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)(srcp1[2]), stride * height, stream));
    src1_d[2].strideEliminator(mem_d+6*width*height, stride);

    GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)(srcp2[0]), stride * height, stream));
    src2_d[0].strideEliminator(mem_d+6*width*height, stride);
    GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)(srcp2[1]), stride * height, stream));
    src2_d[1].strideEliminator(mem_d+6*width*height, stride);
    GPU_CHECK(hipMemcpyHtoDAsync(mem_d+6*width*height, (void*)(srcp2[2]), stride * height, stream));
    src2_d[2].strideEliminator(mem_d+6*width*height, stride);

    //to XYB
    opsinDynamicsImage(src1_d, temp, temp2[0], gaussiankernel_dmem, intensity_multiplier);
    opsinDynamicsImage(src2_d, temp, temp2[0], gaussiankernel_dmem, intensity_multiplier);
    GPU_CHECK(hipGetLastError());

    separateFrequencies(src1_d, temp, lf1, mf1, hf1, uhf1, gaussiankernel_dmem);
    separateFrequencies(src2_d, temp, lf2, mf2, hf2, uhf2, gaussiankernel_dmem);

    //no more needs for src1_d and src2_d so we reuse them as masks for butter
    Plane_d* block_diff_dc = src1_d; //size 3
    Plane_d* block_diff_ac = src2_d; //size 3

    //set the accumulators to 0
    for (int c = 0; c < 3; c++){
        block_diff_ac[c].fill0();
        block_diff_dc[c].fill0();
    }

    const float hf_asymmetry_ = 1.;

    const float wUhfMalta = 5.1409625726;
    const float norm1Uhf = 58.5001247061;
    MaltaDiffMap(uhf1[1].mem_d, uhf2[1].mem_d, block_diff_ac[1].mem_d, width, height, wUhfMalta * hf_asymmetry_, wUhfMalta / hf_asymmetry_, norm1Uhf, stream);

    const float wUhfMaltaX = 4.91743441556;
    const float norm1UhfX = 687196.39002;
    MaltaDiffMap(uhf1[0].mem_d, uhf2[0].mem_d, block_diff_ac[0].mem_d, width, height, wUhfMaltaX * hf_asymmetry_, wUhfMaltaX / hf_asymmetry_, norm1UhfX, stream);

    const float wHfMalta = 153.671655716;
    const float norm1Hf = 83150785.9592;
    MaltaDiffMapLF(hf1[1].mem_d, hf2[1].mem_d, block_diff_ac[1].mem_d, width, height, wHfMalta * sqrt(hf_asymmetry_), wHfMalta / sqrt(hf_asymmetry_), norm1Hf, stream);

    const float wHfMaltaX = 668.358918152;
    const float norm1HfX = 0.882954368025;
    MaltaDiffMapLF(hf1[0].mem_d, hf2[0].mem_d, block_diff_ac[0].mem_d, width, height, wHfMaltaX * sqrt(hf_asymmetry_), wHfMaltaX / sqrt(hf_asymmetry_), norm1HfX, stream);

    const float wMfMalta = 6841.81248144;
    const float norm1Mf = 0.0135134962487;
    MaltaDiffMapLF(mf1[1].mem_d, mf2[1].mem_d, block_diff_ac[1].mem_d, width, height, wMfMalta, wMfMalta, norm1Mf, stream);

    const float wMfMaltaX = 813.901703816;
    const float norm1MfX = 16792.9322251;
    MaltaDiffMapLF(mf1[0].mem_d, mf2[0].mem_d, block_diff_ac[0].mem_d, width, height, wMfMaltaX, wMfMaltaX, norm1MfX, stream);

    hipEventRecord(event_d, stream); //place an event in the stream at the end of all our operations
    hipEventSynchronize(event_d); //when the event is complete, we know our gpu result is ready!
    GPU_CHECK(hipGetLastError());

    hipFree(mem_d);
    hipEventDestroy(event_d);
    
    return 0.;
}

typedef struct {
    VSNode *reference;
    VSNode *distorted;
    float intensity_multiplier;
    int maxshared;
    hipStream_t streams[STREAMNUM];
    int oldthreadnum;
} ButterData;

static const VSFrame *VS_CC butterGetFrame(int n, int activationReason, void *instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    ButterData *d = (ButterData *)instanceData;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->reference, frameCtx);
        vsapi->requestFrameFilter(n, d->distorted, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *src1 = vsapi->getFrameFilter(n, d->reference, frameCtx);
        const VSFrame *src2 = vsapi->getFrameFilter(n, d->distorted, frameCtx);
        
        int height = vsapi->getFrameHeight(src1, 0);
        int width = vsapi->getFrameWidth(src1, 0);
        int stride = vsapi->getStride(src1, 0);

        VSFrame *dst = vsapi->copyFrame(src2, core);

        const uint8_t *srcp1[3] = {
            vsapi->getReadPtr(src1, 0),
            vsapi->getReadPtr(src1, 1),
            vsapi->getReadPtr(src1, 2),
        };

        const uint8_t *srcp2[3] = {
            vsapi->getReadPtr(src2, 0),
            vsapi->getReadPtr(src2, 1),
            vsapi->getReadPtr(src2, 2),
        };
        const double val = butterprocess(srcp1, srcp2, stride, width, height, d->intensity_multiplier, d->maxshared, d->streams[n%STREAMNUM]);

        vsapi->mapSetFloat(vsapi->getFramePropertiesRW(dst), "_BUTTERAUGLI", val, maReplace);

        // Release the source frame
        vsapi->freeFrame(src1);
        vsapi->freeFrame(src2);

        // A reference is consumed when it is returned, so saving the dst reference somewhere
        // and reusing it is not allowed.
        return dst;
    }

    return NULL;
}

// Free all allocated data on filter destruction
static void VS_CC butterFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    ButterData *d = (ButterData *)instanceData;
    vsapi->freeNode(d->reference);
    vsapi->freeNode(d->distorted);

    for (int i = 0; i < STREAMNUM; i++){
        hipStreamDestroy(d->streams[i]);
    }
    //vsapi->setThreadCount(d->oldthreadnum, core);

    free(d);
}

// This function is responsible for validating arguments and creating a new filter  
static void VS_CC butterCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    ButterData d;
    ButterData *data;

    // Get a clip reference from the input arguments. This must be freed later.
    d.reference = vsapi->mapGetNode(in, "reference", 0, 0);
    d.distorted = vsapi->mapGetNode(in, "distorted", 0, 0);
    const VSVideoInfo *viref = vsapi->getVideoInfo(d.reference);
    const VSVideoInfo *vidis = vsapi->getVideoInfo(d.distorted);

    if (!(vsh::isSameVideoInfo(viref, vidis))){
        vsapi->mapSetError(out, "BUTTERAUGLI: both clips must have the same format and dimensions");
        vsapi->freeNode(d.reference);
        vsapi->freeNode(d.distorted);
        return;
    }

    if ((viref->format.colorFamily != cfRGB) || viref->format.sampleType != stFloat){
        vsapi->mapSetError(out, "BUTTERAUGLI: only works with RGBS format");
        vsapi->freeNode(d.reference);
        vsapi->freeNode(d.distorted);
        return;
    }

    int error;
    d.intensity_multiplier = vsapi->mapGetFloat(in, "intensity_multiplier", 0, &error);
    if (error != peSuccess){
        d.intensity_multiplier = 1.;
    }

    int count;
    if (hipGetDeviceCount(&count) != 0){
        vsapi->mapSetError(out, "could not detect devices, check gpu permissions\n");
    };
    if (count == 0){
        vsapi->mapSetError(out, "No GPU was found on the system for a given compilation type. Try switch nvidia/amd binary\n");
    }

    hipDeviceSetCacheConfig(hipFuncCachePreferNone);
    int device;
    hipDeviceProp_t devattr;
    hipGetDevice(&device);
    hipGetDeviceProperties(&devattr, device);

    //int videowidth = viref->width;
    //int videoheight = viref->height;
    //put optimal thread number
    //VSCoreInfo infos;
    //vsapi->getCoreInfo(core, &infos);
    //d.oldthreadnum = infos.numThreads;
    //size_t freemem, totalmem;
    //hipMemGetInfo (&freemem, &totalmem);

    //vsapi->setThreadCount(std::min((int)((float)(freemem - 20*(1llu << 20))/(8*sizeof(float3)*videowidth*videoheight*(1.33333))), d.oldthreadnum), core);

    for (int i = 0; i < STREAMNUM; i++){
        hipStreamCreate(d.streams + i);
    }

    data = (ButterData *)malloc(sizeof(d));
    *data = d;

    for (int i = 0; i < STREAMNUM; i++){
        data->streams[i] = d.streams[i];
    }
    data->maxshared = devattr.sharedMemPerBlock;    

    VSFilterDependency deps[] = {{d.reference, rpStrictSpatial}, {d.distorted, rpStrictSpatial}};
    vsapi->createVideoFilter(out, "vship", viref, butterGetFrame, butterFree, fmParallel, deps, 2, data, core);
}

}