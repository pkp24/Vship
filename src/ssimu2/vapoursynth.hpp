#pragma once

#include "../util/torgbs.hpp"
#include "main.hpp"

namespace ssimu2{

    typedef struct Ssimulacra2Data{
        VSNode *reference;
        VSNode *distorted;
        float3** PinnedMemPool;
        int maxshared;
        float* gaussiankernel_d;
        hipStream_t* streams;
        int streamnum = 0;
        threadSet* streamSet;
    } Ssimulacra2Data;
    
    static const VSFrame *VS_CC ssimulacra2GetFrame(int n, int activationReason, void *instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
        Ssimulacra2Data *d = (Ssimulacra2Data *)instanceData;
    
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
    
            double val;
            const int stream = d->streamSet->pop();
            try{
                val = ssimu2process<FLOAT>(srcp1, srcp2, d->PinnedMemPool[stream], stride, width, height, d->gaussiankernel_d, d->maxshared, d->streams[stream]);
            } catch (const VshipError& e){
                vsapi->setFilterError(e.getErrorMessage().c_str(), frameCtx);
                d->streamSet->insert(stream);
                vsapi->freeFrame(src1);
                vsapi->freeFrame(src2);
                return NULL;
            }
            d->streamSet->insert(stream);
    
            vsapi->mapSetFloat(vsapi->getFramePropertiesRW(dst), "_SSIMULACRA2", val, maReplace);
    
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
    static void VS_CC ssimulacra2Free(void *instanceData, VSCore *core, const VSAPI *vsapi) {
        Ssimulacra2Data *d = (Ssimulacra2Data *)instanceData;
        vsapi->freeNode(d->reference);
        vsapi->freeNode(d->distorted);
    
        for (int i = 0; i < d->streamnum; i++){
            hipHostFree(d->PinnedMemPool[i]);
            hipStreamDestroy(d->streams[i]);
        }
        free(d->PinnedMemPool);
        hipFree(d->gaussiankernel_d);
        free(d->streams);
        delete d->streamSet;
        //vsapi->setThreadCount(d->oldthreadnum, core);
    
        free(d);
    }
    
    // This function is responsible for validating arguments and creating a new filter  
    static void VS_CC ssimulacra2Create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
        Ssimulacra2Data d;
        Ssimulacra2Data *data;
    
        // Get a clip reference from the input arguments. This must be freed later.
        d.reference = toRGBS(vsapi->mapGetNode(in, "reference", 0, 0), core, vsapi, false);
        d.distorted = toRGBS(vsapi->mapGetNode(in, "distorted", 0, 0), core, vsapi, false);
        const VSVideoInfo *viref = vsapi->getVideoInfo(d.reference);
        const VSVideoInfo *vidis = vsapi->getVideoInfo(d.distorted);
    
        if (!(vsh::isSameVideoInfo(viref, vidis))){
            vsapi->mapSetError(out, VshipError(DifferingInputType, __FILE__, __LINE__).getErrorMessage().c_str());
            vsapi->freeNode(d.reference);
            vsapi->freeNode(d.distorted);
            return;
        }
    
        if ((viref->format.bitsPerSample != 32) || (viref->format.colorFamily != cfRGB) || viref->format.sampleType != stFloat){
            vsapi->mapSetError(out, VshipError(NonRGBSInput, __FILE__, __LINE__).getErrorMessage().c_str());
            vsapi->freeNode(d.reference);
            vsapi->freeNode(d.distorted);
            return;
        }
    
        int error;
        int gpuid = vsapi->mapGetInt(in, "gpu_id", 0, &error);
        if (error != peSuccess){
            gpuid = 0;
        }
    
        try{
            //if succeed, this function also does hipSetDevice
            helper::gpuFullCheck(gpuid);
        } catch (const VshipError& e){
            vsapi->mapSetError(out, e.getErrorMessage().c_str());
            return;
        }
    
        //hipSetDevice(gpuid);
    
        hipDeviceSetCacheConfig(hipFuncCachePreferNone);
        int device;
        hipDeviceProp_t devattr;
        hipGetDevice(&device);
        hipGetDeviceProperties(&devattr, device);
        d.maxshared = devattr.sharedMemPerBlock;
    
        //int videowidth = viref->width;
        //int videoheight = viref->height;
        //put optimal thread number
        VSCoreInfo infos;
        vsapi->getCoreInfo(core, &infos);
        //d.oldthreadnum = infos.numThreads;
        //size_t freemem, totalmem;
        //hipMemGetInfo (&freemem, &totalmem);
    
        //vsapi->setThreadCount(std::min((int)((float)(freemem - 20*(1llu << 20))/(8*sizeof(float3)*videowidth*videoheight*(1.33333))), d.oldthreadnum), core);
    
        d.streamnum = vsapi->mapGetInt(in, "numStream", 0, &error);
        if (error != peSuccess){
            d.streamnum = infos.numThreads;
        }
    
        d.streamnum = std::min(d.streamnum, infos.numThreads); // vs threads < numStream would make no sense
        d.streamnum = std::min((size_t)d.streamnum, (size_t)((size_t)devattr.totalGlobalMem/(32llu*4llu*(size_t)viref->width*(size_t)viref->height))); //VRAM overcommit partial protection
        d.streamnum = std::max(d.streamnum, 1); //at least one stream to not just wait indefinitely
        d.streams = (hipStream_t*)malloc(sizeof(hipStream_t)*d.streamnum);
        for (int i = 0; i < d.streamnum; i++){
            hipStreamCreate(d.streams + i);
        }
    
        std::set<int> newstreamset;
        for (int i = 0; i < d.streamnum; i++){
            newstreamset.insert(i);
        }
        d.streamSet = new threadSet(newstreamset);
    
        const int pinnedsize = allocsizeScore(viref->width, viref->height, d.maxshared);
        d.PinnedMemPool = (float3**)malloc(sizeof(float3*)*d.streamnum);
        hipError_t erralloc;
        for (int i = 0; i < d.streamnum; i++){
            erralloc = hipHostMalloc(d.PinnedMemPool+i, sizeof(float3)*pinnedsize);
            if (erralloc != hipSuccess){
                vsapi->mapSetError(out, VshipError(OutOfRAM, __FILE__, __LINE__).getErrorMessage().c_str());
                vsapi->freeNode(d.reference);
                vsapi->freeNode(d.distorted);
                return;
            }
        }
    
        data = (Ssimulacra2Data *)malloc(sizeof(d));
        *data = d;
    
        for (int i = 0; i < d.streamnum; i++){
            data->streams[i] = d.streams[i];
        }
    
        float gaussiankernel[2*GAUSSIANSIZE+1];
        for (int i = 0; i < 2*GAUSSIANSIZE+1; i++){
            gaussiankernel[i] = std::exp(-(GAUSSIANSIZE-i)*(GAUSSIANSIZE-i)/(2*SIGMA*SIGMA))/(std::sqrt(TAU*SIGMA*SIGMA));
        }
    
        hipMalloc(&(data->gaussiankernel_d), sizeof(float)*(2*GAUSSIANSIZE+1));
        hipMemcpyHtoD((hipDeviceptr_t)data->gaussiankernel_d, gaussiankernel, (2*GAUSSIANSIZE+1)*sizeof(float));
    
        VSFilterDependency deps[] = {{d.reference, rpStrictSpatial}, {d.distorted, rpStrictSpatial}};
    
        vsapi->createVideoFilter(out, "vship", viref, ssimulacra2GetFrame, ssimulacra2Free, fmParallel, deps, 2, data, core);
    }

}