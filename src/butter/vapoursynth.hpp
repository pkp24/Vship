#pragma once

#include "../util/torgbs.hpp"
#include "main.hpp"

namespace butter{
    typedef struct ButterData{
        VSNode *reference;
        VSNode *distorted;
        float intensity_multiplier;
        float** PinnedMemPool;
        GaussianHandle gaussianHandle;
        int maxshared;
        int diffmap;
        hipStream_t* streams;
        int streamnum = 0;
        threadSet* streamSet;
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
    
            VSFrame *dst;
            if (d->diffmap){
                VSVideoFormat formatout;
                vsapi->queryVideoFormat(&formatout, cfGray, stFloat, 32, 0, 0, core);
                dst = vsapi->newVideoFrame(&formatout, width, height, NULL, core);
            } else {
                dst = vsapi->copyFrame(src2, core);
            }
    
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
            
            std::tuple<float, float, float> val;
            
            const int stream = d->streamSet->pop();
            try{
                if (d->diffmap){
                    val = butterprocess<FLOAT>(vsapi->getWritePtr(dst, 0), vsapi->getStride(dst, 0), srcp1, srcp2, d->PinnedMemPool[stream], d->gaussianHandle, stride, width, height, d->intensity_multiplier, d->maxshared, d->streams[stream]);
                } else {
                    val = butterprocess<FLOAT>(NULL, 0, srcp1, srcp2, d->PinnedMemPool[stream], d->gaussianHandle, stride, width, height, d->intensity_multiplier, d->maxshared, d->streams[stream]);
                }
            } catch (const VshipError& e){
                vsapi->setFilterError(e.getErrorMessage().c_str(), frameCtx);
                d->streamSet->insert(stream);
                vsapi->freeFrame(src1);
                vsapi->freeFrame(src2);
                return NULL;
            }
            d->streamSet->insert(stream);
    
            vsapi->mapSetFloat(vsapi->getFramePropertiesRW(dst), "_BUTTERAUGLI_2Norm", std::get<0>(val), maReplace);
            vsapi->mapSetFloat(vsapi->getFramePropertiesRW(dst), "_BUTTERAUGLI_3Norm", std::get<1>(val), maReplace);
            vsapi->mapSetFloat(vsapi->getFramePropertiesRW(dst), "_BUTTERAUGLI_INFNorm", std::get<2>(val), maReplace);
    
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
    
        for (int i = 0; i < d->streamnum; i++){
            hipHostFree(d->PinnedMemPool[i]);
            hipStreamDestroy(d->streams[i]);
        }
        free(d->PinnedMemPool);
        free(d->streams);
        d->gaussianHandle.destroy();
        delete d->streamSet;
        //vsapi->setThreadCount(d->oldthreadnum, core);
    
        free(d);
    }
    
    // This function is responsible for validating arguments and creating a new filter  
    static void VS_CC butterCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
        ButterData d;
        ButterData *data;
    
        // Get a clip reference from the input arguments. This must be freed later.
        d.reference = toRGBS(vsapi->mapGetNode(in, "reference", 0, 0), core, vsapi);
        d.distorted = toRGBS(vsapi->mapGetNode(in, "distorted", 0, 0), core, vsapi);
        const VSVideoInfo *viref = vsapi->getVideoInfo(d.reference);
        const VSVideoInfo *vidis = vsapi->getVideoInfo(d.distorted);
        VSVideoFormat formatout;
        vsapi->queryVideoFormat(&formatout, cfGray, stFloat, 32, 0, 0, core);
        VSVideoInfo viout = *viref;
    
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
        d.intensity_multiplier = vsapi->mapGetFloat(in, "intensity_multiplier", 0, &error);
        if (error != peSuccess){
            d.intensity_multiplier = 80.0f;
        }
        int gpuid = vsapi->mapGetInt(in, "gpu_id", 0, &error);
        if (error != peSuccess){
            gpuid = 0;
        }
        d.diffmap = vsapi->mapGetInt(in, "distmap", 0, &error);
        if (error != peSuccess){
            d.diffmap = 0.;
        }
    
        if (d.diffmap){
            viout.format = formatout;
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
    
        try {
            d.gaussianHandle.init();
        } catch (const VshipError& e){
            vsapi->mapSetError(out, e.getErrorMessage().c_str());
            return;
        }
    
        d.streamnum = std::min(d.streamnum, infos.numThreads);
        d.streamnum = std::min(d.streamnum, (int)(devattr.totalGlobalMem/(31*4*viref->width*viref->height))); //VRAM overcommit partial protection.
        d.streamnum = std::max(d.streamnum, 1);
        d.streams = (hipStream_t*)malloc(sizeof(hipStream_t)*d.streamnum);
        for (int i = 0; i < d.streamnum; i++){
            hipStreamCreate(d.streams + i);
        }
    
        std::set<int> newstreamset;
        for (int i = 0; i < d.streamnum; i++){
            newstreamset.insert(i);
        }
        d.streamSet = new threadSet(newstreamset);
    
        const int pinnedsize = allocsizeScore(viref->width, viref->height);
        d.PinnedMemPool = (float**)malloc(sizeof(float*)*d.streamnum);
        hipError_t erralloc;
        for (int i = 0; i < d.streamnum; i++){
            erralloc = hipHostMalloc(d.PinnedMemPool+i, sizeof(float)*pinnedsize);
            if (erralloc != hipSuccess){
                vsapi->mapSetError(out, VshipError(OutOfRAM, __FILE__, __LINE__).getErrorMessage().c_str());
                vsapi->freeNode(d.reference);
                vsapi->freeNode(d.distorted);
                return;
            }
        }
    
        data = (ButterData *)malloc(sizeof(d));
        *data = d;
    
        for (int i = 0; i < d.streamnum; i++){
            data->streams[i] = d.streams[i];
        }
        data->maxshared = devattr.sharedMemPerBlock;
    
        VSFilterDependency deps[] = {{d.reference, rpStrictSpatial}, {d.distorted, rpStrictSpatial}};
        vsapi->createVideoFilter(out, "vship", &viout, butterGetFrame, butterFree, fmParallel, deps, 2, data, core);
    }
}