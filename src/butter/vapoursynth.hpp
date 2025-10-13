#pragma once

#include "../util/torgbs.hpp"
#include "main.hpp"

namespace butter{
    typedef struct ButterData{
        VSNode *reference;
        VSNode *distorted;
        ButterComputingImplementation* butterStreams;
        int diffmap;
        int streamnum = 0;
        threadSet<int>* streamSet;
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
            int stride2 = vsapi->getStride(src2, 0);
    
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
            ButterComputingImplementation& butterstream = d->butterStreams[stream];
            try{
                if (d->diffmap){
                    val = butterstream.run<FLOAT>(vsapi->getWritePtr(dst, 0), vsapi->getStride(dst, 0), srcp1, srcp2, stride, stride2);
                } else {
                    val = butterstream.run<FLOAT>(NULL, 0, srcp1, srcp2, stride, stride2);
                }
            } catch (const VshipError& e){
                vsapi->setFilterError(e.getErrorMessage().c_str(), frameCtx);
                d->streamSet->insert(stream);
                vsapi->freeFrame(src1);
                vsapi->freeFrame(src2);
                return NULL;
            }
            d->streamSet->insert(stream);
    
            vsapi->mapSetFloat(vsapi->getFramePropertiesRW(dst), "_BUTTERAUGLI_QNorm", std::get<0>(val), maReplace);
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
            d->butterStreams[i].destroy();
        }
        free(d->butterStreams);
        delete d->streamSet;
    
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
        float intensity_multiplier = vsapi->mapGetFloat(in, "intensity_multiplier", 0, &error);
        if (error != peSuccess){
            intensity_multiplier = 203.0f;
        }
        int gpuid = vsapi->mapGetInt(in, "gpu_id", 0, &error);
        if (error != peSuccess){
            gpuid = 0;
        }
        d.diffmap = vsapi->mapGetInt(in, "distmap", 0, &error);
        if (error != peSuccess){
            d.diffmap = 0.;
        }
        int Qnorm = vsapi->mapGetInt(in, "qnorm", 0, &error);
        if (error != peSuccess){
            Qnorm = 2;
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
    
        d.streamnum = vsapi->mapGetInt(in, "numStream", 0, &error);
        if (error != peSuccess){
            d.streamnum = 4;
        }

        VSCoreInfo infos;
        vsapi->getCoreInfo(core, &infos);
    
        d.streamnum = std::min(d.streamnum, infos.numThreads);
        d.streamnum = std::max(d.streamnum, 1);
    
        std::set<int> newstreamset;
        for (int i = 0; i < d.streamnum; i++){
            newstreamset.insert(i);
        }
        d.streamSet = new threadSet(newstreamset);
    
        data = (ButterData *)malloc(sizeof(d));
        *data = d;

        try{
            data->butterStreams = (ButterComputingImplementation*)malloc(sizeof(ButterComputingImplementation)*d.streamnum);
            if (data->butterStreams == NULL) throw VshipError(OutOfRAM, __FILE__, __LINE__);
            for (int i = 0; i < d.streamnum; i++){
                data->butterStreams[i].init(viref->width, viref->height, Qnorm, intensity_multiplier);
            }
        } catch (const VshipError& e){
            vsapi->mapSetError(out, e.getErrorMessage().c_str());
            return;
        }
    
        VSFilterDependency deps[] = {{d.reference, rpStrictSpatial}, {d.distorted, rpStrictSpatial}};
        vsapi->createVideoFilter(out, "vship", &viout, butterGetFrame, butterFree, fmParallel, deps, 2, data, core);
    }
}