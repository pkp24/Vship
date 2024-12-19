//////////////////////////////////////////
// This file contains a simple invert
// filter that's commented to show
// the basics of the filter api.
// This file may make more sense when
// read from the bottom and up.

#include <stdlib.h>
#include <stdio.h>
#include "VapourSynth4.h"
#include "VSHelper4.h"

typedef struct {
    VSNode *reference;
    VSNode *distorted;
} Ssimulacra2Data;

// This is the main function that gets called when a frame should be produced. It will, in most cases, get
// called several times to produce one frame. This state is being kept track of by the value of
// activationReason. The first call to produce a certain frame n is always arInitial. In this state
// you should request all the input frames you need. Always do it in ascending order to play nice with the
// upstream filters.
// Once all frames are ready, the filter will be called with arAllFramesReady. It is now time to
// do the actual processing.
static const VSFrame *VS_CC ssimulacra2GetFrame(int n, int activationReason, void *instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    Ssimulacra2Data *d = (Ssimulacra2Data *)instanceData;

    if (activationReason == arInitial) {
        // Request the source frame on the first call
        vsapi->requestFrameFilter(n, d->reference, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame *src = vsapi->getFrameFilter(n, d->reference, frameCtx);
        // The reason we query this on a per frame basis is because we want our filter
        // to accept clips with varying dimensions. If we reject such content using d->vi
        // would be easier.
        const VSVideoFormat *fi = vsapi->getVideoFrameFormat(src);
        int height = vsapi->getFrameHeight(src, 0);
        int width = vsapi->getFrameWidth(src, 0);


        // When creating a new frame for output it is VERY EXTREMELY SUPER IMPORTANT to
        // supply the "dominant" source frame to copy properties from. Frame props
        // are an essential part of the filter chain and you should NEVER break it.
        VSFrame *dst = vsapi->newVideoFrame(fi, width, height, src, core);

        // It's processing loop time!
        // Loop over all the planes
        int plane;
        for (plane = 0; plane < fi->numPlanes; plane++) {
            const uint8_t * VS_RESTRICT srcp = vsapi->getReadPtr(src, plane);
            ptrdiff_t src_stride = vsapi->getStride(src, plane);
            uint8_t * VS_RESTRICT dstp = vsapi->getWritePtr(dst, plane);
            ptrdiff_t dst_stride = vsapi->getStride(dst, plane); // note that if a frame has the same dimensions and format, the stride is guaranteed to be the same. int dst_stride = src_stride would be fine too in this filter.
            // Since planes may be subsampled you have to query the height of them individually
            int h = vsapi->getFrameHeight(src, plane);
            int y;
            int w = vsapi->getFrameWidth(src, plane);
            int x;

            for (y = 0; y < h; y++) {
                for (x = 0; x < w; x++)
                    dstp[x] = ~srcp[x];

                dstp += dst_stride;
                srcp += src_stride;
            }
        }

        // Release the source frame
        vsapi->freeFrame(src);

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
    free(d);
}

// This function is responsible for validating arguments and creating a new filter  
static void VS_CC ssimulacra2Create(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    Ssimulacra2Data d;
    Ssimulacra2Data *data;
    int err;

    // Get a clip reference from the input arguments. This must be freed later.
    d.reference = vsapi->mapGetNode(in, "reference", 0, 0);
    d.distorted = vsapi->mapGetNode(in, "distorted", 0, 0);
    const VSVideoInfo *viref = vsapi->getVideoInfo(d.reference);
    const VSVideoInfo *vidis = vsapi->getVideoInfo(d.distorted);

    if (!(vsh::isSameVideoInfo(viref, vidis))){
        vsapi->mapSetError(out, "SSIMULACRA2: both clips must have the same format and dimensions");
        vsapi->freeNode(d.reference);
        vsapi->freeNode(d.distorted);
        return;
    }

    if ((viref->format.colorFamily != cfRGB) || viref->format.sampleType != stFloat){
        vsapi->mapSetError(out, "SSIMULACRA2: only works with RGBS format");
        vsapi->freeNode(d.reference);
        vsapi->freeNode(d.distorted);
        return;
    }

    data = (Ssimulacra2Data *)malloc(sizeof(d));
    *data = d;

    VSFilterDependency deps[] = {{d.reference, rpStrictSpatial}, {d.distorted, rpStrictSpatial}};

    vsapi->createVideoFilter(out, "vshipssimu2", viref, ssimulacra2GetFrame, ssimulacra2Free, fmParallel, deps, 2, data, core);
}


VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("com.lumen.vshipssimu2", "vshipssimu2", "VapourSynth SSIMULACRA2 on GPU", VS_MAKE_VERSION(3, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("SSIMULACRA2", "reference:vnode;distorted:vnode;", "clip:vnode;", ssimulacra2Create, NULL, plugin);
}
