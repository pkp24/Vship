double butterprocess(const uint8_t *srcp1[3], const uint8_t *srcp2[3], int stride, int width, int height, float* gaussiankernel, int maxshared, hipStream_t stream){
    return 0.;
}

typedef struct {
    VSNode *reference;
    VSNode *distorted;
    int maxshared;
    float* gaussiankernel_d;
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

        const double val = butterprocess(srcp1, srcp2, stride, width, height, d->gaussiankernel_d, d->maxshared, d->streams[n%STREAMNUM]);

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
    hipFree(d->gaussiankernel_d);
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

    float gaussiankernel[2*GAUSSIANSIZE+1];
    for (int i = 0; i < 2*GAUSSIANSIZE+1; i++){
        gaussiankernel[i] = std::exp(-(GAUSSIANSIZE-i)*(GAUSSIANSIZE-i)/(2*SIGMA*SIGMA))/(std::sqrt(2*PI*SIGMA*SIGMA));
    }

    data->maxshared = devattr.sharedMemPerBlock;    

    hipMalloc(&(data->gaussiankernel_d), sizeof(float)*(2*GAUSSIANSIZE+1));
    hipMemcpyHtoD((hipDeviceptr_t)data->gaussiankernel_d, gaussiankernel, (2*GAUSSIANSIZE+1)*sizeof(float));

    VSFilterDependency deps[] = {{d.reference, rpStrictSpatial}, {d.distorted, rpStrictSpatial}};

    vsapi->createVideoFilter(out, "vship", viref, butterGetFrame, butterFree, fmParallel, deps, 2, data, core);
}