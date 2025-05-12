//translated from zig from vapoursynth-zip. Thank you again julek

#ifndef TORGBSH
#define TORGBSH

#include "../util/preprocessor.hpp"

VSNode * toRGBS(VSNode * source, VSCore *core, const VSAPI *vsapi){
    const VSVideoInfo *vi = vsapi->getVideoInfo(source);

    if ((vi->format.bitsPerSample == 32) && (vi->format.colorFamily == cfRGB) && vi->format.sampleType == stFloat){
        return source;
    }

    const int matrix = (vi->height > 650)? 1 : 6;
    VSMap* args = vsapi->createMap();
    vsapi->mapConsumeNode(args, "clip", source, maReplace);
    vsapi->mapSetInt(args, "matrix_in", matrix, maReplace);
    vsapi->mapSetInt(args, "format", pfRGBS, maReplace);

    VSPlugin* vsplugin = vsapi->getPluginByID(VSH_RESIZE_PLUGIN_ID, core);
    VSMap* ret = vsapi->invoke(vsplugin, "Bicubic", args);
    VSNode* out = vsapi->mapGetNode(ret, "clip", 0, NULL);

    vsapi->freeMap(ret);
    vsapi->freeMap(args);

    return out;
}

#endif
