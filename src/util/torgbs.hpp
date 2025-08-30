//translated from zig from vapoursynth-zip. Thank you again julek

#ifndef TORGBSH
#define TORGBSH

#include "../util/preprocessor.hpp"

VSNode * toRGBS(VSNode * source, VSCore *core, const VSAPI *vsapi, bool mode16bit = false){
    const VSVideoInfo *vi = vsapi->getVideoInfo(source);

    if ((vi->format.bitsPerSample == ((mode16bit)? 16:32)) && (vi->format.colorFamily == cfRGB) && vi->format.sampleType == stFloat){
        return source;
    }

    const int matrix = (vi->height > 650)? 1 : 6;
    VSMap* args = vsapi->createMap();
    vsapi->mapConsumeNode(args, "clip", source, maReplace);
    vsapi->mapSetInt(args, "matrix_in", matrix, maReplace); //BT709 and the old one
    vsapi->mapSetInt(args, "transfer_in", 1, maReplace); //BT709
    vsapi->mapSetInt(args, "primaries_in", 1, maReplace); //BT709
    vsapi->mapSetInt(args, "range_in", 0, maReplace); //limited
    vsapi->mapSetInt(args, "matrix", 0, maReplace); //RGB
    vsapi->mapSetInt(args, "transfer", 1, maReplace); //BT709
    vsapi->mapSetInt(args, "primaries", 1, maReplace); //BT709/RGB primaries
    vsapi->mapSetInt(args, "range", 1, maReplace); //full range
    vsapi->mapSetInt(args, "format", mode16bit?pfRGBH:pfRGBS, maReplace);

    VSPlugin* vsplugin = vsapi->getPluginByID(VSH_RESIZE_PLUGIN_ID, core);
    VSMap* ret = vsapi->invoke(vsplugin, "Bicubic", args);
    VSNode* out = vsapi->mapGetNode(ret, "clip", 0, NULL);

    vsapi->freeMap(ret);
    vsapi->freeMap(args);

    return out;
}

#endif
