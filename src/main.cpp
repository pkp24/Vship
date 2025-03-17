#include "butter/main.hpp"
#include "ssimu2/main.hpp"
#include "util/gpuhelper.hpp"

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("com.lumen.vship", "vship", "VapourSynth SSIMULACRA2 on GPU", VS_MAKE_VERSION(3, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("SSIMULACRA2", "reference:vnode;distorted:vnode;gpu_id:int:opt", "clip:vnode;", ssimu2::ssimulacra2Create, NULL, plugin);
    vspapi->registerFunction("BUTTERAUGLI", "reference:vnode;distorted:vnode;intensity_multiplier:float:opt;distmap:int:opt;gpu_id:int:opt", "clip:vnode;", butter::butterCreate, NULL, plugin);
    vspapi->registerFunction("GpuInfo", "gpu_id:int:opt;", "gpu_human_data:data;", helper::GpuInfo, NULL, plugin);
}
