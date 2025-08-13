#include "VapourSynth4.h"
#include "VSHelper4.h"
#include "butter/vapoursynth.hpp"
#include "ssimu2/vapoursynth.hpp"
#include "util/gpuhelper.hpp"

static void VS_CC GpuInfo(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    std::stringstream ss;
    int count, device;
    hipDeviceProp_t devattr;

    //we don't need a full check at that point
    try{
        count = helper::checkGpuCount();
    } catch (const VshipError& e){
        vsapi->mapSetError(out, e.getErrorMessage().c_str());
        return;
    }

    int error;
    int gpuid = vsapi->mapGetInt(in, "gpu_id", 0, &error);
    if (error != peSuccess){
        gpuid = 0;
    }
    
    if (count <= gpuid || gpuid < 0){
        vsapi->mapSetError(out, VshipError(BadDeviceArgument, __FILE__, __LINE__).getErrorMessage().c_str());
        return;
    }

    if (error != peSuccess){
        //no gpu_id was selected
        for (int i = 0; i < count; i++){
            hipSetDevice(i);
            hipGetDevice(&device);
            hipGetDeviceProperties(&devattr, device);
            ss << "GPU " << i << ": " << devattr.name << std::endl;
        }
    } else {
        hipSetDevice(gpuid);
        hipGetDevice(&device);
        hipGetDeviceProperties(&devattr, device);
        ss << "Name: " << devattr.name << std::endl;
        ss << "MultiProcessorCount: " << devattr.multiProcessorCount << std::endl;
        //ss << "ClockRate: " << ((float)devattr.clockRate)/1000000 << " Ghz" << std::endl; deprecated, removed in cuda 13
        ss << "MaxSharedMemoryPerBlock: " << devattr.sharedMemPerBlock << " bytes" << std::endl;
        ss << "WarpSize: " << devattr.warpSize << std::endl;
        ss << "VRAMCapacity: " << ((float)devattr.totalGlobalMem)/1000000000 << " GB" << std::endl;
        ss << "MemoryBusWidth: " << devattr.memoryBusWidth << " bits" << std::endl;
        //ss << "MemoryClockRate: " << ((float)devattr.memoryClockRate)/1000000 << " Ghz" << std::endl; deprecated, removed in cuda13
        ss << "Integrated: " << devattr.integrated << std::endl;
        ss << "PassKernelCheck : " << (int)helper::gpuKernelCheck() << std::endl;
    }
    vsapi->mapSetData(out, "gpu_human_data", ss.str().data(), ss.str().size(), dtUtf8, maReplace);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
    vspapi->configPlugin("com.lumen.vship", "vship", "VapourSynth SSIMULACRA2 on GPU", VS_MAKE_VERSION(3, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("SSIMULACRA2", "reference:vnode;distorted:vnode;numStream:int:opt;gpu_id:int:opt;", "clip:vnode;", ssimu2::ssimulacra2Create, NULL, plugin);
    vspapi->registerFunction("BUTTERAUGLI", "reference:vnode;distorted:vnode;intensity_multiplier:float:opt;distmap:int:opt;numStream:int:opt;gpu_id:int:opt;", "clip:vnode;", butter::butterCreate, NULL, plugin);
    vspapi->registerFunction("GpuInfo", "gpu_id:int:opt;", "gpu_human_data:data;", GpuInfo, NULL, plugin);
}
