#include "VapourSynth4.h"
#include "VSHelper4.h"
#include "butter/vapoursynth.hpp"
#include "ssimu2/vapoursynth.hpp"
#include "cvvdp/vapoursynth.hpp"
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
    vspapi->configPlugin("com.lumen.vship", "vship", "VapourSynth GPU Perceptual Metrics", VS_MAKE_VERSION(3, 3), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("SSIMULACRA2", "reference:vnode;distorted:vnode;numStream:int:opt;gpu_id:int:opt;", "clip:vnode;", ssimu2::ssimulacra2Create, NULL, plugin);
    vspapi->registerFunction("BUTTERAUGLI", "reference:vnode;distorted:vnode;qnorm:int:opt;intensity_multiplier:float:opt;distmap:int:opt;numStream:int:opt;gpu_id:int:opt;", "clip:vnode;", butter::butterCreate, NULL, plugin);
    vspapi->registerFunction("CVVDP", "reference:vnode;distorted:vnode;display:data:opt;numStream:int:opt;gpu_id:int:opt;", "clip:vnode;", cvvdp::cvvdpCreate, NULL, plugin);
    vspapi->registerFunction("GpuInfo", "gpu_id:int:opt;", "gpu_human_data:data;", GpuInfo, NULL, plugin);
}

//let's define the API
#define EXPORTVSHIPLIB //to use dllexport for windows
#include "VshipAPI.h"

extern "C"{
Vship_Version Vship_GetVersion(){
    Vship_Version res;
    res.major = 3; res.minor = 1; res.minorMinor = 0;
    #if defined __CUDACC__
    res.backend = Vship_Cuda;
    #else
    res.backend = Vship_HIP;
    #endif
    return res;
}

Vship_Exception Vship_GetDeviceCount(int* number){
    int count;
    try{
        count = helper::checkGpuCount();
    } catch (const VshipError& e){
        return (Vship_Exception)e.type;
    }
    *number = count;
    return Vship_NoError;
}

Vship_Exception Vship_GetDeviceInfo(Vship_DeviceInfo* device_info, int gpu_id){
    int count;
    try{
        count = helper::checkGpuCount();
    } catch (const VshipError& e){
        return (Vship_Exception)e.type;
    }
    if (gpu_id >= count){
        return Vship_BadDeviceArgument;
    }
    hipDeviceProp_t devattr;
    hipGetDeviceProperties(&devattr, gpu_id);
    memcpy(device_info->name, devattr.name, 256); //256 char to copy
    device_info->VRAMSize = devattr.totalGlobalMem;
    device_info->integrated = devattr.integrated;
    device_info->MultiProcessorCount = devattr.multiProcessorCount;
    device_info->WarpSize = devattr.warpSize;
    return Vship_NoError;
}

Vship_Exception Vship_GPUFullCheck(int gpu_id){
    try{
        helper::gpuFullCheck(gpu_id);
    } catch (const VshipError& e){
        return (Vship_Exception)e.type;
    }
    return Vship_NoError;
}

int Vship_GetErrorMessage(Vship_Exception exception, char* out_message, int len){
    std::string cppstr = errorMessage((VSHIPEXCEPTTYPE)exception);
    if (len == 0) return cppstr.size()+1; //required size to fit the whole message
    memcpy(out_message, cppstr.c_str(), std::min(len-1, (int)cppstr.size()));
    out_message[len-1] = '\0'; //end character
    return cppstr.size()+1;
}

Vship_Exception Vship_SetDevice(int gpu_id){
    int numgpu;
    Vship_Exception errcount = Vship_GetDeviceCount(&numgpu);
    if (errcount != Vship_NoError){
        return errcount;
    }
    if (gpu_id >= numgpu){
        return Vship_BadDeviceArgument;
    }
    hipSetDevice(gpu_id);
    return Vship_NoError;
}

Vship_Exception Vship_PinnedMalloc(void** ptr, uint64_t size){
    hipError_t erralloc = hipHostMalloc(ptr, size);
    if (erralloc != hipSuccess){
        return Vship_OutOfRAM;
    }
    return Vship_NoError;
}

Vship_Exception Vship_PinnedFree(void* ptr){
    hipError_t err = hipHostFree(ptr);
    if (err != hipSuccess){
        return Vship_BadPointer;
    }
    return Vship_NoError;
}

RessourceManager<ssimu2::SSIMU2ComputingImplementation> HandlerManagerSSIMU2;
RessourceManager<butter::ButterComputingImplementation> HandlerManagerButteraugli;

Vship_Exception Vship_SSIMU2Init(Vship_SSIMU2Handler* handler, int width, int height){
    Vship_Exception err = Vship_NoError;
    handler->id = HandlerManagerSSIMU2.allocate();
    HandlerManagerSSIMU2.lock.lock();
    try{
        HandlerManagerSSIMU2.elements[handler->id].init(width, height);
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    HandlerManagerSSIMU2.lock.unlock();
    return err;
}

Vship_Exception Vship_SSIMU2Free(Vship_SSIMU2Handler handler){
    Vship_Exception err = Vship_NoError;
    HandlerManagerSSIMU2.lock.lock();
    if (handler.id >= HandlerManagerSSIMU2.elements.size()){
        HandlerManagerSSIMU2.lock.unlock();
        return Vship_BadHandler;
    }
    try{
        HandlerManagerSSIMU2.elements[handler.id].destroy();
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    HandlerManagerSSIMU2.lock.unlock();
    HandlerManagerSSIMU2.free(handler.id);
    return err;
}

Vship_Exception Vship_ComputeSSIMU2Float(Vship_SSIMU2Handler handler, double* score, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride){
    Vship_Exception err = Vship_NoError;
    HandlerManagerSSIMU2.lock.lock();
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    ssimu2::SSIMU2ComputingImplementation ssimu2computingimplem = HandlerManagerSSIMU2.elements[handler.id];
    HandlerManagerSSIMU2.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        *score = ssimu2computingimplem.run<InputMemType::FLOAT>(srcp1, srcp2, stride, stride);
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    return err;
}

Vship_Exception Vship_ComputeSSIMU2Floatv2(Vship_SSIMU2Handler handler, double* score, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride, int64_t stride2){
    Vship_Exception err = Vship_NoError;
    HandlerManagerSSIMU2.lock.lock();
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    ssimu2::SSIMU2ComputingImplementation ssimu2computingimplem = HandlerManagerSSIMU2.elements[handler.id];
    HandlerManagerSSIMU2.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        *score = ssimu2computingimplem.run<FLOAT>(srcp1, srcp2, stride, stride2);
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    return err;
}

Vship_Exception Vship_ComputeSSIMU2Uint16(Vship_SSIMU2Handler handler, double* score, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride){
    Vship_Exception err = Vship_NoError;
    HandlerManagerSSIMU2.lock.lock();
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    ssimu2::SSIMU2ComputingImplementation ssimu2computingimplem = HandlerManagerSSIMU2.elements[handler.id];
    HandlerManagerSSIMU2.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        *score = ssimu2computingimplem.run<InputMemType::UINT16>(srcp1, srcp2, stride, stride);
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    return err;
}

Vship_Exception Vship_ComputeSSIMU2Uint16v2(Vship_SSIMU2Handler handler, double* score, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride, int64_t stride2){
    Vship_Exception err = Vship_NoError;
    HandlerManagerSSIMU2.lock.lock();
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    ssimu2::SSIMU2ComputingImplementation ssimu2computingimplem = HandlerManagerSSIMU2.elements[handler.id];
    HandlerManagerSSIMU2.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        *score = ssimu2computingimplem.run<UINT16>(srcp1, srcp2, stride, stride2);
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    return err;
}

Vship_Exception Vship_ButteraugliInit(Vship_ButteraugliHandler* handler, int width, int height, float intensity_multiplier){
    Vship_Exception err = Vship_NoError;
    handler->id = HandlerManagerButteraugli.allocate();
    HandlerManagerButteraugli.lock.lock();
    try{
        //Qnorm = 2 by default to mimic old behavior
        HandlerManagerButteraugli.elements[handler->id].init(width, height, 2, intensity_multiplier);
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    HandlerManagerButteraugli.lock.unlock();
    return err;
}

//allows specifying Qnorm
Vship_Exception Vship_ButteraugliInitv2(Vship_ButteraugliHandler* handler, int width, int height, int Qnorm, float intensity_multiplier){
    Vship_Exception err = Vship_NoError;
    handler->id = HandlerManagerButteraugli.allocate();
    HandlerManagerButteraugli.lock.lock();
    try{
        HandlerManagerButteraugli.elements[handler->id].init(width, height, Qnorm, intensity_multiplier);
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    HandlerManagerButteraugli.lock.unlock();
    return err;
}

Vship_Exception Vship_ButteraugliFree(Vship_ButteraugliHandler handler){
    Vship_Exception err = Vship_NoError;
    HandlerManagerButteraugli.lock.lock();
    if (handler.id >= HandlerManagerButteraugli.elements.size()){
        HandlerManagerButteraugli.lock.unlock();
        return Vship_BadHandler;
    }
    try{
        HandlerManagerButteraugli.elements[handler.id].destroy();
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    HandlerManagerButteraugli.lock.unlock();
    HandlerManagerButteraugli.free(handler.id);
    return err;
}

Vship_Exception Vship_ComputeButteraugliFloat(Vship_ButteraugliHandler handler, Vship_ButteraugliScore* score, const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride){
    Vship_Exception err = Vship_NoError;
    HandlerManagerButteraugli.lock.lock();
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    butter::ButterComputingImplementation buttercomputingimplem = HandlerManagerButteraugli.elements[handler.id];
    HandlerManagerButteraugli.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        std::tuple<float, float, float> res = buttercomputingimplem.run<InputMemType::FLOAT>(dstp, dststride, srcp1, srcp2, stride, stride);
        score->norm2 = std::get<0>(res);
        score->norm3 = std::get<1>(res);
        score->norminf = std::get<2>(res);
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    return err;
}

Vship_Exception Vship_ComputeButteraugliFloatv2(Vship_ButteraugliHandler handler, Vship_ButteraugliScore* score, const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride, int64_t stride2){
    Vship_Exception err = Vship_NoError;
    HandlerManagerButteraugli.lock.lock();
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    butter::ButterComputingImplementation buttercomputingimplem = HandlerManagerButteraugli.elements[handler.id];
    HandlerManagerButteraugli.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        std::tuple<float, float, float> res = buttercomputingimplem.run<FLOAT>(dstp, dststride, srcp1, srcp2, stride, stride2);
        score->norm2 = std::get<0>(res);
        score->norm3 = std::get<1>(res);
        score->norminf = std::get<2>(res);
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    return err;
}

Vship_Exception Vship_ComputeButteraugliUint16(Vship_ButteraugliHandler handler, Vship_ButteraugliScore* score, const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride){
    Vship_Exception err = Vship_NoError;
    HandlerManagerButteraugli.lock.lock();
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    butter::ButterComputingImplementation buttercomputingimplem = HandlerManagerButteraugli.elements[handler.id];
    HandlerManagerButteraugli.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        std::tuple<float, float, float> res = buttercomputingimplem.run<InputMemType::UINT16>(dstp, dststride, srcp1, srcp2, stride, stride);
        score->norm2 = std::get<0>(res);
        score->norm3 = std::get<1>(res);
        score->norminf = std::get<2>(res);
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    return err;
}

Vship_Exception Vship_ComputeButteraugliUint16v2(Vship_ButteraugliHandler handler, Vship_ButteraugliScore* score, const uint8_t *dstp, int64_t dststride, const uint8_t* srcp1[3], const uint8_t* srcp2[3], int64_t stride, int64_t stride2){
    Vship_Exception err = Vship_NoError;
    HandlerManagerButteraugli.lock.lock();
    //we have this value by copy to be able to run with the mutex unlocked, the pointer could be invalidated if the vector was to change size
    butter::ButterComputingImplementation buttercomputingimplem = HandlerManagerButteraugli.elements[handler.id];
    HandlerManagerButteraugli.lock.unlock();
    //there is no safety feature to prevent using twice at the same time a single computingimplem
    try{
        std::tuple<float, float, float> res = buttercomputingimplem.run<UINT16>(dstp, dststride, srcp1, srcp2, stride, stride2);
        score->norm2 = std::get<0>(res);
        score->norm3 = std::get<1>(res);
        score->norminf = std::get<2>(res);
    } catch (const VshipError& e){
        err = (Vship_Exception)e.type;
    }
    return err;
}

} //extern "C"