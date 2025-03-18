#ifndef GPUHELPERHPP
#define GPUHELPERHPP

#include "preprocessor.hpp"

namespace helper{

    static void VS_CC GpuInfo(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
        std::stringstream ss;
        int count, device;
        hipDeviceProp_t devattr;

        int error;
        int gpuid = vsapi->mapGetInt(in, "gpu_id", 0, &error);
        if (error != peSuccess){
            gpuid = 0;
        }

        if (hipGetDeviceCount(&count) != 0){
            ss << "Devices could not be detected, check permissions" << std::endl;
        } else if (count == 0) {
            ss << "We detected 0 devices" << std::endl;
        } else {
            for (int i = 0; i < count; i++){
                hipSetDevice(i);
                hipGetDevice(&device);
                hipGetDeviceProperties(&devattr, device);
                ss << "GPU " << i << " : " << devattr.name << std::endl;
            }
            if (error == peSuccess && gpuid < count){
                ss << "---------------------" << std::endl;
                hipSetDevice(gpuid);
                hipGetDevice(&device);
                hipGetDeviceProperties(&devattr, device);
                ss << "gpu_id : " << gpuid << " selected" << std::endl;
                ss << devattr.name << std::endl << std::endl;
                ss << "Global memory: " << ((float)devattr.totalGlobalMem / (1<<30)) << " GiB" << std::endl;
            } else if (gpuid >= count){ 
                ss << "Bad gpu_id selected" << std::endl;
            }
        }
        vsapi->mapSetData(out, "gpu_human_data", ss.str().data(), ss.str().size(), dtUtf8, maReplace);
    }
}

#endif