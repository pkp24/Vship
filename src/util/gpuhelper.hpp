#ifndef GPUHELPERHPP
#define GPUHELPERHPP

#include "preprocessor.hpp"
#include "VshipExceptions.hpp"

//here is the format of the answer:

//case where gpu_id is not specified:

//GPU 0: {GPU Name}
//...

//case where gpu_id is specified:

//Name: {GPU Name string}
//MultiProcessorCount: {multiprocessor count integer}
//ClockRate: {clockRate float} Ghz
//MaxSharedMemoryPerBlock: {Max Shared Memory Per Block integer} bytes
//WarpSize: {Warp Size integer}
//VRAMCapacity: {GPU VRAM Capacity float} GB
//MemoryBusWidth: {memory bus width integer} bits
//MemoryClockRate: {memory clock rate float} Ghz
//Integrated: {0|1}

namespace helper{

    int checkGpuCount(){
        int count;
        if (hipGetDeviceCount(&count) != 0){
            throw VshipError(DeviceCountError, __FILE__, __LINE__);
        };
        if (count == 0){
            throw VshipError(NoDeviceDetected, __FILE__, __LINE__);
        }
        return count;
    }

    static void VS_CC GpuInfo(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
        std::stringstream ss;
        int count, device;
        hipDeviceProp_t devattr;

        //we don't need a full check at that point
        try{
            count = checkGpuCount();
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
            ss << "ClockRate: " << ((float)devattr.clockRate)/1000000 << " Ghz" << std::endl;
            ss << "MaxSharedMemoryPerBlock: " << devattr.sharedMemPerBlock << " bytes" << std::endl;
            ss << "WarpSize: " << devattr.warpSize << std::endl;
            ss << "VRAMCapacity: " << ((float)devattr.totalGlobalMem)/1000000000 << " GB" << std::endl;
            ss << "MemoryBusWidth: " << devattr.memoryBusWidth << " bits" << std::endl;
            ss << "MemoryClockRate: " << ((float)devattr.memoryClockRate)/1000000 << " Ghz" << std::endl;
            ss << "Integrated: " << devattr.integrated << std::endl;
        }
        vsapi->mapSetData(out, "gpu_human_data", ss.str().data(), ss.str().size(), dtUtf8, maReplace);
    }

    __global__ void kernelTest(int* inputtest){
        inputtest[0] = 4320984;
    }

    bool gpuKernelCheck(){
        int inputtest = 0;
        int* inputtest_d;
        hipMalloc(&inputtest_d, sizeof(int)*1);
        hipMemset(inputtest_d, 0, sizeof(int)*1);
        kernelTest<<<dim3(1), dim3(1), 0, 0>>>(inputtest_d);
        hipMemcpyDtoH(&inputtest, inputtest_d, sizeof(int));
        hipFree(inputtest_d);
        return (inputtest == 4320984);
    }

    void gpuFullCheck(int gpuid = 0){
        int count = checkGpuCount();

        if (count <= gpuid || gpuid < 0){
            throw VshipError(BadDeviceArgument, __FILE__, __LINE__);
        }
        hipSetDevice(gpuid);
        if (!gpuKernelCheck()){
            throw VshipError(BadDeviceCode, __FILE__, __LINE__);
        }
    }

}

#endif