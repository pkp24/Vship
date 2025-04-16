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
//PassKernelCheck : {0|1}

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