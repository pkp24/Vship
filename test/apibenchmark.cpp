#include "VshipAPI.h"
#include<iostream>
#include<cstdint>
#include<chrono>

template <typename T>
class Image{
private:
    uint8_t* srcp[3] = {NULL, NULL, NULL};
public:
    const uint8_t* csrcp[3];
    uint64_t width;
    uint64_t height;
    uint64_t stride;
    Image(uint64_t width, uint64_t height, uint64_t stride) : width(width), height(height), stride(stride){
        srcp[0] = (uint8_t*)malloc(stride*height*3); //stride takes into account type of T
        srcp[1] = srcp[0] + stride*height;
        srcp[2] = srcp[1] + stride*height;
        csrcp[0] = srcp[0];
        csrcp[1] = srcp[1];
        csrcp[2] = srcp[2];
    }
    void fillZero(){
        for (uint64_t i = 0; i < stride*height*3; i++){
            srcp[0][i] = 0;
        }
    }
    void fillZeroStrideOne(){
        fillZero();
        for (uint64_t j = 0; j < 3*height; j++){
            for (uint64_t i = width*sizeof(T); i < stride; i++){
                srcp[0][j*stride+i] = (uint8_t)255;
            }
        }
    }
    ~Image(){
        if (srcp[0]) free(srcp[0]);
        srcp[0] = NULL;
    }
};

int main(){
    std::cout << "Tests performed at 1080p" << std::endl;
    int width = 1920; int height = 1080; int stride = 2048;
    Image<uint16_t> original(width, height, stride);
    Image<uint16_t> other(width, height, stride);
    original.fillZeroStrideOne();
    other.fillZeroStrideOne();

    Vship_Version apiver = Vship_GetVersion();
    std::cout << "Vship version: " << apiver.major << "." << apiver.minor << "." << apiver.minorMinor << std::endl;
    switch (apiver.backend){
        case Vship_HIP:
            std::cout << "Running on HIP" << std::endl;
            break;
        case Vship_Cuda:
            std::cout << "Running on Cuda" << std::endl;
            break;
    }

    //first initilization time measure
    std::cout << "--------------------API-Testing-----------------------" << std::endl;
    int numgpu;
    auto time1 = std::chrono::system_clock::now();
    Vship_GetDeviceCount(&numgpu);
    auto time2 = std::chrono::system_clock::now();
    Vship_GetDeviceCount(&numgpu);
    auto time3 = std::chrono::system_clock::now();
    std::cout << "Cuda/HIP initialization time: " << (double)((time2 - time1).count() - (time3 - time2).count())/1000000000. << "s" << std::endl;
    std::cout << "Check Device Count time: " << (time3 - time2).count()/1000000000. << "s" << std::endl;
    time1 = std::chrono::system_clock::now();
    Vship_GPUFullCheck(0);
    time2 = std::chrono::system_clock::now();
    std::cout << "GPUFullCheck time: " << (time2 - time1).count()/1000000000. << "s" << std::endl;

    std::cout << "------------------SSIMU2-Testing-----------------------" << std::endl;
    //SSIMU2 testing
    Vship_SSIMU2Handler ssimu2handler;
    Vship_SSIMU2Handler ssimu2handler2;
    Vship_SSIMU2Handler ssimu2handler3;
    time1 = std::chrono::system_clock::now();
    Vship_SSIMU2Init(&ssimu2handler, width, height);
    time2 = std::chrono::system_clock::now();
    Vship_SSIMU2Init(&ssimu2handler2, width, height);
    Vship_SSIMU2Init(&ssimu2handler3, width, height);
    time3 = std::chrono::system_clock::now();
    std::cout << "Preprocess SSIMU2 Handler time ((latency + 2*process)/2): " << (time3 - time2).count()/2000000000. << "s" << std::endl;
    std::cout << "Preprocess SSIMU2 Handler time (latency + process): " << (time2 - time1).count()/1000000000. << "s" << std::endl;
    std::cout << "=> handler preprocess \"process time\": " << (time3 - time2).count()/1000000000. - (time2 - time1).count()/1000000000. << "s" << std::endl;
    std::cout << "=> handler preprocess \"latency time\": " << (time2 - time3).count()/1000000000. + 2*(time2 - time1).count()/1000000000. << "s" << std::endl;

    double score;
    time1 = std::chrono::system_clock::now();
    Vship_ComputeSSIMU2Uint16(&ssimu2handler, &score, original.csrcp, other.csrcp, stride);
    time2 = std::chrono::system_clock::now();
    Vship_ComputeSSIMU2Uint16(&ssimu2handler, &score, original.csrcp, other.csrcp, stride);
    Vship_ComputeSSIMU2Uint16(&ssimu2handler, &score, original.csrcp, other.csrcp, stride);
    time3 = std::chrono::system_clock::now();
    std::cout << "SSIMU2 Process Uint16 time ((latency + 2*process)/2): " << (time3 - time2).count()/2000000000. << "s" << std::endl;
    std::cout << "SSIMU2 Process Uint16 time (latency + process): " << (time2 - time1).count()/1000000000. << "s" << std::endl;
    std::cout << "=> SSIMU2 Process Uint16 \"process time\": " << (time3 - time2).count()/1000000000. - (time2 - time1).count()/1000000000. << "s" << std::endl;
    std::cout << "=> SSIMU2 Process Uint16 \"latency time\": " << (time2 - time3).count()/1000000000. + 2*(time2 - time1).count()/1000000000. << "s" << std::endl;

    time1 = std::chrono::system_clock::now();
    Vship_SSIMU2Free(&ssimu2handler);
    time2 = std::chrono::system_clock::now();
    Vship_SSIMU2Free(&ssimu2handler2);
    Vship_SSIMU2Free(&ssimu2handler3);
    time3 = std::chrono::system_clock::now();
    std::cout << "Free SSIMU2 Handler time ((latency + 2*process)/2): " << (time3 - time2).count()/2000000000. << "s" << std::endl;
    std::cout << "Free SSIMU2 Handler time (latency + process): " << (time2 - time1).count()/1000000000. << "s" << std::endl;
    std::cout << "=> handler Free \"process time\": " << (time3 - time2).count()/1000000000. - (time2 - time1).count()/1000000000. << "s" << std::endl;
    std::cout << "=> handler Free \"latency time\": " << (time2 - time3).count()/1000000000. + 2*(time2 - time1).count()/1000000000. << "s" << std::endl;


    std::cout << "---------------Butteraugli-Testing-----------------------" << std::endl;
    //Butteraugli testing
    Vship_ButteraugliHandler butterhandler;
    Vship_ButteraugliHandler butterhandler2;
    Vship_ButteraugliHandler butterhandler3;
    time1 = std::chrono::system_clock::now();
    Vship_ButteraugliInit(&butterhandler, width, height, 80.);
    time2 = std::chrono::system_clock::now();
    Vship_ButteraugliInit(&butterhandler2, width, height, 80.);
    Vship_ButteraugliInit(&butterhandler3, width, height, 80.);
    time3 = std::chrono::system_clock::now();
    std::cout << "Preprocess Butteraugli Handler time ((latency + 2*process)/2): " << (time3 - time2).count()/2000000000. << "s" << std::endl;
    std::cout << "Preprocess Butteraugli Handler time (latency + process): " << (time2 - time1).count()/1000000000. << "s" << std::endl;
    std::cout << "=> handler preprocess \"process time\": " << (time3 - time2).count()/1000000000. - (time2 - time1).count()/1000000000. << "s" << std::endl;
    std::cout << "=> handler preprocess \"latency time\": " << (time2 - time3).count()/1000000000. + 2*(time2 - time1).count()/1000000000. << "s" << std::endl;

    Vship_ButteraugliScore scorebutter;
    time1 = std::chrono::system_clock::now();
    Vship_ComputeButteraugliUint16(&butterhandler, &scorebutter, NULL, 0, original.csrcp, other.csrcp, stride);
    time2 = std::chrono::system_clock::now();
    Vship_ComputeButteraugliUint16(&butterhandler, &scorebutter, NULL, 0, original.csrcp, other.csrcp, stride);
    Vship_ComputeButteraugliUint16(&butterhandler, &scorebutter, NULL, 0, original.csrcp, other.csrcp, stride);
    time3 = std::chrono::system_clock::now();
    std::cout << "Butteraugli Process Uint16 time ((latency + 2*process)/2): " << (time3 - time2).count()/2000000000. << "s" << std::endl;
    std::cout << "Butteraugli Process Uint16 time (latency + process): " << (time2 - time1).count()/1000000000. << "s" << std::endl;
    std::cout << "=> Butteraugli Process Uint16 \"process time\": " << (time3 - time2).count()/1000000000. - (time2 - time1).count()/1000000000. << "s" << std::endl;
    std::cout << "=> Butteraugli Process Uint16 \"latency time\": " << (time2 - time3).count()/1000000000. + 2*(time2 - time1).count()/1000000000. << "s" << std::endl;

    time1 = std::chrono::system_clock::now();
    Vship_ButteraugliFree(&butterhandler);
    time2 = std::chrono::system_clock::now();
    Vship_ButteraugliFree(&butterhandler2);
    Vship_ButteraugliFree(&butterhandler3);
    time3 = std::chrono::system_clock::now();
    std::cout << "Free Butteraugli Handler time ((latency + 2*process)/2): " << (time3 - time2).count()/2000000000. << "s" << std::endl;
    std::cout << "Free Butteraugli Handler time (latency + process): " << (time2 - time1).count()/1000000000. << "s" << std::endl;
    std::cout << "=> handler Free \"process time\": " << (time3 - time2).count()/1000000000. - (time2 - time1).count()/1000000000. << "s" << std::endl;
    std::cout << "=> handler Free \"latency time\": " << (time2 - time3).count()/1000000000. + 2*(time2 - time1).count()/1000000000. << "s" << std::endl;
    return 0;
}