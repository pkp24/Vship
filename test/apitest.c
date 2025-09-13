#include "../src/VshipAPI.h"
#include<stdio.h>

int main(){
    Vship_Version v = Vship_GetVersion();
    printf("Vship API version : %i.%i.%i\n", v.major, v.minor, v.minorMinor);
    switch (v.backend){
        case Vship_HIP:
            printf("Running on HIP\n");
            break;
        case Vship_Cuda:
            printf("Running on Cuda\n");
            break;
    }
    Vship_Exception err;
    int numgpu;
    err = Vship_GetDeviceCount(&numgpu);
    if (err != 0){
        printf("Failed to Retrieve number of GPU\n");
        return 0;
    }
    printf("%i GPUs are detected on the system\n", numgpu);
    if (numgpu == 0){
        printf("As no GPU is detected, I will exit\n");
        return 0;
    }
    for (int i = 0; i < numgpu; i++){
        Vship_DeviceInfo deviceInfo;
        err = Vship_GetDeviceInfo(&deviceInfo, i);
        printf("-----------\n");
        printf("GPU %i is %s\n", i, deviceInfo.name);
        printf("It has %f GB of VRAM\n", (float)deviceInfo.VRAMSize/1000000000.);
    }
    printf("Performing full test on GPU 0\n");
    err = Vship_GPUFullCheck(0);
    if (err != 0){
        printf("Error detected while checking GPU 0\n");
        return 0;
    }
    printf("GPU Check success\n");
    err = Vship_BadDeviceCode;
    char errmsg[1024];
    int optimallsize = Vship_GetErrorMessage(err, errmsg, 1024);
    printf("BadDeviceCode gives the following error message: %s\n", errmsg);
    return 0;
}