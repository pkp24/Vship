#ifndef EXCEPTIONSHPP
#define EXCEPTIONSHPP

#include "preprocessor.hpp"

//to parse the error message automatically: here is the recipe

//VshipException
//{Error Enum Name}: {message related to error type}
// - At line {line} of {file}

enum VSHIPEXCEPTTYPE{
    //vship internal issues
    OutOfVRAM,
    OutOfRAM,
    
    //input issues
    DifferingInputType,
    NonRGBSInput, //should never happen since .resize should give RGBS always
    
    //Device related
    DeviceCountError,
    NoDeviceDetected,
    BadDeviceArgument,
    BadDeviceCode,

    //should not be used
    BadErrorType,
};

std::string errorMessage(VSHIPEXCEPTTYPE type){
    switch (type){
        case OutOfVRAM:
        return "OutOfVRAM: Vship was not able to perform GPU memory allocation. (Advice) Reduce the number of vapoursynth threads";

        case OutOfRAM:
        return "OutOfRAM: Vship was not able to allocate CPU memory. This is a rare error that should be reported. Check your RAM usage";
        
        case DifferingInputType:
        return "DifferingInputType: Vship received 2 videos with different properties. (Advice) verify that they have the same width, height and length";

        case NonRGBSInput:
        return "NonRGBSInput: Vship did not manage to get RGBS format of your inputs. This should not happen. (Advice) try converting yourself to RGBS";
    
        case DeviceCountError:
        return "DeviceCountError: Vship was unable to verify the number of GPU on your system. (Advice) Did you select the correct binary for your device AMD/NVIDIA. (Advice) if linux AMD, are you in video and render groups?";
        
        case NoDeviceDetected:
        return "NoDeviceDetected: Vship found no device on your system. (Advice) Did you select the correct binary for your device AMD/NVIDIA. (Advice) if linux AMD, are you in video and render groups?";
        
        case BadDeviceArgument:
        return "BadDeviceArgument: Vship received a bad gpu_id argument either you specified a number >= to your gpu count, either it was negative";

        case BadDeviceCode:
        return "BadDeviceCode: Vship was unable to run a simple GPU Kernel. This usually indicate that the code was compiled for the wrong architecture. (Advice) Try to compile vship yourself, eventually replace --offload-arch=native to your arch";
    
        case BadErrorType:
        return "BadErrorType: There was an unknown error";
    }
    return "BadErrorType: There was an unknown error"; //this will not happen but the compiler will be happy
}

class VshipError : public std::exception
{
    VSHIPEXCEPTTYPE type;
    std::string file;
    int line;
public:
    VshipError(VSHIPEXCEPTTYPE type, const std::string filename, const int line) : std::exception(), type(type), file(filename), line(line){
    }
    
    std::string getErrorMessage() const
    {
        std::stringstream ss;
        ss << "VshipException" << std::endl;
        ss << errorMessage(type) << std::endl;
        ss << " - At line " << line << " of " << file << std::endl;
        return ss.str();
    }
};

#endif