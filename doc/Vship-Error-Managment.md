Though Error managment is optional since the error message given by Vship should already give advice to the user. But sometimes, it can useful to some developpers to have nice and consistent error messages.

### Related Code : [VshipExceptions.hpp](../src/util/VshipExceptions.hpp)

## Error Format:

Due to the Vapoursynth plugin nature of Vship, only Vapoursynth.Error can be sent to python. As such, actual error description and format will be contained within the string.

```
VshipException
{Error Enum Name}: {message related to error type}
 - At line {line} of {file}
```

# Error types:

## Vship Internal Errors

### OutOfVRAM: 

This issue shows that a VRAM allocation (hipMalloc) failed which indicates that your VRAM is full. The common way around it is lowering `vapoursynth.core.num_threads` corresponding to the VRAM requirements described in vship.BUTTERAUGLI and vship.SSIMULACRA2

The VRAM amount of a gpu can be obtained in advance running vship.GpuInfo(gpu_id=?)

### OutOfRAM:

This issue should never happen or is super rare and happens if a RAM allocation (malloc) fails. This likely means that your RAM. And as with VRAM, lowering `vapoursynth.core.num_threads` can help reducing RAM allocation.

## Input Errors

### DifferingInputType:

This issue arise when the 2 clips given in vship.BUTTERAUGLI or vship.SSIMULACRA2 are different. Colorspace conversion is done automatically, but vship does nothing if they don't share the same width, height and frame count.

### NonRGBSInput:

This issue should never happen, ever. (Since vship is handling the conversion to RGBS)

If you manage to get it, congrats! You can come speak to me haha (not that you cannot otherwise)

## Device errors

### DeviceCountError:

This error happens when HIP or CUDA fails to check your number of GPU with hipGetDeviceCount.

This can happen as a result of insufficient permission from your user which prevents you from using vship. It can also happen if hip_runtime or cuda drivers are not installed properly.

### NoDeviceDetected:

This error happens when the call to hipGetDeviceCount works but it returns 0 GPU.

This can happen as a result of insufficient permission from your user which prevents you from using vship. It can also happen if... you have no GPU, or if your GPU is not of the type you compiled VSHIP for.
If you have an AMD GPU and you use the NVIDIA Binary, this can happen.

### BadDeviceArgument:

This happens when you specify a gpu_id argument that is incorrect (either < 0 or >= gpu_count)

### BadDeviceCode:

This happens when vship manages to get all the infos about your gpu but is unable to execute any code on it. This usually shows that vship was not compiled for your GPU's architecture.

## Other error type

### BadErrorType:

This error is not used and only serve to satisfy the compiler, making it seems like it is the default exception.

# Error Messages

I strongly encourage you to go to [VshipExceptions.hpp](../src/util/VshipExceptions.hpp)

And check the errorMessage function.

