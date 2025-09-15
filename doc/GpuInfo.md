GpuInfo possess 2 different behaviours depending or the supplied arguments.

### The related code is in [gpuhelper.hpp](../src/util/gpuhelper.hpp)

## GpuInfo Argument and return type 1

This case is what happens when no argument is given : `vship.GpuInfo()`

This mode allows to list the available GPUs. Beware to manage [Errors](Vship-Error-Managment.md) as this function can return 2 errors: `DeviceCountError` and `NoDeviceDetected`

Here is the return format:

```
GPU 0: {GPU 0 Name}
GPU 1: {GPU 1 Name}
...

```

## GpuInfo Argument and return type 2

This case is what happens when one argument is given : `vship.GpuInfo(gpu_id = x)`

Note that by default, HIP and CUDA will use GPU 0

### gpu_id : integer

In that case, vship understands that you want more details about the GPU specified. The same as before, beware to manage [Errors](https://codeberg.org/Line-fr/Vship/wiki/Vship-Error-Managment). This function can return 3 error types : `DeviceCountError`, `NoDeviceDetected` and `BadDeviceArgument`

Here is the return format and all the infos that vship will give : 

```
Name: {GPU Name string}
MultiProcessorCount: {multiprocessor count integer}
MaxSharedMemoryPerBlock: {Max Shared Memory Per Block integer} bytes
WarpSize: {Warp Size integer}
VRAMCapacity: {GPU VRAM Capacity float} GB
MemoryBusWidth: {memory bus width integer} bits
Integrated: {0|1}
PassKernelCheck : {0|1}
```
PassKernelCheck is special because it is not given by HIP or CUDA, it corresponds to a Sanity Check and gives clues showing that vship **will** fail if it gives 0 and the given gpu is used to compute. The corresponding error that would arise if unmanaged at that step is `BadDeviceCode`