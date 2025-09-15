# Introduction

Vship API has been made to allow low level access to Vship interal metric without having to use FFVship or vapoursynth. It typically allow to retrieve SSIMU2 scores, butteraugli distortion map. It also benefits from not requiring complex compilation processes by being a simple shared library.

# Compilation and Usage

## How to install VshipAPI

VshipAPI is included when installing Vship the vapoursynth plugin. The API itself is contained within vship.dll which also contains the vapoursynth API support. As such, installing VshipAPI only requires doing `make build` (or `make buildcuda` for nvidia GPUs) and `make install`. For windows, there is no standard lib installation folder. As such, the compiled .dll which can also be found on the release page and [VshipAPI Header](../src/VshipAPI.h) need to be installed manually at a known place for use and compilation of your software.

## Usage of VshipAPI inside the code

Including the header and using VshipAPI function is all that is required. As a .h header, it is directly usable in C or C++ but can also be ported to work in other languages.

Example:

```Cpp
#include "VshipAPI.h

int main(){
    int number_of_gpu;
    Vship_Exception err = Vship_GetDeviceCount(&number_of_gpu);
    return 0;
}
```

Later in the document there will be more concrete examples and detailed description of each functions

## Compilation of a software using VshipAPI

As stated before, you only need the header file and the shared library.
As such compilation is as simple as 

`gcc mycode.c -I HeaderDirectory -L LibraryDirectory -l:vship.so`

Note that the -I and -L arguments are optional depending on the placement of the header and library. 
For the resulting executable to work, it will need to be able to find vship.so. On Ubuntu, it might be necessary to add the library path (/usr/local/lib) to LD_LIBRARY_PATH environment variable. Another option is to put vship.so/dll next to your executable.

To verify that everything works for you, you can try compiling yourself [this example](../test/apitest.c)

# API detailed tutorial and description

## Interactive Tutorial

For this tutorial, we will suppose that we wish to retrieve a Butteraugli distortion map using Vship and that you possess the planar RGB with BT709 transfer frames for which we will compute this distmap.

### Error Managment

For this tutorial, we will not bother too much with errors with will create a simple preprocessor wrapper to handle vship errors:

```Cpp
#include<stdio.h>
#include<stdlib.h>

Vship_Exception err;
char errmsg[1024];
#define ErrorCheck(line) err = line;\
if (err != 0){\
    Vship_GetErrorMessage(err, errmsg, 1024);\
    printf("Vship Error occured: %s", errmsg);\
    exit(1);\
}
```

This macro will simply print vship error message if an error was to occur

### Verify that vship will work before using it

```Cpp
//check if gpu 0 exist and if it can work with vship
ErrorCheck(Vship_GPUFullCheck(0));
```

Thanks to our macro, the error message will be displayed to the user before exiting, and nothing will happen if it goes well.

### Create a Butteraugli Handler

Vship being made for maximal throughput, it does some preprocessing. A handler may be used for processing a lot of frames (but only one at a time). To create a handler, you can do this

```Cpp
Vship_ButteraugliHandler butterhandler;
//This will initialize a handler with intensity_target 203 in butteraugli
Vship_ButteraugliInit(&butterhandler, imaage_width, image_height, 203.);
```

### Frame Processing

Our RGB planar BT709 transfer frames must be of type: `const uint8*[3]` but the actual data inside the plane will be in uint16_t in this example

```Cpp
//I suppose that you have already set the data of the frame
const uint8* image1[3];
const uint8* image2[3];

//the result will be here!
//let's allocate the space required to store the distortion map. It is always float
//you can use image_width as a stride in this case
const uint8* distortionMap = (uint8_t*)malloc(sizeof(float)*image_height*image_stride);

Vship_ButteraugliScore score;

//ask vship to compute! We use the Uint16 version since our frames are coded as uint16_t
ErrorCheck(Vship_ComputeButteraugliUint16(&butterhandler, &score, distortionMap, image_stride, image1, image2, image_stride));

//now score contains butteraugli scores and distortionMap contains the distortion map!
```

### Exiting properly

We want to avoid leaks so there is a little step to add

```Cpp
//free the distortion map we just allocated
free(distortionMap);

//free the butteraugli handler
Vship_ButteraugliFree(&butterhandler);
```

## Good practices

## Details on every function