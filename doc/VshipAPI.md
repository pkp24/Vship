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

WIP