#pragma once

namespace VshipColorConvert{

enum Sample_Type{
    COLOR_FLOAT,
    COLOR_HALF,
    COLOR_8BIT,
    COLOR_9BIT,
    COLOR_10BIT,
    COLOR_12BIT,
    COLOR_14BIT,
    COLOR_16BIT,
};

template<Sample_Type T>
__device__ float inline PickValue(const uint8_t* const source_plane, const int i, const int stride, const int width);

template<>
__device__ float inline PickValue<COLOR_FLOAT>(const uint8_t* const source_plane, const int i, const int stride, const int width){
    const int line = i/width;
    const int column = i%width;
    return ((float*)(source_plane+line*stride))[column];
}

template<>
__device__ float inline PickValue<COLOR_HALF>(const uint8_t* const source_plane, const int i, const int stride, const int width){
    const int line = i/width;
    const int column = i%width;
    return ((__half*)(source_plane+line*stride))[column];
}

//the compiler is able to optimize and unroll the loop by itself
template<int bitwidth>
__device__ float getBitIntegerArray(const uint8_t* const source_plane, const int i, const int stride, const int width){
    const int x = i%width;
    const int y = i/width;
    const uint8_t* byte_ptr = source_plane+stride*y;
    const uint8_t bitoffset = (bitwidth*x)%8;
    byte_ptr += (x*bitwidth)/8;
    float val = (byte_ptr[0]&((1 << (8-bitoffset)) -1)) << (bitwidth-8+bitoffset); //first value
    byte_ptr++;
    int remain = bitwidth-bitoffset-8;
    while (remain >= 8){
        val += byte_ptr[0] << (remain - 8);
        remain -= 8;
        byte_ptr++;
    }
    if (remain > 0) val += byte_ptr[0] >> (8 - remain);
    return val/((1 << bitwidth)-1);
}

template<>
__device__ float inline PickValue<COLOR_8BIT>(const uint8_t* const source_plane, const int i, const int stride, const int width){
    return getBitIntegerArray<8>(source_plane, i, stride, width);
}

template<>
__device__ float inline PickValue<COLOR_9BIT>(const uint8_t* const source_plane, const int i, const int stride, const int width){
    return getBitIntegerArray<9>(source_plane, i, stride, width);
}

template<>
__device__ float inline PickValue<COLOR_10BIT>(const uint8_t* const source_plane, const int i, const int stride, const int width){
    return getBitIntegerArray<10>(source_plane, i, stride, width);
}

template<>
__device__ float inline PickValue<COLOR_12BIT>(const uint8_t* const source_plane, const int i, const int stride, const int width){
    return getBitIntegerArray<12>(source_plane, i, stride, width);
}

template<>
__device__ float inline PickValue<COLOR_14BIT>(const uint8_t* const source_plane, const int i, const int stride, const int width){
    return getBitIntegerArray<14>(source_plane, i, stride, width);
}

template<>
__device__ float inline PickValue<COLOR_16BIT>(const uint8_t* const source_plane, const int i, const int stride, const int width){
    return getBitIntegerArray<16>(source_plane, i, stride, width);
}


}