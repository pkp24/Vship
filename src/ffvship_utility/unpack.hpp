#pragma once

//the compiler is able to optimize and unroll the loop by itself
template<int bitwidth>
__device__ __host__ uint32_t getBitIntegerArray(const uint8_t* const source_plane, const int i, const int stride, const int width){
    const int x = i%width;
    const int y = i/width;
    const uint8_t* byte_ptr = source_plane+stride*y;
    const uint8_t bitoffset = (bitwidth*x)%8;
    byte_ptr += (x*bitwidth)/8;
    uint32_t val = (byte_ptr[0]&((1 << (8-bitoffset)) -1)) << (bitwidth-8+bitoffset); //first value
    byte_ptr++;
    int remain = bitwidth+bitoffset-8;
    while (remain >= 8){
        val += byte_ptr[0] << (remain - 8);
        remain -= 8;
        byte_ptr++;
    }
    if (remain > 0) val += byte_ptr[0] >> (8 - remain);
    return val;
}

template<int bitwidth>
__device__ __host__ void setBitIntegerArray(uint8_t* const out_plane, const int i, const int stride, const int width, uint32_t val){
    const int x = i%width;
    const int y = i/width;
    uint8_t* byte_ptr = out_plane+stride*y;
    const uint8_t bitoffset = (bitwidth*x)%8;
    byte_ptr += (x*bitwidth)/8;
    byte_ptr[0] &= 256 - (1 << (8 - bitoffset));
    byte_ptr[0] |= (val >> (bitwidth-8+bitoffset)); //first value
    byte_ptr++;
    int remain = bitwidth-8+bitoffset;
    while (remain >= 8){
        byte_ptr[0] = val >> (remain - 8);
        remain -= 8;
        byte_ptr++;
    }
    if (remain > 0){
        byte_ptr[0] &= (1 << (8 - remain)) - 1;
        byte_ptr[0] |= (val << (8 - remain))%256;
    }
}

//we only support const bitdepth per plane because of zimg
//real width of input is numPlane * width (stride should take this into account)
template<int bitwidth>
__device__ __host__ void unpack3(uint8_t* out[3], const int out_stride, const uint8_t* const in, const int in_stride, const int width, const int height){
    for (int i = 0; i < width*height; i++){
        setBitIntegerArray<bitwidth>(out[0], i, out_stride, width, getBitIntegerArray<bitwidth>(in, 3*i, in_stride, 3*width));
        setBitIntegerArray<bitwidth>(out[1], i, out_stride, width, getBitIntegerArray<bitwidth>(in, 3*i+1, in_stride, 3*width));
        setBitIntegerArray<bitwidth>(out[2], i, out_stride, width, getBitIntegerArray<bitwidth>(in, 3*i+2, in_stride, 3*width));
    }
}

//we only support const bitdepth per plane because of zimg
//real width of input is numPlane * width (stride should take this into account)
template<int bitwidth>
__device__ __host__ void unpack4(uint8_t* out[4], const int out_stride, const uint8_t* const in, const int in_stride, const int width, const int height){
    for (int i = 0; i < width*height; i++){
        setBitIntegerArray<bitwidth>(out[0], i, out_stride, width, getBitIntegerArray<bitwidth>(in, 4*i, in_stride, 4*width));
        setBitIntegerArray<bitwidth>(out[1], i, out_stride, width, getBitIntegerArray<bitwidth>(in, 4*i+1, in_stride, 4*width));
        setBitIntegerArray<bitwidth>(out[2], i, out_stride, width, getBitIntegerArray<bitwidth>(in, 4*i+2, in_stride, 4*width));
        setBitIntegerArray<bitwidth>(out[3], i, out_stride, width, getBitIntegerArray<bitwidth>(in, 4*i+3, in_stride, 4*width));
    }
}