#pragma once

namespace VshipColorConvert{

template <FFMS_ColorRanges RANGE_TYPE>
__device__ void inline RangeLinearize(float& a);

template <FFMS_ColorRanges RANGE_TYPE>
__device__ void inline RangeLinearize(float3& a){
    RangeLinearize<RANGE_TYPE>(a.x);
    RangeLinearize<RANGE_TYPE>(a.y);
    RangeLinearize<RANGE_TYPE>(a.z);
}

template<>
__device__ void inline RangeLinearize<FFMS_CR_JPEG>(float& a){

}

template<>
__device__ void inline RangeLinearize<FFMS_CR_MPEG>(float& a){
    a = (255.*a - 16.)/235.;
}

}