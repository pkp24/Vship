#pragma once

#include "../util/preprocessor.hpp"
#include "../util/float3operations.hpp"
#include <cmath>

namespace cvvdp {

// Color space conversion matrices and functions for CVVDP

// sRGB to Linear RGB conversion (EOTF)
__device__ __forceinline__ float srgb_to_linear(float v) {
    if (v <= 0.04045f) {
        return v / 12.92f;
    } else {
        return powf((v + 0.055f) / 1.055f, 2.4f);
    }
}

// BT.709 RGB to XYZ (D65 white point)
__device__ __forceinline__ float3 rgb_to_xyz(float3 rgb) {
    float3 xyz;
    xyz.x = 0.4124564f * rgb.x + 0.3575761f * rgb.y + 0.1804375f * rgb.z;
    xyz.y = 0.2126729f * rgb.x + 0.7151522f * rgb.y + 0.0721750f * rgb.z;
    xyz.z = 0.0193339f * rgb.x + 0.1191920f * rgb.y + 0.9503041f * rgb.z;
    return xyz;
}

// XYZ to LMS (cone space using Hunt-Pointer-Estevez transformation)
__device__ __forceinline__ float3 xyz_to_lms2006(float3 xyz) {
    float3 lms;
    lms.x =  0.4002f * xyz.x + 0.7075f * xyz.y - 0.0807f * xyz.z;  // L
    lms.y = -0.2280f * xyz.x + 1.1500f * xyz.y + 0.0612f * xyz.z;  // M
    lms.z =  0.0000f * xyz.x + 0.0000f * xyz.y + 0.9184f * xyz.z;  // S
    return lms;
}

// LMS to DKL opponent color space (Derrington-Krauskopf-Lennie)
// DKL with D65 white point
__device__ __forceinline__ float3 lms_to_dkl_d65(float3 lms) {
    // Normalized to D65 white point: L=1.0, M=1.0, S=1.0
    // These values represent the LMS for D65 white
    const float L_w = 1.0f;
    const float M_w = 1.0f;
    const float S_w = 1.0f;

    // Normalize by white point
    float l_norm = lms.x / L_w;
    float m_norm = lms.y / M_w;
    float s_norm = lms.z / S_w;

    // DKL transformation
    float3 dkl;
    // Achromatic (luminance) channel
    dkl.x = l_norm + m_norm;
    // Red-Green opponent channel
    dkl.y = l_norm - m_norm;
    // Blue-Yellow (S-cone) opponent channel
    dkl.z = s_norm - (l_norm + m_norm);

    return dkl;
}

// Combined RGB to DKL transformation
__device__ __forceinline__ float3 rgb_to_dkl_d65(float3 rgb) {
    float3 xyz = rgb_to_xyz(rgb);
    float3 lms = xyz_to_lms2006(xyz);
    return lms_to_dkl_d65(lms);
}

// Log transformation for contrast encoding
__device__ __forceinline__ float3 log_transform(float3 lms, float epsilon = 1e-5f) {
    float3 result;
    result.x = log10f(fmaxf(lms.x, epsilon));
    result.y = log10f(fmaxf(lms.y, epsilon));
    result.z = log10f(fmaxf(lms.z, epsilon));
    return result;
}

// Apply display photometry (gain-offset-gamma model for SDR)
__device__ __forceinline__ float apply_gog_eotf(float v, float gamma, float L_max, float L_black) {
    // Gain-offset-gamma (GOG) display model
    // L = L_black + (L_max - L_black) * v^gamma
    float linear = powf(fmaxf(v, 0.0f), gamma);
    return L_black + (L_max - L_black) * linear;
}

// PQ (Perceptual Quantizer) EOTF for HDR content (SMPTE ST 2084)
__device__ __forceinline__ float pq_eotf(float v) {
    const float m1 = 0.1593017578125f;  // 2610/16384
    const float m2 = 78.84375f;          // 2523/32
    const float c1 = 0.8359375f;         // 3424/4096 = c3 - c2 + 1
    const float c2 = 18.8515625f;        // 2413/128
    const float c3 = 18.6875f;           // 2392/128

    float vp = powf(fmaxf(v, 0.0f), 1.0f / m2);
    float num = fmaxf(vp - c1, 0.0f);
    float den = c2 - c3 * vp;
    float L = powf(num / fmaxf(den, 1e-10f), 1.0f / m1);
    return L * 10000.0f; // Scale to cd/m²
}

// HLG (Hybrid Log-Gamma) EOTF for HDR content (ITU-R BT.2100)
__device__ __forceinline__ float hlg_eotf(float v, float L_max = 1000.0f) {
    const float a = 0.17883277f;
    const float b = 0.28466892f;  // 1 - 4a
    const float c = 0.55991073f;  // 0.5 - a * ln(4a)

    float Y;
    if (v <= 0.5f) {
        Y = (v * v) / 3.0f;
    } else {
        Y = (expf((v - c) / a) + b) / 12.0f;
    }

    // Scale to luminance
    return Y * L_max;
}

// Kernel to convert RGB to linear luminance values
__launch_bounds__(256)
__global__ void rgb_to_linear_kernel(float3* data, int64_t size, float L_max, float L_black, float gamma) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float3 rgb = data[idx];

    // Apply EOTF (sRGB to linear for each channel)
    rgb.x = srgb_to_linear(rgb.x);
    rgb.y = srgb_to_linear(rgb.y);
    rgb.z = srgb_to_linear(rgb.z);

    // Apply display model if needed
    if (gamma != 1.0f) {
        rgb.x = apply_gog_eotf(rgb.x, gamma, L_max, L_black);
        rgb.y = apply_gog_eotf(rgb.y, gamma, L_max, L_black);
        rgb.z = apply_gog_eotf(rgb.z, gamma, L_max, L_black);
    } else {
        // Linear scaling
        rgb.x = L_black + (L_max - L_black) * rgb.x;
        rgb.y = L_black + (L_max - L_black) * rgb.y;
        rgb.z = L_black + (L_max - L_black) * rgb.z;
    }

    data[idx] = rgb;
}

// Kernel to convert RGB to DKL color space
__launch_bounds__(256)
__global__ void rgb_to_dkl_kernel(float3* data, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float3 rgb = data[idx];
    data[idx] = rgb_to_dkl_d65(rgb);
}

// Kernel to apply log transformation to LMS values
__launch_bounds__(256)
__global__ void apply_log_transform_kernel(float3* data, int64_t size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float3 lms = data[idx];
    data[idx] = log_transform(lms);
}

// Host functions to launch kernels
inline void rgb_to_linear(float3* data, int64_t size, float L_max, float L_black, float gamma, hipStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    rgb_to_linear_kernel<<<blocks, threads, 0, stream>>>(data, size, L_max, L_black, gamma);
}

inline void rgb_to_dkl(float3* data, int64_t size, hipStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    rgb_to_dkl_kernel<<<blocks, threads, 0, stream>>>(data, size);
}

inline void apply_log_transform(float3* data, int64_t size, hipStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    apply_log_transform_kernel<<<blocks, threads, 0, stream>>>(data, size);
}

} // namespace cvvdp
