#pragma once

#include "../util/preprocessor.hpp"
#include "../util/VshipExceptions.hpp"
#include "../util/gpuhelper.hpp"
#include "config.hpp"
#include "colorspace.hpp"
#include "lpyr.hpp"
#include "csf.hpp"
#include "temporal.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <vector>
#include <utility>

#include "../../../third_party/rapidjson/include/rapidjson/stringbuffer.h"
#include "../../../third_party/rapidjson/include/rapidjson/writer.h"

namespace cvvdp {

constexpr int kMaxChannels = 4;

// Helper: safe_pow - differentiable power function that handles zeros
__device__ __forceinline__ float safe_pow(float x, float p) {
    const float epsilon = 0.00001f;
    return powf(x + epsilon, p) - powf(epsilon, p);
}

// Helper: clamp_diffs with soft clamping (matches Python "soft" mode)
__device__ __forceinline__ float clamp_diffs(float D, float d_max) {
    float max_v = powf(10.0f, d_max);
    return max_v * D / (max_v + D);
}

// Extract log10(luminance) from Gaussian pyramid Y-channel for local adaptation
// This provides per-pixel background luminance for Weber contrast encoding
__launch_bounds__(256)
__global__ void extract_log_luminance_kernel(
    const float3* test_gauss,
    const float3* ref_gauss,
    float* log_L_bkg_test,
    float* log_L_bkg_ref,
    int pixels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) return;

    const float min_bkg = 0.01f; // Minimum background to avoid log(0)

    // Extract Y-channel (luminance) from DKL representation
    // In DKL space, the first channel (x) is achromatic luminance
    float L_test = fmaxf(test_gauss[idx].x, min_bkg);
    float L_ref = fmaxf(ref_gauss[idx].x, min_bkg);

    log_L_bkg_test[idx] = log10f(L_test);
    log_L_bkg_ref[idx] = log10f(L_ref);
}

// Baseband difference kernel - for lowest frequency residual
// Applies CSF but without masking or high-pass filtering
__launch_bounds__(256)
__global__ void baseband_difference_kernel(
    const float* test_contrast, const float* ref_contrast,
    const float* log_L_bkg_test, const float* log_L_bkg_ref,
    float4* D_out,
    int width, int height,
    float rho_cpd,
    const float* log_L_bkg_lut, const float* log_rho_lut,
    const float* logS_c0, const float* logS_c1, const float* logS_c2, const float* logS_transient,
    int num_L_bkg, int num_rho,
    int channel_count,
    float sensitivity_correction
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixels = width * height;
    if (idx >= pixels) return;

    const int base = idx * 3;
    float3 T = make_float3(
        test_contrast[base + 0],
        test_contrast[base + 1],
        test_contrast[base + 2]
    );
    float3 R = make_float3(
        ref_contrast[base + 0],
        ref_contrast[base + 1],
        ref_contrast[base + 2]
    );

    // Use per-pixel background luminance if available
    float log_L_bkg = (log_L_bkg_test && log_L_bkg_ref)
        ? 0.5f * (log_L_bkg_test[idx] + log_L_bkg_ref[idx])
        : log_L_bkg_ref ? log_L_bkg_ref[idx]
                        : log_L_bkg_test ? log_L_bkg_test[idx]
                                         : 0.0f;

    float log_rho = log10f(rho_cpd);

    // Find LUT indices (same as apply_csf_and_gain_kernel)
    float rho_idx = 0.0f;
    for (int i = 0; i < num_rho - 1; i++) {
        if (log_rho_lut[i] <= log_rho && log_rho <= log_rho_lut[i + 1]) {
            rho_idx = i + (log_rho - log_rho_lut[i]) / (log_rho_lut[i + 1] - log_rho_lut[i]);
            break;
        }
    }

    float L_idx = 0.0f;
    for (int i = 0; i < num_L_bkg - 1; i++) {
        if (log_L_bkg_lut[i] <= log_L_bkg && log_L_bkg <= log_L_bkg_lut[i + 1]) {
            L_idx = i + (log_L_bkg - log_L_bkg_lut[i]) / (log_L_bkg_lut[i + 1] - log_L_bkg_lut[i]);
            break;
        }
    }

    // Interpolate CSF sensitivity
    float logS_y = interp2d(logS_c0, num_rho, num_L_bkg, rho_idx, L_idx);
    float logS_rg = interp2d(logS_c1, num_rho, num_L_bkg, rho_idx, L_idx);
    float logS_by = interp2d(logS_c2, num_rho, num_L_bkg, rho_idx, L_idx);
    float logS_trans = interp2d(logS_transient, num_rho, num_L_bkg, rho_idx, L_idx);

    float S_y = powf(10.0f, logS_y + sensitivity_correction / 20.0f);
    float S_rg = powf(10.0f, logS_rg + sensitivity_correction / 20.0f);
    float S_by = powf(10.0f, logS_by + sensitivity_correction / 20.0f);
    float S_trans = powf(10.0f, logS_trans + sensitivity_correction / 20.0f);

    // Channel gains
    const float ch_gain_y = 1.0f;
    const float ch_gain_rg = 1.45f;
    const float ch_gain_by = 1.0f;
    const float ch_gain_trans = 1.0f;

    float4 diff = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (channel_count > 0) diff.x = fabsf(T.x - R.x) * S_y * ch_gain_y;
    if (channel_count > 1) diff.y = fabsf(T.y - R.y) * S_rg * ch_gain_rg;
    if (channel_count > 2) diff.z = fabsf(T.z - R.z) * S_by * ch_gain_by;
    if (channel_count > 3) diff.w = fabsf(T.x - R.x) * S_trans * ch_gain_trans;

    D_out[idx] = diff;
}

// Apply CSF sensitivity and channel gain (for mult-mutual masking model)
// This kernel prepares T_p and R_p = contrast * sensitivity * ch_gain
__launch_bounds__(256)
__global__ void apply_csf_and_gain_kernel(
    const float* test_contrast, const float* ref_contrast,
    const float* log_L_bkg_test, const float* log_L_bkg_ref,
    float4* T_p, float4* R_p,
    int pixels,
    float rho_cpd,
    const float* log_L_bkg_lut, const float* log_rho_lut,
    const float* logS_c0, const float* logS_c1, const float* logS_c2,
    const float* logS_transient,
    int num_L_bkg, int num_rho,
    int channel_count,
    float sensitivity_correction
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) return;

    const int base = idx * 3;
    float3 T = make_float3(
        test_contrast[base + 0],
        test_contrast[base + 1],
        test_contrast[base + 2]
    );
    float3 R = make_float3(
        ref_contrast[base + 0],
        ref_contrast[base + 1],
        ref_contrast[base + 2]
    );

    float log_L_bkg = (log_L_bkg_test && log_L_bkg_ref)
        ? 0.5f * (log_L_bkg_test[idx] + log_L_bkg_ref[idx])
        : log_L_bkg_ref ? log_L_bkg_ref[idx]
                        : log_L_bkg_test ? log_L_bkg_test[idx]
                                         : 0.0f;

    // Find position in CSF LUT (same logic as before)
    float log_rho = log10f(rho_cpd);

    float rho_idx = 0.0f;
    for (int i = 0; i < num_rho - 1; i++) {
        if (log_rho_lut[i] <= log_rho && log_rho <= log_rho_lut[i + 1]) {
            rho_idx = i + (log_rho - log_rho_lut[i]) / (log_rho_lut[i + 1] - log_rho_lut[i]);
            break;
        }
    }

    float L_idx = 0.0f;
    for (int i = 0; i < num_L_bkg - 1; i++) {
        if (log_L_bkg_lut[i] <= log_L_bkg && log_L_bkg <= log_L_bkg_lut[i + 1]) {
            L_idx = i + (log_L_bkg - log_L_bkg_lut[i]) / (log_L_bkg_lut[i + 1] - log_L_bkg_lut[i]);
            break;
        }
    }

    // Interpolate sensitivity for each channel
    float logS_y = interp2d(logS_c0, num_rho, num_L_bkg, rho_idx, L_idx);
    float logS_rg = interp2d(logS_c1, num_rho, num_L_bkg, rho_idx, L_idx);
    float logS_by = interp2d(logS_c2, num_rho, num_L_bkg, rho_idx, L_idx);
    float logS_trans = interp2d(logS_transient, num_rho, num_L_bkg, rho_idx, L_idx);

    // Convert from log to linear and apply correction
    float S_y = powf(10.0f, logS_y + sensitivity_correction / 20.0f);
    float S_rg = powf(10.0f, logS_rg + sensitivity_correction / 20.0f);
    float S_by = powf(10.0f, logS_by + sensitivity_correction / 20.0f);
    float S_trans = powf(10.0f, logS_trans + sensitivity_correction / 20.0f);

    // Channel gains from Python: ch_gain = [1, 1.45, 1, 1.] for [Y, RG, BY, trans-Y]
    const float ch_gain_y = 1.0f;
    const float ch_gain_rg = 1.45f;
    const float ch_gain_by = 1.0f;
    const float ch_gain_trans = 1.0f;

    float4 Tp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 Rp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (channel_count > 0) {
        Tp.x = T.x * S_y * ch_gain_y;
        Rp.x = R.x * S_y * ch_gain_y;
    }
    if (channel_count > 1) {
        Tp.y = T.y * S_rg * ch_gain_rg;
        Rp.y = R.y * S_rg * ch_gain_rg;
    }
    if (channel_count > 2) {
        Tp.z = T.z * S_by * ch_gain_by;
        Rp.z = R.z * S_by * ch_gain_by;
    }
    if (channel_count > 3) {
        Tp.w = T.x * S_trans * ch_gain_trans;
        Rp.w = R.x * S_trans * ch_gain_trans;
    }

    T_p[idx] = Tp;
    R_p[idx] = Rp;
}

// Compute mutual masking term: M_mm = min(|T_p|, |R_p|)
__launch_bounds__(256)
__global__ void compute_mutual_masking_kernel(
    const float4* T_p, const float4* R_p,
    float4* M_mm,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float4 Tp = T_p[idx];
    float4 Rp = R_p[idx];

    M_mm[idx] = make_float4(
        fminf(fabsf(Tp.x), fabsf(Rp.x)),
        fminf(fabsf(Tp.y), fabsf(Rp.y)),
        fminf(fabsf(Tp.z), fabsf(Rp.z)),
        fminf(fabsf(Tp.w), fabsf(Rp.w))
    );
}

// Gaussian blur kernels used in phase-uncertainty masking
__launch_bounds__(256)
__global__ void gaussian_blur_horizontal_kernel(
    const float4* input,
    float4* output,
    int width,
    int height,
    const float* kernel,
    int radius
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    int y = idx / width;
    int x = idx % width;

    float4 accum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int k = -radius; k <= radius; ++k) {
        int sample_x = min(max(x + k, 0), width - 1);
        float weight = kernel[k + radius];
        float4 sample = input[y * width + sample_x];
        accum.x += sample.x * weight;
        accum.y += sample.y * weight;
        accum.z += sample.z * weight;
        accum.w += sample.w * weight;
    }
    output[idx] = accum;
}

__launch_bounds__(256)
__global__ void gaussian_blur_vertical_kernel(
    const float4* input,
    float4* output,
    int width,
    int height,
    const float* kernel,
    int radius
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    int y = idx / width;
    int x = idx % width;

    float4 accum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int k = -radius; k <= radius; ++k) {
        int sample_y = min(max(y + k, 0), height - 1);
        float weight = kernel[k + radius];
        float4 sample = input[sample_y * width + x];
        accum.x += sample.x * weight;
        accum.y += sample.y * weight;
        accum.z += sample.z * weight;
        accum.w += sample.w * weight;
    }
    output[idx] = accum;
}

// Apply phase uncertainty: optional blur + multiplicative bias
__launch_bounds__(256)
__global__ void apply_phase_uncertainty_kernel(
    float4* M_mm,
    float multiplier,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    M_mm[idx].x *= multiplier;
    M_mm[idx].y *= multiplier;
    M_mm[idx].z *= multiplier;
    M_mm[idx].w *= multiplier;
}

// Apply cross-channel masking with xcm_weights matrix (already exponentiated)
__launch_bounds__(256)
__global__ void apply_cross_channel_masking_kernel(
    const float4* M_pu,
    float4* M_xcm,
    const float* xcm_weights,
    int channel_count,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float in_values[kMaxChannels] = {
        M_pu[idx].x,
        M_pu[idx].y,
        M_pu[idx].z,
        M_pu[idx].w
    };
    float out_values[kMaxChannels] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int dest = 0; dest < channel_count; ++dest) {
        float sum = 0.0f;
        for (int src = 0; src < channel_count; ++src) {
            float weight = xcm_weights[src * 4 + dest];
            sum += in_values[src] * weight;
        }
        out_values[dest] = sum;
    }

    M_xcm[idx] = make_float4(out_values[0], out_values[1], out_values[2], out_values[3]);
}

// Element-wise safe power of absolute values: dst = safe_pow(|src|, q[channel])
__launch_bounds__(256)
__global__ void apply_channel_power_kernel(
    const float4* input,
    float4* output,
    float4 mask_q,
    int channel_count,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    const float4 value = input[idx];
    float4 result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (channel_count > 0) result.x = safe_pow(fabsf(value.x), mask_q.x);
    if (channel_count > 1) result.y = safe_pow(fabsf(value.y), mask_q.y);
    if (channel_count > 2) result.z = safe_pow(fabsf(value.z), mask_q.z);
    if (channel_count > 3) result.w = safe_pow(fabsf(value.w), mask_q.w);
    output[idx] = result;
}


enum class ContrastMode : int {
    WEBER_G1_REF = 0,
    WEBER_G1 = 1,
    WEBER_G0_REF = 2
};

inline ContrastMode parse_contrast_mode(const std::string& value) {
    if (value == "weber_g1_ref") return ContrastMode::WEBER_G1_REF;
    if (value == "weber_g1") return ContrastMode::WEBER_G1;
    if (value == "weber_g0_ref") return ContrastMode::WEBER_G0_REF;
    throw VshipError(ConfigurationError, __FILE__, __LINE__);
}

__launch_bounds__(256)
__global__ void compute_weber_contrast_kernel(
    const float* lap_test,
    const float* lap_ref,
    const float* gauss_test_curr,
    const float* gauss_ref_curr,
    const float* gauss_test_exp,
    const float* gauss_ref_exp,
    float* contrast_test_out,
    float* contrast_ref_out,
    float* log_bkg_test_out,
    float* log_bkg_ref_out,
    int pixels,
    int contrast_mode,
    int is_baseband
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) return;

    const float min_bkg = 0.01f;
    const float max_contrast = 1000.0f;

    const float* test_exp_ptr = gauss_test_exp ? gauss_test_exp : gauss_test_curr;
    const float* ref_exp_ptr = gauss_ref_exp ? gauss_ref_exp : gauss_ref_curr;

    float test_y_curr = gauss_test_curr[idx * 3 + 0];
    float ref_y_curr = gauss_ref_curr[idx * 3 + 0];
    float test_y_exp = test_exp_ptr[idx * 3 + 0];
    float ref_y_exp = ref_exp_ptr[idx * 3 + 0];

    float background_test = min_bkg;
    float background_ref = min_bkg;

    switch (static_cast<ContrastMode>(contrast_mode)) {
        case ContrastMode::WEBER_G1_REF:
            background_ref = fmaxf(is_baseband ? ref_y_curr : ref_y_exp, min_bkg);
            background_test = background_ref;
            break;
        case ContrastMode::WEBER_G1:
            background_test = fmaxf(is_baseband ? test_y_curr : test_y_exp, min_bkg);
            background_ref = fmaxf(is_baseband ? ref_y_curr : ref_y_exp, min_bkg);
            break;
        case ContrastMode::WEBER_G0_REF:
        default:
            background_ref = fmaxf(ref_y_curr, min_bkg);
            background_test = background_ref;
            break;
    }

    float* out_test = contrast_test_out + idx * 3;
    float* out_ref = contrast_ref_out + idx * 3;

    float ty = lap_test[idx * 3 + 0] / background_test;
    float trg = lap_test[idx * 3 + 1] / background_test;
    float tby = lap_test[idx * 3 + 2] / background_test;

    float ry = lap_ref[idx * 3 + 0] / background_ref;
    float rrg = lap_ref[idx * 3 + 1] / background_ref;
    float rby = lap_ref[idx * 3 + 2] / background_ref;

    out_test[0] = fminf(ty, max_contrast);
    out_test[1] = fminf(trg, max_contrast);
    out_test[2] = fminf(tby, max_contrast);

    out_ref[0] = fminf(ry, max_contrast);
    out_ref[1] = fminf(rrg, max_contrast);
    out_ref[2] = fminf(rby, max_contrast);

    log_bkg_test_out[idx] = log10f(background_test);
    log_bkg_ref_out[idx] = log10f(background_ref);
}

enum class ClampMode : int {
    SOFT = 0,
    HARD = 1,
    NONE = 2
};

// Apply masking model: D_u = |T_p - R_p|^p / (1 + M^q), then clamp
__launch_bounds__(256)
__global__ void apply_masking_kernel(
    const float4* T_p, const float4* R_p,
    const float4* M_xcm, float4* D_d,
    float mask_p,
    int channel_count,
    int clamp_mode,
    float d_max,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    const float4 Tp = T_p[idx];
    const float4 Rp = R_p[idx];
    const float4 M = M_xcm[idx];

    float values[kMaxChannels] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float components[kMaxChannels] = {
        fabsf(Tp.x - Rp.x),
        fabsf(Tp.y - Rp.y),
        fabsf(Tp.z - Rp.z),
        fabsf(Tp.w - Rp.w)
    };

    for (int c = 0; c < channel_count; ++c) {
        const float denom = 1.0f + fmaxf(((&M.x)[c]), 0.0f);
        values[c] = safe_pow(components[c], mask_p) / denom;
    }

    ClampMode mode = static_cast<ClampMode>(clamp_mode);
    float max_v = powf(10.0f, d_max);

    float4 result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int c = 0; c < channel_count; ++c) {
        float value = values[c];
        switch (mode) {
            case ClampMode::HARD:
                value = fminf(value, max_v);
                break;
            case ClampMode::NONE:
                break;
            case ClampMode::SOFT:
            default:
                value = max_v * value / (max_v + value);
                break;
        }
        (&result.x)[c] = value;
    }

    D_d[idx] = result;
}

// Reduction kernel for accumulating |input|^beta per channel
__launch_bounds__(256)
__global__ void reduce_lp_sum_per_channel_kernel(
    const float4* input,
    float4* output,
    float beta,
    int channel_count,
    int size
) {
    extern __shared__ float4 sdata_vec[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float4 value = input[idx];
        float4 accum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (channel_count > 0) accum.x = powf(fabsf(value.x), beta);
        if (channel_count > 1) accum.y = powf(fabsf(value.y), beta);
        if (channel_count > 2) accum.z = powf(fabsf(value.z), beta);
        if (channel_count > 3) accum.w = powf(fabsf(value.w), beta);
        sdata_vec[tid] = accum;
    } else {
        sdata_vec[tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_vec[tid].x += sdata_vec[tid + s].x;
            sdata_vec[tid].y += sdata_vec[tid + s].y;
            sdata_vec[tid].z += sdata_vec[tid + s].z;
            sdata_vec[tid].w += sdata_vec[tid + s].w;
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata_vec[0];
    }
}

// Main CVVDP processing class
class CVVDPProcessor {
private:
    int width, height;
    float ppd; // pixels per degree
    DisplayModel display;
    CVVDPParameters params;
    LaplacianPyramid lpyr;
    CastleCSF csf;
    TemporalBuffer temporal;
    hipStream_t stream;
    std::filesystem::path data_root;
    float frame_rate = 30.0f; // Default, can be updated
    bool temporal_filtering_enabled = true;
    bool cross_channel_masking_enabled = true;

    // Cross-channel masking weights (4×4 matrix on GPU)
    float* xcm_weights_d = nullptr;
    float* pu_kernel_d = nullptr;
    int pu_kernel_radius = 0;
    int pu_padsize = 0;
    float phase_uncertainty_multiplier = 1.0f;
    float* rgb2xyz_d = nullptr;
    EotfType eotf_type = EotfType::SRGB;
    float eotf_gamma = 2.2f;

public:
    void init(int w, int h, const std::string& display_name, const std::string& data_dir_override = "") {
        width = w;
        height = h;

        // Resolve configuration directory
        data_root = resolve_cvvdp_data_root(data_dir_override);

        const std::filesystem::path display_model_path = data_root / "display_models.json";
        const std::filesystem::path params_path = data_root / "cvvdp_parameters.json";

        display = DisplayModel::load(display_model_path, display_name);
        display.color_space = load_color_space_parameters(data_root / "color_spaces.json", display.colorspace);
        params = CVVDPParameters::load(params_path);

        const std::string eotf_name = display.color_space.eotf;
        if (eotf_name == "sRGB") {
            eotf_type = EotfType::SRGB;
            eotf_gamma = display.color_space.gamma;
        } else if (eotf_name == "linear") {
            eotf_type = EotfType::LINEAR;
            eotf_gamma = 1.0f;
        } else if (eotf_name == "PQ") {
            eotf_type = EotfType::PQ;
            eotf_gamma = 1.0f;
        } else if (eotf_name == "HLG") {
            eotf_type = EotfType::HLG;
            eotf_gamma = (display.color_space.gamma > 0.0f) ? display.color_space.gamma : 1.2f;
        } else if (eotf_name == "gamma") {
            eotf_type = EotfType::GAMMA;
            eotf_gamma = display.color_space.gamma;
        } else {
            eotf_type = EotfType::SRGB;
            eotf_gamma = 2.2f;
        }

        if (!rgb2xyz_d) {
            GPU_CHECK(hipMalloc(&rgb2xyz_d, 9 * sizeof(float)));
        }
        GPU_CHECK(hipMemcpyHtoD(rgb2xyz_d, display.color_space.rgb2xyz, 9 * sizeof(float)));

        ppd = display.get_ppd();

        // Initialize Laplacian pyramid (start with 3 sustained channels; transient added later)
        lpyr.init(width, height, ppd, 3);

        // Initialize CSF
        hipStreamCreate(&stream);
        const std::filesystem::path csf_path = data_root / ("csf_lut_" + params.csf + ".json");
        csf.init(csf_path.string(), stream);

        // Initialize temporal filtering
        if (temporal_filtering_enabled && !params.sigma_tf.empty()) {
            temporal.init(
                lpyr.num_bands,
                lpyr.band_widths,
                lpyr.band_heights,
                params.sigma_tf,
                params.beta_tf,
                frame_rate,
                stream
            );
        }

        // Initialize cross-channel masking weights
        if (cross_channel_masking_enabled && !params.xcm_weights.empty()) {
            if (params.xcm_weights.size() != 16) {
                std::cerr << "[CVVDP] Warning: xcm_weights should have 16 elements (4×4 matrix), got "
                          << params.xcm_weights.size() << ". Disabling cross-channel masking." << std::endl;
                cross_channel_masking_enabled = false;
            } else {
                std::array<float, 16> weights_pow{};
                for (size_t i = 0; i < 16; ++i) {
                    weights_pow[i] = std::pow(2.0f, params.xcm_weights[i]);
                }
                GPU_CHECK(hipMalloc(&xcm_weights_d, 16 * sizeof(float)));
                GPU_CHECK(hipMemcpyHtoD(xcm_weights_d, weights_pow.data(), 16 * sizeof(float)));
            }
        }

        phase_uncertainty_multiplier = std::pow(10.0f, params.mask_c);
        pu_padsize = params.pu_dilate * 2;
        pu_kernel_radius = 0;
        if (pu_kernel_d) {
            hipFree(pu_kernel_d);
            pu_kernel_d = nullptr;
        }
        if (params.pu_dilate != 0) {
            const int kernel_size = static_cast<int>(params.pu_dilate * 4) + 1;
            pu_kernel_radius = kernel_size / 2;
            std::vector<float> kernel(kernel_size);
            const float sigma = static_cast<float>(params.pu_dilate);
            float sum = 0.0f;
            for (int i = -pu_kernel_radius; i <= pu_kernel_radius; ++i) {
                float weight = std::exp(-(static_cast<float>(i * i)) / (2.0f * sigma * sigma));
                kernel[i + pu_kernel_radius] = weight;
                sum += weight;
            }
            if (sum > 0.0f) {
                for (float& weight : kernel) {
                    weight /= sum;
                }
            }
            GPU_CHECK(hipMalloc(&pu_kernel_d, kernel_size * sizeof(float)));
            GPU_CHECK(hipMemcpyHtoD(pu_kernel_d, kernel.data(), kernel_size * sizeof(float)));
        }
    }

    void destroy() {
        if (xcm_weights_d) {
            hipFree(xcm_weights_d);
            xcm_weights_d = nullptr;
        }
        if (pu_kernel_d) {
            hipFree(pu_kernel_d);
            pu_kernel_d = nullptr;
            pu_kernel_radius = 0;
            pu_padsize = 0;
        }
        if (rgb2xyz_d) {
            hipFree(rgb2xyz_d);
            rgb2xyz_d = nullptr;
        }
        temporal.destroy();
        csf.destroy();
        hipStreamDestroy(stream);
    }

    // Allow setting frame rate for temporal filtering
    void set_frame_rate(float fps) {
        frame_rate = fps;
    }

    // Allow disabling temporal filtering (for testing)
    void set_temporal_filtering(bool enabled) {
        temporal_filtering_enabled = enabled;
    }

    // Process a single frame pair and return JOD score
    // Input: test and reference frames in planar RGB format (float3 arrays)
    float process_frame(const float3* test_d, const float3* ref_d, int frame_index = -1, std::string* debug_json_out = nullptr) {
        const bool collect_debug = (debug_json_out != nullptr);

        struct KernelTimings {
            float apply_csf_and_gain = 0.0f;
            float mutual_mask = 0.0f;
            float phase_uncertainty = 0.0f;
            float cross_channel_mask = 0.0f;
            float masking = 0.0f;
            float baseband = 0.0f;
            float spatial_pool = 0.0f;
            float reduce = 0.0f;
            float host_copy = 0.0f;
        };

        struct ChannelStats {
            float min = std::numeric_limits<float>::max();
            float max = std::numeric_limits<float>::lowest();
            double sum = 0.0;
        };

        struct BandDebugInfo {
            int index = 0;
            bool is_baseband = false;
            int width = 0;
            int height = 0;
            float rho_cpd = 0.0f;
            float beta_score = 0.0f;
            float3 beta_scores = make_float3(0.0f, 0.0f, 0.0f);
            int sample_count = 0;
            std::array<ChannelStats, 3> channels{};
        };

        KernelTimings timings;
        std::vector<BandDebugInfo> band_debug_infos;
        if (collect_debug) {
            band_debug_infos.reserve(lpyr.num_bands);
        }

        hipEvent_t timing_start = nullptr;
        hipEvent_t timing_end = nullptr;
        if (collect_debug) {
            hipEventCreateWithFlags(&timing_start, hipEventDefault);
            hipEventCreateWithFlags(&timing_end, hipEventDefault);
        }

        ClampMode clamp_mode = ClampMode::SOFT;
        if (params.dclamp_type == "hard") {
            clamp_mode = ClampMode::HARD;
        } else if (params.dclamp_type == "none") {
            clamp_mode = ClampMode::NONE;
        }
        const int clamp_mode_value = static_cast<int>(clamp_mode);

        int frame_size = width * height;
        const int contrast_mode = static_cast<int>(parse_contrast_mode(params.contrast));

        // Allocate working buffers
        float3 *test_linear_d, *ref_linear_d;
        GPU_CHECK(hipMallocAsync(&test_linear_d, frame_size * sizeof(float3), stream));
        GPU_CHECK(hipMallocAsync(&ref_linear_d, frame_size * sizeof(float3), stream));

        // Copy input
        GPU_CHECK(hipMemcpyDtoDAsync(test_linear_d, test_d, frame_size * sizeof(float3), stream));
        GPU_CHECK(hipMemcpyDtoDAsync(ref_linear_d, ref_d, frame_size * sizeof(float3), stream));

        // Step 1: Convert to linear luminance
        const float L_max = display.max_luminance;
        const float L_black = display.get_black_level();

        rgb_to_linear(test_linear_d, frame_size, eotf_type, eotf_gamma, L_max, L_black, stream);
        rgb_to_linear(ref_linear_d, frame_size, eotf_type, eotf_gamma, L_max, L_black, stream);

        // Step 2: Convert to DKL opponent color space
        rgb_to_dkl(test_linear_d, frame_size, rgb2xyz_d, stream);
        rgb_to_dkl(ref_linear_d, frame_size, rgb2xyz_d, stream);

        // Step 3: Build Laplacian pyramid for both images
        float** test_laplacian = new float*[lpyr.num_bands];
        float** ref_laplacian = new float*[lpyr.num_bands];
        float** test_gaussian = new float*[lpyr.num_bands];
        float** ref_gaussian = new float*[lpyr.num_bands];

        lpyr.decompose(reinterpret_cast<const float*>(test_linear_d),
                       test_laplacian,
                       test_gaussian,
                       stream);
        lpyr.decompose(reinterpret_cast<const float*>(ref_linear_d),
                       ref_laplacian,
                       ref_gaussian,
                       stream);

        // Step 4: Apply CSF and masking model for each band
        // Store per-channel scores for hierarchical pooling
        std::vector<float4> band_scores(lpyr.num_bands, float4{0.0f, 0.0f, 0.0f, 0.0f});
        std::vector<int> band_channel_counts(lpyr.num_bands, 3);

        const float beta_tch_value =
            (params.beta_tch > 0.0f) ? params.beta_tch
                                     : ((params.beta > 0.0f) ? params.beta : 1.0f);

        float channel_weights[kMaxChannels] = {1.0f, params.ch_chrom_w, params.ch_chrom_w, params.ch_trans_w};

        float4 mask_q_vec{1.0f, 1.0f, 1.0f, 1.0f};
        if (!params.mask_q.empty()) {
            mask_q_vec.x = params.mask_q.size() > 0 ? params.mask_q[0] : 1.0f;
            mask_q_vec.y = params.mask_q.size() > 1 ? params.mask_q[1] : mask_q_vec.x;
            mask_q_vec.z = params.mask_q.size() > 2 ? params.mask_q[2] : mask_q_vec.y;
            mask_q_vec.w = params.mask_q.size() > 3 ? params.mask_q[3] : mask_q_vec.x;
        }

        auto combine_channels = [&](const float4& channel_values, int active_channels) -> float {
            float sum = 0.0f;
            for (int c = 0; c < active_channels; ++c) {
                const float value = fmaxf(((&channel_values.x)[c]), 0.0f);
                sum += channel_weights[c] * powf(value, beta_tch_value);
            }
            return (sum > 0.0f) ? powf(sum, 1.0f / beta_tch_value) : 0.0f;
        };

        for (int band = 0; band < lpyr.num_bands; band++) {
            int band_size = lpyr.band_widths[band] * lpyr.band_heights[band];
            float rho = lpyr.band_freqs[band]; // Spatial frequency in cpd
            bool is_baseband = (band == lpyr.num_bands - 1);

            int threads = 256;
            int blocks = (band_size + threads - 1) / threads;

            // Pyramid bands already contain contrast (clamped to [-1000, 1000])
            float* test_contrast = reinterpret_cast<float*>(test_laplacian[band]);
            float* ref_contrast = reinterpret_cast<float*>(ref_laplacian[band]);

            // Allocate output buffer for D (perceptually weighted differences)
            float4* D_d;
            GPU_CHECK(hipMallocAsync(&D_d, band_size * sizeof(float4), stream));

            // Extract per-pixel background luminance from Gaussian pyramid Y-channel
            // This provides local adaptation for Weber contrast
            float* log_L_bkg_test_d = nullptr;
            float* log_L_bkg_ref_d = nullptr;

            GPU_CHECK(hipMallocAsync(&log_L_bkg_test_d, band_size * sizeof(float), stream));
            GPU_CHECK(hipMallocAsync(&log_L_bkg_ref_d, band_size * sizeof(float), stream));

            // Launch kernel to extract Y-channel and compute log10(max(Y, min_bkg))
            extract_log_luminance_kernel<<<blocks, threads, 0, stream>>>(
                reinterpret_cast<float3*>(test_gaussian[band]),
                reinterpret_cast<float3*>(ref_gaussian[band]),
                log_L_bkg_test_d,
                log_L_bkg_ref_d,
                band_size
            );

            TemporalBuffer::BandBuffer* temporal_band =
                (temporal_filtering_enabled && band < temporal.num_bands)
                    ? &temporal.bands[band]
                    : nullptr;
            int band_channel_count = 3;

            if (is_baseband) {
                band_channel_count = (temporal_band && temporal_band->frames_processed > 1) ? 4 : 3;
                if (collect_debug) GPU_CHECK(hipEventRecord(timing_start, stream));
                baseband_difference_kernel<<<blocks, threads, 0, stream>>>(
                    test_contrast, ref_contrast,
                    log_L_bkg_test_d, log_L_bkg_ref_d,
                    D_d,
                    lpyr.band_widths[band], lpyr.band_heights[band],
                    rho,
                    csf.log_L_bkg_d, csf.log_rho_d,
                    csf.logS_o0_c0_d, csf.logS_o0_c1_d, csf.logS_o0_c2_d, csf.logS_o1_c0_d,
                    csf.num_L_bkg, csf.num_rho, band_channel_count, params.sensitivity_correction
                );
                if (collect_debug) {
                    GPU_CHECK(hipEventRecord(timing_end, stream));
                    GPU_CHECK(hipEventSynchronize(timing_end));
                    float elapsed = 0.0f;
                    GPU_CHECK(hipEventElapsedTime(&elapsed, timing_start, timing_end));
                    timings.baseband += elapsed;
                }
            } else {
                float4 *T_p_d, *R_p_d;
                GPU_CHECK(hipMallocAsync(&T_p_d, band_size * sizeof(float4), stream));
                GPU_CHECK(hipMallocAsync(&R_p_d, band_size * sizeof(float4), stream));

                if (collect_debug) GPU_CHECK(hipEventRecord(timing_start, stream));
                apply_csf_and_gain_kernel<<<blocks, threads, 0, stream>>>(
                    test_contrast, ref_contrast,
                    log_L_bkg_test_d, log_L_bkg_ref_d,
                    T_p_d, R_p_d,
                    band_size,
                    rho,
                    csf.log_L_bkg_d, csf.log_rho_d,
                    csf.logS_o0_c0_d, csf.logS_o0_c1_d, csf.logS_o0_c2_d, csf.logS_o1_c0_d,
                    csf.num_L_bkg, csf.num_rho, temporal_band ? 4 : 3, params.sensitivity_correction
                );
                if (collect_debug) {
                    GPU_CHECK(hipEventRecord(timing_end, stream));
                    GPU_CHECK(hipEventSynchronize(timing_end));
                    float elapsed = 0.0f;
                    GPU_CHECK(hipEventElapsedTime(&elapsed, timing_start, timing_end));
                    timings.apply_csf_and_gain += elapsed;
                }

                // Apply temporal filtering if enabled
                if (temporal_band) {
                    apply_temporal_filtering(
                        T_p_d, R_p_d,
                        *temporal_band,
                        temporal,
                        lpyr.band_widths[band], lpyr.band_heights[band],
                        stream
                    );
                }

                band_channel_count = (temporal_band && temporal_band->frames_processed > 1) ? 4 : 3;

                float4* M_mm_d;
                GPU_CHECK(hipMallocAsync(&M_mm_d, band_size * sizeof(float4), stream));

                if (collect_debug) GPU_CHECK(hipEventRecord(timing_start, stream));
                compute_mutual_masking_kernel<<<blocks, threads, 0, stream>>>(
                    T_p_d, R_p_d, M_mm_d, band_size
                );
                if (collect_debug) {
                    GPU_CHECK(hipEventRecord(timing_end, stream));
                    GPU_CHECK(hipEventSynchronize(timing_end));
                    float elapsed = 0.0f;
                    GPU_CHECK(hipEventElapsedTime(&elapsed, timing_start, timing_end));
                    timings.mutual_mask += elapsed;
                }

                const bool do_phase_blur =
                    (pu_kernel_d != nullptr) &&
                    (lpyr.band_widths[band] > pu_padsize) &&
                    (lpyr.band_heights[band] > pu_padsize);

                if (collect_debug) GPU_CHECK(hipEventRecord(timing_start, stream));
                if (do_phase_blur) {
                    float4* blur_tmp_d;
                    GPU_CHECK(hipMallocAsync(&blur_tmp_d, band_size * sizeof(float4), stream));
                    gaussian_blur_horizontal_kernel<<<blocks, threads, 0, stream>>>(
                        M_mm_d, blur_tmp_d,
                        lpyr.band_widths[band], lpyr.band_heights[band],
                        pu_kernel_d, pu_kernel_radius
                    );
                    gaussian_blur_vertical_kernel<<<blocks, threads, 0, stream>>>(
                        blur_tmp_d, M_mm_d,
                        lpyr.band_widths[band], lpyr.band_heights[band],
                        pu_kernel_d, pu_kernel_radius
                    );
                    GPU_CHECK(hipFreeAsync(blur_tmp_d, stream));
                }
                apply_phase_uncertainty_kernel<<<blocks, threads, 0, stream>>>(
                    M_mm_d, phase_uncertainty_multiplier, band_size
                );
                if (collect_debug) {
                    GPU_CHECK(hipEventRecord(timing_end, stream));
                    GPU_CHECK(hipEventSynchronize(timing_end));
                    float elapsed = 0.0f;
                    GPU_CHECK(hipEventElapsedTime(&elapsed, timing_start, timing_end));
                    timings.phase_uncertainty += elapsed;
                }

                float4* M_pow_d;
                GPU_CHECK(hipMallocAsync(&M_pow_d, band_size * sizeof(float4), stream));
                apply_channel_power_kernel<<<blocks, threads, 0, stream>>>(
                    M_mm_d, M_pow_d,
                    mask_q_vec,
                    band_channel_count,
                    band_size
                );

                float4* M_xcm_d;
                GPU_CHECK(hipMallocAsync(&M_xcm_d, band_size * sizeof(float4), stream));

                if (cross_channel_masking_enabled && xcm_weights_d) {
                    if (collect_debug) GPU_CHECK(hipEventRecord(timing_start, stream));
                    apply_cross_channel_masking_kernel<<<blocks, threads, 0, stream>>>(
                        M_pow_d, M_xcm_d, xcm_weights_d,
                        band_channel_count,
                        band_size
                    );
                    if (collect_debug) {
                        GPU_CHECK(hipEventRecord(timing_end, stream));
                        GPU_CHECK(hipEventSynchronize(timing_end));
                        float elapsed = 0.0f;
                        GPU_CHECK(hipEventElapsedTime(&elapsed, timing_start, timing_end));
                        timings.cross_channel_mask += elapsed;
                    }
                } else {
                    GPU_CHECK(hipMemcpyDtoDAsync(M_xcm_d, M_pow_d, band_size * sizeof(float4), stream));
                }

                if (collect_debug) GPU_CHECK(hipEventRecord(timing_start, stream));
                apply_masking_kernel<<<blocks, threads, 0, stream>>>(
                    T_p_d, R_p_d, M_xcm_d, D_d,
                    params.mask_p,
                    band_channel_count,
                    clamp_mode_value,
                    params.d_max,
                    band_size
                );
                if (collect_debug) {
                    GPU_CHECK(hipEventRecord(timing_end, stream));
                    GPU_CHECK(hipEventSynchronize(timing_end));
                    float elapsed = 0.0f;
                    GPU_CHECK(hipEventElapsedTime(&elapsed, timing_start, timing_end));
                    timings.masking += elapsed;
                }

                GPU_CHECK(hipFreeAsync(T_p_d, stream));
                GPU_CHECK(hipFreeAsync(R_p_d, stream));
                GPU_CHECK(hipFreeAsync(M_mm_d, stream));
                GPU_CHECK(hipFreeAsync(M_pow_d, stream));
                GPU_CHECK(hipFreeAsync(M_xcm_d, stream));
            }

            float4* partial_sums_d;
            int num_blocks = (band_size + 255) / 256;
            GPU_CHECK(hipMallocAsync(&partial_sums_d, num_blocks * sizeof(float4), stream));

            if (collect_debug) GPU_CHECK(hipEventRecord(timing_start, stream));
            reduce_lp_sum_per_channel_kernel<<<num_blocks, 256, 256 * sizeof(float4), stream>>>(
                D_d, partial_sums_d, params.beta, band_channel_count, band_size
            );
            if (collect_debug) {
                GPU_CHECK(hipEventRecord(timing_end, stream));
                GPU_CHECK(hipEventSynchronize(timing_end));
                float elapsed = 0.0f;
                GPU_CHECK(hipEventElapsedTime(&elapsed, timing_start, timing_end));
                timings.reduce += elapsed;
            }

            std::vector<float4> partial_sums_h(num_blocks);
            std::vector<float4> band_values_h;
            if (collect_debug) {
                band_values_h.resize(band_size);
                GPU_CHECK(hipEventRecord(timing_start, stream));
                GPU_CHECK(hipMemcpyDtoHAsync(band_values_h.data(), D_d, band_size * sizeof(float4), stream));
            }
            GPU_CHECK(hipMemcpyDtoHAsync(partial_sums_h.data(), partial_sums_d,
                                      num_blocks * sizeof(float4), stream));
            if (collect_debug) GPU_CHECK(hipEventRecord(timing_end, stream));

            hipStreamSynchronize(stream);

            if (collect_debug) {
                GPU_CHECK(hipEventSynchronize(timing_end));
                float elapsed = 0.0f;
                GPU_CHECK(hipEventElapsedTime(&elapsed, timing_start, timing_end));
                timings.host_copy += elapsed;
            }

            double channel_sums[kMaxChannels] = {0.0, 0.0, 0.0, 0.0};
            for (const float4& value : partial_sums_h) {
                channel_sums[0] += static_cast<double>(value.x);
                channel_sums[1] += static_cast<double>(value.y);
                channel_sums[2] += static_cast<double>(value.z);
                channel_sums[3] += static_cast<double>(value.w);
            }

            const int sample_count = std::max(1, band_size);
            const float beta_value = (params.beta > 0.0f) ? params.beta : 1.0f;
            const float inv_beta = 1.0f / beta_value;

            float4 band_lp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int c = 0; c < band_channel_count; ++c) {
                const float avg = fmaxf(static_cast<float>(channel_sums[c]) / static_cast<float>(sample_count), 0.0f);
                (&band_lp.x)[c] = powf(avg, inv_beta);
            }
            band_scores[band] = band_lp;
            band_channel_counts[band] = band_channel_count;

            std::cerr << "DEBUG_BAND[" << band << "]: lp_norm=("
                      << band_lp.x << ", " << band_lp.y << ", " << band_lp.z << ", " << band_lp.w << ") "
                      << "beta=" << beta_value << std::endl;

            if (collect_debug) {
                BandDebugInfo info;
                info.index = band;
                info.is_baseband = is_baseband;
                info.width = lpyr.band_widths[band];
                info.height = lpyr.band_heights[band];
                info.rho_cpd = rho;
                info.beta_scores = make_float3(band_scores[band].x, band_scores[band].y, band_scores[band].z);
                info.beta_score = combine_channels(band_scores[band], band_channel_count);
                info.sample_count = band_size;

                for (const float4& value : band_values_h) {
                    const float channels[kMaxChannels] = {value.x, value.y, value.z, value.w};
                    for (int c = 0; c < 3; ++c) {
                        ChannelStats& stats = info.channels[c];
                        stats.min = std::min(stats.min, channels[c]);
                        stats.max = std::max(stats.max, channels[c]);
                        stats.sum += channels[c];
                    }
                }

                band_debug_infos.push_back(std::move(info));
            }

            GPU_CHECK(hipFreeAsync(D_d, stream));
            GPU_CHECK(hipFreeAsync(partial_sums_d, stream));
            GPU_CHECK(hipFreeAsync(log_L_bkg_test_d, stream));
            GPU_CHECK(hipFreeAsync(log_L_bkg_ref_d, stream));
        }

        lpyr.free_levels(test_laplacian, stream);
        lpyr.free_levels(ref_laplacian, stream);
        lpyr.free_levels(test_gaussian, stream);
        lpyr.free_levels(ref_gaussian, stream);
        delete[] test_laplacian;
        delete[] ref_laplacian;
        delete[] test_gaussian;
        delete[] ref_gaussian;

        GPU_CHECK(hipFreeAsync(test_linear_d, stream));
        GPU_CHECK(hipFreeAsync(ref_linear_d, stream));

        // Hierarchical pooling across spatial bands using beta_sch with per-channel/baseband weights
        const float beta_sch = (params.beta_sch > 0.0f)
                                   ? params.beta_sch
                                   : ((params.beta > 0.0f) ? params.beta : 1.0f);

        float baseband_weights[kMaxChannels] = {1.0f, 1.0f, 1.0f, 1.0f};
        if (!params.baseband_weight.empty()) {
            for (int c = 0; c < static_cast<int>(params.baseband_weight.size()) && c < kMaxChannels; ++c) {
                baseband_weights[c] = params.baseband_weight[c];
            }
        }

        float channel_band_sums[kMaxChannels] = {0.0f, 0.0f, 0.0f, 0.0f};
        int active_channel_count = 3;
        for (int band = 0; band < lpyr.num_bands; ++band) {
            const bool is_baseband_band = (band == lpyr.num_bands - 1);
            const int band_channel_count = band_channel_counts[band];
            active_channel_count = std::max(active_channel_count, band_channel_count);
            for (int c = 0; c < band_channel_count; ++c) {
                const float component = fmaxf(((&band_scores[band].x)[c]), 0.0f);
                const float weight = channel_weights[c] * (is_baseband_band ? baseband_weights[c] : 1.0f);
                channel_band_sums[c] += powf(component * weight, beta_sch);
            }
        }

        float channel_q_s[kMaxChannels] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int c = 0; c < active_channel_count; ++c) {
            const float sum = channel_band_sums[c];
            channel_q_s[c] = (sum > 0.0f) ? powf(sum, 1.0f / beta_sch) : 0.0f;
        }

        float sum_channels = 0.0f;
        for (int c = 0; c < active_channel_count; ++c) {
            sum_channels += powf(fmaxf(channel_q_s[c], 0.0f), beta_tch_value);
        }
        float Q_tc = (sum_channels > 0.0f)
                         ? powf(sum_channels, 1.0f / beta_tch_value)
                         : 0.0f;

        bool image_mode = true;
        if (temporal_filtering_enabled && temporal.num_bands > 0) {
            image_mode = true;
            for (const auto& band : temporal.bands) {
                if (band.frames_processed > 1) {
                    image_mode = false;
                    break;
                }
            }
        }

        const float image_int_factor = image_mode ? params.image_int : 1.0f;
        const float Q_per_ch = Q_tc * image_int_factor;

        // DEBUG: Print aggregated scores
        std::cerr << "DEBUG_AGG: sum_bands=("
                  << channel_band_sums[0] << ", " << channel_band_sums[1] << ", "
                  << channel_band_sums[2] << ", " << channel_band_sums[3] << ")" << std::endl;
        std::cerr << "DEBUG_AGG: beta_sch=" << beta_sch << " num_bands=" << lpyr.num_bands << std::endl;
        std::cerr << "DEBUG_AGG: Q_spatial_per_channel=("
                  << channel_q_s[0] << ", " << channel_q_s[1] << ", "
                  << channel_q_s[2] << ", " << channel_q_s[3] << ")" << std::endl;
        std::cerr << "DEBUG_AGG: beta_tch=" << beta_tch_value << " ch_chrom_w=" << params.ch_chrom_w << std::endl;
        std::cerr << "DEBUG_AGG: Q_per_ch=" << Q_per_ch << std::endl;

        // Convert Q to JOD scale (matches Python: Q_JOD = 10 - jod_a * Q^jod_exp)
        // Use linearization for small Q values to maintain differentiability
        const float Q_t = 0.1f;  // Threshold for linearization
        float Q_jod;

        if (Q_per_ch <= Q_t) {
            // Linearized version: jod_a_p = jod_a * Q_t^(jod_exp-1)
            float jod_a_p = params.jod_a * powf(Q_t, params.jod_exp - 1.0f);
            Q_jod = 10.0f - jod_a_p * Q_per_ch;
        } else {
            Q_jod = 10.0f - params.jod_a * powf(Q_per_ch, params.jod_exp);
        }

        Q_jod = fminf(fmaxf(Q_jod, 0.0f), 10.0f);

        // DEBUG: Print final JOD calculation
        std::cerr << "DEBUG_JOD: jod_a=" << params.jod_a << " jod_exp=" << params.jod_exp << std::endl;
        std::cerr << "DEBUG_JOD: Q_per_ch=" << Q_per_ch << " Q_t=" << Q_t << std::endl;
        std::cerr << "DEBUG_JOD: using_linearization=" << (Q_per_ch <= Q_t ? "true" : "false") << std::endl;
        std::cerr << "DEBUG_JOD: Q_jod (final)=" << Q_jod << std::endl;

        if (collect_debug) {
            GPU_CHECK(hipEventDestroy(timing_start));
            GPU_CHECK(hipEventDestroy(timing_end));
        }

        if (collect_debug && debug_json_out) {
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

            const std::string data_root_utf8 = data_root.u8string();

            writer.StartObject();
            writer.Key("metric");
            writer.String("cvvdp");
            writer.Key("frame");
            writer.Int(frame_index);
            writer.Key("jod");
            writer.Double(static_cast<double>(Q_jod));
            writer.Key("width");
            writer.Int(width);
            writer.Key("height");
            writer.Int(height);
            writer.Key("ppd");
            writer.Double(static_cast<double>(ppd));
            writer.Key("display");
            writer.StartObject();
            writer.Key("name");
            writer.String(display.name.c_str());
            writer.Key("max_luminance");
            writer.Double(static_cast<double>(display.max_luminance));
            writer.Key("min_luminance");
            writer.Double(static_cast<double>(display.min_luminance));
            writer.Key("contrast");
            writer.Double(static_cast<double>(display.contrast));
            writer.EndObject();
            writer.Key("config_root");
            writer.String(data_root_utf8.c_str());
            writer.Key("temporal");
            writer.StartObject();
            writer.Key("buffer_fill");
            writer.Double(0.0); // Temporal filtering not yet implemented
            writer.Key("filter_len");
            writer.Int(params.filter_len);
            writer.EndObject();
            writer.Key("bands");
            writer.StartArray();
            for (const BandDebugInfo& info : band_debug_infos) {
                writer.StartObject();
                writer.Key("index");
                writer.Int(info.index);
                writer.Key("is_baseband");
                writer.Bool(info.is_baseband);
                writer.Key("rho_cpd");
                writer.Double(static_cast<double>(info.rho_cpd));
                writer.Key("width");
                writer.Int(info.width);
                writer.Key("height");
                writer.Int(info.height);
                writer.Key("beta_score");
                writer.Double(static_cast<double>(info.beta_score));
                writer.Key("beta_scores");
                writer.StartObject();
                const char* channel_names[3] = {"Y", "RG", "BY"};
                const float beta_components[3] = {
                    info.beta_scores.x,
                    info.beta_scores.y,
                    info.beta_scores.z
                };
                for (int c = 0; c < 3; ++c) {
                    writer.Key(channel_names[c]);
                    writer.Double(static_cast<double>(beta_components[c]));
                }
                writer.EndObject();
                writer.Key("channels");
                writer.StartObject();
                for (int c = 0; c < 3; ++c) {
                    writer.Key(channel_names[c]);
                    writer.StartObject();
                    writer.Key("min");
                    writer.Double(static_cast<double>(info.channels[c].min));
                    writer.Key("max");
                    writer.Double(static_cast<double>(info.channels[c].max));
                    double mean = (info.sample_count > 0)
                                      ? info.channels[c].sum / static_cast<double>(info.sample_count)
                                      : 0.0;
                    writer.Key("mean");
                    writer.Double(mean);
                    writer.EndObject();
                }
                writer.EndObject();
                writer.EndObject();
            }
            writer.EndArray();
            writer.Key("kernels_ms");
            writer.StartObject();
            writer.Key("apply_csf_and_gain");
            writer.Double(static_cast<double>(timings.apply_csf_and_gain));
            writer.Key("mutual_mask");
            writer.Double(static_cast<double>(timings.mutual_mask));
            writer.Key("phase_uncertainty");
            writer.Double(static_cast<double>(timings.phase_uncertainty));
            writer.Key("cross_channel_mask");
            writer.Double(static_cast<double>(timings.cross_channel_mask));
            writer.Key("masking");
            writer.Double(static_cast<double>(timings.masking));
            writer.Key("baseband");
            writer.Double(static_cast<double>(timings.baseband));
            writer.Key("spatial_pool");
            writer.Double(static_cast<double>(timings.spatial_pool));
            writer.Key("reduce");
            writer.Double(static_cast<double>(timings.reduce));
            writer.Key("host_copy");
            writer.Double(static_cast<double>(timings.host_copy));
            writer.EndObject();
            writer.EndObject();

            *debug_json_out = buffer.GetString();
        }

        return Q_jod;
    }
};

// Simplified API for integration
struct CVVDPHandle {
    CVVDPProcessor processor;
    int width, height;
};

inline CVVDPHandle* cvvdp_init(int width, int height, const char* display_name) {
    try {
        CVVDPHandle* handle = new CVVDPHandle();
        handle->width = width;
        handle->height = height;
        handle->processor.init(width, height, std::string(display_name));
        return handle;
    } catch (const VshipError& e) {
        return nullptr;
    }
}

inline float cvvdp_compute(CVVDPHandle* handle, const float3* test_d, const float3* ref_d) {
    if (!handle) return -1.0f;

    try {
        return handle->processor.process_frame(test_d, ref_d, -1, nullptr);
    } catch (const VshipError& e) {
        return -1.0f;
    }
}

inline void cvvdp_free(CVVDPHandle* handle) {
    if (handle) {
        handle->processor.destroy();
        delete handle;
    }
}

// Kernel to convert planar uint16 RGB to packed float3
__global__ void cvvdp_convert_planar_uint16_to_float3(
    float3* output,
    const uint8_t* r_plane,
    const uint8_t* g_plane,
    const uint8_t* b_plane,
    int width, int height, int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    int y = idx / width;
    int x = idx % width;
    int plane_idx = y * (stride / sizeof(uint16_t)) + x;

    const uint16_t* r_ptr = (const uint16_t*)r_plane;
    const uint16_t* g_ptr = (const uint16_t*)g_plane;
    const uint16_t* b_ptr = (const uint16_t*)b_plane;

    float3 rgb;
    // Convert uint16 (0-65535) to float (0.0-1.0)
    rgb.x = r_ptr[plane_idx] / 65535.0f;
    rgb.y = g_ptr[plane_idx] / 65535.0f;
    rgb.z = b_ptr[plane_idx] / 65535.0f;

    output[idx] = rgb;
}

// Computing implementation wrapper for FFVship integration
class CVVDPComputingImplementation {
private:
    CVVDPProcessor processor;
    int width, height;
    hipStream_t stream;
    std::string display_name;
    std::string data_dir_override;
    bool debug_enabled = false;

    // Temporary buffers for converting uint16 planar RGB to float3
    float3* temp_src_d = nullptr;
    float3* temp_dist_d = nullptr;

public:
    void init(int w, int h, const std::string& display = "standard_4k",
              const std::string& data_dir = "", bool enable_debug = false) {
        width = w;
        height = h;
        display_name = display;
        data_dir_override = data_dir;
        debug_enabled = enable_debug;

        // Initialize the processor
        processor.init(width, height, display_name, data_dir_override);

        // Create stream
        hipStreamCreate(&stream);

        // Allocate temporary conversion buffers
        size_t buffer_size = width * height * sizeof(float3);
        GPU_CHECK(hipMalloc(&temp_src_d, buffer_size));
        GPU_CHECK(hipMalloc(&temp_dist_d, buffer_size));
    }

    void destroy() {
        processor.destroy();

        if (temp_src_d) {
            hipFree(temp_src_d);
            temp_src_d = nullptr;
        }
        if (temp_dist_d) {
            hipFree(temp_dist_d);
            temp_dist_d = nullptr;
        }

        hipStreamDestroy(stream);
    }

    template <InputMemType T>
    std::tuple<float, float, float> run(
        const uint8_t* distmap,  // Unused for CVVDP (no distortion map output yet)
        int distmapstride,       // Unused
        const uint8_t* src_planes[3],
        const uint8_t* dist_planes[3],
        int64_t stride_src,
        int64_t stride_dist,
        int frame_index = -1
    ) {
        // Only support UINT16 for now
        static_assert(T == InputMemType::UINT16, "CVVDP currently only supports UINT16 input");

        // Convert planar uint16 RGB to packed float3
        int threads = 256;
        int blocks = (width * height + threads - 1) / threads;

        cvvdp_convert_planar_uint16_to_float3<<<blocks, threads, 0, stream>>>(
            temp_src_d, src_planes[0], src_planes[1], src_planes[2],
            width, height, stride_src
        );

        cvvdp_convert_planar_uint16_to_float3<<<blocks, threads, 0, stream>>>(
            temp_dist_d, dist_planes[0], dist_planes[1], dist_planes[2],
            width, height, stride_dist
        );

        GPU_CHECK(hipStreamSynchronize(stream));

        // Process with CVVDP
        // Note: CVVDP expects test first, then reference (opposite of other metrics)
        std::string telemetry_blob;
        float jod_score = processor.process_frame(
            temp_dist_d,
            temp_src_d,
            frame_index,
            debug_enabled ? &telemetry_blob : nullptr);

        if (debug_enabled && !telemetry_blob.empty()) {
            static std::mutex telemetry_mutex;
            std::lock_guard<std::mutex> lock(telemetry_mutex);
            std::cout << telemetry_blob << std::endl;
        }

        // Return JOD score in all three positions (CVVDP only outputs one score)
        return std::make_tuple(jod_score, jod_score, jod_score);
    }
};

} // namespace cvvdp










