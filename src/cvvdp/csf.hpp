#pragma once

#include "../util/preprocessor.hpp"
#include "config.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

namespace cvvdp {

// Contrast Sensitivity Function (castleCSF) implementation.
// Loads calibrated lookup tables from ColorVideoVDP JSON assets and exposes
// GPU-resident buffers for kernels that interpolate sensitivities.
struct CastleCSF {
    float* log_L_bkg_d = nullptr;   // log10 background luminance samples
    float* log_rho_d = nullptr;     // log10 spatial frequency samples
    float* logS_o0_c0_d = nullptr;  // sustained Y
    float* logS_o0_c1_d = nullptr;  // sustained RG
    float* logS_o0_c2_d = nullptr;  // sustained BY
    float* logS_o1_c0_d = nullptr;  // transient Y

    int num_L_bkg = 0;
    int num_rho = 0;

    hipStream_t stream = nullptr;

    std::vector<float> log_L_bkg_h;
    std::vector<float> log_rho_h;
    std::vector<float> logS_o0_c0;  // Host-side data for debugging
    std::vector<float> logS_o0_c1;
    std::vector<float> logS_o0_c2;
    std::vector<float> logS_o1_c0;

    void init(const std::string& csf_lut_path, hipStream_t s) {
        stream = s;

        const std::filesystem::path filepath(csf_lut_path);
        rapidjson::Document doc;
        parse_json_document(filepath, doc);

        auto load_scalar_array = [&](const char* key) -> std::vector<float> {
            std::vector<float> values = get_float_array(doc, key, filepath, true);
            if (values.empty()) {
                std::cerr << "[CVVDP] CSF LUT '" << filepath.string()
                          << "' contains empty array for key '" << key << "'" << std::endl;
                throw VshipError(ConfigurationError, __FILE__, __LINE__);
            }
            return values;
        };

        const std::vector<float> L_bkg = load_scalar_array("L_bkg");
        const std::vector<float> rho = load_scalar_array("rho");

        num_L_bkg = static_cast<int>(L_bkg.size());
        num_rho = static_cast<int>(rho.size());

        auto load_plane = [&](const char* key) -> std::vector<float> {
            const rapidjson::Value& value = require_member(doc, key, filepath);
            if (!value.IsArray()) {
                std::cerr << "[CVVDP] CSF LUT '" << filepath.string()
                          << "' key '" << key << "' must be an array of arrays." << std::endl;
                throw VshipError(ConfigurationError, __FILE__, __LINE__);
            }
            if (value.Size() != static_cast<rapidjson::SizeType>(num_L_bkg)) {
                std::cerr << "[CVVDP] CSF LUT '" << filepath.string()
                          << "' key '" << key << "' expected " << num_L_bkg
                          << " rows but received " << value.Size() << std::endl;
                throw VshipError(ConfigurationError, __FILE__, __LINE__);
            }

            std::vector<float> result;
            result.reserve(static_cast<size_t>(num_L_bkg) * static_cast<size_t>(num_rho));

            for (rapidjson::SizeType row = 0; row < value.Size(); ++row) {
                const rapidjson::Value& row_array = value[row];
                if (!row_array.IsArray()) {
                    std::cerr << "[CVVDP] CSF LUT '" << filepath.string()
                              << "' key '" << key << "' row " << row
                              << " must be an array." << std::endl;
                    throw VshipError(ConfigurationError, __FILE__, __LINE__);
                }
                if (row_array.Size() != static_cast<rapidjson::SizeType>(num_rho)) {
                    std::cerr << "[CVVDP] CSF LUT '" << filepath.string()
                              << "' key '" << key << "' row " << row
                              << " expected " << num_rho << " columns but received "
                              << row_array.Size() << std::endl;
                    throw VshipError(ConfigurationError, __FILE__, __LINE__);
                }
                for (rapidjson::SizeType col = 0; col < row_array.Size(); ++col) {
                    const rapidjson::Value& cell = row_array[col];
                    if (!cell.IsNumber()) {
                        std::cerr << "[CVVDP] Non-numeric CSF value at key '" << key
                                  << "' (" << row << "," << col << ") in "
                                  << filepath.string() << std::endl;
                        throw VshipError(ConfigurationError, __FILE__, __LINE__);
                    }
                    result.push_back(static_cast<float>(cell.GetDouble()));
                }
            }
            return result;
        };

        log_L_bkg_h.resize(num_L_bkg);
        log_rho_h.resize(num_rho);
        for (int i = 0; i < num_L_bkg; ++i) {
            log_L_bkg_h[i] = log10f(std::max(L_bkg[i], 1e-6f));
        }
        for (int i = 0; i < num_rho; ++i) {
            log_rho_h[i] = log10f(std::max(rho[i], 1e-6f));
        }

        const size_t axis_L_bytes = static_cast<size_t>(num_L_bkg) * sizeof(float);
        const size_t axis_rho_bytes = static_cast<size_t>(num_rho) * sizeof(float);
        const size_t plane_bytes =
            static_cast<size_t>(num_L_bkg) * static_cast<size_t>(num_rho) * sizeof(float);

        GPU_CHECK(hipMalloc(&log_L_bkg_d, axis_L_bytes));
        GPU_CHECK(hipMalloc(&log_rho_d, axis_rho_bytes));
        GPU_CHECK(hipMemcpyHtoDAsync(log_L_bkg_d, log_L_bkg_h.data(), axis_L_bytes, stream));
        GPU_CHECK(hipMemcpyHtoDAsync(log_rho_d, log_rho_h.data(), axis_rho_bytes, stream));

        logS_o0_c0 = load_plane("o0_c1");
        logS_o0_c1 = load_plane("o0_c2");
        logS_o0_c2 = load_plane("o0_c3");
        logS_o1_c0 = load_plane("o5_c1");

        GPU_CHECK(hipMalloc(&logS_o0_c0_d, plane_bytes));
        GPU_CHECK(hipMalloc(&logS_o0_c1_d, plane_bytes));
        GPU_CHECK(hipMalloc(&logS_o0_c2_d, plane_bytes));
        GPU_CHECK(hipMalloc(&logS_o1_c0_d, plane_bytes));

        GPU_CHECK(hipMemcpyHtoDAsync(logS_o0_c0_d, logS_o0_c0.data(), plane_bytes, stream));
        GPU_CHECK(hipMemcpyHtoDAsync(logS_o0_c1_d, logS_o0_c1.data(), plane_bytes, stream));
        GPU_CHECK(hipMemcpyHtoDAsync(logS_o0_c2_d, logS_o0_c2.data(), plane_bytes, stream));
        GPU_CHECK(hipMemcpyHtoDAsync(logS_o1_c0_d, logS_o1_c0.data(), plane_bytes, stream));

        GPU_CHECK(hipStreamSynchronize(stream));
    }

    void destroy() {
        if (log_L_bkg_d) {
            hipFree(log_L_bkg_d);
            log_L_bkg_d = nullptr;
        }
        if (log_rho_d) {
            hipFree(log_rho_d);
            log_rho_d = nullptr;
        }
        if (logS_o0_c0_d) {
            hipFree(logS_o0_c0_d);
            logS_o0_c0_d = nullptr;
        }
        if (logS_o0_c1_d) {
            hipFree(logS_o0_c1_d);
            logS_o0_c1_d = nullptr;
        }
        if (logS_o0_c2_d) {
            hipFree(logS_o0_c2_d);
            logS_o0_c2_d = nullptr;
        }
        if (logS_o1_c0_d) {
            hipFree(logS_o1_c0_d);
            logS_o1_c0_d = nullptr;
        }
        num_L_bkg = 0;
        num_rho = 0;
        log_L_bkg_h.clear();
        log_rho_h.clear();
        logS_o0_c0.clear();
        logS_o0_c1.clear();
        logS_o0_c2.clear();
        logS_o1_c0.clear();
    }

    // Helper method to log CSF sensitivity values for debugging
    void log_csf_samples(float rho, const float* log_L_bkg_samples, int num_samples, hipStream_t stream) {
        std::vector<float> log_L_host(num_samples);
        GPU_CHECK(hipMemcpyDtoHAsync(log_L_host.data(), log_L_bkg_samples, num_samples * sizeof(float), stream));
        GPU_CHECK(hipStreamSynchronize(stream));
        
        std::cerr << "DEBUG_CSF_LOOKUP: rho=" << rho << "cpd" << std::endl;
        for (int i = 0; i < std::min(5, num_samples); i++) {
            // Perform CSF lookup for this luminance
            float log_rho = log10f(rho);
            
            // Find indices for interpolation (same logic as in kernels)
            float rho_idx = 0.0f;
            for (int j = 0; j < num_rho - 1; j++) {
                if (log_rho_h[j] <= log_rho && log_rho <= log_rho_h[j + 1]) {
                    rho_idx = j + (log_rho - log_rho_h[j]) / (log_rho_h[j + 1] - log_rho_h[j]);
                    break;
                }
            }

            float L_idx = 0.0f;
            for (int j = 0; j < num_L_bkg - 1; j++) {
                if (log_L_bkg_h[j] <= log_L_host[i] && log_L_host[i] <= log_L_bkg_h[j + 1]) {
                    L_idx = j + (log_L_host[i] - log_L_bkg_h[j]) / (log_L_bkg_h[j + 1] - log_L_bkg_h[j]);
                    break;
                }
            }

            // Interpolate sensitivity values (host-side version)
            auto interp2d_host = [](const std::vector<float>& lut, int width, int height, float x, float y) -> float {
                int x0 = static_cast<int>(floorf(x));
                int y0 = static_cast<int>(floorf(y));
                int x1 = std::min(x0 + 1, width - 1);
                int y1 = std::min(y0 + 1, height - 1);
                
                float fx = x - x0;
                float fy = y - y0;
                
                float f00 = lut[y0 * width + x0];
                float f01 = lut[y1 * width + x0];
                float f10 = lut[y0 * width + x1];
                float f11 = lut[y1 * width + x1];
                
                float f0 = f00 * (1.0f - fx) + f10 * fx;
                float f1 = f01 * (1.0f - fx) + f11 * fx;
                
                return f0 * (1.0f - fy) + f1 * fy;
            };
            
            float logS_y = interp2d_host(logS_o0_c0, num_rho, num_L_bkg, rho_idx, L_idx);
            float logS_rg = interp2d_host(logS_o0_c1, num_rho, num_L_bkg, rho_idx, L_idx);
            float logS_by = interp2d_host(logS_o0_c2, num_rho, num_L_bkg, rho_idx, L_idx);
            
            // Convert to linear sensitivity
            float S_y = powf(10.0f, logS_y);
            float S_rg = powf(10.0f, logS_rg);
            float S_by = powf(10.0f, logS_by);
            
            std::cerr << "  log_L_bkg[" << i << "]=" << log_L_host[i] 
                      << " -> logS=(y:" << logS_y << ", rg:" << logS_rg << ", by:" << logS_by << ")"
                      << " S=(y:" << S_y << ", rg:" << S_rg << ", by:" << S_by << ")" << std::endl;
        }
    }
};

// Bilinear interpolation in 2D lookup table
__device__ __forceinline__ float interp2d(const float* lut, int width, int height,
                                          float x, float y) {
    // Clamp coordinates
    x = fmaxf(0.0f, fminf(x, width - 1.0f));
    y = fmaxf(0.0f, fminf(y, height - 1.0f));

    int x0 = static_cast<int>(floorf(x));
    int y0 = static_cast<int>(floorf(y));
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);

    float fx = x - static_cast<float>(x0);
    float fy = y - static_cast<float>(y0);

    float v00 = lut[y0 * width + x0];
    float v10 = lut[y0 * width + x1];
    float v01 = lut[y1 * width + x0];
    float v11 = lut[y1 * width + x1];

    float v0 = v00 * (1.0f - fx) + v10 * fx;
    float v1 = v01 * (1.0f - fx) + v11 * fx;

    return v0 * (1.0f - fy) + v1 * fy;
}

// Kernel to apply CSF weighting to pyramid bands
__launch_bounds__(256)
__global__ void apply_csf_kernel(float3* band, int width, int height,
                                 float rho_cpd, float log_L_bkg,
                                 const float* log_L_bkg_lut, const float* log_rho_lut,
                                 const float* logS_c0, const float* logS_c1, const float* logS_c2,
                                 int num_L_bkg, int num_rho,
                                 float sensitivity_correction) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    // Find position in LUT
    float log_rho = log10f(rho_cpd);

    // Find indices for interpolation
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

    // Convert from log to linear and apply correction
    float S_y = powf(10.0f, logS_y + sensitivity_correction / 20.0f);
    float S_rg = powf(10.0f, logS_rg + sensitivity_correction / 20.0f);
    float S_by = powf(10.0f, logS_by + sensitivity_correction / 20.0f);

    // Weight the DKL channels by sensitivity
    float3 val = band[idx];
    val.x /= S_y;  // Y channel
    val.y /= S_rg; // RG channel
    val.z /= S_by; // BY channel

    band[idx] = val;
}

} // namespace cvvdp
