#pragma once

#include "../util/preprocessor.hpp"
#include "config.hpp"
#include <vector>
#include <cmath>

namespace cvvdp {

// Contrast Sensitivity Function (castleCSF) implementation
// Uses lookup tables for efficient GPU computation

struct CastleCSF {
    // CSF lookup tables (loaded from JSON)
    float* log_L_bkg_d;  // Background luminance values (log scale)
    float* log_rho_d;    // Spatial frequency values (log scale)
    float* logS_o0_c0_d; // Sustained Y channel
    float* logS_o0_c1_d; // Sustained RG channel
    float* logS_o0_c2_d; // Sustained BY channel
    float* logS_o1_c0_d; // Transient Y channel

    int num_L_bkg;
    int num_rho;

    hipStream_t stream;

    void init(const std::string& csf_lut_path, hipStream_t s) {
        stream = s;

        // Load CSF lookup tables from JSON
        // For now, using placeholder values - need to load from actual CSF LUT files
        // The actual implementation would parse csf_lut_weber_fixed_size.json

        // Typical dimensions from the JSON files
        num_L_bkg = 41; // Number of luminance samples
        num_rho = 41;   // Number of frequency samples

        // Allocate device memory for LUTs
        int lut_size = num_L_bkg * num_rho;

        GPU_CHECK(hipMalloc(&log_L_bkg_d, num_L_bkg * sizeof(float)));
        GPU_CHECK(hipMalloc(&log_rho_d, num_rho * sizeof(float)));
        GPU_CHECK(hipMalloc(&logS_o0_c0_d, lut_size * sizeof(float)));
        GPU_CHECK(hipMalloc(&logS_o0_c1_d, lut_size * sizeof(float)));
        GPU_CHECK(hipMalloc(&logS_o0_c2_d, lut_size * sizeof(float)));
        GPU_CHECK(hipMalloc(&logS_o1_c0_d, lut_size * sizeof(float)));

        // TODO: Load actual CSF data from JSON files
        // For now, using simple approximations
        std::vector<float> log_L_bkg_h(num_L_bkg);
        std::vector<float> log_rho_h(num_rho);
        std::vector<float> logS_o0_c0_h(lut_size);
        std::vector<float> logS_o0_c1_h(lut_size);
        std::vector<float> logS_o0_c2_h(lut_size);
        std::vector<float> logS_o1_c0_h(lut_size);

        // Generate sample points (log scale from 0.01 to 10000 cd/m²)
        for (int i = 0; i < num_L_bkg; i++) {
            log_L_bkg_h[i] = log10f(0.01f) + i * (log10f(10000.0f) - log10f(0.01f)) / (num_L_bkg - 1);
        }

        // Spatial frequencies (log scale from 0.1 to 100 cpd)
        for (int i = 0; i < num_rho; i++) {
            log_rho_h[i] = log10f(0.1f) + i * (log10f(100.0f) - log10f(0.1f)) / (num_rho - 1);
        }

        // CSF model based on typical human contrast sensitivity
        // These are approximate values - actual CSF should be loaded from JSON
        for (int l = 0; l < num_L_bkg; l++) {
            for (int r = 0; r < num_rho; r++) {
                int idx = l * num_rho + r;
                float L = powf(10.0f, log_L_bkg_h[l]);
                float rho = powf(10.0f, log_rho_h[r]);

                // CSF model parameters
                // Sensitivity should be "how many JNDs per unit contrast"
                // Higher values = more sensitive = smaller detectable contrasts
                float rho_peak = 4.0f; // Peak around 4 cpd

                // Base sensitivity at 100 cd/m^2 and peak frequency
                // Typical peak CSF is around 100-200 in proper units
                // But we need MUCH higher values to normalize our large contrast values
                float L_adapt = fmaxf(L, 0.1f);
                float base_sensitivity = 50000.0f * powf(L_adapt / 100.0f, 0.3f);

                // Frequency response (band-pass filter shape)
                float log_ratio = log10f(rho / rho_peak);
                float freq_response = expf(-powf(log_ratio, 2.0f) / 0.4f);

                // Low frequency cut-off (reduced sensitivity below 0.5 cpd)
                if (rho < 0.5f) {
                    freq_response *= rho / 0.5f;
                }

                // High frequency roll-off (more aggressive above 30 cpd)
                if (rho > 30.0f) {
                    freq_response *= expf(-(rho - 30.0f) / 20.0f);
                }

                float sensitivity = base_sensitivity * freq_response;

                // Clamp to reasonable range
                sensitivity = fmaxf(1000.0f, fminf(sensitivity, 1000000.0f));

                logS_o0_c0_h[idx] = log10f(sensitivity); // Y sustained
                logS_o0_c1_h[idx] = log10f(sensitivity * 0.6f); // RG (chromatic, lower)
                logS_o0_c2_h[idx] = log10f(sensitivity * 0.4f); // BY (chromatic, even lower)
                logS_o1_c0_h[idx] = log10f(sensitivity * 0.9f); // Y transient (slightly lower)
            }
        }

        // Copy to device
        GPU_CHECK(hipMemcpyHtoDAsync(log_L_bkg_d, log_L_bkg_h.data(), num_L_bkg * sizeof(float), stream));
        GPU_CHECK(hipMemcpyHtoDAsync(log_rho_d, log_rho_h.data(), num_rho * sizeof(float), stream));
        GPU_CHECK(hipMemcpyHtoDAsync(logS_o0_c0_d, logS_o0_c0_h.data(), lut_size * sizeof(float), stream));
        GPU_CHECK(hipMemcpyHtoDAsync(logS_o0_c1_d, logS_o0_c1_h.data(), lut_size * sizeof(float), stream));
        GPU_CHECK(hipMemcpyHtoDAsync(logS_o0_c2_d, logS_o0_c2_h.data(), lut_size * sizeof(float), stream));
        GPU_CHECK(hipMemcpyHtoDAsync(logS_o1_c0_d, logS_o1_c0_h.data(), lut_size * sizeof(float), stream));
    }

    void destroy() {
        if (log_L_bkg_d) hipFree(log_L_bkg_d);
        if (log_rho_d) hipFree(log_rho_d);
        if (logS_o0_c0_d) hipFree(logS_o0_c0_d);
        if (logS_o0_c1_d) hipFree(logS_o0_c1_d);
        if (logS_o0_c2_d) hipFree(logS_o0_c2_d);
        if (logS_o1_c0_d) hipFree(logS_o1_c0_d);
    }
};

// Bilinear interpolation in 2D lookup table
__device__ __forceinline__ float interp2d(const float* lut, int width, int height,
                                           float x, float y) {
    // Clamp coordinates
    x = fmaxf(0.0f, fminf(x, width - 1.0f));
    y = fmaxf(0.0f, fminf(y, height - 1.0f));

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = min(x0 + 1, width - 1);
    int y1 = min(y0 + 1, height - 1);

    float fx = x - x0;
    float fy = y - y0;

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
