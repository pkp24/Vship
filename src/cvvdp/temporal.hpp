#pragma once

#include "../util/preprocessor.hpp"
#include "../util/VshipExceptions.hpp"
#include <vector>
#include <cmath>
#include <deque>

namespace cvvdp {

// Temporal buffer for storing previous frames
// Implements 4-channel IIR filtering for sustained/transient channels
struct TemporalBuffer {
    // Per-band, per-channel buffers
    // Structure: [band][channel][pixel]
    // Channels: 0=Y-sustained, 1=RG, 2=BY, 3=Y-transient
    // Need separate buffers for test and reference
    struct BandBuffer {
        float* test_buffers[4] = {nullptr, nullptr, nullptr, nullptr};
        float* ref_buffers[4] = {nullptr, nullptr, nullptr, nullptr};
        int width = 0;
        int height = 0;
        bool initialized = false;

        void allocate(int w, int h, hipStream_t stream) {
            width = w;
            height = h;
            size_t size = static_cast<size_t>(w) * static_cast<size_t>(h) * sizeof(float);

            for (int c = 0; c < 4; c++) {
                GPU_CHECK(hipMalloc(&test_buffers[c], size));
                GPU_CHECK(hipMalloc(&ref_buffers[c], size));
                GPU_CHECK(hipMemsetAsync(test_buffers[c], 0, size, stream));
                GPU_CHECK(hipMemsetAsync(ref_buffers[c], 0, size, stream));
            }
            GPU_CHECK(hipStreamSynchronize(stream));
            initialized = true;
        }

        void free() {
            for (int c = 0; c < 4; c++) {
                if (test_buffers[c]) {
                    hipFree(test_buffers[c]);
                    test_buffers[c] = nullptr;
                }
                if (ref_buffers[c]) {
                    hipFree(ref_buffers[c]);
                    ref_buffers[c] = nullptr;
                }
            }
            initialized = false;
        }
    };

    std::vector<BandBuffer> bands;
    int num_bands = 0;
    float frame_rate = 30.0f; // Default, will be updated
    float dt = 1.0f / 30.0f;  // Time between frames in seconds

    // IIR filter alphas (calculated from sigma_tf)
    // alpha = 1 - exp(-dt / sigma_tf)
    float alpha[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // beta parameters for transient channel
    float beta_tf[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    void init(int n_bands, const std::vector<int>& widths, const std::vector<int>& heights,
              const std::vector<float>& sigma_tf_params, const std::vector<float>& beta_tf_params,
              float fps, hipStream_t stream) {
        num_bands = n_bands;
        frame_rate = fps;
        dt = 1.0f / fps;

        // Calculate alpha values from sigma_tf
        for (int c = 0; c < 4; c++) {
            if (c < static_cast<int>(sigma_tf_params.size())) {
                float sigma_tf = sigma_tf_params[c];
                // Convert from frames to seconds if needed
                // sigma_tf is given in frames in the parameters
                float sigma_seconds = sigma_tf * dt;
                alpha[c] = 1.0f - expf(-dt / sigma_seconds);
            } else {
                alpha[c] = 1.0f; // No filtering
            }

            if (c < static_cast<int>(beta_tf_params.size())) {
                beta_tf[c] = beta_tf_params[c];
            }
        }

        // Allocate buffers for each band
        bands.resize(n_bands);
        for (int b = 0; b < n_bands; b++) {
            bands[b].allocate(widths[b], heights[b], stream);
        }
    }

    void destroy() {
        for (auto& band : bands) {
            band.free();
        }
        bands.clear();
        num_bands = 0;
    }
};

// GPU kernel: Apply 4-channel IIR temporal filtering
// Filters T_p (test) and R_p (reference) through sustained/transient channels
// Outputs: T_filtered[4], R_filtered[4] for the 4 temporal channels
__launch_bounds__(256)
__global__ void apply_temporal_filter_kernel(
    const float3* T_p,           // Input: CSF-weighted test signal (Y, RG, BY)
    const float3* R_p,           // Input: CSF-weighted reference signal (Y, RG, BY)
    float* T_prev_0,             // Previous filtered state: Y-sustained
    float* T_prev_1,             // Previous filtered state: RG
    float* T_prev_2,             // Previous filtered state: BY
    float* T_prev_3,             // Previous filtered state: Y-transient
    float* R_prev_0,             // Previous filtered state: Y-sustained (ref)
    float* R_prev_1,             // Previous filtered state: RG (ref)
    float* R_prev_2,             // Previous filtered state: BY (ref)
    float* R_prev_3,             // Previous filtered state: Y-transient (ref)
    float alpha_0, float alpha_1, float alpha_2, float alpha_3, // Filter coefficients
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float3 T = T_p[idx];
    float3 R = R_p[idx];

    // Channel 0: Y-sustained (low-pass filtered Y)
    float T_y_sustained = alpha_0 * T.x + (1.0f - alpha_0) * T_prev_0[idx];
    float R_y_sustained = alpha_0 * R.x + (1.0f - alpha_0) * R_prev_0[idx];
    T_prev_0[idx] = T_y_sustained;
    R_prev_0[idx] = R_y_sustained;

    // Channel 1: RG (low-pass filtered RG)
    float T_rg = alpha_1 * T.y + (1.0f - alpha_1) * T_prev_1[idx];
    float R_rg = alpha_1 * R.y + (1.0f - alpha_1) * R_prev_1[idx];
    T_prev_1[idx] = T_rg;
    R_prev_1[idx] = R_rg;

    // Channel 2: BY (low-pass filtered BY)
    float T_by = alpha_2 * T.z + (1.0f - alpha_2) * T_prev_2[idx];
    float R_by = alpha_2 * R.z + (1.0f - alpha_2) * R_prev_2[idx];
    T_prev_2[idx] = T_by;
    R_prev_2[idx] = R_by;

    // Channel 3: Y-transient (high-pass: original - sustained)
    float T_y_transient = T.x - T_y_sustained;
    float R_y_transient = R.x - R_y_sustained;

    // Apply IIR to transient channel (very fast response)
    T_y_transient = alpha_3 * T_y_transient + (1.0f - alpha_3) * T_prev_3[idx];
    R_y_transient = alpha_3 * R_y_transient + (1.0f - alpha_3) * R_prev_3[idx];
    T_prev_3[idx] = T_y_transient;
    R_prev_3[idx] = R_y_transient;
}

// Kernel: Combine temporal channels back to float3 for masking
// Takes sustained channels (Y, RG, BY) and outputs as float3
__launch_bounds__(256)
__global__ void combine_sustained_channels_kernel(
    const float* T_y, const float* T_rg, const float* T_by,
    const float* R_y, const float* R_rg, const float* R_by,
    float3* T_out, float3* R_out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    T_out[idx] = make_float3(T_y[idx], T_rg[idx], T_by[idx]);
    R_out[idx] = make_float3(R_y[idx], R_rg[idx], R_by[idx]);
}

// Helper function: Apply temporal filtering to a band
// This integrates into the existing CVVDP pipeline
inline void apply_temporal_filtering(
    float3* T_p_d, float3* R_p_d,
    TemporalBuffer::BandBuffer& buffer,
    const float* alpha,
    int width, int height,
    hipStream_t stream
) {
    if (!buffer.initialized) {
        // First frame - buffers already initialized to zero, just proceed
        // This will properly initialize the state
    }

    int size = width * height;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Apply IIR filtering
    // This updates the buffer state and outputs filtered sustained channels
    apply_temporal_filter_kernel<<<blocks, threads, 0, stream>>>(
        T_p_d, R_p_d,
        buffer.test_buffers[0], buffer.test_buffers[1],
        buffer.test_buffers[2], buffer.test_buffers[3],
        buffer.ref_buffers[0], buffer.ref_buffers[1],
        buffer.ref_buffers[2], buffer.ref_buffers[3],
        alpha[0], alpha[1], alpha[2], alpha[3],
        size
    );

    // Update T_p and R_p with sustained channels (channels 0-2)
    // The transient channel (3) is handled separately if needed
    combine_sustained_channels_kernel<<<blocks, threads, 0, stream>>>(
        buffer.test_buffers[0], buffer.test_buffers[1], buffer.test_buffers[2],
        buffer.ref_buffers[0], buffer.ref_buffers[1], buffer.ref_buffers[2],
        T_p_d, R_p_d,
        size
    );
}

} // namespace cvvdp
