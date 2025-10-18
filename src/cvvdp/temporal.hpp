#pragma once

#include "../util/preprocessor.hpp"
#include "../util/VshipExceptions.hpp"
#include <cmath>
#include <vector>

namespace cvvdp {

// Temporal buffer for storing previous filtered state per band/channel.
// Matches the original IIR implementation used in Python (alpha-based).
struct TemporalBuffer {
    struct BandBuffer {
        float* test_buffers[4] = {nullptr, nullptr, nullptr, nullptr};
        float* ref_buffers[4] = {nullptr, nullptr, nullptr, nullptr};
        int width = 0;
        int height = 0;
        bool initialized = false;
        int frames_processed = 0;

        void allocate(int w, int h, hipStream_t stream) {
            width = w;
            height = h;
            size_t pixels = static_cast<size_t>(w) * static_cast<size_t>(h);
            size_t bytes = pixels * sizeof(float);

            for (int c = 0; c < 4; ++c) {
                GPU_CHECK(hipMalloc(&test_buffers[c], bytes));
                GPU_CHECK(hipMalloc(&ref_buffers[c], bytes));
                GPU_CHECK(hipMemsetAsync(test_buffers[c], 0, bytes, stream));
                GPU_CHECK(hipMemsetAsync(ref_buffers[c], 0, bytes, stream));
            }
            GPU_CHECK(hipStreamSynchronize(stream));
            initialized = true;
            frames_processed = 0;
        }

        void free() {
            for (int c = 0; c < 4; ++c) {
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
            frames_processed = 0;
        }
    };

    std::vector<BandBuffer> bands;
    int num_bands = 0;
    float frame_rate = 30.0f;
    float dt = 1.0f / 30.0f;

    // IIR filter coefficients (alpha per channel)
    float alpha[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float beta_tf[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    void init(int n_bands,
              const std::vector<int>& widths,
              const std::vector<int>& heights,
              const std::vector<float>& sigma_tf_params,
              const std::vector<float>& beta_tf_params,
              float fps,
              hipStream_t stream) {
        num_bands = n_bands;
        frame_rate = fps;
        dt = 1.0f / fps;

        for (int c = 0; c < 4; ++c) {
            if (c < static_cast<int>(sigma_tf_params.size())) {
                float sigma_tf = sigma_tf_params[c];
                float sigma_seconds = sigma_tf * dt;
                alpha[c] = 1.0f - expf(-dt / sigma_seconds);
            } else {
                alpha[c] = 1.0f;
            }

            if (c < static_cast<int>(beta_tf_params.size())) {
                beta_tf[c] = beta_tf_params[c];
            }
        }

        bands.resize(n_bands);
        for (int b = 0; b < n_bands; ++b) {
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

// GPU kernel: Apply temporal filtering (IIR) to sustained/transient channels.
__launch_bounds__(256)
__global__ void apply_temporal_filter_kernel(
    float4* T_p,
    float4* R_p,
    float* T_prev_0,
    float* T_prev_1,
    float* T_prev_2,
    float* T_prev_3,
    float* R_prev_0,
    float* R_prev_1,
    float* R_prev_2,
    float* R_prev_3,
    float alpha_0,
    float alpha_1,
    float alpha_2,
    float alpha_3,
    int first_frame,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float4 T = T_p[idx];
    float4 R = R_p[idx];

    float a0 = first_frame ? 1.0f : alpha_0;
    float a1 = first_frame ? 1.0f : alpha_1;
    float a2 = first_frame ? 1.0f : alpha_2;
    float a3 = first_frame ? 1.0f : alpha_3;

    float T_y_sust = a0 * T.x + (1.0f - a0) * T_prev_0[idx];
    float R_y_sust = a0 * R.x + (1.0f - a0) * R_prev_0[idx];

    float T_rg = a1 * T.y + (1.0f - a1) * T_prev_1[idx];
    float R_rg = a1 * R.y + (1.0f - a1) * R_prev_1[idx];

    float T_by = a2 * T.z + (1.0f - a2) * T_prev_2[idx];
    float R_by = a2 * R.z + (1.0f - a2) * R_prev_2[idx];

    float T_y_trans = T.x - T_y_sust;
    float R_y_trans = R.x - R_y_sust;

    T_y_trans = a3 * T_y_trans + (1.0f - a3) * T_prev_3[idx];
    R_y_trans = a3 * R_y_trans + (1.0f - a3) * R_prev_3[idx];

    T_prev_0[idx] = T_y_sust;
    T_prev_1[idx] = T_rg;
    T_prev_2[idx] = T_by;
    T_prev_3[idx] = T_y_trans;

    R_prev_0[idx] = R_y_sust;
    R_prev_1[idx] = R_rg;
    R_prev_2[idx] = R_by;
    R_prev_3[idx] = R_y_trans;

    T_p[idx] = make_float4(T_y_sust, T_rg, T_by, T_y_trans);
    R_p[idx] = make_float4(R_y_sust, R_rg, R_by, R_y_trans);
}

inline void apply_temporal_filtering(
    float4* T_p_d,
    float4* R_p_d,
    TemporalBuffer::BandBuffer& buffer,
    const TemporalBuffer& temporal,
    int width,
    int height,
    hipStream_t stream
) {
    if (!buffer.initialized) {
        return;
    }

    int size = width * height;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    int first_frame = (buffer.frames_processed == 0) ? 1 : 0;

    apply_temporal_filter_kernel<<<blocks, threads, 0, stream>>>(
        T_p_d, R_p_d,
        buffer.test_buffers[0], buffer.test_buffers[1],
        buffer.test_buffers[2], buffer.test_buffers[3],
        buffer.ref_buffers[0], buffer.ref_buffers[1],
        buffer.ref_buffers[2], buffer.ref_buffers[3],
        temporal.alpha[0], temporal.alpha[1], temporal.alpha[2], temporal.alpha[3],
        first_frame,
        size
    );

    buffer.frames_processed += 1;
}

} // namespace cvvdp
