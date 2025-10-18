#pragma once

#include "../util/preprocessor.hpp"
#include "../util/VshipExceptions.hpp"
#include <array>
#include <cmath>
#include <vector>
#include <algorithm>

namespace cvvdp {

namespace detail {

inline int compute_temporal_filter_length(float fps, int override_len) {
    const float safe_fps = (fps > 0.0f) ? fps : 30.0f;
    if (override_len > 0) {
        const int ensured_odd = (override_len % 2 == 0) ? override_len + 1 : override_len;
        return std::max(3, ensured_odd);
    }

    const double half_window = 0.250 * static_cast<double>(safe_fps) / 2.0;
    int tap_count = static_cast<int>(std::ceil(half_window));
    tap_count = std::max(1, tap_count);
    tap_count = tap_count * 2 + 1; // Ensure odd length
    return tap_count;
}

inline void compute_temporal_kernels(std::array<std::vector<float>, 4>& kernels,
                                     const std::vector<float>& sigma_tf_params,
                                     const std::vector<float>& beta_tf_params,
                                     float fps,
                                     int filter_len,
                                     bool use_high_pass_transient) {
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    const int N = filter_len;
    const int N_omega = N / 2 + 1;

    std::vector<double> omega(N_omega, 0.0);
    if (N_omega > 1) {
        const double step = (fps / 2.0) / static_cast<double>(N_omega - 1);
        for (int k = 0; k < N_omega; ++k) {
            omega[k] = step * k;
        }
    }

    std::array<std::vector<double>, 4> freq_response;
    for (auto& channel : freq_response) {
        channel.assign(N_omega, 0.0);
    }

    for (int c = 0; c < 3; ++c) {
        const double sigma = std::max(1e-6, static_cast<double>(
            c < static_cast<int>(sigma_tf_params.size()) ? sigma_tf_params[c] : 1.0f));
        const double beta = static_cast<double>(
            c < static_cast<int>(beta_tf_params.size()) ? beta_tf_params[c] : 1.0f);

        for (int k = 0; k < N_omega; ++k) {
            const double term = std::pow(omega[k], beta) / sigma;
            freq_response[c][k] = std::exp(-term);
        }
    }

    if (use_high_pass_transient) {
        freq_response[3] = freq_response[0];
        for (int k = 0; k < N_omega; ++k) {
            freq_response[3][k] = 1.0 - freq_response[0][k];
        }
    } else {
        const double sigma = std::max(1e-6, static_cast<double>(
            3 < static_cast<int>(sigma_tf_params.size()) ? sigma_tf_params[3] : 1.0f));
        const double beta = static_cast<double>(
            3 < static_cast<int>(beta_tf_params.size()) ? beta_tf_params[3] : 1.0f);
        const double center = 5.0;
        for (int k = 0; k < N_omega; ++k) {
            const double term = std::pow(omega[k], beta) - std::pow(center, beta);
            freq_response[3][k] = std::exp(-(term * term) / sigma);
        }
    }

    for (int c = 0; c < 4; ++c) {
        kernels[c].resize(N);
        for (int n = 0; n < N; ++n) {
            double sum = freq_response[c][0];
            for (int k = 1; k < N_omega; ++k) {
                const double angle = (kTwoPi * k * n) / static_cast<double>(N);
                sum += 2.0 * freq_response[c][k] * std::cos(angle);
            }
            const int shifted_index = (n + N / 2) % N;
            kernels[c][shifted_index] = static_cast<float>(sum);
        }
    }
}

} // namespace detail

struct TemporalBuffer {
    struct BandBuffer {
        float* test_history[3] = {nullptr, nullptr, nullptr};
        float* ref_history[3] = {nullptr, nullptr, nullptr};
        int width = 0;
        int height = 0;
        int history_len = 0;
        int write_index = 0;
        bool initialized = false;
        int frames_processed = 0;

        void allocate(int w, int h, int history, hipStream_t stream) {
            width = w;
            height = h;
            history_len = history;
            write_index = 0;
            frames_processed = 0;

            const size_t pixels = static_cast<size_t>(w) * static_cast<size_t>(h);
            const size_t bytes = pixels * static_cast<size_t>(history_len) * sizeof(float);

            for (int c = 0; c < 3; ++c) {
                GPU_CHECK(hipMalloc(&test_history[c], bytes));
                GPU_CHECK(hipMalloc(&ref_history[c], bytes));
                GPU_CHECK(hipMemsetAsync(test_history[c], 0, bytes, stream));
                GPU_CHECK(hipMemsetAsync(ref_history[c], 0, bytes, stream));
            }
            GPU_CHECK(hipStreamSynchronize(stream));
            initialized = true;
        }

        void free() {
            for (int c = 0; c < 3; ++c) {
                if (test_history[c]) {
                    hipFree(test_history[c]);
                    test_history[c] = nullptr;
                }
                if (ref_history[c]) {
                    hipFree(ref_history[c]);
                    ref_history[c] = nullptr;
                }
            }
            width = 0;
            height = 0;
            history_len = 0;
            write_index = 0;
            frames_processed = 0;
            initialized = false;
        }
    };

    std::vector<BandBuffer> bands;
    int num_bands = 0;
    float frame_rate = 30.0f;
    float dt = 1.0f / 30.0f;
    int filter_len = 0;
    std::array<std::vector<float>, 4> filter_kernels;
    float* filter_kernel_dev[4] = {nullptr, nullptr, nullptr, nullptr};
    bool use_high_pass_transient = true;

    void init(int n_bands,
              const std::vector<int>& widths,
              const std::vector<int>& heights,
              const std::vector<float>& sigma_tf_params,
              const std::vector<float>& beta_tf_params,
              float fps,
              int filter_len_override,
              hipStream_t stream) {
        destroy();

        num_bands = n_bands;
        frame_rate = (fps > 0.0f) ? fps : 30.0f;
        dt = 1.0f / frame_rate;
        use_high_pass_transient = true;

        filter_len = detail::compute_temporal_filter_length(frame_rate, filter_len_override);
        detail::compute_temporal_kernels(filter_kernels,
                                         sigma_tf_params,
                                         beta_tf_params,
                                         frame_rate,
                                         filter_len,
                                         use_high_pass_transient);

        const size_t kernel_bytes = static_cast<size_t>(filter_len) * sizeof(float);
        for (int c = 0; c < 4; ++c) {
            GPU_CHECK(hipMalloc(&filter_kernel_dev[c], kernel_bytes));
            GPU_CHECK(hipMemcpyAsync(filter_kernel_dev[c],
                                     filter_kernels[c].data(),
                                     kernel_bytes,
                                     hipMemcpyHostToDevice,
                                     stream));
        }
        GPU_CHECK(hipStreamSynchronize(stream));

        bands.resize(n_bands);
        for (int b = 0; b < n_bands; ++b) {
            bands[b].allocate(widths[b], heights[b], filter_len, stream);
        }
    }

    void destroy() {
        for (auto& band : bands) {
            band.free();
        }
        bands.clear();
        num_bands = 0;

        for (int c = 0; c < 4; ++c) {
            if (filter_kernel_dev[c]) {
                hipFree(filter_kernel_dev[c]);
                filter_kernel_dev[c] = nullptr;
            }
        }
        filter_len = 0;
    }
};

__launch_bounds__(256)
__global__ void apply_temporal_fir_kernel(
    float4* __restrict__ T_p,
    float4* __restrict__ R_p,
    float* __restrict__ T_hist_y,
    float* __restrict__ T_hist_rg,
    float* __restrict__ T_hist_yv,
    float* __restrict__ R_hist_y,
    float* __restrict__ R_hist_rg,
    float* __restrict__ R_hist_yv,
    const float* __restrict__ kernel_y,
    const float* __restrict__ kernel_rg,
    const float* __restrict__ kernel_yv,
    const float* __restrict__ kernel_transient,
    int filter_len,
    int write_index,
    int is_first_frame,
    int size) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    const int stride = size;
    const float4 T_in = T_p[idx];
    const float4 R_in = R_p[idx];

    if (is_first_frame) {
        for (int tap = 0; tap < filter_len; ++tap) {
            const int offset = tap * stride + idx;
            T_hist_y[offset] = T_in.x;
            T_hist_rg[offset] = T_in.y;
            T_hist_yv[offset] = T_in.z;
            R_hist_y[offset] = R_in.x;
            R_hist_rg[offset] = R_in.y;
            R_hist_yv[offset] = R_in.z;
        }
    } else {
        const int offset = write_index * stride + idx;
        T_hist_y[offset] = T_in.x;
        T_hist_rg[offset] = T_in.y;
        T_hist_yv[offset] = T_in.z;
        R_hist_y[offset] = R_in.x;
        R_hist_rg[offset] = R_in.y;
        R_hist_yv[offset] = R_in.z;
    }

    const int latest = write_index;

    float T_y_sust = 0.0f;
    float T_rg_sust = 0.0f;
    float T_yv_sust = 0.0f;
    float T_y_trans = 0.0f;

    float R_y_sust = 0.0f;
    float R_rg_sust = 0.0f;
    float R_yv_sust = 0.0f;
    float R_y_trans = 0.0f;

    for (int tap = 0; tap < filter_len; ++tap) {
        const int hist_index = (latest - tap + filter_len) % filter_len;
        const int offset = hist_index * stride + idx;

        const float T_y_sample = T_hist_y[offset];
        const float T_rg_sample = T_hist_rg[offset];
        const float T_yv_sample = T_hist_yv[offset];

        const float R_y_sample = R_hist_y[offset];
        const float R_rg_sample = R_hist_rg[offset];
        const float R_yv_sample = R_hist_yv[offset];

        const float k_y = kernel_y[tap];
        const float k_rg = kernel_rg[tap];
        const float k_yv = kernel_yv[tap];
        const float k_trans = kernel_transient[tap];

        T_y_sust += k_y * T_y_sample;
        T_rg_sust += k_rg * T_rg_sample;
        T_yv_sust += k_yv * T_yv_sample;
        T_y_trans += k_trans * T_y_sample;

        R_y_sust += k_y * R_y_sample;
        R_rg_sust += k_rg * R_rg_sample;
        R_yv_sust += k_yv * R_yv_sample;
        R_y_trans += k_trans * R_y_sample;
    }

    T_p[idx] = make_float4(T_y_sust, T_rg_sust, T_yv_sust, T_y_trans);
    R_p[idx] = make_float4(R_y_sust, R_rg_sust, R_yv_sust, R_y_trans);
}

inline void apply_temporal_filtering(
    float4* T_p_d,
    float4* R_p_d,
    TemporalBuffer::BandBuffer& buffer,
    const TemporalBuffer& temporal,
    int width,
    int height,
    hipStream_t stream) {

    if (!buffer.initialized || temporal.filter_len <= 0) {
        return;
    }

    const int size = width * height;
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    const int is_first_frame = (buffer.frames_processed == 0) ? 1 : 0;

    apply_temporal_fir_kernel<<<blocks, threads, 0, stream>>>(
        T_p_d, R_p_d,
        buffer.test_history[0],
        buffer.test_history[1],
        buffer.test_history[2],
        buffer.ref_history[0],
        buffer.ref_history[1],
        buffer.ref_history[2],
        temporal.filter_kernel_dev[0],
        temporal.filter_kernel_dev[1],
        temporal.filter_kernel_dev[2],
        temporal.filter_kernel_dev[3],
        temporal.filter_len,
        buffer.write_index,
        is_first_frame,
        size);

    buffer.frames_processed += 1;
    buffer.write_index = (buffer.write_index + 1) % temporal.filter_len;
}

} // namespace cvvdp

