#pragma once

#include "../util/preprocessor.hpp"
#include <vector>
#include <cmath>
#include <iostream>

namespace cvvdp {

// -----------------------------
// Utility: reflect-101 indexing (used for REDUCE like Python)
// -----------------------------
__device__ __forceinline__ int reflect101(int p, int n) {
    if (n <= 1) return 0;
    if (p < 0)   return -p;             // ... 2,1,0,0,1,2 ...
    if (p >= n)  return 2*n - 2 - p;    // ... n-3,n-2,n-1,n-1,n-2 ...
    return p;
}

// 1D 5-tap kernel used by Python with a=0.4
// K = [0.25 - a/2, 0.25, a, 0.25, 0.25 - a/2] with a = 0.4
__constant__ float k5[5] = { 0.05f, 0.25f, 0.40f, 0.25f, 0.05f };

// -----------------------------
// REDUCE (vertical), stride 2 in rows, reflect-101 borders
// Output height = ceil(in_height/2), width unchanged
// -----------------------------
__launch_bounds__(256)
__global__ void reduce_vert_kernel(const float* __restrict__ input,
                                   float* __restrict__ tmp,  // size: out_h x in_w x C
                                   int in_w,
                                   int in_h,
                                   int out_h,
                                   int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // 0..in_w-1
    int oy = blockIdx.y * blockDim.y + threadIdx.y;  // 0..out_h-1
    if (x >= in_w || oy >= out_h) return;

    const int iy_center = oy * 2; // source row index for center tap

    for (int c = 0; c < channels; ++c) {
        float acc = 0.0f;
        #pragma unroll
        for (int k = -2; k <= 2; ++k) {
            int iy = reflect101(iy_center + k, in_h);
            const int idx = (iy * in_w + x) * channels + c;
            acc += input[idx] * k5[k + 2];
        }
        const int oidx = (oy * in_w + x) * channels + c;
        tmp[oidx] = acc; // no extra scaling; DC preserved
    }
}

// -----------------------------
// REDUCE (horizontal), stride 2 in cols, reflect-101 borders
// Input tmp: out_h x in_w x C, Output: out_h x out_w x C where out_w = ceil(in_w/2)
// -----------------------------
__launch_bounds__(256)
__global__ void reduce_horiz_kernel(const float* __restrict__ tmp,
                                    float* __restrict__ output,
                                    int in_w,
                                    int out_h,
                                    int out_w,
                                    int channels) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;  // 0..out_w-1
    int oy = blockIdx.y * blockDim.y + threadIdx.y;  // 0..out_h-1
    if (ox >= out_w || oy >= out_h) return;

    const int ix_center = ox * 2; // source col index for center tap

    for (int c = 0; c < channels; ++c) {
        float acc = 0.0f;
        #pragma unroll
        for (int k = -2; k <= 2; ++k) {
            int ix = reflect101(ix_center + k, in_w);
            const int idx = (oy * in_w + ix) * channels + c;
            acc += tmp[idx] * k5[k + 2];
        }
        const int oidx = (oy * out_w + ox) * channels + c;
        output[oidx] = acc;
    }
}

// -----------------------------
// EXPAND by 2 using zero-insertion + 5-tap separable Gaussian
// DC gain = 1. Endpoint REPLICATION like Python's interleave_zeros_and_pad
// -----------------------------
__launch_bounds__(256)
__global__ void expand_gaussian_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int in_w,
                                       int in_h,
                                       int out_w,
                                       int out_h,
                                       int channels) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= out_w || oy >= out_h) return;

    // Parity-aware sampling with endpoint REPLICATION (clamp), not reflect
    // Only even coordinates map to real source samples after zero insertion
    for (int c = 0; c < channels; ++c) {
        float acc = 0.0f;
        #pragma unroll
        for (int j = -2; j <= 2; ++j) {
            int y2 = oy - j;
            if (y2 & 1) continue; // odd => zero slot
            int iy = y2 >> 1;      // divide by 2
            // replicate endpoints
            if (iy < 0) iy = 0; else if (iy >= in_h) iy = in_h - 1;

            #pragma unroll
            for (int i = -2; i <= 2; ++i) {
                int x2 = ox - i;
                if (x2 & 1) continue; // odd => zero slot
                int ix = x2 >> 1;
                if (ix < 0) ix = 0; else if (ix >= in_w) ix = in_w - 1;

                const float w = k5[j + 2] * k5[i + 2];
                const int idx = (iy * in_w + ix) * channels + c;
                acc += input[idx] * w;
            }
        }
        // Compensate sparsity: ×4 total (2 per axis) for unity DC
        const int oidx = (oy * out_w + ox) * channels + c;
        output[oidx] = acc * 4.0f;
    }
}

// -----------------------------
// Subtract images (for Laplacian computation)
// -----------------------------
__launch_bounds__(256)
__global__ void subtract_images_kernel(const float* a,
                                       const float* b,
                                       float* result,
                                       int pixels,
                                       int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) return;
    const int offset = idx * channels;
    for (int c = 0; c < channels; ++c) {
        result[offset + c] = a[offset + c] - b[offset + c];
    }
}

struct LaplacianPyramid {
    int width = 0;
    int height = 0;
    int channels = 0;
    int num_bands = 0;

    std::vector<int> band_widths;
    std::vector<int> band_heights;
    std::vector<float> band_freqs;

    void init(int w, int h, float ppd, int channel_count) {
        width = w;
        height = h;
        channels = channel_count;

        const int max_levels = static_cast<int>(std::floor(std::log2(std::min(width, height)))) - 1;
        num_bands = std::max(1, max_levels);

        band_widths.resize(num_bands);
        band_heights.resize(num_bands);
        band_freqs.resize(num_bands);

        int curr_w = width;
        int curr_h = height;

        for (int i = 0; i < num_bands; ++i) {
            band_widths[i] = curr_w;
            band_heights[i] = curr_h;
            // Band centers (cpd). Baseband ≈ 0.1 cpd like Python
            band_freqs[i] = (i == num_bands - 1)
                ? 0.1f
                : (i == 0)
                    ? (1.0f * ppd / 2.0f)
                    : (0.3228f * powf(2.0f, -static_cast<float>(i - 1)) * (ppd / 2.0f));

            curr_w = std::max(1, curr_w / 2);
            curr_h = std::max(1, curr_h / 2);
        }
    }

    int band_size(int band) const { return band_widths[band] * band_heights[band]; }

    void decompose(const float* input,
                   float** laplacian_out,
                   float** gaussian_out,
                   hipStream_t stream) {
        std::vector<float*> gaussian_levels(num_bands, nullptr);

        const size_t level0_bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * channels * sizeof(float);
        GPU_CHECK(hipMallocAsync(&gaussian_levels[0], level0_bytes, stream));
        GPU_CHECK(hipMemcpyDtoDAsync(gaussian_levels[0], input, level0_bytes, stream));

        // Build Gaussian pyramid with true REDUCE (vertical then horizontal)
        for (int i = 1; i < num_bands; ++i) {
            const int in_w = band_widths[i - 1];
            const int in_h = band_heights[i - 1];
            const int out_w = band_widths[i];
            const int out_h = band_heights[i];

            // tmp buffer: (out_h x in_w x C)
            float* tmp = nullptr;
            const size_t tmp_bytes = static_cast<size_t>(out_h) * static_cast<size_t>(in_w) * channels * sizeof(float);
            GPU_CHECK(hipMallocAsync(&tmp, tmp_bytes, stream));

            const size_t out_bytes = static_cast<size_t>(out_w) * static_cast<size_t>(out_h) * channels * sizeof(float);
            GPU_CHECK(hipMallocAsync(&gaussian_levels[i], out_bytes, stream));

            dim3 block(16,16);
            dim3 grid_v((in_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);
            reduce_vert_kernel<<<grid_v, block, 0, stream>>>(
                gaussian_levels[i - 1], tmp, in_w, in_h, out_h, channels);

            dim3 grid_h((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);
            reduce_horiz_kernel<<<grid_h, block, 0, stream>>>(
                tmp, gaussian_levels[i], in_w, out_h, out_w, channels);

            GPU_CHECK(hipFreeAsync(tmp, stream));
        }

        // Build Laplacian pyramid using Gaussian EXPAND
        for (int i = 0; i < num_bands - 1; ++i) {
            const int w = band_widths[i];
            const int h = band_heights[i];
            const int pixels = w * h;
            const size_t bytes = static_cast<size_t>(pixels) * channels * sizeof(float);

            GPU_CHECK(hipMallocAsync(&laplacian_out[i], bytes, stream));

            float* expanded = nullptr;
            GPU_CHECK(hipMallocAsync(&expanded, bytes, stream));

            dim3 block(16,16);
            dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
            expand_gaussian_kernel<<<grid, block, 0, stream>>>(
                gaussian_levels[i + 1], expanded, band_widths[i + 1], band_heights[i + 1], w, h, channels);

            const int threads = 256;
            const int blocks = (pixels + threads - 1) / threads;
            subtract_images_kernel<<<blocks, threads, 0, stream>>>(
                gaussian_levels[i], expanded, laplacian_out[i], pixels, channels);

            GPU_CHECK(hipFreeAsync(expanded, stream));
        }

        // Baseband = last Gaussian level
        const int last = num_bands - 1;
        const int last_pixels = band_widths[last] * band_heights[last];
        const size_t last_bytes = static_cast<size_t>(last_pixels) * channels * sizeof(float);
        GPU_CHECK(hipMallocAsync(&laplacian_out[last], last_bytes, stream));
        GPU_CHECK(hipMemcpyDtoDAsync(laplacian_out[last], gaussian_levels[last], last_bytes, stream));

        if (gaussian_out) {
            for (int i = 0; i < num_bands; ++i) {
                gaussian_out[i] = gaussian_levels[i];
                gaussian_levels[i] = nullptr;
            }
        }
        for (float* ptr : gaussian_levels) {
            if (ptr) GPU_CHECK(hipFreeAsync(ptr, stream));
        }
    }

    void free_levels(float** levels, hipStream_t stream) const {
        if (!levels) return;
        for (int i = 0; i < num_bands; ++i) {
            if (levels[i]) GPU_CHECK(hipFreeAsync(levels[i], stream));
        }
    }
};

} // namespace cvvdp

