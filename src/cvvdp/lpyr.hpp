#pragma once

#include "../util/preprocessor.hpp"
#include <vector>
#include <cmath>

namespace cvvdp {

// Downsample with Gaussian filtering (2x decimation)
__launch_bounds__(256)
__global__ void downsample_kernel(const float* input,
                                  float* output,
                                  int in_width,
                                  int in_height,
                                  int out_width,
                                  int out_height,
                                  int channels) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= out_width || out_y >= out_height) {
        return;
    }

    const float kernel[5][5] = {
        {0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f},
        {0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f},
        {0.023792f, 0.094907f, 0.150342f, 0.094907f, 0.023792f},
        {0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f},
        {0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f}
    };

    const int out_offset = (out_y * out_width + out_x) * channels;

    for (int c = 0; c < channels; ++c) {
        float num = 0.0f;
        float den = 0.0f;

        const int in_center_x = out_x * 2;
        const int in_center_y = out_y * 2;

        for (int ky = 0; ky < 5; ++ky) {
            for (int kx = 0; kx < 5; ++kx) {
                int in_x = in_center_x + kx - 2;
                int in_y = in_center_y + ky - 2;

                in_x = max(0, min(in_x, in_width - 1));
                in_y = max(0, min(in_y, in_height - 1));

                const float w = kernel[ky][kx];
                const int in_index = (in_y * in_width + in_x) * channels + c;

                num += input[in_index] * w;
                den += w;
            }
        }

        output[out_offset + c] = num / den;
    }
}

// Upsample (2x expansion with bilinear interpolation)
__launch_bounds__(256)
__global__ void upsample_kernel(const float* input,
                                float* output,
                                int in_width,
                                int in_height,
                                int out_width,
                                int out_height,
                                int channels) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= out_width || out_y >= out_height) {
        return;
    }

    const float in_x_f = (out_x + 0.5f) * 0.5f - 0.5f;
    const float in_y_f = (out_y + 0.5f) * 0.5f - 0.5f;

    int in_x0 = static_cast<int>(floorf(in_x_f));
    int in_y0 = static_cast<int>(floorf(in_y_f));
    int in_x1 = in_x0 + 1;
    int in_y1 = in_y0 + 1;

    const float fx = in_x_f - in_x0;
    const float fy = in_y_f - in_y0;

    in_x0 = max(0, min(in_x0, in_width - 1));
    in_x1 = max(0, min(in_x1, in_width - 1));
    in_y0 = max(0, min(in_y0, in_height - 1));
    in_y1 = max(0, min(in_y1, in_height - 1));

    const int out_offset = (out_y * out_width + out_x) * channels;

    for (int c = 0; c < channels; ++c) {
        const int idx00 = (in_y0 * in_width + in_x0) * channels + c;
        const int idx10 = (in_y0 * in_width + in_x1) * channels + c;
        const int idx01 = (in_y1 * in_width + in_x0) * channels + c;
        const int idx11 = (in_y1 * in_width + in_x1) * channels + c;

        const float v00 = input[idx00];
        const float v10 = input[idx10];
        const float v01 = input[idx01];
        const float v11 = input[idx11];

        const float v0 = v00 * (1.0f - fx) + v10 * fx;
        const float v1 = v01 * (1.0f - fx) + v11 * fx;
        const float result = (v0 * (1.0f - fy) + v1 * fy) * 4.0f;

        output[out_offset + c] = result;
    }
}

// Subtract images (for Laplacian computation)
__launch_bounds__(256)
__global__ void subtract_images_kernel(const float* a,
                                       const float* b,
                                       float* result,
                                       int pixels,
                                       int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) {
        return;
    }

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
            band_freqs[i] = (i == num_bands - 1)
                ? 0.1f
                : (0.3228f * powf(2.0f, -static_cast<float>(i)) * (ppd / 2.0f));

            curr_w = std::max(1, curr_w / 2);
            curr_h = std::max(1, curr_h / 2);
        }
    }

    int band_size(int band) const {
        return band_widths[band] * band_heights[band];
    }

    void decompose(const float* input,
                   float** laplacian_out,
                   float** gaussian_out,
                   hipStream_t stream) {
        std::vector<float*> gaussian_levels(num_bands, nullptr);

        const size_t level0_bytes =
            static_cast<size_t>(width) * static_cast<size_t>(height) * channels * sizeof(float);
        GPU_CHECK(hipMallocAsync(&gaussian_levels[0], level0_bytes, stream));
        GPU_CHECK(hipMemcpyDtoDAsync(gaussian_levels[0], input, level0_bytes, stream));

        for (int i = 1; i < num_bands; ++i) {
            const int in_w = band_widths[i - 1];
            const int in_h = band_heights[i - 1];
            const int out_w = band_widths[i];
            const int out_h = band_heights[i];

            const size_t bytes = static_cast<size_t>(out_w) * static_cast<size_t>(out_h) * channels * sizeof(float);
            GPU_CHECK(hipMallocAsync(&gaussian_levels[i], bytes, stream));

            dim3 block(16, 16);
            dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);

            downsample_kernel<<<grid, block, 0, stream>>>(
                gaussian_levels[i - 1],
                gaussian_levels[i],
                in_w,
                in_h,
                out_w,
                out_h,
                channels);
        }

        for (int i = 0; i < num_bands - 1; ++i) {
            const int w = band_widths[i];
            const int h = band_heights[i];
            const int pixels = w * h;
            const size_t bytes = static_cast<size_t>(pixels) * channels * sizeof(float);

            GPU_CHECK(hipMallocAsync(&laplacian_out[i], bytes, stream));

            float* upsampled = nullptr;
            GPU_CHECK(hipMallocAsync(&upsampled, bytes, stream));

            dim3 block(16, 16);
            dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

            upsample_kernel<<<grid, block, 0, stream>>>(
                gaussian_levels[i + 1],
                upsampled,
                band_widths[i + 1],
                band_heights[i + 1],
                w,
                h,
                channels);

            const int threads = 256;
            const int blocks = (pixels + threads - 1) / threads;

            subtract_images_kernel<<<blocks, threads, 0, stream>>>(
                gaussian_levels[i],
                upsampled,
                laplacian_out[i],
                pixels,
                channels);

            GPU_CHECK(hipFreeAsync(upsampled, stream));
        }

        const int last = num_bands - 1;
        const int last_pixels = band_widths[last] * band_heights[last];
        const size_t last_bytes = static_cast<size_t>(last_pixels) * channels * sizeof(float);

        GPU_CHECK(hipMallocAsync(&laplacian_out[last], last_bytes, stream));
        GPU_CHECK(hipMemcpyDtoDAsync(laplacian_out[last],
                                     gaussian_levels[last],
                                     last_bytes,
                                     stream));

        if (gaussian_out) {
            for (int i = 0; i < num_bands; ++i) {
                gaussian_out[i] = gaussian_levels[i];
                gaussian_levels[i] = nullptr;
            }
        }

        for (float* ptr : gaussian_levels) {
            if (ptr) {
                GPU_CHECK(hipFreeAsync(ptr, stream));
            }
        }
    }

    void free_levels(float** levels, hipStream_t stream) const {
        if (!levels) {
            return;
        }
        for (int i = 0; i < num_bands; ++i) {
            if (levels[i]) {
                GPU_CHECK(hipFreeAsync(levels[i], stream));
            }
        }
    }
};

} // namespace cvvdp
