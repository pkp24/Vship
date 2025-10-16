#pragma once

#include "../util/preprocessor.hpp"
#include "../util/float3operations.hpp"
#include <vector>
#include <cmath>

namespace cvvdp {

// Laplacian Pyramid Decomposition for CVVDP
// This implements a decimated Laplacian pyramid with Gaussian filtering

// Gaussian blur kernel for pyramid downsampling
__device__ __forceinline__ float gaussian_weight(float distance, float sigma) {
    return expf(-(distance * distance) / (2.0f * sigma * sigma));
}

// Downsample with Gaussian filtering (2x decimation)
__launch_bounds__(256)
__global__ void downsample_kernel(const float3* input, float3* output,
                                   int in_width, int in_height,
                                   int out_width, int out_height) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= out_width || out_y >= out_height) return;

    // 5x5 Gaussian kernel (sigma ≈ 1.0)
    const float kernel[5][5] = {
        {0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f},
        {0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f},
        {0.023792f, 0.094907f, 0.150342f, 0.094907f, 0.023792f},
        {0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f},
        {0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f}
    };

    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;

    // Map output pixel to input pixel (center of 2x2 region)
    int in_center_x = out_x * 2;
    int in_center_y = out_y * 2;

    // Apply 5x5 Gaussian filter
    for (int ky = 0; ky < 5; ky++) {
        for (int kx = 0; kx < 5; kx++) {
            int in_x = in_center_x + kx - 2;
            int in_y = in_center_y + ky - 2;

            // Clamp to image boundaries
            in_x = max(0, min(in_x, in_width - 1));
            in_y = max(0, min(in_y, in_height - 1));

            float w = kernel[ky][kx];
            float3 val = input[in_y * in_width + in_x];

            sum.x += val.x * w;
            sum.y += val.y * w;
            sum.z += val.z * w;
            weight_sum += w;
        }
    }

    output[out_y * out_width + out_x] = sum / weight_sum;
}

// Upsample (2x expansion with bilinear interpolation)
__launch_bounds__(256)
__global__ void upsample_kernel(const float3* input, float3* output,
                                 int in_width, int in_height,
                                 int out_width, int out_height) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= out_width || out_y >= out_height) return;

    // Map to input coordinates
    float in_x_f = (out_x + 0.5f) * 0.5f - 0.5f;
    float in_y_f = (out_y + 0.5f) * 0.5f - 0.5f;

    int in_x0 = (int)floorf(in_x_f);
    int in_y0 = (int)floorf(in_y_f);
    int in_x1 = in_x0 + 1;
    int in_y1 = in_y0 + 1;

    float fx = in_x_f - in_x0;
    float fy = in_y_f - in_y0;

    // Clamp coordinates
    in_x0 = max(0, min(in_x0, in_width - 1));
    in_x1 = max(0, min(in_x1, in_width - 1));
    in_y0 = max(0, min(in_y0, in_height - 1));
    in_y1 = max(0, min(in_y1, in_height - 1));

    // Bilinear interpolation
    float3 v00 = input[in_y0 * in_width + in_x0];
    float3 v10 = input[in_y0 * in_width + in_x1];
    float3 v01 = input[in_y1 * in_width + in_x0];
    float3 v11 = input[in_y1 * in_width + in_x1];

    float3 v0 = v00 * (1.0f - fx) + v10 * fx;
    float3 v1 = v01 * (1.0f - fx) + v11 * fx;
    float3 result = v0 * (1.0f - fy) + v1 * fy;

    output[out_y * out_width + out_x] = result * 4.0f; // Scale by 4 to compensate for decimation
}

// Subtract images (for Laplacian computation)
__launch_bounds__(256)
__global__ void subtract_images_kernel(const float3* a, const float3* b, float3* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    result[idx] = a[idx] - b[idx];
}

// Laplacian Pyramid structure
struct LaplacianPyramid {
    int width;
    int height;
    float ppd;
    int num_bands;

    std::vector<int> band_widths;
    std::vector<int> band_heights;
    std::vector<float> band_freqs;

    void init(int w, int h, float pixels_per_degree) {
        width = w;
        height = h;
        ppd = pixels_per_degree;

        // Calculate number of pyramid levels
        int max_levels = (int)floor(log2(fmin(height, width))) - 1;

        // Calculate frequency bands (cycles per degree)
        std::vector<float> bands;
        bands.push_back(1.0f * ppd / 2.0f); // Nyquist frequency
        for (int i = 0; i < 14; i++) {
            bands.push_back(0.3228f * pow(2.0f, -i) * ppd / 2.0f);
        }

        // Find valid bands (above minimum frequency threshold of 0.2 cpd)
        const float min_freq = 0.2f;
        num_bands = 0;
        for (size_t i = 0; i < bands.size() && i < (size_t)max_levels; i++) {
            if (bands[i] > min_freq) {
                num_bands++;
            } else {
                break;
            }
        }

        num_bands = fmin(num_bands, max_levels);

        // Set up band dimensions and frequencies
        band_widths.resize(num_bands);
        band_heights.resize(num_bands);
        band_freqs.resize(num_bands);

        int curr_w = width;
        int curr_h = height;
        for (int i = 0; i < num_bands; i++) {
            band_widths[i] = curr_w;
            band_heights[i] = curr_h;
            if (i == 0) {
                band_freqs[i] = 1.0f * ppd / 2.0f;
            } else {
                band_freqs[i] = 0.3228f * pow(2.0f, -(i - 1)) * ppd / 2.0f;
            }
            curr_w = (curr_w + 1) / 2;
            curr_h = (curr_h + 1) / 2;
        }
    }

    // Decompose image into Laplacian pyramid
    // Returns array of band pointers (allocated on device)
    // bands[0] = highest frequency, bands[num_bands-1] = lowest frequency (residual)
    void decompose(const float3* input, float3** bands, hipStream_t stream) {
        // Allocate temporary buffers for Gaussian pyramid
        std::vector<float3*> gaussian_pyramid(num_bands);

        // First level is the input image
        GPU_CHECK(hipMallocAsync(&gaussian_pyramid[0], width * height * sizeof(float3), stream));
        GPU_CHECK(hipMemcpyAsync(gaussian_pyramid[0], input, width * height * sizeof(float3), hipMemcpyDeviceToDevice, stream));

        // Build Gaussian pyramid by successive downsampling
        for (int i = 1; i < num_bands; i++) {
            int in_w = band_widths[i - 1];
            int in_h = band_heights[i - 1];
            int out_w = band_widths[i];
            int out_h = band_heights[i];

            GPU_CHECK(hipMallocAsync(&gaussian_pyramid[i], out_w * out_h * sizeof(float3), stream));

            dim3 block(16, 16);
            dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);
            downsample_kernel<<<grid, block, 0, stream>>>(gaussian_pyramid[i - 1], gaussian_pyramid[i],
                                                            in_w, in_h, out_w, out_h);
        }

        // Build Laplacian pyramid (difference between levels)
        for (int i = 0; i < num_bands - 1; i++) {
            int w = band_widths[i];
            int h = band_heights[i];

            // Allocate band
            GPU_CHECK(hipMallocAsync(&bands[i], w * h * sizeof(float3), stream));

            // Upsample next level
            float3* upsampled;
            GPU_CHECK(hipMallocAsync(&upsampled, w * h * sizeof(float3), stream));

            dim3 block(16, 16);
            dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
            upsample_kernel<<<grid, block, 0, stream>>>(gaussian_pyramid[i + 1], upsampled,
                                                         band_widths[i + 1], band_heights[i + 1], w, h);

            // Compute difference (Laplacian)
            int threads = 256;
            int blocks = (w * h + threads - 1) / threads;
            subtract_images_kernel<<<blocks, threads, 0, stream>>>(gaussian_pyramid[i], upsampled, bands[i], w * h);

            GPU_CHECK(hipFreeAsync(upsampled, stream));
        }

        // Last band is the residual (low-pass)
        int last_idx = num_bands - 1;
        int last_size = band_widths[last_idx] * band_heights[last_idx];
        GPU_CHECK(hipMallocAsync(&bands[last_idx], last_size * sizeof(float3), stream));
        GPU_CHECK(hipMemcpyAsync(bands[last_idx], gaussian_pyramid[last_idx],
                                  last_size * sizeof(float3), hipMemcpyDeviceToDevice, stream));

        // Free Gaussian pyramid
        for (int i = 0; i < num_bands; i++) {
            GPU_CHECK(hipFreeAsync(gaussian_pyramid[i], stream));
        }
    }

    // Free pyramid bands
    void free_bands(float3** bands, hipStream_t stream) {
        for (int i = 0; i < num_bands; i++) {
            if (bands[i]) {
                GPU_CHECK(hipFreeAsync(bands[i], stream));
            }
        }
    }
};

} // namespace cvvdp
