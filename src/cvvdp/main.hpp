#pragma once

#include "../util/preprocessor.hpp"
#include "../util/VshipExceptions.hpp"
#include "../util/gpuhelper.hpp"
#include "../util/float3operations.hpp"
#include "config.hpp"
#include "colorspace.hpp"
#include "lpyr.hpp"
#include "csf.hpp"
#include <cmath>
#include <vector>
#include <fstream>
#include <cstdlib>

namespace cvvdp {

// Weber contrast encoding kernel
__launch_bounds__(256)
__global__ void weber_contrast_kernel(const float3* test, const float3* ref, float3* contrast_out,
                                       int size, float epsilon = 1e-5f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float3 T = test[idx];
    float3 R = ref[idx];

    // Weber contrast: (T - R) / (R + epsilon)
    float3 result;
    result.x = (T.x - R.x) / fmaxf(R.x, epsilon);
    result.y = (T.y - R.y) / fmaxf(R.y, epsilon);
    result.z = (T.z - R.z) / fmaxf(R.z, epsilon);

    contrast_out[idx] = result;
}

// Spatial pooling kernel (p-norm)
__launch_bounds__(256)
__global__ void spatial_pooling_kernel(const float3* input, float* output,
                                        int width, int height, float p) {
    // Simple mean pooling across spatial dimensions for now
    // TODO: Implement proper p-norm spatial pooling
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    float3 val = input[idx];
    // Combine channels (TODO: apply proper channel weights)
    float combined = powf(fabsf(val.x), p) + powf(fabsf(val.y), p) + powf(fabsf(val.z), p);
    output[idx] = combined;
}

// Reduction kernel for computing mean
__launch_bounds__(256)
__global__ void reduce_mean_kernel(const float* input, float* output, int size) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (idx < size) ? input[idx] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
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
    hipStream_t stream;

public:
    std::string find_config_dir() {
        // Search for CVVDP config files in multiple locations
        std::vector<std::string> search_paths;

        #ifdef _WIN32
        // Windows: Check multiple locations
        search_paths.push_back("C:\\Tools\\config\\cvvdp_data\\");
        search_paths.push_back("C:\\Tools\\lib\\vapoursynth\\cvvdp_data\\");

        // Check relative to current directory
        search_paths.push_back("config\\cvvdp_data\\");
        search_paths.push_back("..\\config\\cvvdp_data\\");

        // Check APPDATA VapourSynth plugins
        const char* appdata = getenv("APPDATA");
        if (appdata) {
            std::string vs_path = std::string(appdata) + "\\VapourSynth\\plugins64\\cvvdp_data\\";
            search_paths.push_back(vs_path);
        }
        #else
        // Linux: Check standard locations
        search_paths.push_back("/usr/local/share/vship/cvvdp_data/");
        search_paths.push_back("/usr/share/vship/cvvdp_data/");
        search_paths.push_back("./config/cvvdp_data/");
        search_paths.push_back("../config/cvvdp_data/");
        #endif

        // Test each path by checking for display_models.json
        for (const auto& path : search_paths) {
            std::string test_file = path + "display_models.json";
            std::ifstream test(test_file);
            if (test.good()) {
                return path;
            }
        }

        // Fallback to default (will fail later if files not found)
        #ifdef _WIN32
        return "C:\\Tools\\config\\cvvdp_data\\";
        #else
        return "/usr/local/share/vship/cvvdp_data/";
        #endif
    }

    void init(int w, int h, const std::string& display_name) {
        width = w;
        height = h;

        // Find configuration directory
        std::string config_dir = find_config_dir();

        display = DisplayModel::load(config_dir + "display_models.json", display_name);
        params = CVVDPParameters::load(config_dir + "cvvdp_parameters.json");

        ppd = display.get_ppd();

        // Initialize Laplacian pyramid
        lpyr.init(width, height, ppd);

        // Initialize CSF
        hipStreamCreate(&stream);
        csf.init(config_dir + "csf_lut_" + params.csf + ".json", stream);
    }

    void destroy() {
        csf.destroy();
        hipStreamDestroy(stream);
    }

    // Process a single frame pair and return JOD score
    // Input: test and reference frames in planar RGB format (float3 arrays)
    float process_frame(const float3* test_d, const float3* ref_d) {
        int frame_size = width * height;

        // Allocate working buffers
        float3 *test_linear_d, *ref_linear_d;
        GPU_CHECK(hipMallocAsync(&test_linear_d, frame_size * sizeof(float3), stream));
        GPU_CHECK(hipMallocAsync(&ref_linear_d, frame_size * sizeof(float3), stream));

        // Copy input
        GPU_CHECK(hipMemcpyAsync(test_linear_d, test_d, frame_size * sizeof(float3),
                                  hipMemcpyDeviceToDevice, stream));
        GPU_CHECK(hipMemcpyAsync(ref_linear_d, ref_d, frame_size * sizeof(float3),
                                  hipMemcpyDeviceToDevice, stream));

        // Step 1: Convert to linear luminance
        float L_max = display.max_luminance;
        float L_black = display.get_black_level();
        float gamma = 2.2f; // TODO: Get from display colorspace

        rgb_to_linear(test_linear_d, frame_size, L_max, L_black, gamma, stream);
        rgb_to_linear(ref_linear_d, frame_size, L_max, L_black, gamma, stream);

        // Step 2: Convert to DKL opponent color space
        rgb_to_dkl(test_linear_d, frame_size, stream);
        rgb_to_dkl(ref_linear_d, frame_size, stream);

        // Step 3: Build Laplacian pyramid for both images
        float3** test_bands = new float3*[lpyr.num_bands];
        float3** ref_bands = new float3*[lpyr.num_bands];

        lpyr.decompose(test_linear_d, test_bands, stream);
        lpyr.decompose(ref_linear_d, ref_bands, stream);

        // Step 4: Compute contrast and apply CSF for each band
        std::vector<float> band_scores(lpyr.num_bands);

        for (int band = 0; band < lpyr.num_bands; band++) {
            int band_size = lpyr.band_widths[band] * lpyr.band_heights[band];
            float rho = lpyr.band_freqs[band]; // Spatial frequency in cpd

            // Allocate contrast buffer
            float3* contrast_d;
            GPU_CHECK(hipMallocAsync(&contrast_d, band_size * sizeof(float3), stream));

            // Compute Weber contrast
            int threads = 256;
            int blocks = (band_size + threads - 1) / threads;
            weber_contrast_kernel<<<blocks, threads, 0, stream>>>(
                test_bands[band], ref_bands[band], contrast_d, band_size);

            // Apply CSF weighting
            // TODO: Properly implement CSF application per-channel
            // apply_csf_kernel<<<blocks, threads, 0, stream>>>(
            //     contrast_d, lpyr.band_widths[band], lpyr.band_heights[band],
            //     rho, log10f(L_max / 2.0f), csf.log_L_bkg_d, csf.log_rho_d,
            //     csf.logS_o0_c0_d, csf.logS_o0_c1_d, csf.logS_o0_c2_d,
            //     csf.num_L_bkg, csf.num_rho, params.sensitivity_correction);

            // Step 5: Spatial pooling (p-norm)
            float* pooled_d;
            GPU_CHECK(hipMallocAsync(&pooled_d, band_size * sizeof(float), stream));

            spatial_pooling_kernel<<<blocks, threads, 0, stream>>>(
                contrast_d, pooled_d, lpyr.band_widths[band], lpyr.band_heights[band], params.beta);

            // Reduce to single value
            float* partial_sums_d;
            int num_blocks = (band_size + 255) / 256;
            GPU_CHECK(hipMallocAsync(&partial_sums_d, num_blocks * sizeof(float), stream));

            reduce_mean_kernel<<<num_blocks, 256, 256 * sizeof(float), stream>>>(
                pooled_d, partial_sums_d, band_size);

            // Final reduction on CPU (for simplicity)
            std::vector<float> partial_sums_h(num_blocks);
            GPU_CHECK(hipMemcpyAsync(partial_sums_h.data(), partial_sums_d,
                                      num_blocks * sizeof(float), hipMemcpyDeviceToHost, stream));
            hipStreamSynchronize(stream);

            float sum = 0.0f;
            for (int i = 0; i < num_blocks; i++) {
                sum += partial_sums_h[i];
            }
            band_scores[band] = powf(sum / band_size, 1.0f / params.beta);

            // Free buffers
            GPU_CHECK(hipFreeAsync(contrast_d, stream));
            GPU_CHECK(hipFreeAsync(pooled_d, stream));
            GPU_CHECK(hipFreeAsync(partial_sums_d, stream));
        }

        // Free pyramid bands
        lpyr.free_bands(test_bands, stream);
        lpyr.free_bands(ref_bands, stream);
        delete[] test_bands;
        delete[] ref_bands;

        GPU_CHECK(hipFreeAsync(test_linear_d, stream));
        GPU_CHECK(hipFreeAsync(ref_linear_d, stream));

        // Step 6: Combine band scores (TODO: apply baseband weights)
        float Q_per_ch = 0.0f;
        for (int band = 0; band < lpyr.num_bands; band++) {
            // Simple averaging for now (TODO: apply proper weights and pooling)
            Q_per_ch += band_scores[band];
        }
        Q_per_ch /= lpyr.num_bands;

        // Step 7: Convert to JOD scale
        // Q_JOD = jod_a * Q^jod_exp
        float Q_jod = params.jod_a * powf(Q_per_ch, params.jod_exp);

        // Clamp to reasonable range
        Q_jod = fminf(fmaxf(Q_jod, 0.0f), 10.0f);

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
        return handle->processor.process_frame(test_d, ref_d);
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

} // namespace cvvdp
