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

// Apply CSF sensitivity and channel gain (for mult-mutual masking model)
// This kernel prepares T_p and R_p = contrast * sensitivity * ch_gain
__launch_bounds__(256)
__global__ void apply_csf_and_gain_kernel(
    const float3* test_contrast, const float3* ref_contrast,
    float3* T_p, float3* R_p,
    int width, int height,
    float rho_cpd, float log_L_bkg,
    const float* log_L_bkg_lut, const float* log_rho_lut,
    const float* logS_c0, const float* logS_c1, const float* logS_c2,
    int num_L_bkg, int num_rho,
    float sensitivity_correction
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    float3 T = test_contrast[idx];
    float3 R = ref_contrast[idx];

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

    // Convert from log to linear and apply correction
    float S_y = powf(10.0f, logS_y + sensitivity_correction / 20.0f);
    float S_rg = powf(10.0f, logS_rg + sensitivity_correction / 20.0f);
    float S_by = powf(10.0f, logS_by + sensitivity_correction / 20.0f);

    // Channel gains from Python: ch_gain = [1, 1.45, 1, 1.] for [Y, RG, BY, trans-Y]
    // We only have sustained channels here (no temporal filtering yet)
    const float ch_gain_y = 1.0f;
    const float ch_gain_rg = 1.45f;
    const float ch_gain_by = 1.0f;

    // T_p = T * S * ch_gain, R_p = R * S * ch_gain
    T_p[idx] = make_float3(
        T.x * S_y * ch_gain_y,
        T.y * S_rg * ch_gain_rg,
        T.z * S_by * ch_gain_by
    );

    R_p[idx] = make_float3(
        R.x * S_y * ch_gain_y,
        R.y * S_rg * ch_gain_rg,
        R.z * S_by * ch_gain_by
    );
}

// Compute mutual masking term: M_mm = min(|T_p|, |R_p|)
__launch_bounds__(256)
__global__ void compute_mutual_masking_kernel(
    const float3* T_p, const float3* R_p,
    float3* M_mm,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float3 Tp = T_p[idx];
    float3 Rp = R_p[idx];

    M_mm[idx] = make_float3(
        fminf(fabsf(Tp.x), fabsf(Rp.x)),
        fminf(fabsf(Tp.y), fabsf(Rp.y)),
        fminf(fabsf(Tp.z), fabsf(Rp.z))
    );
}

// Apply phase uncertainty: multiply by 10^mask_c
// (Gaussian blur would go here if image is large enough, but we'll skip for now)
__launch_bounds__(256)
__global__ void apply_phase_uncertainty_kernel(
    float3* M_mm,
    float mask_c,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float mult = powf(10.0f, mask_c);
    M_mm[idx].x *= mult;
    M_mm[idx].y *= mult;
    M_mm[idx].z *= mult;
}

// Apply masking model: D_u = |T_p - R_p|^p / (1 + M^q), then clamp
__launch_bounds__(256)
__global__ void apply_masking_kernel(
    const float3* T_p, const float3* R_p,
    const float3* M_pu,  // Phase uncertainty applied masking
    float3* D_out,
    float mask_p, float mask_q_y, float mask_q_rg, float mask_q_by,
    float d_max,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float3 Tp = T_p[idx];
    float3 Rp = R_p[idx];
    float3 M = M_pu[idx];

    // For simplicity, we're skipping cross-channel masking (mask_pool)
    // and using the per-pixel masking directly
    // In Python: M = mask_pool(safe_pow(|M_mm|, q))
    // Here we approximate: M = |M_mm|^q
    float M_y = safe_pow(M.x, mask_q_y);
    float M_rg = safe_pow(M.y, mask_q_rg);
    float M_by = safe_pow(M.z, mask_q_by);

    // D_u = |T_p - R_p|^p / (1 + M)
    float D_y = safe_pow(fabsf(Tp.x - Rp.x), mask_p) / (1.0f + M_y);
    float D_rg = safe_pow(fabsf(Tp.y - Rp.y), mask_p) / (1.0f + M_rg);
    float D_by = safe_pow(fabsf(Tp.z - Rp.z), mask_p) / (1.0f + M_by);

    // Soft clamp
    D_out[idx] = make_float3(
        clamp_diffs(D_y, d_max),
        clamp_diffs(D_rg, d_max),
        clamp_diffs(D_by, d_max)
    );
}

// For baseband: D = |T_f - R_f| * S (no masking)
__launch_bounds__(256)
__global__ void baseband_difference_kernel(
    const float3* test_contrast, const float3* ref_contrast,
    float3* D_out,
    int width, int height,
    float rho_cpd, float log_L_bkg,
    const float* log_L_bkg_lut, const float* log_rho_lut,
    const float* logS_c0, const float* logS_c1, const float* logS_c2,
    int num_L_bkg, int num_rho,
    float sensitivity_correction
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    float3 T = test_contrast[idx];
    float3 R = ref_contrast[idx];

    // Find position in CSF LUT
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

    // Interpolate sensitivity
    float logS_y = interp2d(logS_c0, num_rho, num_L_bkg, rho_idx, L_idx);
    float logS_rg = interp2d(logS_c1, num_rho, num_L_bkg, rho_idx, L_idx);
    float logS_by = interp2d(logS_c2, num_rho, num_L_bkg, rho_idx, L_idx);

    float S_y = powf(10.0f, logS_y + sensitivity_correction / 20.0f);
    float S_rg = powf(10.0f, logS_rg + sensitivity_correction / 20.0f);
    float S_by = powf(10.0f, logS_by + sensitivity_correction / 20.0f);

    // For baseband: D = |T - R| * S
    D_out[idx] = make_float3(
        fabsf(T.x - R.x) * S_y,
        fabsf(T.y - R.y) * S_rg,
        fabsf(T.z - R.z) * S_by
    );
}

// Spatial pooling kernel (beta-norm across DKL channels)
// Computes: sum over pixels of (|D_y|^beta + |D_rg|^beta + |D_by|^beta)
// The Python code does: lp_norm(D, beta, dim=(-2,-1), normalize=True)
// normalize=True means: divide by number of pixels
__launch_bounds__(256)
__global__ void spatial_pooling_kernel(const float3* input, float* output,
                                        int width, int height, float beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    float3 val = input[idx];

    // Compute beta-norm contribution for this pixel across DKL channels
    // Using Minkowski p-norm: (|x|^p + |y|^p + |z|^p)
    float sum = safe_pow(fabsf(val.x), beta) + safe_pow(fabsf(val.y), beta) + safe_pow(fabsf(val.z), beta);

    // Store the sum (will reduce across pixels and normalize later)
    output[idx] = sum;
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
        GPU_CHECK(hipMemcpyDtoDAsync(test_linear_d, test_d, frame_size * sizeof(float3), stream));
        GPU_CHECK(hipMemcpyDtoDAsync(ref_linear_d, ref_d, frame_size * sizeof(float3), stream));

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

        // Step 4: Apply CSF and masking model for each band
        // In Python: Q_per_ch_block[:,:,:,bb] = lp_norm(D, beta, dim=(-2,-1), normalize=True)
        std::vector<float> band_scores(lpyr.num_bands);

        for (int band = 0; band < lpyr.num_bands; band++) {
            int band_size = lpyr.band_widths[band] * lpyr.band_heights[band];
            float rho = lpyr.band_freqs[band]; // Spatial frequency in cpd
            bool is_baseband = (band == lpyr.num_bands - 1);

            int threads = 256;
            int blocks = (band_size + threads - 1) / threads;

            // Pyramid bands already contain contrast (clamped to [-1000, 1000])
            float3* test_contrast = test_bands[band];
            float3* ref_contrast = ref_bands[band];

            // Allocate output buffer for D (perceptually weighted differences)
            float3* D_d;
            GPU_CHECK(hipMallocAsync(&D_d, band_size * sizeof(float3), stream));

            float log_L_bkg = log10f(L_max / 2.0f); // Use average luminance

            if (is_baseband) {
                // For baseband: D = |T_f - R_f| * S (no masking)
                baseband_difference_kernel<<<blocks, threads, 0, stream>>>(
                    test_contrast, ref_contrast, D_d,
                    lpyr.band_widths[band], lpyr.band_heights[band],
                    rho, log_L_bkg,
                    csf.log_L_bkg_d, csf.log_rho_d,
                    csf.logS_o0_c0_d, csf.logS_o0_c1_d, csf.logS_o0_c2_d,
                    csf.num_L_bkg, csf.num_rho, params.sensitivity_correction
                );
            } else {
                // For other bands: Apply masking model
                // Step 1: Apply CSF and channel gain -> T_p, R_p
                float3 *T_p_d, *R_p_d;
                GPU_CHECK(hipMallocAsync(&T_p_d, band_size * sizeof(float3), stream));
                GPU_CHECK(hipMallocAsync(&R_p_d, band_size * sizeof(float3), stream));

                apply_csf_and_gain_kernel<<<blocks, threads, 0, stream>>>(
                    test_contrast, ref_contrast, T_p_d, R_p_d,
                    lpyr.band_widths[band], lpyr.band_heights[band],
                    rho, log_L_bkg,
                    csf.log_L_bkg_d, csf.log_rho_d,
                    csf.logS_o0_c0_d, csf.logS_o0_c1_d, csf.logS_o0_c2_d,
                    csf.num_L_bkg, csf.num_rho, params.sensitivity_correction
                );

                // Step 2: Compute mutual masking: M_mm = min(|T_p|, |R_p|)
                float3* M_mm_d;
                GPU_CHECK(hipMallocAsync(&M_mm_d, band_size * sizeof(float3), stream));

                compute_mutual_masking_kernel<<<blocks, threads, 0, stream>>>(
                    T_p_d, R_p_d, M_mm_d, band_size
                );

                // Step 3: Apply phase uncertainty: M_pu = M_mm * 10^mask_c
                // (Gaussian blur omitted for simplicity)
                apply_phase_uncertainty_kernel<<<blocks, threads, 0, stream>>>(
                    M_mm_d, params.mask_c, band_size
                );

                // Step 4: Apply masking model: D = safe_pow(|T_p - R_p|, p) / (1 + M^q)
                apply_masking_kernel<<<blocks, threads, 0, stream>>>(
                    T_p_d, R_p_d, M_mm_d, D_d,
                    params.mask_p,
                    params.mask_q[0], params.mask_q[1], params.mask_q[2],
                    params.d_max,
                    band_size
                );

                GPU_CHECK(hipFreeAsync(T_p_d, stream));
                GPU_CHECK(hipFreeAsync(R_p_d, stream));
                GPU_CHECK(hipFreeAsync(M_mm_d, stream));
            }

            // Step 5: Spatial pooling with normalize=True
            // Python: lp_norm(D, beta, dim=(-2,-1), normalize=True, keepdim=False)
            // This computes: (sum(|D|^beta) / N)^(1/beta) where N = num_pixels
            float* pooled_d;
            GPU_CHECK(hipMallocAsync(&pooled_d, band_size * sizeof(float), stream));

            spatial_pooling_kernel<<<blocks, threads, 0, stream>>>(
                D_d, pooled_d, lpyr.band_widths[band], lpyr.band_heights[band], params.beta
            );

            // Reduce to single value (sum across all pixels)
            float* partial_sums_d;
            int num_blocks = (band_size + 255) / 256;
            GPU_CHECK(hipMallocAsync(&partial_sums_d, num_blocks * sizeof(float), stream));

            reduce_mean_kernel<<<num_blocks, 256, 256 * sizeof(float), stream>>>(
                pooled_d, partial_sums_d, band_size
            );

            // Final reduction on CPU
            std::vector<float> partial_sums_h(num_blocks);
            GPU_CHECK(hipMemcpyDtoHAsync(partial_sums_h.data(), partial_sums_d,
                                      num_blocks * sizeof(float), stream));
            hipStreamSynchronize(stream);

            float sum = 0.0f;
            for (int i = 0; i < num_blocks; i++) {
                sum += partial_sums_h[i];
            }

            // Normalize by number of pixels and take beta root
            // This matches Python: lp_norm with normalize=True
            float normalized_sum = sum / band_size;
            band_scores[band] = powf(normalized_sum, 1.0f / params.beta);

            // DEBUG: Print band info
            if (band == 0 || band == lpyr.num_bands - 1) {
                printf("[CVVDP DEBUG] Band %d: rho=%.2f cpd, normalized_sum=%.8f, score=%.8f, is_baseband=%d\n",
                       band, rho, normalized_sum, band_scores[band], is_baseband);
            }

            // Free buffers
            GPU_CHECK(hipFreeAsync(D_d, stream));
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

        // Step 6: Combine band scores with proper weighting
        // Use beta pooling across bands
        float sum_bands = 0.0f;
        for (int band = 0; band < lpyr.num_bands; band++) {
            // Apply baseband weight if available
            float weight = 1.0f;
            if (band < params.baseband_weight.size()) {
                weight = params.baseband_weight[band];
            }

            // Accumulate weighted contributions
            sum_bands += weight * powf(band_scores[band], params.beta);
        }

        // Take the beta root of the pooled value
        float Q_per_ch = powf(sum_bands, 1.0f / params.beta);

        // Step 7: Convert to JOD scale
        // Q_JOD = jod_a * Q^jod_exp
        float Q_jod = params.jod_a * powf(Q_per_ch, params.jod_exp);

        // DEBUG: Print final conversion
        printf("[CVVDP DEBUG] Q_per_ch=%.8f, Q_jod(raw)=%.8f, jod_a=%.6f, jod_exp=%.6f\n",
               Q_per_ch, Q_jod, params.jod_a, params.jod_exp);

        // Clamp to reasonable range (0-10 JOD)
        Q_jod = fminf(fmaxf(Q_jod, 0.0f), 10.0f);

        printf("[CVVDP DEBUG] Q_jod(final)=%.8f\n", Q_jod);

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

    // Temporary buffers for converting uint16 planar RGB to float3
    float3* temp_src_d;
    float3* temp_dist_d;

public:
    void init(int w, int h, const std::string& display = "standard_4k") {
        width = w;
        height = h;
        display_name = display;

        // Initialize the processor
        processor.init(width, height, display_name);

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
        int64_t stride_dist
    ) {
        // Only support UINT16 for now
        static_assert(T == UINT16, "CVVDP currently only supports UINT16 input");

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
        float jod_score = processor.process_frame(temp_dist_d, temp_src_d);

        // Return JOD score in all three positions (CVVDP only outputs one score)
        return std::make_tuple(jod_score, jod_score, jod_score);
    }
};

} // namespace cvvdp
