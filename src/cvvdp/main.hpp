#pragma once

#include "../util/preprocessor.hpp"
#include "../util/VshipExceptions.hpp"
#include "../util/gpuhelper.hpp"
#include "../util/float3operations.hpp"
#include "config.hpp"
#include "colorspace.hpp"
#include "lpyr.hpp"
#include "csf.hpp"
#include "temporal.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <vector>
#include <utility>

#include "../../../third_party/rapidjson/include/rapidjson/stringbuffer.h"
#include "../../../third_party/rapidjson/include/rapidjson/writer.h"

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

// Apply cross-channel masking with xcm_weights matrix
// M_out = 10^(xcm_weights @ log10(M_in))
// This models how masking in one channel affects others
__launch_bounds__(256)
__global__ void apply_cross_channel_masking_kernel(
    const float3* M_pu,  // Phase uncertainty applied masking (3 sustained channels)
    float3* M_xcm,       // Output: cross-channel masked values
    const float* xcm_weights,  // 4×4 matrix (16 elements)
    float mask_q_y, float mask_q_rg, float mask_q_by, float mask_q_trans,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float3 M = M_pu[idx];

    // Apply mask_q exponent to get M^q for each channel
    float M_raw[4];
    M_raw[0] = safe_pow(M.x, mask_q_y);      // Y-sustained
    M_raw[1] = safe_pow(M.y, mask_q_rg);     // RG
    M_raw[2] = safe_pow(M.z, mask_q_by);     // BY
    M_raw[3] = safe_pow(M.x, mask_q_trans);  // Y-transient (use Y for now)

    // Convert to log space (with small epsilon to avoid log(0))
    float log_M[4];
    const float epsilon = 1e-8f;
    for (int i = 0; i < 4; i++) {
        log_M[i] = log10f(M_raw[i] + epsilon);
    }

    // Apply cross-channel masking: log_M_out = xcm_weights @ log_M
    // xcm_weights is row-major 4×4 matrix
    float log_M_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            log_M_out[row] += xcm_weights[row * 4 + col] * log_M[col];
        }
    }

    // Convert back from log space: M_out = 10^log_M_out
    float M_out[4];
    for (int i = 0; i < 4; i++) {
        M_out[i] = powf(10.0f, log_M_out[i]);
    }

    // Output sustained channels only (Y, RG, BY)
    // Transient channel pooling will be added later
    M_xcm[idx] = make_float3(M_out[0], M_out[1], M_out[2]);
}

// Apply masking model: D_u = |T_p - R_p|^p / (1 + M), then clamp
__launch_bounds__(256)
__global__ void apply_masking_kernel(
    const float3* T_p, const float3* R_p,
    const float3* M_masked,  // Cross-channel masked values
    float3* D_out,
    float mask_p,
    float d_max,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float3 Tp = T_p[idx];
    float3 Rp = R_p[idx];
    float3 M = M_masked[idx];

    // D_u = |T_p - R_p|^p / (1 + M)
    float D_y = safe_pow(fabsf(Tp.x - Rp.x), mask_p) / (1.0f + M.x);
    float D_rg = safe_pow(fabsf(Tp.y - Rp.y), mask_p) / (1.0f + M.y);
    float D_by = safe_pow(fabsf(Tp.z - Rp.z), mask_p) / (1.0f + M.z);

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

// Spatial pooling kernel - per-channel version for hierarchical pooling
// Computes beta-norm separately for each channel: (|D_y|^beta, |D_rg|^beta, |D_by|^beta)
// The Python code does: lp_norm(D, beta, dim=(-2,-1), normalize=True) per channel
// This maintains per-channel tracking for later beta_tch pooling
__launch_bounds__(256)
__global__ void spatial_pooling_per_channel_kernel(const float3* input, float3* output,
                                                     int width, int height, float beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    float3 val = input[idx];

    // Compute beta-norm contribution for each channel separately
    // This allows hierarchical pooling: spatial → band → channel
    output[idx] = make_float3(
        safe_pow(fabsf(val.x), beta),  // Y channel
        safe_pow(fabsf(val.y), beta),  // RG channel
        safe_pow(fabsf(val.z), beta)   // BY channel
    );
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

// Reduction kernel for per-channel mean (float3 version)
// Used for hierarchical pooling to maintain per-channel scores
__launch_bounds__(256)
__global__ void reduce_mean_per_channel_kernel(const float3* input, float3* output, int size) {
    extern __shared__ float3 sdata_vec[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (idx < size) {
        sdata_vec[tid] = input[idx];
    } else {
        sdata_vec[tid] = make_float3(0.0f, 0.0f, 0.0f);
    }
    __syncthreads();

    // Reduction in shared memory (component-wise)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_vec[tid].x += sdata_vec[tid + s].x;
            sdata_vec[tid].y += sdata_vec[tid + s].y;
            sdata_vec[tid].z += sdata_vec[tid + s].z;
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata_vec[0];
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
    TemporalBuffer temporal;
    hipStream_t stream;
    std::filesystem::path data_root;
    float frame_rate = 30.0f; // Default, can be updated
    bool temporal_filtering_enabled = true;
    bool cross_channel_masking_enabled = true;

    // Cross-channel masking weights (4×4 matrix on GPU)
    float* xcm_weights_d = nullptr;

public:
    void init(int w, int h, const std::string& display_name, const std::string& data_dir_override = "") {
        width = w;
        height = h;

        // Resolve configuration directory
        data_root = resolve_cvvdp_data_root(data_dir_override);

        const std::filesystem::path display_model_path = data_root / "display_models.json";
        const std::filesystem::path params_path = data_root / "cvvdp_parameters.json";

        display = DisplayModel::load(display_model_path, display_name);
        params = CVVDPParameters::load(params_path);

        ppd = display.get_ppd();

        // Initialize Laplacian pyramid
        lpyr.init(width, height, ppd);

        // Initialize CSF
        hipStreamCreate(&stream);
        const std::filesystem::path csf_path = data_root / ("csf_lut_" + params.csf + ".json");
        csf.init(csf_path.string(), stream);

        // Initialize temporal filtering
        if (temporal_filtering_enabled && !params.sigma_tf.empty()) {
            temporal.init(
                lpyr.num_bands,
                lpyr.band_widths,
                lpyr.band_heights,
                params.sigma_tf,
                params.beta_tf,
                frame_rate,
                stream
            );
        }

        // Initialize cross-channel masking weights
        if (cross_channel_masking_enabled && !params.xcm_weights.empty()) {
            if (params.xcm_weights.size() != 16) {
                std::cerr << "[CVVDP] Warning: xcm_weights should have 16 elements (4×4 matrix), got "
                          << params.xcm_weights.size() << ". Disabling cross-channel masking." << std::endl;
                cross_channel_masking_enabled = false;
            } else {
                // Allocate GPU memory and copy weights
                GPU_CHECK(hipMalloc(&xcm_weights_d, 16 * sizeof(float)));
                GPU_CHECK(hipMemcpyHtoD(xcm_weights_d, params.xcm_weights.data(), 16 * sizeof(float)));
            }
        }
    }

    void destroy() {
        if (xcm_weights_d) {
            hipFree(xcm_weights_d);
            xcm_weights_d = nullptr;
        }
        temporal.destroy();
        csf.destroy();
        hipStreamDestroy(stream);
    }

    // Allow setting frame rate for temporal filtering
    void set_frame_rate(float fps) {
        frame_rate = fps;
    }

    // Allow disabling temporal filtering (for testing)
    void set_temporal_filtering(bool enabled) {
        temporal_filtering_enabled = enabled;
    }

    // Process a single frame pair and return JOD score
    // Input: test and reference frames in planar RGB format (float3 arrays)
    float process_frame(const float3* test_d, const float3* ref_d, int frame_index = -1, std::string* debug_json_out = nullptr) {
        const bool collect_debug = (debug_json_out != nullptr);

        struct KernelTimings {
            float apply_csf_and_gain = 0.0f;
            float mutual_mask = 0.0f;
            float phase_uncertainty = 0.0f;
            float cross_channel_mask = 0.0f;
            float masking = 0.0f;
            float baseband = 0.0f;
            float spatial_pool = 0.0f;
            float reduce = 0.0f;
            float host_copy = 0.0f;
        };

        struct ChannelStats {
            float min = std::numeric_limits<float>::max();
            float max = std::numeric_limits<float>::lowest();
            double sum = 0.0;
        };

        struct BandDebugInfo {
            int index = 0;
            bool is_baseband = false;
            int width = 0;
            int height = 0;
            float rho_cpd = 0.0f;
            float beta_score = 0.0f;
            int sample_count = 0;
            std::array<ChannelStats, 3> channels{};
        };

        KernelTimings timings;
        std::vector<BandDebugInfo> band_debug_infos;
        if (collect_debug) {
            band_debug_infos.reserve(lpyr.num_bands);
        }

        hipEvent_t timing_start = nullptr;
        hipEvent_t timing_end = nullptr;
        if (collect_debug) {
            hipEventCreateWithFlags(&timing_start, hipEventDefault);
            hipEventCreateWithFlags(&timing_end, hipEventDefault);
        }

        int frame_size = width * height;

        // Allocate working buffers
        float3 *test_linear_d, *ref_linear_d;
        GPU_CHECK(hipMallocAsync(&test_linear_d, frame_size * sizeof(float3), stream));
        GPU_CHECK(hipMallocAsync(&ref_linear_d, frame_size * sizeof(float3), stream));

        // Copy input
        GPU_CHECK(hipMemcpyDtoDAsync(test_linear_d, test_d, frame_size * sizeof(float3), stream));
        GPU_CHECK(hipMemcpyDtoDAsync(ref_linear_d, ref_d, frame_size * sizeof(float3), stream));

        // Step 1: Convert to linear luminance
        const float L_max = display.max_luminance;
        const float L_black = display.get_black_level();
        const float gamma = 2.2f; // TODO: derive from display colorspace

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
        // Store per-channel scores for hierarchical pooling
        std::vector<float3> band_scores(lpyr.num_bands);

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

            float log_L_bkg = log10f(std::max(L_max / 2.0f, 1e-4f));

            if (is_baseband) {
                if (collect_debug) GPU_CHECK(hipEventRecord(timing_start, stream));
                baseband_difference_kernel<<<blocks, threads, 0, stream>>>(
                    test_contrast, ref_contrast, D_d,
                    lpyr.band_widths[band], lpyr.band_heights[band],
                    rho, log_L_bkg,
                    csf.log_L_bkg_d, csf.log_rho_d,
                    csf.logS_o0_c0_d, csf.logS_o0_c1_d, csf.logS_o0_c2_d,
                    csf.num_L_bkg, csf.num_rho, params.sensitivity_correction
                );
                if (collect_debug) {
                    GPU_CHECK(hipEventRecord(timing_end, stream));
                    GPU_CHECK(hipEventSynchronize(timing_end));
                    float elapsed = 0.0f;
                    GPU_CHECK(hipEventElapsedTime(&elapsed, timing_start, timing_end));
                    timings.baseband += elapsed;
                }
            } else {
                float3 *T_p_d, *R_p_d;
                GPU_CHECK(hipMallocAsync(&T_p_d, band_size * sizeof(float3), stream));
                GPU_CHECK(hipMallocAsync(&R_p_d, band_size * sizeof(float3), stream));

                if (collect_debug) GPU_CHECK(hipEventRecord(timing_start, stream));
                apply_csf_and_gain_kernel<<<blocks, threads, 0, stream>>>(
                    test_contrast, ref_contrast, T_p_d, R_p_d,
                    lpyr.band_widths[band], lpyr.band_heights[band],
                    rho, log_L_bkg,
                    csf.log_L_bkg_d, csf.log_rho_d,
                    csf.logS_o0_c0_d, csf.logS_o0_c1_d, csf.logS_o0_c2_d,
                    csf.num_L_bkg, csf.num_rho, params.sensitivity_correction
                );
                if (collect_debug) {
                    GPU_CHECK(hipEventRecord(timing_end, stream));
                    GPU_CHECK(hipEventSynchronize(timing_end));
                    float elapsed = 0.0f;
                    GPU_CHECK(hipEventElapsedTime(&elapsed, timing_start, timing_end));
                    timings.apply_csf_and_gain += elapsed;
                }

                // Apply temporal filtering if enabled
                if (temporal_filtering_enabled && temporal.num_bands > 0 && band < temporal.num_bands) {
                    apply_temporal_filtering(
                        T_p_d, R_p_d,
                        temporal.bands[band],
                        temporal.alpha,
                        lpyr.band_widths[band], lpyr.band_heights[band],
                        stream
                    );
                }

                float3* M_mm_d;
                GPU_CHECK(hipMallocAsync(&M_mm_d, band_size * sizeof(float3), stream));

                if (collect_debug) GPU_CHECK(hipEventRecord(timing_start, stream));
                compute_mutual_masking_kernel<<<blocks, threads, 0, stream>>>(
                    T_p_d, R_p_d, M_mm_d, band_size
                );
                if (collect_debug) {
                    GPU_CHECK(hipEventRecord(timing_end, stream));
                    GPU_CHECK(hipEventSynchronize(timing_end));
                    float elapsed = 0.0f;
                    GPU_CHECK(hipEventElapsedTime(&elapsed, timing_start, timing_end));
                    timings.mutual_mask += elapsed;
                }

                if (collect_debug) GPU_CHECK(hipEventRecord(timing_start, stream));
                apply_phase_uncertainty_kernel<<<blocks, threads, 0, stream>>>(
                    M_mm_d, params.mask_c, band_size
                );
                if (collect_debug) {
                    GPU_CHECK(hipEventRecord(timing_end, stream));
                    GPU_CHECK(hipEventSynchronize(timing_end));
                    float elapsed = 0.0f;
                    GPU_CHECK(hipEventElapsedTime(&elapsed, timing_start, timing_end));
                    timings.phase_uncertainty += elapsed;
                }

                // Apply cross-channel masking if enabled
                const float mask_q_y = (params.mask_q.size() > 0) ? params.mask_q[0] : 1.0f;
                const float mask_q_rg = (params.mask_q.size() > 1) ? params.mask_q[1] : mask_q_y;
                const float mask_q_by = (params.mask_q.size() > 2) ? params.mask_q[2] : mask_q_rg;
                const float mask_q_trans = (params.mask_q.size() > 3) ? params.mask_q[3] : mask_q_by;

                float3* M_xcm_d;
                GPU_CHECK(hipMallocAsync(&M_xcm_d, band_size * sizeof(float3), stream));

                if (cross_channel_masking_enabled && xcm_weights_d) {
                    if (collect_debug) GPU_CHECK(hipEventRecord(timing_start, stream));
                    apply_cross_channel_masking_kernel<<<blocks, threads, 0, stream>>>(
                        M_mm_d, M_xcm_d, xcm_weights_d,
                        mask_q_y, mask_q_rg, mask_q_by, mask_q_trans,
                        band_size
                    );
                    if (collect_debug) {
                        GPU_CHECK(hipEventRecord(timing_end, stream));
                        GPU_CHECK(hipEventSynchronize(timing_end));
                        float elapsed = 0.0f;
                        GPU_CHECK(hipEventElapsedTime(&elapsed, timing_start, timing_end));
                        timings.cross_channel_mask += elapsed;
                    }
                } else {
                    // No cross-channel masking - just copy M_mm_d to M_xcm_d
                    GPU_CHECK(hipMemcpyDtoDAsync(M_xcm_d, M_mm_d, band_size * sizeof(float3), stream));
                }

                if (collect_debug) GPU_CHECK(hipEventRecord(timing_start, stream));
                apply_masking_kernel<<<blocks, threads, 0, stream>>>(
                    T_p_d, R_p_d, M_xcm_d, D_d,
                    params.mask_p,
                    params.d_max,
                    band_size
                );
                if (collect_debug) {
                    GPU_CHECK(hipEventRecord(timing_end, stream));
                    GPU_CHECK(hipEventSynchronize(timing_end));
                    float elapsed = 0.0f;
                    GPU_CHECK(hipEventElapsedTime(&elapsed, timing_start, timing_end));
                    timings.masking += elapsed;
                }

                GPU_CHECK(hipFreeAsync(T_p_d, stream));
                GPU_CHECK(hipFreeAsync(R_p_d, stream));
                GPU_CHECK(hipFreeAsync(M_mm_d, stream));
                GPU_CHECK(hipFreeAsync(M_xcm_d, stream));
            }

            // Step 5: Spatial pooling with normalize=True (per-channel)
            // Pool each channel separately for hierarchical pooling
            float3* pooled_d;
            GPU_CHECK(hipMallocAsync(&pooled_d, band_size * sizeof(float3), stream));

            if (collect_debug) GPU_CHECK(hipEventRecord(timing_start, stream));
            spatial_pooling_per_channel_kernel<<<blocks, threads, 0, stream>>>(
                D_d, pooled_d, lpyr.band_widths[band], lpyr.band_heights[band], params.beta
            );
            if (collect_debug) {
                GPU_CHECK(hipEventRecord(timing_end, stream));
                GPU_CHECK(hipEventSynchronize(timing_end));
                float elapsed = 0.0f;
                GPU_CHECK(hipEventElapsedTime(&elapsed, timing_start, timing_end));
                timings.spatial_pool += elapsed;
            }

            // Reduce per-channel sums
            float3* partial_sums_d;
            int num_blocks = (band_size + 255) / 256;
            GPU_CHECK(hipMallocAsync(&partial_sums_d, num_blocks * sizeof(float3), stream));

            if (collect_debug) GPU_CHECK(hipEventRecord(timing_start, stream));
            reduce_mean_per_channel_kernel<<<num_blocks, 256, 256 * sizeof(float3), stream>>>(
                pooled_d, partial_sums_d, band_size
            );
            if (collect_debug) {
                GPU_CHECK(hipEventRecord(timing_end, stream));
                GPU_CHECK(hipEventSynchronize(timing_end));
                float elapsed = 0.0f;
                GPU_CHECK(hipEventElapsedTime(&elapsed, timing_start, timing_end));
                timings.reduce += elapsed;
            }

            std::vector<float3> partial_sums_h(num_blocks);
            std::vector<float3> band_values_h;
            if (collect_debug) {
                band_values_h.resize(band_size);
                GPU_CHECK(hipEventRecord(timing_start, stream));
                GPU_CHECK(hipMemcpyDtoHAsync(band_values_h.data(), D_d, band_size * sizeof(float3), stream));
            }
            GPU_CHECK(hipMemcpyDtoHAsync(partial_sums_h.data(), partial_sums_d,
                                      num_blocks * sizeof(float3), stream));
            if (collect_debug) GPU_CHECK(hipEventRecord(timing_end, stream));

            hipStreamSynchronize(stream);

            if (collect_debug) {
                GPU_CHECK(hipEventSynchronize(timing_end));
                float elapsed = 0.0f;
                GPU_CHECK(hipEventElapsedTime(&elapsed, timing_start, timing_end));
                timings.host_copy += elapsed;
            }

            // Sum per-channel partial sums
            float3 sum = make_float3(0.0f, 0.0f, 0.0f);
            for (int i = 0; i < num_blocks; i++) {
                sum.x += partial_sums_h[i].x;
                sum.y += partial_sums_h[i].y;
                sum.z += partial_sums_h[i].z;
            }

            // Normalize and take beta-th root per channel
            float3 normalized_sum = make_float3(
                sum.x / std::max(1, band_size),
                sum.y / std::max(1, band_size),
                sum.z / std::max(1, band_size)
            );
            band_scores[band] = make_float3(
                powf(normalized_sum.x, 1.0f / params.beta),
                powf(normalized_sum.y, 1.0f / params.beta),
                powf(normalized_sum.z, 1.0f / params.beta)
            );

            if (collect_debug) {
                BandDebugInfo info;
                info.index = band;
                info.is_baseband = is_baseband;
                info.width = lpyr.band_widths[band];
                info.height = lpyr.band_heights[band];
                info.rho_cpd = rho;
                info.beta_score = band_scores[band];
                info.sample_count = band_size;

                for (const float3& value : band_values_h) {
                    const float channels[3] = {value.x, value.y, value.z};
                    for (int c = 0; c < 3; ++c) {
                        ChannelStats& stats = info.channels[c];
                        stats.min = std::min(stats.min, channels[c]);
                        stats.max = std::max(stats.max, channels[c]);
                        stats.sum += channels[c];
                    }
                }

                band_debug_infos.push_back(std::move(info));
            }

            GPU_CHECK(hipFreeAsync(D_d, stream));
            GPU_CHECK(hipFreeAsync(pooled_d, stream));
            GPU_CHECK(hipFreeAsync(partial_sums_d, stream));
        }

        lpyr.free_bands(test_bands, stream);
        lpyr.free_bands(ref_bands, stream);
        delete[] test_bands;
        delete[] ref_bands;

        GPU_CHECK(hipFreeAsync(test_linear_d, stream));
        GPU_CHECK(hipFreeAsync(ref_linear_d, stream));

        // Hierarchical pooling across spatial bands using beta_sch
        // Python: Q_sc = lp_norm(Q_per_ch * per_ch_w * per_sband_w, beta_sch, dim=spatial_bands)
        // We already applied per_ch_w in spatial pooling, now apply per_sband_w (baseband_weight)
        const float beta_sch = (params.beta_sch > 0.0f) ? params.beta_sch : params.beta;

        float sum_bands = 0.0f;
        for (int band = 0; band < lpyr.num_bands; band++) {
            float weight = 1.0f;
            if (band < static_cast<int>(params.baseband_weight.size())) {
                weight = params.baseband_weight[band];
            }
            sum_bands += weight * powf(band_scores[band], beta_sch);
        }

        float Q_per_ch = powf(sum_bands, 1.0f / beta_sch);

        // Convert Q to JOD scale (matches Python: Q_JOD = 10 - jod_a * Q^jod_exp)
        // Use linearization for small Q values to maintain differentiability
        const float Q_t = 0.1f;  // Threshold for linearization
        float Q_jod;

        if (Q_per_ch <= Q_t) {
            // Linearized version: jod_a_p = jod_a * Q_t^(jod_exp-1)
            float jod_a_p = params.jod_a * powf(Q_t, params.jod_exp - 1.0f);
            Q_jod = 10.0f - jod_a_p * Q_per_ch;
        } else {
            Q_jod = 10.0f - params.jod_a * powf(Q_per_ch, params.jod_exp);
        }

        Q_jod = fminf(fmaxf(Q_jod, 0.0f), 10.0f);

        if (collect_debug) {
            GPU_CHECK(hipEventDestroy(timing_start));
            GPU_CHECK(hipEventDestroy(timing_end));
        }

        if (collect_debug && debug_json_out) {
            rapidjson::StringBuffer buffer;
            rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

            const std::string data_root_utf8 = data_root.u8string();

            writer.StartObject();
            writer.Key("metric");
            writer.String("cvvdp");
            writer.Key("frame");
            writer.Int(frame_index);
            writer.Key("jod");
            writer.Double(static_cast<double>(Q_jod));
            writer.Key("width");
            writer.Int(width);
            writer.Key("height");
            writer.Int(height);
            writer.Key("ppd");
            writer.Double(static_cast<double>(ppd));
            writer.Key("display");
            writer.StartObject();
            writer.Key("name");
            writer.String(display.name.c_str());
            writer.Key("max_luminance");
            writer.Double(static_cast<double>(display.max_luminance));
            writer.Key("min_luminance");
            writer.Double(static_cast<double>(display.min_luminance));
            writer.Key("contrast");
            writer.Double(static_cast<double>(display.contrast));
            writer.EndObject();
            writer.Key("config_root");
            writer.String(data_root_utf8.c_str());
            writer.Key("temporal");
            writer.StartObject();
            writer.Key("buffer_fill");
            writer.Double(0.0); // Temporal filtering not yet implemented
            writer.Key("filter_len");
            writer.Int(params.filter_len);
            writer.EndObject();
            writer.Key("bands");
            writer.StartArray();
            for (const BandDebugInfo& info : band_debug_infos) {
                writer.StartObject();
                writer.Key("index");
                writer.Int(info.index);
                writer.Key("is_baseband");
                writer.Bool(info.is_baseband);
                writer.Key("rho_cpd");
                writer.Double(static_cast<double>(info.rho_cpd));
                writer.Key("width");
                writer.Int(info.width);
                writer.Key("height");
                writer.Int(info.height);
                writer.Key("beta_score");
                writer.Double(static_cast<double>(info.beta_score));
                writer.Key("channels");
                writer.StartObject();
                const char* channel_names[3] = {"Y", "RG", "BY"};
                for (int c = 0; c < 3; ++c) {
                    writer.Key(channel_names[c]);
                    writer.StartObject();
                    writer.Key("min");
                    writer.Double(static_cast<double>(info.channels[c].min));
                    writer.Key("max");
                    writer.Double(static_cast<double>(info.channels[c].max));
                    double mean = (info.sample_count > 0)
                                      ? info.channels[c].sum / static_cast<double>(info.sample_count)
                                      : 0.0;
                    writer.Key("mean");
                    writer.Double(mean);
                    writer.EndObject();
                }
                writer.EndObject();
                writer.EndObject();
            }
            writer.EndArray();
            writer.Key("kernels_ms");
            writer.StartObject();
            writer.Key("apply_csf_and_gain");
            writer.Double(static_cast<double>(timings.apply_csf_and_gain));
            writer.Key("mutual_mask");
            writer.Double(static_cast<double>(timings.mutual_mask));
            writer.Key("phase_uncertainty");
            writer.Double(static_cast<double>(timings.phase_uncertainty));
            writer.Key("cross_channel_mask");
            writer.Double(static_cast<double>(timings.cross_channel_mask));
            writer.Key("masking");
            writer.Double(static_cast<double>(timings.masking));
            writer.Key("baseband");
            writer.Double(static_cast<double>(timings.baseband));
            writer.Key("spatial_pool");
            writer.Double(static_cast<double>(timings.spatial_pool));
            writer.Key("reduce");
            writer.Double(static_cast<double>(timings.reduce));
            writer.Key("host_copy");
            writer.Double(static_cast<double>(timings.host_copy));
            writer.EndObject();
            writer.EndObject();

            *debug_json_out = buffer.GetString();
        }

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
        return handle->processor.process_frame(test_d, ref_d, -1, nullptr);
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
    std::string data_dir_override;
    bool debug_enabled = false;

    // Temporary buffers for converting uint16 planar RGB to float3
    float3* temp_src_d = nullptr;
    float3* temp_dist_d = nullptr;

public:
    void init(int w, int h, const std::string& display = "standard_4k",
              const std::string& data_dir = "", bool enable_debug = false) {
        width = w;
        height = h;
        display_name = display;
        data_dir_override = data_dir;
        debug_enabled = enable_debug;

        // Initialize the processor
        processor.init(width, height, display_name, data_dir_override);

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
        int64_t stride_dist,
        int frame_index = -1
    ) {
        // Only support UINT16 for now
        static_assert(T == InputMemType::UINT16, "CVVDP currently only supports UINT16 input");

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
        std::string telemetry_blob;
        float jod_score = processor.process_frame(
            temp_dist_d,
            temp_src_d,
            frame_index,
            debug_enabled ? &telemetry_blob : nullptr);

        if (debug_enabled && !telemetry_blob.empty()) {
            static std::mutex telemetry_mutex;
            std::lock_guard<std::mutex> lock(telemetry_mutex);
            std::cout << telemetry_blob << std::endl;
        }

        // Return JOD score in all three positions (CVVDP only outputs one score)
        return std::make_tuple(jod_score, jod_score, jod_score);
    }
};

} // namespace cvvdp








