#pragma once

#include "VapourSynth4.h"
#include "VSHelper4.h"
#include "../util/preprocessor.hpp"
#include "../util/VshipExceptions.hpp"
#include "../util/gpuhelper.hpp"
#include "../util/concurrency.hpp"
#include "main.hpp"
#include <string>
#include <sstream>

namespace cvvdp {

template <InputMemType T>
__launch_bounds__(256)
__global__ void pack_rgb_kernel(float3* out, const uint8_t* srcp0, const uint8_t* srcp1, const uint8_t* srcp2,
                                 int64_t stride, int64_t width, int64_t height) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    int x = idx % width;
    int y = idx / width;

    float r = convertPointer<T>(srcp0, y, x, stride);
    float g = convertPointer<T>(srcp1, y, x, stride);
    float b = convertPointer<T>(srcp2, y, x, stride);

    out[idx] = make_float3(r, g, b);
}

struct CVVDPData {
    VSNode* reference;
    VSNode* distorted;
    int gpu_id;
    std::string display_name;
    int numStream;

    // Per-stream data
    struct StreamData {
        hipStream_t stream;
        CVVDPHandle* cvvdp_handle;
        float3 *test_d, *ref_d;
        int width, height;
    };

    std::vector<StreamData> streams;
    ThreadSafeQueue<int> available_streams;

    const VSVideoInfo* vi;
};

template <InputMemType T>
static const VSFrame* VS_CC cvvdpGetFrame(int n, int activationReason, void* instanceData,
                                           void** frameData, VSFrameContext* frameCtx,
                                           VSCore* core, const VSAPI* vsapi) {
    CVVDPData* d = static_cast<CVVDPData*>(instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->reference, frameCtx);
        vsapi->requestFrameFilter(n, d->distorted, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrame* ref_frame = vsapi->getFrameFilter(n, d->reference, frameCtx);
        const VSFrame* dist_frame = vsapi->getFrameFilter(n, d->distorted, frameCtx);

        int width = vsapi->getFrameWidth(ref_frame, 0);
        int height = vsapi->getFrameHeight(ref_frame, 0);
        int stride = vsapi->getStride(ref_frame, 0);

        // Get an available stream
        int stream_id = d->available_streams.pop().value();
        auto& stream_data = d->streams[stream_id];

        try {
            // Copy reference frame to device
            const uint8_t* ref_planes[3] = {
                vsapi->getReadPtr(ref_frame, 0),
                vsapi->getReadPtr(ref_frame, 1),
                vsapi->getReadPtr(ref_frame, 2)
            };

            // Copy distorted frame to device
            const uint8_t* dist_planes[3] = {
                vsapi->getReadPtr(dist_frame, 0),
                vsapi->getReadPtr(dist_frame, 1),
                vsapi->getReadPtr(dist_frame, 2)
            };

            // Upload and pack RGB data
            uint8_t *temp_planes_d[3];
            for (int p = 0; p < 3; p++) {
                GPU_CHECK(hipMallocAsync(&temp_planes_d[p], stride * height, stream_data.stream));
                GPU_CHECK(hipMemcpyHtoDAsync(temp_planes_d[p], ref_planes[p], stride * height, stream_data.stream));
            }

            int threads = 256;
            int blocks = (width * height + threads - 1) / threads;
            pack_rgb_kernel<T><<<blocks, threads, 0, stream_data.stream>>>(
                stream_data.ref_d, temp_planes_d[0], temp_planes_d[1], temp_planes_d[2],
                stride, width, height);

            // Upload distorted
            for (int p = 0; p < 3; p++) {
                GPU_CHECK(hipMemcpyHtoDAsync(temp_planes_d[p], dist_planes[p], stride * height, stream_data.stream));
            }

            pack_rgb_kernel<T><<<blocks, threads, 0, stream_data.stream>>>(
                stream_data.test_d, temp_planes_d[0], temp_planes_d[1], temp_planes_d[2],
                stride, width, height);

            // Free temp buffers
            for (int p = 0; p < 3; p++) {
                GPU_CHECK(hipFreeAsync(temp_planes_d[p], stream_data.stream));
            }

            // Compute CVVDP score
            float jod_score = cvvdp_compute(stream_data.cvvdp_handle, stream_data.test_d, stream_data.ref_d);

            hipStreamSynchronize(stream_data.stream);

            // Create output frame (copy reference)
            VSFrame* dst = vsapi->copyFrame(ref_frame, core);

            // Set frame property with JOD score
            VSMap* props = vsapi->getFramePropertiesRW(dst);
            vsapi->mapSetFloat(props, "_CVVDP", jod_score, maReplace);

            // Return stream to pool
            d->available_streams.push(stream_id);

            vsapi->freeFrame(ref_frame);
            vsapi->freeFrame(dist_frame);

            return dst;

        } catch (const VshipError& e) {
            d->available_streams.push(stream_id);
            vsapi->freeFrame(ref_frame);
            vsapi->freeFrame(dist_frame);
            vsapi->setFilterError(e.getErrorMessage().c_str(), frameCtx);
            return nullptr;
        }
    }

    return nullptr;
}

static void VS_CC cvvdpFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    CVVDPData* d = static_cast<CVVDPData*>(instanceData);

    // Free all stream resources
    for (auto& stream_data : d->streams) {
        cvvdp_free(stream_data.cvvdp_handle);
        hipFree(stream_data.test_d);
        hipFree(stream_data.ref_d);
        hipStreamDestroy(stream_data.stream);
    }

    vsapi->freeNode(d->reference);
    vsapi->freeNode(d->distorted);

    delete d;
}

static void VS_CC cvvdpCreate(const VSMap* in, VSMap* out, void* userData,
                               VSCore* core, const VSAPI* vsapi) {
    std::unique_ptr<CVVDPData> d(new CVVDPData());
    int err;

    // Get input nodes
    d->reference = vsapi->mapGetNode(in, "reference", 0, nullptr);
    d->distorted = vsapi->mapGetNode(in, "distorted", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->reference);

    // Validate inputs
    const VSVideoInfo* vi_dist = vsapi->getVideoInfo(d->distorted);
    if (!vsh::isConstantVideoFormat(d->vi) || !vsh::isConstantVideoFormat(vi_dist)) {
        vsapi->mapSetError(out, "CVVDP: Only constant format input supported");
        vsapi->freeNode(d->reference);
        vsapi->freeNode(d->distorted);
        return;
    }

    if (d->vi->format.colorFamily != cfRGB || d->vi->format.numPlanes != 3) {
        vsapi->mapSetError(out, "CVVDP: Only planar RGB input supported");
        vsapi->freeNode(d->reference);
        vsapi->freeNode(d->distorted);
        return;
    }

    if (d->vi->width != vi_dist->width || d->vi->height != vi_dist->height) {
        vsapi->mapSetError(out, "CVVDP: Reference and distorted must have same dimensions");
        vsapi->freeNode(d->reference);
        vsapi->freeNode(d->distorted);
        return;
    }

    // Get optional parameters
    d->numStream = vsapi->mapGetIntSaturated(in, "numStream", 0, &err);
    if (err) d->numStream = 4;
    d->numStream = std::max(1, std::min(d->numStream, 8));

    d->gpu_id = vsapi->mapGetIntSaturated(in, "gpu_id", 0, &err);
    if (err) d->gpu_id = 0;

    const char* display_name_cstr = vsapi->mapGetData(in, "display", 0, &err);
    if (err || !display_name_cstr) {
        d->display_name = "standard_4k";
    } else {
        d->display_name = display_name_cstr;
    }

    // Set GPU device
    try {
        int count = helper::checkGpuCount();
        if (d->gpu_id >= count || d->gpu_id < 0) {
            vsapi->mapSetError(out, VshipError(BadDeviceArgument, __FILE__, __LINE__).getErrorMessage().c_str());
            vsapi->freeNode(d->reference);
            vsapi->freeNode(d->distorted);
            return;
        }
        GPU_CHECK(hipSetDevice(d->gpu_id));

        // Initialize streams
        d->streams.resize(d->numStream);
        for (int i = 0; i < d->numStream; i++) {
            auto& stream_data = d->streams[i];
            stream_data.width = d->vi->width;
            stream_data.height = d->vi->height;

            GPU_CHECK(hipStreamCreate(&stream_data.stream));

            // Initialize CVVDP processor
            stream_data.cvvdp_handle = cvvdp_init(stream_data.width, stream_data.height,
                                                   d->display_name.c_str());
            if (!stream_data.cvvdp_handle) {
                vsapi->mapSetError(out, "CVVDP: Failed to initialize processor");
                vsapi->freeNode(d->reference);
                vsapi->freeNode(d->distorted);
                return;
            }

            // Allocate frame buffers
            int frame_size = stream_data.width * stream_data.height;
            GPU_CHECK(hipMalloc(&stream_data.test_d, frame_size * sizeof(float3)));
            GPU_CHECK(hipMalloc(&stream_data.ref_d, frame_size * sizeof(float3)));

            d->available_streams.push(i);
        }

    } catch (const VshipError& e) {
        vsapi->mapSetError(out, e.getErrorMessage().c_str());
        vsapi->freeNode(d->reference);
        vsapi->freeNode(d->distorted);
        return;
    }

    VSFilterDependency deps[] = {
        {d->reference, rpStrictSpatial},
        {d->distorted, rpStrictSpatial}
    };

    // Determine frame callback based on input format
    VSFilterGetFrame getFrame;
    if (d->vi->format.sampleType == stInteger && d->vi->format.bitsPerSample == 8) {
        getFrame = cvvdpGetFrame<InputMemType::UINT8>;
    } else if (d->vi->format.sampleType == stInteger && d->vi->format.bitsPerSample == 16) {
        getFrame = cvvdpGetFrame<InputMemType::UINT16>;
    } else if (d->vi->format.sampleType == stFloat && d->vi->format.bitsPerSample == 32) {
        getFrame = cvvdpGetFrame<InputMemType::FLOAT>;
    } else {
        vsapi->mapSetError(out, "CVVDP: Unsupported input format");
        vsapi->freeNode(d->reference);
        vsapi->freeNode(d->distorted);
        return;
    }

    vsapi->createVideoFilter(out, "CVVDP", d->vi, getFrame, cvvdpFree, fmParallel, deps, 2, d.release(), core);
}

} // namespace cvvdp
