#pragma once

extern "C" {
#include <ffms.h>
#include <libavutil/pixfmt.h>
#include <zimg.h>
}

// #include "util/CLI_Parser.hpp"
#include "ffmpegToZimgFormat.hpp"
#include "unpack.hpp"
#include "../util/preprocessor.hpp"

#include <cstdio>
#include <cstdlib>
#include <optional>

#ifndef ASSERT_WITH_MESSAGE
#define ASSERT_WITH_MESSAGE(condition, message)\
if (!(condition)) {\
    std::fprintf(stderr, "Assertion failed!\nExpression : %s\nFile       : %s\n  Line       : %d\nMessage    : %s\n", #condition, __FILE__, __LINE__, message);\
    std::abort();\
}
#endif

enum class MetricType { SSIMULACRA2, Butteraugli, Unknown };

struct CropRectangle{
    int top = 0;
    int bottom = 0;
    int left = 0;
    int right = 0;
};

int inline align_stride(int stride){
    return ((stride-1)/32+1)*32;
}

static void print_zimg_error(void) {
    char err_msg[1024];
    int err_code = zimg_get_last_error(err_msg, sizeof(err_msg));

    fprintf(stderr, "zimg error %d: %s\n", err_code, err_msg);
}

class GpuWorker {
  private:
    int image_width;
    int image_height;
    int image_stride;

    CropRectangle cropSource;
    CropRectangle cropEncoded;

    int widthEncoded;
    int heightEncoded;
    int strideEncoded;

    MetricType selected_metric;

    ssimu2::SSIMU2ComputingImplementation ssimu2worker;
    butter::ButterComputingImplementation butterworker;

  public:
    GpuWorker(MetricType metric, int width, int height, int stride, int strideEncoded, CropRectangle cropSource, CropRectangle cropEncoded, int Qnorm, float intensity_multiplier)
        : image_width(width), image_height(height), selected_metric(metric), image_stride(stride), strideEncoded(strideEncoded), cropSource(cropSource), cropEncoded(cropEncoded){
        widthEncoded = width - cropSource.left - cropSource.right + cropEncoded.left + cropEncoded.right;
        heightEncoded = height - cropSource.top - cropSource.bottom + cropEncoded.top + cropEncoded.bottom;
        allocate_gpu_memory(Qnorm, intensity_multiplier);
    }
    ~GpuWorker(){
        deallocate_gpu_memory();
    }

    std::tuple<float, float, float>
    compute_metric_score(uint8_t *source_frame, uint8_t *encoded_frame) {
        const int channel_offset_bytes =
            image_stride * image_height;
        const int channel_offset_bytesEncoded =
            strideEncoded * heightEncoded;
        
        //for crop
        int64_t topleftOffset = cropSource.top * image_stride + sizeof(uint16_t)*cropSource.left;
        int64_t topleftOffsetEncoded = cropEncoded.top * strideEncoded + sizeof(uint16_t)*cropEncoded.left;

        const uint8_t *source_channels[3] = {
            source_frame + topleftOffset, source_frame + channel_offset_bytes + topleftOffset,
            source_frame + 2 * channel_offset_bytes + topleftOffset};

        const uint8_t *encoded_channels[3] = {
            encoded_frame + topleftOffsetEncoded, encoded_frame + channel_offset_bytesEncoded + topleftOffsetEncoded,
            encoded_frame + 2 * channel_offset_bytesEncoded + topleftOffsetEncoded};

        if (selected_metric == MetricType::SSIMULACRA2) {
            const double score = ssimu2worker.run<UINT16>(
                source_channels, encoded_channels, image_stride, strideEncoded); 
            float s = static_cast<float>(score);
            return {s, s, s};
        }

        if (selected_metric == MetricType::Butteraugli) {
            return butterworker.run<UINT16>(
                nullptr, 0, source_channels, encoded_channels, image_stride, strideEncoded);
        }

        ASSERT_WITH_MESSAGE(false, "Unknown metric specified for GpuWorker.");
        return {0.0f, 0.0f, 0.0f};
    }

    static uint8_t *allocate_external_rgb_buffer(int bytes) {
        const size_t buffer_size_bytes = bytes * 3;
        uint8_t *buffer_ptr = nullptr;

        const hipError_t result = hipHostMalloc(
            reinterpret_cast<void **>(&buffer_ptr), buffer_size_bytes);

        ASSERT_WITH_MESSAGE(
            result == hipSuccess && buffer_ptr != nullptr,
            "Pinned buffer allocation failed in allocate_external_rgb_buffer");

        return buffer_ptr;
    }

    static void deallocate_external_buffer(uint8_t *buffer_ptr) {
        if (buffer_ptr != nullptr) {
            hipHostFree(buffer_ptr);
        }
    }

  private:
    void allocate_gpu_memory(int Qnorm = 2, float intensity_multiplier = 203) {
        if (selected_metric == MetricType::SSIMULACRA2) {
            try {
                ssimu2worker.init(image_width-cropSource.left-cropSource.right, image_height-cropSource.top-cropSource.bottom);
            } catch (const VshipError& e){
                std::cerr << e.getErrorMessage() << std::endl;
                ASSERT_WITH_MESSAGE(false, "Failed to initialize SSIMULACRA2 Worker");
                return;
            }
        } else if (selected_metric == MetricType::Butteraugli) {
            try {
                butterworker.init(image_width-cropSource.left-cropSource.right, image_height-cropSource.top-cropSource.bottom, Qnorm, intensity_multiplier);
            } catch (const VshipError& e){
                std::cerr << e.getErrorMessage() << std::endl;
                ASSERT_WITH_MESSAGE(false, "Failed to initialize Butteraugli Worker");
                return;
            }
        } else {
            ASSERT_WITH_MESSAGE(false,
                                "Unknown metric during memory allocation.");
        }
    }

    void deallocate_gpu_memory() {
        if (selected_metric == MetricType::SSIMULACRA2) {
            ssimu2worker.destroy();
        } else if (selected_metric == MetricType::Butteraugli) {
            butterworker.destroy();
        }
    }
};

int IndexCallback(int64_t current, int64_t total, void* progp){
    auto prog = (ProgressBar<100, false>*)progp;
    prog->set_values(current/1000000, total/1000000);
    return 0;
}

class FFMSIndexResult {
  private:
    static constexpr int error_message_buffer_size = 1024;

    char error_message_buffer[error_message_buffer_size]{};
    FFMS_ErrorInfo error_info;

  public:
    std::string file_path;
    std::string index_file_path;
    FFMS_Index* index = NULL;
    FFMS_Track* track = NULL;
    int selected_video_track = -1;
    int numFrame = 0;
    int write = 0;

    explicit FFMSIndexResult(const std::string& input_file_path, std::string input_index_file_path, const bool cache_index, const bool debug_out = false, const bool show_progress = true) {
        file_path = input_file_path;
        index_file_path = input_index_file_path;
        FFMS_Init(0, 0);

        error_info.Buffer = error_message_buffer;
        error_info.BufferSize = error_message_buffer_size;
        error_info.ErrorType = FFMS_ERROR_SUCCESS;
        error_info.SubType = FFMS_ERROR_SUCCESS;

        bool from_file_success = false;

        //if cached but no path specified, we default to this path
        if (input_index_file_path == "" && cache_index) input_index_file_path = input_file_path + ".ffindex";

        //if path not empty, we try to read
        if (input_index_file_path != ""){
            index = FFMS_ReadIndex(input_index_file_path.c_str(), &error_info);
            if (index != nullptr && !FFMS_IndexBelongsToFile(index, input_file_path.c_str(), &error_info)) {
                from_file_success = true;
                if (debug_out) std::cout << "Successfully read index from [" << input_index_file_path << "]" << std::endl;
            } else {
                if (debug_out) std::cout << "Index file at [" << input_index_file_path << "] is invalid or does not exist, creating" << std::endl;
            }
        }

        //if failed, we will need to compute ourself
        if (!from_file_success){
            FFMS_Indexer *indexer = FFMS_CreateIndexer(input_file_path.c_str(), &error_info);
            ASSERT_WITH_MESSAGE(indexer != nullptr,
                            ("FFMS2: Failed to create indexer for file [" +
                            input_file_path + "] - " + error_message_buffer)
                                .c_str());

            ProgressBar<100, false>* prog;
            if (show_progress){
                std::cout << "Indexing Progress (MB) " << input_file_path << std::endl;
                prog = new ProgressBar<100, false>(0);
                FFMS_SetProgressCallback(indexer, IndexCallback, prog);
            }

            index = FFMS_DoIndexing2(indexer, FFMS_IEH_ABORT, &error_info);
            ASSERT_WITH_MESSAGE(index != nullptr,
                            ("FFMS2: Failed to index file [" + input_file_path +
                            "] - " + error_message_buffer)
                                .c_str());
            
            if (show_progress){
                prog->refresh(true);
                delete prog;
                std::cout << std::endl; //end of progressbar
            }
        }

        //if we need to cache and it s computed (compiler will optimize automatically)
        //we will write the cache file
        if (!from_file_success && cache_index){
            write = FFMS_WriteIndex(input_index_file_path.c_str(), index, &error_info);
            ASSERT_WITH_MESSAGE(write == 0,
                                ("FFMS2: Failed to write index to file [" + input_index_file_path +
                                "] - " + error_message_buffer)
                                    .c_str());
            if (debug_out) std::cout << "Successfully wrote index to [" << input_index_file_path << "]" << std::endl;
        }

        selected_video_track =
            FFMS_GetFirstTrackOfType(index, FFMS_TYPE_VIDEO, &error_info);
        ASSERT_WITH_MESSAGE(selected_video_track >= 0,
                            ("FFMS2: No video track found in file [" +
                             input_file_path + "] - " + error_message_buffer)
                                .c_str());

        track =
            FFMS_GetTrackFromIndex(index, selected_video_track);
        ASSERT_WITH_MESSAGE(track != NULL,
                            ("FFMS2: Failed to get FFMS_Track in file [" +
                             input_file_path + "]").c_str());

        numFrame = FFMS_GetNumFrames(track);
        ASSERT_WITH_MESSAGE(numFrame != 0, ("FFMS2: Got 0 frames in [" +
                             input_file_path + "]").c_str());

    }

    std::set<int> getKeyFrameIndices(){
        std::set<int> out;
        for (int i = 0; i < numFrame; i++){
            const FFMS_FrameInfo* frameinfo = FFMS_GetFrameInfo(track, i);
            ASSERT_WITH_MESSAGE(frameinfo != NULL, ("Failed to retrieve KeyFrame information in the indexer for file [" + file_path + "]").c_str());
            if (frameinfo->KeyFrame != 0){
                out.insert(i);
            }
        }
        return std::move(out);
    }

    ~FFMSIndexResult() {
        if (index != nullptr) {
            FFMS_DestroyIndex(index);
            index = nullptr;
        }

        selected_video_track = -1;
    }
};

class FFMSFrameReader {
  private:
    int num_decoder_threads = std::thread::hardware_concurrency();
    int seek_mode = FFMS_SEEK_NORMAL;

  public:
    const FFMS_VideoProperties *video_properties = nullptr;
    const FFMS_Frame *current_frame = nullptr;

    AVPixelFormat video_pixel_format = AV_PIX_FMT_NONE;
    int frame_width = 0;
    int frame_height = 0;
    int total_frame_count = 0;

    FFMS_VideoSource *video_source = nullptr;

    FFMS_ErrorInfo error_info;
    char error_message_buffer[1024] = {};

    explicit FFMSFrameReader(const std::string &file_path, FFMS_Index *index,
                             int video_track_index) {
        initialize_error_info();
        create_video_source(file_path, index, video_track_index);
        load_video_properties();
        load_first_frame();
        configure_pixel_format(file_path);
    }

    ~FFMSFrameReader() {
        if (video_source != nullptr) {
            FFMS_DestroyVideoSource(video_source);
            video_source = nullptr;
        }
    }

    void fetch_frame(int frame_index) {
        current_frame = FFMS_GetFrame(video_source, frame_index, &error_info);
        ASSERT_WITH_MESSAGE(current_frame != nullptr,
                            ("FFMS2: Failed to fetch frame [" +
                             std::to_string(frame_index) + "] - " +
                             error_message_buffer)
                                .c_str());
    }

  private:
    void initialize_error_info() {
        error_info.Buffer = error_message_buffer;
        error_info.BufferSize = sizeof(error_message_buffer);
        error_info.ErrorType = FFMS_ERROR_SUCCESS;
        error_info.SubType = FFMS_ERROR_SUCCESS;
    }

    void create_video_source(const std::string &file_path, FFMS_Index *index,
                             int track_index) {

        video_source =
            FFMS_CreateVideoSource(file_path.c_str(), track_index, index,
                                   num_decoder_threads, seek_mode, &error_info);

        ASSERT_WITH_MESSAGE(video_source != nullptr,
                            ("FFMS2: Failed to create video source for [" +
                             file_path + "] - " + error_message_buffer)
                                .c_str());
    }

    void load_video_properties() {
        video_properties = FFMS_GetVideoProperties(video_source);
        ASSERT_WITH_MESSAGE(video_properties != nullptr,
                            "FFMS2: FFMS_GetVideoProperties returned null.");

        total_frame_count = video_properties->NumFrames;
        ASSERT_WITH_MESSAGE(total_frame_count > 0,
                            "FFMS2: No frames found in video stream.");
    }

    void load_first_frame() {
        current_frame = FFMS_GetFrame(video_source, 0, &error_info);
        ASSERT_WITH_MESSAGE(current_frame != nullptr,
                            ("FFMS2: Failed to fetch first frame - " +
                             std::string(error_message_buffer))
                                .c_str());

        video_pixel_format =
            static_cast<AVPixelFormat>(current_frame->EncodedPixelFormat);
        frame_width = current_frame->EncodedWidth;
        frame_height = current_frame->EncodedHeight;
    }

    void configure_pixel_format(const std::string &file_path) {
        int output_formats[] = {static_cast<int>(video_pixel_format), -1};

        int result = FFMS_SetOutputFormatV2(video_source, output_formats,
                                            frame_width, frame_height,
                                            FFMS_RESIZER_BICUBIC, &error_info);

        ASSERT_WITH_MESSAGE(result == 0,
                            ("FFMS2: Failed to set output format for [" +
                             file_path + "] - " + error_message_buffer)
                                .c_str());
    }
};

class ZimgProcessor {
  public:
    zimg_filter_graph *graph = nullptr;
    zimg_image_format src_format = {};
    zimg_image_format dst_format = {};
    zimg_image_buffer_const src_buffer = {ZIMG_API_VERSION};
    zimg_image_buffer dst_buffer = {ZIMG_API_VERSION};
    void *tmp_buffer = nullptr;
    size_t tmp_size = 0;

    AVPixelFormat src_pixfmt;
    uint8_t* unpack_buffer[3] = {nullptr, nullptr, nullptr};
    int unpack_stride[3] = {0, 0, 0};

    ZimgProcessor(const FFMS_Frame *ref_frame, int target_width,
                  int target_height) {
        initialize_formats(ref_frame, target_width, target_height);
        build_unpack();
        build_graph();
        allocate_tmp_buffer();
    }

    ~ZimgProcessor() {
        if (unpack_buffer[0]) {
            free(unpack_buffer[0]);
            unpack_buffer[0] = NULL;
        }
        if (graph) zimg_filter_graph_free(graph);
        if (tmp_buffer) {
            free(tmp_buffer);
            tmp_buffer = NULL;
        }
    }

    void process(const FFMS_Frame *src, uint8_t *dst, int stride,
                 int plane_size) {

        if (unpack_buffer[0] != NULL){
            unpack(src);
        }

        for (int p = 0; p < 3; ++p) {
            dst_buffer.plane[p].data = dst + p * plane_size;
            dst_buffer.plane[p].stride = stride;
            dst_buffer.plane[p].mask = ZIMG_BUFFER_MAX;

            src_buffer.plane[p].mask = ZIMG_BUFFER_MAX;
            if (unpack_buffer[0] == NULL){
                src_buffer.plane[p].data = src->Data[p];
                src_buffer.plane[p].stride = src->Linesize[p];
            } else {
                //we have unpacked so we choose the data in unpack_buffer
                src_buffer.plane[p].data = unpack_buffer[p];
                src_buffer.plane[p].stride = unpack_stride[p];
            }
        }

        int ret = zimg_filter_graph_process(graph, &src_buffer, &dst_buffer,
                                            tmp_buffer, 0, 0, 0, 0);

        ASSERT_WITH_MESSAGE(ret == 0, "zimg: Filter graph processing failed.");
    }

  private:
    void initialize_formats(const FFMS_Frame *frame, int width, int height) {
        int result = ffmpegToZimgFormat(src_format, frame);
        ASSERT_WITH_MESSAGE(
            result == 0, "zimg: Failed to convert source format from FFmpeg.");

        zimg_image_format_default(&dst_format, ZIMG_API_VERSION);
        dst_format.width = width;
        dst_format.height = height;
        dst_format.pixel_type = ZIMG_PIXEL_WORD;
        dst_format.subsample_w = 0;
        dst_format.subsample_h = 0;
        dst_format.color_family = ZIMG_COLOR_RGB;
        dst_format.matrix_coefficients = ZIMG_MATRIX_RGB;
        dst_format.transfer_characteristics = ZIMG_TRANSFER_BT709;
        dst_format.color_primaries = ZIMG_PRIMARIES_BT709;
        dst_format.depth = 16;
        dst_format.pixel_range = ZIMG_RANGE_FULL;

        src_pixfmt = (AVPixelFormat)frame->EncodedPixelFormat;
    }

    void build_unpack(){
        int depth;
        //list supported formats (they are repeated in ffmpegToZimgFormat to handle them correctly)
        switch (src_pixfmt){

            //bitdepth 8, 422
            case AV_PIX_FMT_YUYV422:
            depth = 8;
            unpack_stride[0] = src_format.width * depth; //in bits
            unpack_stride[1] = (src_format.width/2) * depth; //in bits
            unpack_stride[0] = ((unpack_stride[0]-1)/256+1)*256; //align to 32 bytes for zimg
            unpack_stride[1] = ((unpack_stride[1]-1)/256+1)*256; //align to 32 bytes for zimg
            unpack_stride[0] >>= 3; //in bytes
            unpack_stride[1] >>= 3; //in bytes

            unpack_stride[2] = unpack_stride[1]; //same for both chroma

            unpack_buffer[0] = (uint8_t*)aligned_alloc(32, sizeof(uint8_t)*(unpack_stride[0]+unpack_stride[1]+unpack_stride[2])*src_format.height);
            unpack_buffer[1] = unpack_buffer[0] + unpack_stride[0]*src_format.height;
            unpack_buffer[2] = unpack_buffer[1] + unpack_stride[1]*src_format.height;
            break;

            //bitdepth 8, 444
            case AV_PIX_FMT_RGB24:
            case AV_PIX_FMT_ARGB:
            case AV_PIX_FMT_RGBA:
            case AV_PIX_FMT_ABGR:
            case AV_PIX_FMT_BGRA:
            depth = 8;
            unpack_stride[0] = src_format.width * depth; //in bits
            unpack_stride[0] = ((unpack_stride[0]-1)/256+1)*256; //align to 32 bytes for zimg
            unpack_stride[0] >>= 3; //in bytes
            unpack_stride[1] = unpack_stride[0]; //444, same stride everywhere
            unpack_stride[2] = unpack_stride[0];

            unpack_buffer[0] = (uint8_t*)aligned_alloc(32, unpack_stride[0]*src_format.height*3);
            unpack_buffer[1] = unpack_buffer[0] + unpack_stride[0]*src_format.height;
            unpack_buffer[2] = unpack_buffer[1] + unpack_stride[0]*src_format.height;
            break;

            //depth 16, 444
            case AV_PIX_FMT_RGB48LE:
            case AV_PIX_FMT_RGBA64LE:
            depth = 16;
            unpack_stride[0] = src_format.width * depth; //in bits
            unpack_stride[0] = ((unpack_stride[0]-1)/256+1)*256; //align to 32 bytes for zimg
            unpack_stride[0] >>= 3; //in bytes
            unpack_stride[1] = unpack_stride[0]; //444, same stride everywhere
            unpack_stride[2] = unpack_stride[0];

            unpack_buffer[0] = (uint8_t*)aligned_alloc(32, unpack_stride[0]*src_format.height*3);
            unpack_buffer[1] = unpack_buffer[0] + unpack_stride[0]*src_format.height;
            unpack_buffer[2] = unpack_buffer[1] + unpack_stride[0]*src_format.height;
            break;

            default:
                return; //no unpack
        }
    }

    void unpack(const FFMS_Frame *src){
        switch (src_pixfmt){
            case AV_PIX_FMT_YUYV422:
                for (int j = 0; j < src_format.height; j++){
                    for (int i = 0; i < src_format.width/2; i++){
                        //Y
                        unpack_buffer[0][j*unpack_stride[0]+2*i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i];
                        //U
                        unpack_buffer[1][j*unpack_stride[1]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+1];
                        //Y
                        unpack_buffer[0][j*unpack_stride[0]+2*i+1] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+2];
                        //V
                        unpack_buffer[2][j*unpack_stride[2]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+3];
                    }
                }
            break;
            case AV_PIX_FMT_RGB48LE:
                for (int j = 0; j < src_format.height; j++){
                    for (int i = 0; i < src_format.width; i++){
                        ((uint16_t*)(unpack_buffer[0]+j*unpack_stride[0]))[i] = ((uint16_t*)(src->Data[0]+j*src->Linesize[0]))[3*i];
                        ((uint16_t*)(unpack_buffer[1]+j*unpack_stride[1]))[i] = ((uint16_t*)(src->Data[0]+j*src->Linesize[0]))[3*i+1];
                        ((uint16_t*)(unpack_buffer[2]+j*unpack_stride[2]))[i] = ((uint16_t*)(src->Data[0]+j*src->Linesize[0]))[3*i+2];
                    }
                }
            break;
            case AV_PIX_FMT_RGBA64LE:
                for (int j = 0; j < src_format.height; j++){
                    for (int i = 0; i < src_format.width; i++){
                        ((uint16_t*)(unpack_buffer[0]+j*unpack_stride[0]))[i] = ((uint16_t*)(src->Data[0]+j*src->Linesize[0]))[4*i];
                        ((uint16_t*)(unpack_buffer[1]+j*unpack_stride[1]))[i] = ((uint16_t*)(src->Data[0]+j*src->Linesize[0]))[4*i+1];
                        ((uint16_t*)(unpack_buffer[2]+j*unpack_stride[2]))[i] = ((uint16_t*)(src->Data[0]+j*src->Linesize[0]))[4*i+2];
                    }
                }
            break;

            case AV_PIX_FMT_RGB24:
                for (int j = 0; j < src_format.height; j++){
                    for (int i = 0; i < src_format.width; i++){
                        unpack_buffer[0][j*unpack_stride[0]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+3*i];
                        unpack_buffer[1][j*unpack_stride[1]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+3*i+1];
                        unpack_buffer[2][j*unpack_stride[2]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+3*i+2];
                    }
                }
            break;
            case AV_PIX_FMT_RGBA:
                for (int j = 0; j < src_format.height; j++){
                    for (int i = 0; i < src_format.width; i++){
                        unpack_buffer[0][j*unpack_stride[0]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i];
                        unpack_buffer[1][j*unpack_stride[1]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+1];
                        unpack_buffer[2][j*unpack_stride[2]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+2];
                    }
                }
            break;
            case AV_PIX_FMT_ARGB:
                for (int j = 0; j < src_format.height; j++){
                    for (int i = 0; i < src_format.width; i++){
                        unpack_buffer[0][j*unpack_stride[0]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+1];
                        unpack_buffer[1][j*unpack_stride[1]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+2];
                        unpack_buffer[2][j*unpack_stride[2]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+3];
                    }
                }
            break;
            case AV_PIX_FMT_ABGR:
                for (int j = 0; j < src_format.height; j++){
                    for (int i = 0; i < src_format.width; i++){
                        unpack_buffer[0][j*unpack_stride[0]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+3];
                        unpack_buffer[1][j*unpack_stride[1]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+2];
                        unpack_buffer[2][j*unpack_stride[2]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+1];
                    }
                }
            break;
            case AV_PIX_FMT_BGRA:
                for (int j = 0; j < src_format.height; j++){
                    for (int i = 0; i < src_format.width; i++){
                        unpack_buffer[0][j*unpack_stride[0]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+2];
                        unpack_buffer[1][j*unpack_stride[1]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+1];
                        unpack_buffer[2][j*unpack_stride[2]+i] = ((uint8_t*)(src->Data[0]))[j*src->Linesize[0]+4*i+0];
                    }
                }
            break;

            default:
                //this should not happen and result of forgiveness of the dev, better to place this in case
                std::cout << "Error: Trying to unpack an unsupported format " << src_pixfmt << " line " << __LINE__ << " of " << __FILE__ << std::endl;
                return;
        }
    }

    void build_graph() {
        zimg_graph_builder_params params;
        zimg_graph_builder_params_default(&params, ZIMG_API_VERSION);

        graph = zimg_filter_graph_build(&src_format, &dst_format, &params);
        ASSERT_WITH_MESSAGE(graph != nullptr,
                            "zimg: Failed to build filter graph.");
    }

    void allocate_tmp_buffer() {
        int result = zimg_filter_graph_get_tmp_size(graph, &tmp_size);
        ASSERT_WITH_MESSAGE(result == 0,
                            "zimg: Failed to get temporary buffer size.");

        tmp_buffer = aligned_alloc(32, tmp_size);
        ASSERT_WITH_MESSAGE(tmp_buffer != nullptr,
                            "zimg: Failed to allocate temporary buffer.");
    }
};

class VideoManager {
  public:
    int plane_size_bytes = 0;
    int plane_stride_bytes = 0;

    std::unique_ptr<FFMSFrameReader> reader;
    std::unique_ptr<ZimgProcessor> processor;

    VideoManager(const std::string &file_path, FFMS_Index *index,
                 int video_track_index, int resize_width = -1,
                 int resize_height = -1) {

        reader = std::make_unique<FFMSFrameReader>(file_path, index,
                                                   video_track_index);

        if (resize_width < 0)
            resize_width = reader->frame_width;
        if (resize_height < 0)
            resize_height = reader->frame_height;

        plane_stride_bytes = align_stride(resize_width * sizeof(uint16_t)); //this needs to be divisible by 32
        plane_size_bytes = plane_stride_bytes * resize_height;

        processor = std::make_unique<ZimgProcessor>(
            reader->current_frame, resize_width, resize_height);

        ASSERT_WITH_MESSAGE(
            processor != nullptr,
            "VideoManager: Failed to initialize ZimgProcessor.");
    }

    void fetch_frame_into_buffer(int frame_index, uint8_t *output_buffer) {
        reader->fetch_frame(frame_index);
        processor->process(reader->current_frame, output_buffer,
                           plane_stride_bytes, plane_size_bytes);
    }
};

struct CommandLineOptions {
    std::string source_file;
    std::string encoded_file;
    std::string json_output_file;
    std::string source_index;
    std::string encoded_index;

    int start_frame = 0;
    int end_frame = -1;
    int every_nth_frame = 1;
    int encoded_offset = 0;

    CropRectangle cropSource;
    CropRectangle cropEncoded;

    std::vector<int> source_indices_list;
    std::vector<int> encoded_indices_list;

    //for butteraugli
    int Qnorm = 2;
    int intensity_target_nits = 203;

    int gpu_id = 0;
    int gpu_threads = 3;
    int cpu_threads = 1;

    bool list_gpus = false;
    bool version = false;
    MetricType metric = MetricType::SSIMULACRA2; //SSIMULACRA2 by default

    bool NoAssertExit = false; //please exit without creating an assertion failed scary error

    bool live_index_score_output = false;

    bool cache_index = false;
};

std::vector<int> splitPerToken(std::string inp){
    std::vector<int> out;
    std::string temp;

    for (const char c: inp){
        switch (c){
            case ',':
                out.push_back(std::stoi(temp));
                temp.clear();
                break;
            case ' ':
                continue;
            default:
                temp.push_back(c);
        }
    }
    if (!temp.empty()) out.push_back(std::stoi(temp));

    return out;
}

MetricType parse_metric_name(const std::string &name) {
    std::string lowered;
    lowered.resize(name.size());
    for (unsigned int i = 0; i < name.size(); i++){
        lowered[i] = std::tolower(name[i]);
    }
    if (lowered == "ssimulacra2" || lowered == "ssimu2") return MetricType::SSIMULACRA2;
    if (lowered == "butteraugli" || lowered == "butter") return MetricType::Butteraugli;
    return MetricType::Unknown;
}

CommandLineOptions parse_command_line_arguments(int argc, char **argv) {
    std::vector<std::string> args(argc);
    for (int i = 0; i < argc; i++){
        args[i] = argv[i];
    }

    helper::ArgParser parser;

    std::string metric_name;
    std::string source_indices_str;
    std::string encoded_indices_str;

    CommandLineOptions opts;

    parser.add_flag({"--source", "-s"}, &opts.source_file, "Reference video to compare to", true);
    parser.add_flag({"--encoded", "-e"}, &opts.encoded_file, "Distorted encode of the source", true);
    parser.add_flag({"--metric", "-m"}, &metric_name, "Which metric to use [SSIMULACRA2, Butteraugli]");
    parser.add_flag({"--json"}, &opts.json_output_file, "Outputs metric results to a json file");
    parser.add_flag({"--live-score-output"}, &opts.live_index_score_output, "replace stdout output with index-score lines");
    parser.add_flag({"--source-index"}, &opts.source_index, "FFMS2 index file for source video");
    parser.add_flag({"--encoded-index"}, &opts.encoded_index, "FFMS2 index file for encoded video");
    parser.add_flag({"--cache-index"}, &opts.cache_index, "Write index files to disk and reuse if available");

    parser.add_flag({"--start"}, &opts.start_frame, "Starting frame of source");
    parser.add_flag({"--end"}, &opts.end_frame, "Ending frame of source");
    parser.add_flag({"--encoded-offset"}, &opts.encoded_offset, "Frame offset of encoded video to source");
    parser.add_flag({"--every"}, &opts.every_nth_frame, "Frame sampling rate");
    parser.add_flag({"--source-indices"}, &source_indices_str, "List of source indices subjective to --start, --end, --every and --encoded-offset. If --encoded-indices isnt specified, this will be applied to encoded-indices too. Format is integers separated by comma");
    parser.add_flag({"--encoded-indices"}, &encoded_indices_str, "List of encoded indices subjective to --start, --end, --every and --encoded-offset. Format is integers separated by comma");
    
    parser.add_flag({"--cropTopSource"}, &opts.cropSource.top, "Allows to crop source");
    parser.add_flag({"--cropBottomSource"}, &opts.cropSource.bottom, "Allows to crop source");
    parser.add_flag({"--cropLeftSource"}, &opts.cropSource.left, "Allows to crop source");
    parser.add_flag({"--cropRightSource"}, &opts.cropSource.right, "Allows to crop source");
    parser.add_flag({"--cropTopEncoded"}, &opts.cropEncoded.top, "Allows to crop Encoded");
    parser.add_flag({"--cropBottomEncoded"}, &opts.cropEncoded.bottom, "Allows to crop Encoded");
    parser.add_flag({"--cropLeftEncoded"}, &opts.cropEncoded.left, "Allows to crop Encoded");
    parser.add_flag({"--cropRightEncoded"}, &opts.cropEncoded.right, "Allows to crop Encoded");

    parser.add_flag({"--intensity-target"}, &opts.intensity_target_nits, "Target nits for Butteraugli");
    parser.add_flag({"--qnorm"}, &opts.Qnorm, "Optional Norm to compute (default to 2)");
    parser.add_flag({"--threads", "-t"}, &opts.cpu_threads, "Number of Decoder process, recommended is 2");
    parser.add_flag({"--gpu-threads", "-g"}, &opts.gpu_threads, "GPU thread count, recommended is 3");
    parser.add_flag({"--gpu-id"}, &opts.gpu_id, "GPU index");
    parser.add_flag({"--list-gpu"}, &opts.list_gpus, "List available GPUs");
    parser.add_flag({"--version"}, &opts.version, "Print FFVship version");

    if (parser.parse_cli_args(args) != 0) { //the parser will have already printed an error
        opts.NoAssertExit = true;
        return opts;
    }

    if (opts.list_gpus || opts.version) return opts;

    try {
        opts.source_indices_list = splitPerToken(source_indices_str);
    } catch (...){
        std::cerr << "Invalid integer found in --source-indices" << std::endl;
        opts.NoAssertExit = true;
        return opts;
    }
    try {
        opts.encoded_indices_list = splitPerToken(encoded_indices_str);
    } catch (...){
        std::cerr << "Invalid integer found in --encoded-indices" << std::endl;
        opts.NoAssertExit = true;
        return opts;
    }

    if (opts.source_file.empty()){
        std::cerr << "Source file is not specified" << std::endl;
        opts.NoAssertExit = true;
    }

    if (opts.encoded_file.empty()){
        std::cerr << "Encoded file is not specified" << std::endl;
        opts.NoAssertExit = true;
    }

    if (!metric_name.empty()) {
        opts.metric = parse_metric_name(metric_name);
        if (opts.metric == MetricType::Unknown){
            std::cerr << "Unknown metric type. Expected 'SSIMULACRA2' or 'Butteraugli'." << std::endl;
            opts.NoAssertExit = true;
        }
    }

    return opts;
}
