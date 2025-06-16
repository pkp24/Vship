#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

#include "util/preprocessor.hpp"
#include "util/gpuhelper.hpp"
#include "util/VshipExceptions.hpp"
#include "util/threadsafeset.hpp"
#include "util/CLI_Parser.hpp"

#include "ssimu2/main.hpp"
#include "butter/main.hpp"

extern "C"{
#include <zimg.h>
#include <ffms.h>
#include<libavutil/pixfmt.h>
}

#include "util/ffmpegToZimgFormat.hpp"
#include "gpuColorToLinear/vshipColor.hpp"

void print_zimg_error(void)
{
	char err_msg[1024];
	int err_code = zimg_get_last_error(err_msg, sizeof(err_msg));

	fprintf(stderr, "zimg error %d: %s\n", err_code, err_msg);
}

class FFmpegVideoManager{
public:
    int ret = 0;
    //zimg data
    zimg_filter_graph *zimg_graph = NULL;
	zimg_image_buffer_const zimg_src_buf = { ZIMG_API_VERSION };
	zimg_image_buffer zimg_dst_buf = { ZIMG_API_VERSION };
	zimg_image_format zimg_src_format;
	zimg_image_format zimg_dst_format;
	size_t zimg_tmp_size = 0;
	void *zimg_tmp = NULL;

    char errmsg[1024];
    FFMS_ErrorInfo errinfo;
    FFMS_VideoSource* ffms_source = NULL;
    const FFMS_VideoProperties* ffms_props = NULL;
    int numframe = 0;
    int width = 0;
    int height = 0;
    AVPixelFormat pix_fmt = AV_PIX_FMT_NONE;
    
    const FFMS_Frame *frame = NULL;
    uint8_t* outputRGB = NULL;
    uint8_t* RGBptrHelper[3] = {NULL, NULL, NULL};
    int RGBstride[3] = {0, 0, 0};
    int error = 0;
    FFmpegVideoManager(std::string file, FFMS_Index* index, int trackno, int resize_width = -1, int resize_height = -1){

        errinfo.Buffer      = errmsg;
        errinfo.BufferSize  = sizeof(errmsg);
        errinfo.ErrorType   = FFMS_ERROR_SUCCESS;
        errinfo.SubType     = FFMS_ERROR_SUCCESS;

        //ffms2 part
        ffms_source = FFMS_CreateVideoSource(file.c_str(), trackno, index, 1, FFMS_SEEK_NORMAL, &errinfo);
        if (ffms_source == NULL) {
            std::cout << "FFMS, failed to create video source of " << file << " with error " << errmsg << std::endl;
            error = 1;
            return;
        }
        ffms_props = FFMS_GetVideoProperties(ffms_source);
        numframe = ffms_props->NumFrames;
        if (numframe == 0){
            std::cout << "Got an empty video..." << std::endl;
            error = 2;
            return;
        }
        frame = FFMS_GetFrame(ffms_source, 0, &errinfo);
        if (frame == NULL){
            std::cout << "Error retrieving frame 0" << " with error " << errmsg << std::endl;
            error = 6;
            return;
        }
        pix_fmt = (AVPixelFormat)frame->EncodedPixelFormat;
        width = frame->EncodedWidth;
        height = frame->EncodedHeight;

        int pixfmts[2];
        pixfmts[0] = (int)pix_fmt;
        pixfmts[1] = -1;

        if (FFMS_SetOutputFormatV2(ffms_source, pixfmts, width, height,
            FFMS_RESIZER_FAST_BILINEAR, &errinfo)) {
            std::cout << "Failed to set the output format in FFMS for file : " << file << " with error " << errmsg << std::endl;
            error = 3;
            return;
        }

        if (resize_height < 0) resize_height = height;
        if (resize_width < 0) resize_width = width;

        hipHostMalloc((void**)&outputRGB, resize_width*resize_height*sizeof(uint16_t)*3); //allocate pinned memory for end buffer for faster gpu send
        if (!outputRGB){
            std::cout << "Failed to allocate Pinned RAM for RGB output for file : " << file << " of size " << resize_width*resize_height*sizeof(uint16_t)*3 << std::endl;
            error = 10;
            return;
        }
        RGBptrHelper[0] = outputRGB;
        RGBptrHelper[1] = outputRGB+resize_width*resize_height*sizeof(uint16_t);
        RGBptrHelper[2] = outputRGB+2*resize_width*resize_height*sizeof(uint16_t);

        RGBstride[0] = resize_width*sizeof(uint16_t);
        RGBstride[1] = resize_width*sizeof(uint16_t);
        RGBstride[2] = resize_width*sizeof(uint16_t);
        
        //zimg init
        if (ffmpegToZimgFormat(zimg_src_format, frame) != 0){
            std::cout << "Failed to convert ffmpeg input format to zimg processing format" << std::endl;
            error = 10;
            return;
        }

        //destination format
        zimg_image_format_default(&zimg_dst_format, ZIMG_API_VERSION);
        zimg_dst_format.width = resize_width;
        zimg_dst_format.height = resize_height;
        zimg_dst_format.pixel_type = ZIMG_PIXEL_WORD;

        zimg_dst_format.subsample_w = 0;
        zimg_dst_format.subsample_h = 0;

        zimg_dst_format.color_family = ZIMG_COLOR_RGB;
        zimg_dst_format.matrix_coefficients = ZIMG_MATRIX_RGB;
        zimg_dst_format.transfer_characteristics = ZIMG_TRANSFER_BT709;
        zimg_dst_format.color_primaries = ZIMG_PRIMARIES_BT709;
        zimg_dst_format.depth = 16;
        zimg_dst_format.pixel_range = ZIMG_RANGE_FULL;

        zimg_graph_builder_params zimgparam;

        zimg_graph_builder_params_default(&zimgparam, ZIMG_API_VERSION);

        zimg_graph = zimg_filter_graph_build(&zimg_src_format, &zimg_dst_format, &zimgparam);
        if (!zimg_graph){
            std::cout << "Failed to generate zimg conversion graph for file : " << file << std::endl;
            print_zimg_error();
            error = 11;
            return;
        }

        zimg_dst_buf = { ZIMG_API_VERSION };
        for (int p = 0; p < 3; p++){
            zimg_dst_buf.plane[p].data = RGBptrHelper[p];
            zimg_dst_buf.plane[p].stride = RGBstride[p];
            zimg_dst_buf.plane[p].mask = ZIMG_BUFFER_MAX;
        }

        if ((ret = zimg_filter_graph_get_tmp_size(zimg_graph, &zimg_tmp_size))) {
            print_zimg_error();
            error = 12;
            return;
        }
        zimg_tmp = aligned_alloc(32, zimg_tmp_size);
    }
    int getFrame(int i, bool convert = true){ //access result with object.frame. return 0 if success, -2 is EndOfVideo

        //ffms2 get frame
        frame = FFMS_GetFrame(ffms_source, i, &errinfo);
        if (frame == NULL){
            std::cout << "Error retrieving frame " << i << " with error " << errmsg << std::endl;
            error = 6;
            return -1;
        }

        if (convert){
            zimg_src_buf = { ZIMG_API_VERSION };
            for (int p = 0; p < 3; p++){
                zimg_src_buf.plane[p].data = frame->Data[p];
                zimg_src_buf.plane[p].stride = frame->Linesize[p];
                zimg_src_buf.plane[p].mask = ZIMG_BUFFER_MAX;
            }
            if ((ret = zimg_filter_graph_process(zimg_graph, &zimg_src_buf, &zimg_dst_buf, zimg_tmp, 0, 0, 0, 0))) {
                print_zimg_error();
                return -1;
            }
        }

        return 0;
    }
    ~FFmpegVideoManager(){
        if (ffms_source != NULL) FFMS_DestroyVideoSource(ffms_source);
        if (outputRGB != NULL) hipHostFree(outputRGB);
        if (zimg_graph != NULL) zimg_filter_graph_free(zimg_graph);
        if (zimg_tmp != NULL) free(zimg_tmp);

        zimg_tmp = NULL;
        zimg_graph = NULL;
        outputRGB = NULL;
        ffms_source = NULL;
    }
};

enum METRICS{SSIMULACRA2, Butteraugli};

void threadwork(std::string file1, FFMS_Index* index1, int trackno1, std::string file2, FFMS_Index* index2, int trackno2, int start, int end, int every, int threadid, int threadnum, METRICS metric, threadSet* gpustreams, int maxshared, float intensity_multiplier, float* gaussiankernel_dssimu2, butter::GaussianHandle* gaussianhandlebutter, void** pinnedmempool, hipStream_t* streams_d, std::vector<float>* output){ //for butteraugli, return 2norm, 3norm, Infnorm, 2norm, ...
    
    FFmpegVideoManager v1(file1, index1, trackno1);
    if (v1.error){
        std::cout << "Thread " << threadid << " Failed to open file " << file1 << std::endl;
        return;
    }
    FFmpegVideoManager v2(file2, index2, trackno2, v1.width, v1.height);
    if (v2.error){
        std::cout << "Thread " << threadid << " Failed to open file " << file2 << std::endl;
        return;
    }
    //if (v1.width != v2.width || v1.height != v2.height){
    //    std::cout << "the 2 videos do not have the same sizes (" << v1.width << "x" << v1.height << " vs " << v2.width << "x" << v2.height <<")" << std::endl;
    //    return;
    //}
    
    if (end < 0) end = v1.numframe;
    if (end < start) end = start;
    if (start < 0) start = 0;
    
    if (std::min(v1.numframe, end) != std::min(v2.numframe, end)){
        std::cout << "both videos do not have the same amount of frame" << std::endl;
        return;
    }

    int pinnedsize = 0;
    switch (metric){
        case SSIMULACRA2:
        pinnedsize = ssimu2::allocsizeScore(v1.width, v1.height, maxshared)*sizeof(float3);
        break;
        case Butteraugli:
        pinnedsize = butter::allocsizeScore(v1.width, v1.height)*sizeof(float);
        break;
    }
    
    int threadbegin = (end-start)*threadid/threadnum-1;
    threadbegin /= every;
    threadbegin += 1;
    if (threadid == 0) threadbegin = 0; //fix negative division not being what we expect
    threadbegin *= every;
    threadbegin += start;
    for (int i = threadbegin ; i < (end-start)*(threadid+1)/threadnum + start; i += every){
        v1.getFrame(i);
        v2.getFrame(i);

        const uint8_t* srcp1[3] = {v1.RGBptrHelper[0], v1.RGBptrHelper[1], v1.RGBptrHelper[2]};
        const uint8_t* srcp2[3] = {v2.RGBptrHelper[0], v2.RGBptrHelper[1], v2.RGBptrHelper[2]};

        int streamid = gpustreams->pop();
        hipStream_t stream = streams_d[streamid];

        if (pinnedmempool[streamid] == NULL){
            //first usage of this stream, let's allocate the pinned mem
            hipError_t erralloc = hipHostMalloc(pinnedmempool+streamid, pinnedsize);
            if (erralloc != hipSuccess){
                std::cout << "Thread " << threadid << " Failed to allocate pinned memory for back buffer" << std::endl;
                return;
            }
        }
        void* pinnedmem = pinnedmempool[streamid];

        try{
            switch (metric){
                case Butteraugli:
                {
                const std::tuple<float, float, float> scorebutter = butter::butterprocess<UINT16>(NULL, 0, srcp1, srcp2, (float*)pinnedmem, *gaussianhandlebutter, v1.RGBstride[0], v1.width, v1.height, intensity_multiplier, maxshared, stream);
                output->push_back(std::get<0>(scorebutter));
                output->push_back(std::get<1>(scorebutter));
                output->push_back(std::get<2>(scorebutter));
                break;
                }
                case SSIMULACRA2:
                {
                const double scoressimu2 = ssimu2::ssimu2process<UINT16>(srcp1, srcp2, (float3*)pinnedmem, v1.RGBstride[0], v1.width, v1.height, gaussiankernel_dssimu2, maxshared, stream);
                output->push_back(scoressimu2);
                break;
                }
            }
        } catch (const VshipError& e){
            std::cout << "Thread " << i << " Got an Vship Exception : " << e.getErrorMessage() << std::endl;
            return;
        }
        gpustreams->insert(streamid);
    }
    return;
}

int main(int argc, char** argv){
    std::vector<std::string> args(argc-1);
    for (int i = 1; i < argc; i++){
        args[i-1] = argv[i];
    } 

    std::string file1, file2, metric_name = "", jsonout = "";
    int start = 0, end = -1, every = 1, gpuid = 0, gputhreads = 6;
    int threads = std::thread::hardware_concurrency();
    bool list_gpu = false;

    int intensity_multiplier = 203;

    helper::ArgParser parser("FFVship");

    // string flags
    parser.add_flag({"--source", "-s"}, &file1,"Reference video to compare to", true);
    parser.add_flag({"--encoded", "-e"}, &file2, "Distorted encode of the source", true);
    parser.add_flag({"--metric", "-m"}, &metric_name, "Which metric to use [SSIMULACRA2, Butteraugli]");
    parser.add_flag({"--json"}, &jsonout, "Outputs metric results to a json file");
    
    // int flags
    parser.add_flag({"--start"}, &start, "Starting frame of the both videos");
    parser.add_flag({"--end"}, &end, "Ending frame of both videos ");
    parser.add_flag({"--every"}, &every, "Only compute the metric score for every X frame");
    parser.add_flag({"--intensity-target"}, &intensity_multiplier, "intensity target to compute butteraugli at. Measured in nits");
    parser.add_flag({"--gpu-id"}, &gpuid, "Which gpu to do metric calculations on");
    parser.add_flag({"--gpu-threads", "-g"}, &gputhreads, "How many number of gpu threads to spawn");
    parser.add_flag({"--threads", "-t"}, &threads, "How many cpu threads are used for video decode and upload");

    // bool switch flags
    parser.add_flag({"--list-gpu"}, &list_gpu, "lists all avaliable gpus");
    

    if (0 != parser.parse_cli_args(args)) return 1;

    if (list_gpu) {
        try{
            std::cout << helper::listGPU();
        } catch (const VshipError& e){
            std::cout << e.getErrorMessage() << std::endl;
        }
        return 1;
    }

    if (file1 == ""){
        std::cerr << "--source is not set" << std::endl;
        return 1;
    }
    if (file2 == ""){
        std::cerr << "--encoded is not set" << std::endl;
        return 1;
    }

    METRICS metric = SSIMULACRA2;

    if (metric_name == "") {
        metric = SSIMULACRA2;
    } else if (metric_name == "SSIMULACRA2") {
        metric = SSIMULACRA2;
    } else if (metric_name == "Butteraugli") {
        metric = Butteraugli;
    } else {
        std::cerr << "unknown metric " << metric_name;
        return 1;
    }

    //gpu sanity check
    try{
        //if succeed, this function also does hipSetDevice
        helper::gpuFullCheck(gpuid);
    } catch (const VshipError& e){
        std::cout << e.getErrorMessage() << std::endl;
        return 0;
    }

    auto init = std::chrono::high_resolution_clock::now();

    //FFMS2 indexer init
    FFMS_Init(0, 0);
    char errmsg[1024];
    FFMS_ErrorInfo errinfo;
    errinfo.Buffer      = errmsg;
    errinfo.BufferSize  = sizeof(errmsg);
    errinfo.ErrorType   = FFMS_ERROR_SUCCESS;
    errinfo.SubType     = FFMS_ERROR_SUCCESS;

    FFMS_Indexer *indexer1 = FFMS_CreateIndexer(file1.c_str(), &errinfo);
    if (indexer1 == NULL) {
        std::cout << "FFMS2, failed to create indexer of file " << file1 << " with error : " << errmsg << std::endl;
        return 0;
    }
    FFMS_Indexer *indexer2 = FFMS_CreateIndexer(file2.c_str(), &errinfo);
    if (indexer2 == NULL) {
        std::cout << "FFMS2, failed to create indexer of file " << file2 << " with error : " << errmsg << std::endl;
        return 0;
    }

    FFMS_Index *index1 = FFMS_DoIndexing2(indexer1, FFMS_IEH_ABORT, &errinfo);
    if (index1 == NULL) {
        std::cout << "FFMS2, failed to index file " << file1 << " with error : " << errmsg << std::endl;
        return 0;
    } 
    FFMS_Index *index2 = FFMS_DoIndexing2(indexer2, FFMS_IEH_ABORT, &errinfo);
    if (index2 == NULL) {
        std::cout << "FFMS2, failed to index file " << file2 << " with error : " << errmsg << std::endl;
        return 0;
    } 

    int trackno1 = FFMS_GetFirstTrackOfType(index1, FFMS_TYPE_VIDEO, &errinfo);
    if (trackno1 < 0) {
        std::cout << "FFMS2, found no video track in " << file1 << " with error : " << errmsg << std::endl;
        return 0;
    }
    int trackno2 = FFMS_GetFirstTrackOfType(index2, FFMS_TYPE_VIDEO, &errinfo);
    if (trackno2 < 0) {
        std::cout << "FFMS2, found no video track in " << file2 << " with error : " << errmsg << std::endl;
        return 0;
    }

    //prepare objects
    void** pinnedmempool = (void**)malloc(sizeof(void*)*gputhreads);
    for (int i = 0; i < gputhreads; i++) pinnedmempool[i] = NULL;

    float* gaussiankernel_dssimu2 = NULL;
    butter::GaussianHandle gaussianhandlebutter;

    switch (metric){
        case SSIMULACRA2:
        float gaussiankernel[2*GAUSSIANSIZE+1];
        for (int i = 0; i < 2*GAUSSIANSIZE+1; i++){
            gaussiankernel[i] = std::exp(-(GAUSSIANSIZE-i)*(GAUSSIANSIZE-i)/(2*SIGMA*SIGMA))/(std::sqrt(TAU*SIGMA*SIGMA));
        }

        hipMalloc(&(gaussiankernel_dssimu2), sizeof(float)*(2*GAUSSIANSIZE+1));
        hipMemcpyHtoD(gaussiankernel_dssimu2, gaussiankernel, (2*GAUSSIANSIZE+1)*sizeof(float));
        break;
        case Butteraugli:
        gaussianhandlebutter.init();
        break;
    }

    int device;
    hipDeviceProp_t devattr;
    hipGetDevice(&device);
    hipGetDeviceProperties(&devattr, device);
    const int maxshared = devattr.sharedMemPerBlock;

    threadSet gpustreams({});
    for (int i = 0; i < gputhreads; i++) gpustreams.insert(i);

    hipStream_t* streams_d = (hipStream_t*)malloc(sizeof(hipStream_t)*gputhreads);
    for (int i = 0; i < gputhreads; i++) hipStreamCreate(streams_d + i);

    //execute
    std::vector<std::thread> threadlist;
    std::vector<std::vector<float>> returnlist(threads);
    for (int i = 0; i < threads; i++){
        threadlist.emplace_back(threadwork, file1, index1, trackno1, file2, index2, trackno2, start, end, every, i, threads, metric, &gpustreams, maxshared, intensity_multiplier, gaussiankernel_dssimu2, &gaussianhandlebutter, pinnedmempool, streams_d, &(returnlist[i]));
    }

    for (int i = 0; i < threads; i++){
        threadlist[i].join();
    }

    //flatten
    std::vector<float> finalreslist;
    for (const auto& el: returnlist){
        for (const auto& el2: el){
            finalreslist.push_back(el2);
        }
    }
    
    auto fin = std::chrono::high_resolution_clock::now();

    int millitaken = std::chrono::duration_cast<std::chrono::milliseconds>(fin-init).count();
    int frames;
    switch (metric){
        case Butteraugli:
        frames = finalreslist.size()/3;
        break;
        case SSIMULACRA2:
        frames = finalreslist.size();
        break;
    }
    float fps = frames*1000/millitaken;
    
    //free
    FFMS_DestroyIndex(index1);
    FFMS_DestroyIndex(index2);
    for (int i = 0; i < gputhreads; i++) if (pinnedmempool[i] != NULL) hipHostFree(pinnedmempool[i]);
    free(pinnedmempool);
    for (int i = 0; i < gputhreads; i++) hipStreamDestroy(streams_d[i]);
    free(streams_d);
    switch (metric){
        case SSIMULACRA2:
        hipFree(gaussiankernel_dssimu2);
        break;
        case Butteraugli:
        gaussianhandlebutter.destroy();
        break;
    }

    if (finalreslist.size() == 0){
        std::cout << "Error: No scores were detected" << std::endl;
        return 0;
    }

    //posttreatment

    //json output
    if (jsonout != ""){
        std::ofstream jsonfile(jsonout, std::ios_base::out);
        if (!jsonfile){
            std::cout << "Failed to open output file" << std::endl;
            return 0;
        }
        jsonfile << "[";
        for (int i = 0; i < frames; i++){
            jsonfile << "[";
            switch (metric){
                case Butteraugli:
                jsonfile << finalreslist[3*i] << ", ";
                jsonfile << finalreslist[3*i+1] << ", ";
                jsonfile << finalreslist[3*i+2];
                break;
                case SSIMULACRA2:
                jsonfile << finalreslist[i];
                break;
            }
            if (i == frames-1) {
                jsonfile << "]";
            } else {
                jsonfile << "], ";
            }
        }
        jsonfile << "]";
    }

    //console output
    switch (metric){
        case Butteraugli:
        {
            std::vector<float> split1(finalreslist.size()/3);
            std::vector<float> split2(finalreslist.size()/3);
            std::vector<float> split3(finalreslist.size()/3);

            for (unsigned int i = 0; i < frames; i++){
                split1[i] = finalreslist[3*i];
                split2[i] = finalreslist[3*i+1];
                split3[i] = finalreslist[3*i+2];
            }

            std::sort(split1.begin(), split1.end()); //2 norm
            std::sort(split2.begin(), split2.end()); //3 norm
            std::sort(split3.begin(), split3.end()); //inf norm

            std::cout << "Butteraugli Result between " << file1 << " and " << file2 << std::endl;
            std::cout << "Computed " << frames << " frames at " << fps << " fps" << std::endl;
            std::cout << std::endl;

            double avg = 0;
            double avg_squared = 0;
            for (unsigned int i = 0; i < frames; i++){
                avg += split1[i];
                avg_squared += split1[i]*split1[i];
            }
            avg /= frames;
            avg_squared /= frames;
            double std_dev = std::sqrt(avg_squared - avg*avg);

            std::cout << "----2-Norm----" << std::endl;
            std::cout << "Average : " << avg << std::endl;
            std::cout << "Standard Deviation : " << std_dev << std::endl;
            std::cout << "Median : " << split1[frames/2] << std::endl;
            std::cout << "5th percentile : " << split1[frames/20] << std::endl;
            std::cout << "95th percentile : " << split1[frames*19/20] << std::endl;
            std::cout << "Maximum : " << split1[frames-1] << std::endl;
            
            avg = 0;
            avg_squared = 0;
            for (unsigned int i = 0; i < frames; i++){
                avg += split2[i];
                avg_squared += split2[i]*split2[i];
            }
            avg /= frames;
            avg_squared /= frames;
            std_dev = std::sqrt(avg_squared - avg*avg);

            std::cout << "----3-Norm----" << std::endl;
            std::cout << "Average : " << avg << std::endl;
            std::cout << "Standard Deviation : " << std_dev << std::endl;
            std::cout << "Median : " << split2[frames/2] << std::endl;
            std::cout << "5th percentile : " << split2[frames/20] << std::endl;
            std::cout << "95th percentile : " << split2[frames*19/20] << std::endl;
            std::cout << "Maximum : " << split2[frames-1] << std::endl;

            avg = 0;
            avg_squared = 0;
            for (unsigned int i = 0; i < frames; i++){
                avg += split3[i];
                avg_squared += split3[i]*split3[i];
            }
            avg /= frames;
            avg_squared /= frames;
            std_dev = std::sqrt(avg_squared - avg*avg);

            std::cout << "--INF-Norm----" << std::endl;
            std::cout << "Average : " << avg << std::endl;
            std::cout << "Standard Deviation : " << std_dev << std::endl;
            std::cout << "Median : " << split3[frames/2] << std::endl;
            std::cout << "5th percentile : " << split3[frames/20] << std::endl;
            std::cout << "95th percentile : " << split3[frames*19/20] << std::endl;
            std::cout << "Maximum : " << split3[frames-1] << std::endl;
        }
        break;
        case SSIMULACRA2:
        {
            std::sort(finalreslist.begin(), finalreslist.end());
            double avg = 0;
            double avg_squared = 0;
            for (unsigned int i = 0; i < finalreslist.size(); i++){
                avg += finalreslist[i];
                avg_squared += finalreslist[i]*finalreslist[i];
            }
            avg /= finalreslist.size();
            avg_squared /= finalreslist.size();
            const double std_dev = std::sqrt(avg_squared - avg*avg);

            std::cout << "SSIMU2 Result between " << file1 << " and " << file2 << std::endl;
            std::cout << "Computed " << frames << " frames at " << fps << " fps" << std::endl;
            std::cout << "Average : " << avg << std::endl;
            std::cout << "Standard Deviation : " << std_dev << std::endl;
            std::cout << "Median : " << finalreslist[finalreslist.size()/2] << std::endl;
            std::cout << "5th percentile : " << finalreslist[finalreslist.size()/20] << std::endl;
            std::cout << "95th percentile : " << finalreslist[19*finalreslist.size()/20] << std::endl;
            std::cout << "Minimum : " << finalreslist[0] << std::endl; 
        }
        break;
    }

    return 0;
}
