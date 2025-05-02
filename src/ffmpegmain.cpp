#include<fstream>
#include<algorithm>

#include "util/preprocessor.hpp"
#include "util/gpuhelper.hpp"
#include "util/VshipExceptions.hpp"
#include "util/threadsafeset.hpp"

#include "ssimu2/main.hpp"
#include "butter/main.hpp"

extern "C"{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

class FFmpegVideoManager{
public:
    SwsContext* sws_ctx = NULL;

    AVFormatContext* fmt_ctx = NULL;
    int streamid;
    AVStream *st;
    AVCodecParameters *dec_param;
    const AVCodec *dec;
    AVCodecContext *video_dec_ctx = NULL;

    //active decoding context
    AVPacket *pkt = NULL;
    bool packet_end = true;
    bool end_of_video = false; //used for flushing last decoder frames

    uint64_t beginTime = 0;
    uint64_t endTime = 0;
    
    AVFrame *frame = NULL;
    uint8_t* outputRGB = NULL;
    uint8_t* RGBptrHelper[3] = {NULL, NULL, NULL};
    int RGBstride[3] = {0, 0, 0};
    int error = 0;
    FFmpegVideoManager(std::string file){
        int ret = 0;

        ret = avformat_open_input(&fmt_ctx, file.c_str(), NULL, NULL);
        if (ret < 0) {
            std::cout << "FFmpeg failed to read file (Error : " << ret << ") : " << file << std::endl;
            error = 1;
            return;
        }

        ret = avformat_find_stream_info(fmt_ctx, NULL);
        if (ret < 0){
            std::cout << "FFmpeg failed to read streams from file (Error : " << ret << ") : " << file << std::endl;
            error = 2;
            return;
        }

        ret = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
        if (ret < 0){
            std::cout << "FFmpeg failed to find a video stream for file (Error : " << ret << ") : " << file << std::endl;
            error = 3;
            return;
        }

        streamid = ret;
        st = fmt_ctx->streams[streamid];
        dec_param = st->codecpar;

        dec = avcodec_find_decoder(dec_param->codec_id);
        if (!dec){
            std::cout << "FFmpeg failed to find " << file << " codec" << std::endl;
            error = 4;
            return;
        }

        video_dec_ctx = avcodec_alloc_context3(dec);
        if (!video_dec_ctx){
            std::cout << "FFmpeg failed to allocate codec context" << std::endl;
            error = 5;
            return;
        }

        avcodec_parameters_to_context(video_dec_ctx, dec_param);

        ret = avcodec_open2(video_dec_ctx, dec, NULL);
        if (ret < 0){
            std::cout << "FFmpeg failed to open " << file << " codec" << std::endl;
            error = 6;
            return;
        }

        frame = av_frame_alloc();
        if (!frame){
            std::cout << "FFmpeg failed to allocate frame" << std::endl;
            error = 7;
            return;
        }
        pkt = av_packet_alloc();
        if (!pkt){
            std::cout << "FFmpeg failed to allocate packet" << std::endl;
            error = 8;
            return;
        }
        beginTime = 0;
        endTime = ((fmt_ctx->duration+1)*st->time_base.den)/(st->time_base.num*AV_TIME_BASE);

        auto outputformat = AV_PIX_FMT_RGB48LE;
        sws_ctx = sws_getContext(video_dec_ctx->width, video_dec_ctx->height, video_dec_ctx->pix_fmt, video_dec_ctx->width, video_dec_ctx->height, outputformat, SWS_FAST_BILINEAR, NULL, NULL, NULL);
        if (!sws_ctx){
            std::cout << "FFmpeg failed to create rescale (color conversion) object for file : " << file << std::endl;
            error = 9;
            return;
        }
        uint64_t bufferSize = av_image_get_buffer_size(outputformat, video_dec_ctx->width, video_dec_ctx->height, 1);
        hipHostMalloc((void**)&outputRGB, bufferSize); //allocate pinned memory for end buffer for faster gpu send
        if (!outputRGB){
            std::cout << "Failed to allocate Pinned RAM for RGB output for file : " << file << " of size " << bufferSize << std::endl;
            error = 10;
            return;
        }
        RGBptrHelper[0] = outputRGB;
        RGBptrHelper[1] = outputRGB+bufferSize/3;
        RGBptrHelper[2] = outputRGB+2*bufferSize/3;

        RGBstride[0] = video_dec_ctx->width;
        RGBstride[1] = video_dec_ctx->width;
        RGBstride[2] = video_dec_ctx->width;
        
    }
    int seek(int num, int den){
        //std::cout << fmt_ctx->duration << " " << st->time_base.num << " " << st->time_base.den << " " << AV_TIME_BASE << std::endl;
        uint64_t totaltimeunit = ((fmt_ctx->duration+1)*st->time_base.den)/(st->time_base.num*AV_TIME_BASE);
        uint64_t requested = ((int64_t)totaltimeunit*(int64_t)num-1)/den+1;
        int ret = avformat_seek_file(fmt_ctx, streamid, INT64_MIN, requested, INT64_MAX, 0);
        beginTime = requested;
        endTime = ((int64_t)totaltimeunit*(int64_t)(num+1)-1)/den+1;
        return ret;
    }
    int getNextFrame(){ //access result with object.frame. return 0 if success, -2 is EndOfVideo
        int ret;

        av_frame_unref(frame); //we remove last frame
        if (packet_end){
            av_packet_unref(pkt); //we remove last packet and search a new one
            while (true){
                if (av_read_frame(fmt_ctx, pkt) < 0){
                    if (end_of_video) return -2; //we ve been there already so we are flushed
                    packet_end = false;
                    ret = avcodec_send_packet(video_dec_ctx, NULL);
                    if (ret < 0){
                        std::cout << "Error submitting a packet to the decoder" << std::endl;
                    }
                    end_of_video = true;
                    return getNextFrame();
                } else {
                    if (pkt->stream_index == streamid){
                        packet_end = false;
                        ret = avcodec_send_packet(video_dec_ctx, pkt);
                        if (ret < 0){
                            std::cout << "Error submitting a packet to the decoder" << std::endl;
                        }
                        break;
                    }
                }
            }
        }
        ret = avcodec_receive_frame(video_dec_ctx, frame);
        if (ret < 0) {
            if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN)){ //end of packet
                packet_end = true;
                return getNextFrame(); //still needs to track the next one
            }
 
            std::cout << "Error during decoding: " <<  av_err2str(ret) << std::endl;
            return -1;
        }

        const uint64_t currentTime = frame->pts;
        //std::cout << pkt->pts << "  " << packet_frame << "  " << st->time_base.num << " / " << st->time_base.den << std::endl;
        if (currentTime < beginTime) return getNextFrame();
        if (currentTime >= endTime) return -2;

        //we finally have our frame to return! Let s convert to RGB

        sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height, RGBptrHelper, RGBstride);

        return 0;
    }
    ~FFmpegVideoManager(){
        if (fmt_ctx != NULL) avformat_close_input(&fmt_ctx);
        if (video_dec_ctx != NULL) avcodec_free_context(&video_dec_ctx);
        if (pkt != NULL) av_packet_free(&pkt);
        if (frame != NULL) av_frame_free(&frame);
        if (sws_ctx != NULL) sws_freeContext(sws_ctx);
        if (outputRGB != NULL) hipFreeHost(outputRGB);

        outputRGB = NULL;
        sws_ctx = NULL;
        frame = NULL;
        pkt = NULL;
        video_dec_ctx = NULL;
        fmt_ctx = NULL;
    }
};

enum METRICS{SSIMULACRA2, Butteraugli};

void threadwork(std::string file1, std::string file2, int threadid, int threadnum, METRICS metric, threadSet* gpustreams, int maxshared, float intensity_multiplier, float* gaussiankernel_dssimu2, butter::GaussianHandle* gaussianhandlebutter, void** pinnedmempool, hipStream_t* streams_d, std::vector<float>* output){ //for butteraugli, return 2norm, 3norm, Infnorm, 2norm, ...

    av_log_set_level(AV_LOG_ERROR);
    
    FFmpegVideoManager v1(file1);
    if (v1.error){
        std::cout << "Thread " << threadid << " Failed to open file " << file1 << std::endl;
        return;
    }
    FFmpegVideoManager v2(file2);
    if (v2.error){
        std::cout << "Thread " << threadid << " Failed to open file " << file2 << std::endl;
        return;
    }
    if (v1.video_dec_ctx->width != v2.video_dec_ctx->width || v1.video_dec_ctx->height != v2.video_dec_ctx->height){
        std::cout << "the 2 videos do not have the same sizes" << std::endl;
        return;
    }
    if (v1.seek(threadid, threadnum) != 0){
        std::cout << "Thread " << threadid << " Failed to seek file " << file1 << std::endl;
        return;
    }
    if (v2.seek(threadid, threadnum) != 0){
        std::cout << "Thread " << threadid << " Failed to seek file " << file2 << std::endl;
        return;
    }

    int pinnedsize = 0;
    switch (metric){
        case SSIMULACRA2:
        pinnedsize = ssimu2::allocsizeScore(v2.video_dec_ctx->width, v1.video_dec_ctx->height, maxshared)*sizeof(float3);
        break;
        case Butteraugli:
        pinnedsize = butter::allocsizeScore(v2.video_dec_ctx->width, v1.video_dec_ctx->height)*sizeof(float);
        break;
    }
    
    int i = 0;
    while (v1.getNextFrame() == 0){
        if (v2.getNextFrame() != 0){
            std::cout << "premature end of distorded, are both stream of the same size?" << std::endl;
            break;
        }

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

        switch (metric){
            case Butteraugli:
            {
            const std::tuple<float, float, float> scorebutter = butter::butterprocess<UINT16>(NULL, 0, srcp1, srcp2, (float*)pinnedmem, *gaussianhandlebutter, v1.RGBstride[0], v1.video_dec_ctx->width, v1.video_dec_ctx->height, intensity_multiplier, maxshared, stream);
            output->push_back(std::get<0>(scorebutter));
            output->push_back(std::get<1>(scorebutter));
            output->push_back(std::get<2>(scorebutter));
            break;
            }
            case SSIMULACRA2:
            {
            const double scoressimu2 = ssimu2::ssimu2process<UINT16>(srcp1, srcp2, (float3*)pinnedmem, v1.RGBstride[0], v1.video_dec_ctx->width, v1.video_dec_ctx->height, gaussiankernel_dssimu2, maxshared, stream);
            output->push_back(scoressimu2);
            break;
            }
        }
        gpustreams->insert(streamid);
        
        i++;
    }
    return;
}

void printUsage(){
    std::cout << R"(usage: ./vship [-h] [--source SOURCE] [--encoded ENCODED]
                    [-m {SSIMULACRA2, Butteraugli}]
                    [-t THREADS] [-g gpuThreads] [-gpu gpu_id]
                    [--json OUTPUT]
                    [--list-gpu]
                    Specific to Butteraugli: 
                    [--intensity-target Intensity(nits)])" << std::endl;
}

int main(int argc, char** argv){
    std::vector<std::string> args(argc-1);
    for (int i = 1; i < argc; i++){
        args[i-1] = argv[i];
    } 

    if (argc == 1){
        printUsage();
        return 0;
    }

    int gpuid = 0;
    int gputhreads = 8;
    int threads = 12;
    bool metric_set = false;
    METRICS metric = SSIMULACRA2;
    std::string file1;
    std::string file2;
    std::string jsonout = "";

    int intensity_multiplier = 80;

    for (unsigned int i = 0; i < args.size(); i++){
        if (args[i] == "-h" || args[i] == "--help"){
            printUsage();
            return 0;
        } else if (args[i] == "--list-gpu"){
            try{
                std::cout << helper::listGPU();
            } catch (const VshipError& e){
                std::cout << e.getErrorMessage() << std::endl;
                return 1;
            }
            return 0;
        } else if (args[i] == "--source") {
            if (i == args.size()-1){
                std::cout << "--source needs an argument" << std::endl;
                return 0;
            }
            file1 = args[i+1];
            i++;
        } else if (args[i] == "--json") {
            if (i == args.size()-1){
                std::cout << "--json needs an argument" << std::endl;
                return 0;
            }
            jsonout = args[i+1];
            i++;
        } else if (args[i] == "--encoded") {
            if (i == args.size()-1){
                std::cout << "--encoded needs an argument" << std::endl;
                return 0;
            }
            file2 = args[i+1];
            i++;
        } else if (args[i] == "-t" || args[i] == "--threads"){
            if (i == args.size()-1){
                std::cout << "-t needs an argument" << std::endl;
                return 0;
            }
            try {
                threads = stoi(args[i+1]);
            } catch (std::invalid_argument& e){
                std::cout << "invalid value for -t" << std::endl;
                return 0;
            }
            i++;
        } else if (args[i] == "--gpu-id"){
            if (i == args.size()-1){
                std::cout << "--gpu-id needs an argument" << std::endl;
                return 0;
            }
            try {
                gpuid = stoi(args[i+1]);
            } catch (std::invalid_argument& e){
                std::cout << "invalid value for --gpu-id" << std::endl;
                return 0;
            }
            i++;
        } else if (args[i] == "-g" || args[i] == "--gputhreads"){
            if (i == args.size()-1){
                std::cout << "-g needs an argument" << std::endl;
                return 0;
            }
            try {
                gputhreads = stoi(args[i+1]);
            } catch (std::invalid_argument& e){
                std::cout << "invalid value for -g" << std::endl;
                return 0;
            }
            i++;
        } else if (args[i] == "--intensity-target"){
            if (i == args.size()-1){
                std::cout << "--intensity-target needs an argument" << std::endl;
                return 0;
            }
            try {
                intensity_multiplier = stoi(args[i+1]);
            } catch (std::invalid_argument& e){
                std::cout << "invalid value for --intensity-target" << std::endl;
                return 0;
            }
            i++;
        } else if (args[i] == "-m" || args[i] == "--metric"){
            if (i == args.size()-1){
                std::cout << "-m needs an argument" << std::endl;
                return 0;
            }
            if (args[i+1] == "SSIMULACRA2"){
                metric_set = true;
                metric = SSIMULACRA2;
            } else if (args[i+1] == "Butteraugli"){
                metric_set = true;
                metric = Butteraugli;
            } else {
                std::cout << "unrecognized metric : " << args[i+1] << std::endl;
                return 0;
            }
            i++;
        } else {
            std::cout << "Unrecognized option: " << args[i] << std::endl;
            return 0;
        }
    }

    //gpu sanity check
    helper::gpuFullCheck(gpuid);

    auto init = std::chrono::high_resolution_clock::now();

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
        threadlist.emplace_back(threadwork, file1, file2, i, threads, metric, &gpustreams, maxshared, intensity_multiplier, gaussiankernel_dssimu2, &gaussianhandlebutter, pinnedmempool, streams_d, &(returnlist[i]));
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
    
    //free
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

    //posttreatment
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
            std::cout << "Computed " << frames << " at " << fps << " fps" << std::endl;
            std::cout << std::endl;

            float avg = 0;
            float avg_squared = 0;
            for (unsigned int i = 0; i < frames; i++){
                avg += split1[i];
                avg_squared += split1[i]*split1[i];
            }
            avg /= frames;
            avg_squared /= frames;
            float std_dev = std::sqrt(avg_squared - avg*avg);

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
            float avg = 0;
            float avg_squared = 0;
            for (unsigned int i = 0; i < finalreslist.size(); i++){
                avg += finalreslist[i];
                avg_squared += finalreslist[i]*finalreslist[i];
            }
            avg /= finalreslist.size();
            avg_squared /= finalreslist.size();
            const float std_dev = std::sqrt(avg_squared - avg*avg);

            std::cout << "SSIMU2 Result between " << file1 << " and " << file2 << std::endl;
            std::cout << "Computed " << frames << " at " << fps << " fps" << std::endl;
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