#include "util/preprocessor.hpp"
#include "util/gpuhelper.hpp"
#include "util/VshipExceptions.hpp"

#include "util/preprocessor.hpp"

extern "C"{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
//#include <libavutil/mem.h>
//#include <libavutil/pixdesc.h>
//#include <libavutil/hwcontext.h>
//#include <libavutil/opt.h>
//#include <libavutil/avassert.h>
#include <libavutil/imgutils.h>
}

class FFmpegVideoManager{
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
    
public:
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

void printUsage(){
    std::cout << R"(usage: ./vship [-h] [--source SOURCE] [--encoded ENCODED]
                    [-m {SSIMULACRA2, Butteraugli}]
                    [-t THREADS] [-g gpuThreads] [-gpu gpu_id]
                    [-e EVERY] [--start START] [--end END]
                    [--list-gpu]
                    Specific to Butteraugli: 
                    [--intensity-target Intensity(nits)])" << std::endl;
}

enum METRICS{SSIMULACRA2, Butteraugli};

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
    int start = 0;
    int end = -1;
    int skip = 1;
    int threads = -1;
    bool metric_set = false;
    METRICS metric = SSIMULACRA2;
    std::string file1;
    std::string file2;

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
        } else if (args[i] == "-e" || args[i] == "--every"){
            if (i == args.size()-1){
                std::cout << "-e needs an argument" << std::endl;
                return 0;
            }
            try {
                skip = stoi(args[i+1]);
            } catch (std::invalid_argument& e){
                std::cout << "invalid value for -e" << std::endl;
                return 0;
            }
            i++;
        } else if (args[i] == "--source") {
            if (i == args.size()-1){
                std::cout << "--source needs an argument" << std::endl;
                return 0;
            }
            file1 = args[i+1];
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
        } else if (args[i] == "--start"){
            if (i == args.size()-1){
                std::cout << "--start needs an argument" << std::endl;
                return 0;
            }
            try {
                start = stoi(args[i+1]);
            } catch (std::invalid_argument& e){
                std::cout << "invalid value for --start" << std::endl;
                return 0;
            }
            i++;
        } else if (args[i] == "--end"){
            if (i == args.size()-1){
                std::cout << "--end needs an argument" << std::endl;
                return 0;
            }
            try {
                end = stoi(args[i+1]);
            } catch (std::invalid_argument& e){
                std::cout << "invalid value for --end" << std::endl;
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
    
    FFmpegVideoManager v1(file1);
    if (v1.error){
        std::cout << "Failed to open file " << file1 << std::endl;
        return 0;
    }
    FFmpegVideoManager v2(file2);
    if (v2.error){
        std::cout << "Failed to open file " << file2 << std::endl;
        return 0;
    }

    //v1.seek(5, 24);
    //v2.seek(5, 24);

    auto init = std::chrono::high_resolution_clock::now();
    int i = 0;
    while (v1.getNextFrame() == 0){
        i++;
    }
    auto fin = std::chrono::high_resolution_clock::now();
    std::cout << "Read " << i << " frames at " << i*1000/(std::chrono::duration_cast<std::chrono::milliseconds>(fin-init)).count() << std::endl;

    init = std::chrono::high_resolution_clock::now();
    i = 0;
    while (v2.getNextFrame() == 0){
        i++;
    }
    fin = std::chrono::high_resolution_clock::now();
    std::cout << "Read " << i << " frames at " << i*1000/(std::chrono::duration_cast<std::chrono::milliseconds>(fin-init)).count() << std::endl;
    
    return 0;
}