#include "util/preprocessor.hpp"
#include "util/gpuhelper.hpp"
#include "util/VshipExceptions.hpp"

#include "util/preprocessor.hpp"

extern "C"{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/mem.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/avassert.h>
#include <libavutil/imgutils.h>
}

#include "ffmpeg_util/base.hpp"

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

    int ret = 0;

    AVFormatContext* fmt_ctx1 = NULL;
    AVFormatContext* fmt_ctx2 = NULL;
    ret = avformat_open_input(&fmt_ctx1, file1.c_str(), NULL, NULL);
    if (ret < 0) {
        std::cout << "FFmpeg failed to read file (Error : " << ret << ") : " << file1 << std::endl;
        return 0;
    }
    ret = avformat_open_input(&fmt_ctx2, file2.c_str(), NULL, NULL);
    if (ret < 0) {
        std::cout << "FFmpeg failed to read file (Error : " << ret << ") : " << file2 << std::endl;
        avformat_close_input(&fmt_ctx1);
        return 0;
    }

    ret = avformat_find_stream_info(fmt_ctx1, NULL);
    if (ret < 0){
        std::cout << "FFmpeg failed to read streams from file (Error : " << ret << ") : " << file1 << std::endl;
        avformat_close_input(&fmt_ctx1);
        avformat_close_input(&fmt_ctx2);
        return 0;
    }
    ret = avformat_find_stream_info(fmt_ctx2, NULL);
    if (ret < 0){
        std::cout << "FFmpeg failed to read streams from file (Error : " << ret << ") : " << file2 << std::endl;
        avformat_close_input(&fmt_ctx1);
        avformat_close_input(&fmt_ctx2);
        return 0;
    }

    ret = av_find_best_stream(fmt_ctx1, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (ret < 0){
        std::cout << "FFmpeg failed to find a video stream for file (Error : " << ret << ") : " << file1 << std::endl;
        avformat_close_input(&fmt_ctx1);
        avformat_close_input(&fmt_ctx2);
    }
    int streamid1 = ret;
    ret = av_find_best_stream(fmt_ctx2, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (ret < 0){
        std::cout << "FFmpeg failed to find a video stream for file (Error : " << ret << ") : " << file2 << std::endl;
        avformat_close_input(&fmt_ctx1);
        avformat_close_input(&fmt_ctx2);
    }
    int streamid2 = ret;
    
    AVStream *st1 = fmt_ctx1->streams[streamid1];
    AVStream *st2 = fmt_ctx2->streams[streamid2];

    AVCodecParameters *dec_param1 = st1->codecpar;
    AVCodecParameters *dec_param2 = st2->codecpar;

    const AVCodec *dec1 = avcodec_find_decoder(dec_param1->codec_id);
    if (!dec1){
        std::cout << "FFmpeg failed to find " << file1 << " codec" << std::endl;
        avformat_close_input(&fmt_ctx1);
        avformat_close_input(&fmt_ctx2);
        return 0;
    }
    const AVCodec *dec2 = avcodec_find_decoder(dec_param2->codec_id);
    if (!dec2){
        std::cout << "FFmpeg failed to find " << file2 << " codec" << std::endl;
        avformat_close_input(&fmt_ctx1);
        avformat_close_input(&fmt_ctx2);
        return 0;
    }


    avformat_close_input(&fmt_ctx1);
    avformat_close_input(&fmt_ctx2);
    
    return 0;
}