#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <numeric>
#include <string>

#include "util/CLI_Parser.hpp"
#include "util/VshipExceptions.hpp"
#include "util/gpuhelper.hpp"
#include "util/preprocessor.hpp"
#include "util/threadsafeset.hpp"

#include "butter/main.hpp"
#include "ssimu2/main.hpp"

#include "ffmpegmain.hpp"
#include "util/concurrency.hpp"

extern "C" {
#include <ffms.h>
#include <libavutil/pixfmt.h>
#include <zimg.h>
}

#include "gpuColorToLinear/vshipColor.hpp"

using score_tuple_t = std::tuple<float, float, float>;
using score_queue_t = ThreadSafeQueue<std::tuple<int, score_tuple_t>>;
using frame_tuple_t = std::tuple<int, uint8_t *, uint8_t *>;
using frame_queue_t = ThreadSafeQueue<frame_tuple_t>;
using frame_pool_t = ThreadSafePoolVecWrapper<uint8_t *>;

void frame_reader_thread(VideoManager &v1, VideoManager &v2, int start, int end,
                         int every, frame_queue_t &queue,
                         frame_pool_t &frame_buffer_pool) {

    for (int i = start; i < end; i += every) {
        uint8_t *src_buffer = frame_buffer_pool.acquire();
        uint8_t *enc_buffer = frame_buffer_pool.acquire();

        auto future_src =
            std::async(std::launch::async, [&v1, i, src_buffer]() {
                v1.fetch_frame_into_buffer(i, src_buffer);
            });

        auto future_enc =
            std::async(std::launch::async, [&v2, i, enc_buffer]() {
                v2.fetch_frame_into_buffer(i, enc_buffer);
            });

        future_src.get();
        future_enc.get();

        frame_tuple_t frame_tuple = std::make_tuple(i, src_buffer, enc_buffer);
        queue.push(frame_tuple);
    }
}

void frame_worker_thread(frame_queue_t &input_queue,
                         frame_pool_t &frame_buffer_pool, GpuWorker &gpu_worker,
                         MetricType metric, float intensity_multiplier,
                         score_queue_t &output_score_queue) {
    while (true) {
        std::optional<std::tuple<int, uint8_t *, uint8_t *>> maybe_task =
            input_queue.pop();
        if (!maybe_task.has_value()) {
            break;
        }
        auto [frame_index, src_buffer, enc_buffer] = *maybe_task;

        std::tuple<float, float, float> scores;
        try {
            scores = gpu_worker.compute_metric_score(src_buffer, enc_buffer,
                                                     intensity_multiplier);
        } catch (const VshipError &e) {
            std::cout << " error: " << e.getErrorMessage() << std::endl;
            frame_buffer_pool.release(src_buffer);
            frame_buffer_pool.release(enc_buffer);
            continue;
        }

        output_score_queue.push(std::make_tuple(frame_index, scores));

        frame_buffer_pool.release(src_buffer);
        frame_buffer_pool.release(enc_buffer);
    }
}

void aggregate_scores_function(score_queue_t &input_score_queue,
                               std::vector<float> &aggregated_scores,
                               MetricType metric) {
    while (true) {
        std::optional<std::tuple<int, std::tuple<float, float, float>>>
            maybe_score = input_score_queue.pop();

        if (!maybe_score.has_value()) {
            break;
        }

        const auto &[frame_index, scores_tuple] = *maybe_score;
        const bool should_store_first_score =
            (metric == MetricType::SSIMULACRA2);

        if (should_store_first_score) {
            aggregated_scores[frame_index] = std::get<0>(scores_tuple);
        } else {
            aggregated_scores[frame_index * 3] = std::get<0>(scores_tuple);
            aggregated_scores[frame_index * 3 + 1] = std::get<1>(scores_tuple);
            aggregated_scores[frame_index * 3 + 2] = std::get<2>(scores_tuple);
        }
    }
}

void print_aggergate_metric_statistics(const std::vector<float> &data,
                                       const std::string &label) {
    if (data.empty())
        return;

    std::vector<float> sorted = data;
    std::sort(sorted.begin(), sorted.end());

    const size_t count = sorted.size();
    const double average =
        std::accumulate(sorted.begin(), sorted.end(), 0.0) / count;
    const double squared_sum =
        std::inner_product(sorted.begin(), sorted.end(), sorted.begin(), 0.0);
    const double stddev = std::sqrt(squared_sum / count - average * average);

    std::vector<std::pair<std::string, double>> stats = {
        {"Average", average},
        {"Standard Deviation", stddev},
        {"Median", sorted[count / 2]},
        {"5th percentile", sorted[count / 20]},
        {"95th percentile", sorted[19 * count / 20]},
        {"Minimum", sorted.front()},
        {"Maximum", sorted.back()}};

    // Dynamically calculate label width
    size_t max_label_width = 0;
    for (const auto &[name, _] : stats) {
        max_label_width = std::max(max_label_width, name.size());
    }

    constexpr int value_width = 12;
    constexpr int precision = 6;
    const int spacing = 3; // " : "
    const int total_width =
        static_cast<int>(max_label_width) + spacing + value_width;

    // Center the label
    const int label_padding =
        std::max(0, (total_width - static_cast<int>(label.size())) / 2);
    std::cout << std::string(label_padding, '-') << label
              << std::string(total_width - label_padding -
                                 static_cast<int>(label.size()),
                             '-')
              << std::endl;

    for (const auto &[name, value] : stats) {
        std::cout << std::setw(static_cast<int>(max_label_width)) << std::right
                  << name << " : " << std::setw(value_width) << std::right
                  << std::fixed << std::setprecision(precision) << value
                  << std::endl;
    }

    std::cout << std::endl;
}

int main(int argc, char **argv) {
    CommandLineOptions cli_args = parse_command_line_arguments(argc, argv);

    if (cli_args.list_gpus) {
        try {
            std::cout << helper::listGPU();
        } catch (const VshipError &e) {
            std::cout << e.getErrorMessage() << std::endl;
        }
        return 1;
    }

    // gpu sanity check
    try {
        // if succeed, this function also does hipSetDevice
        helper::gpuFullCheck(cli_args.gpu_id);
    } catch (const VshipError &e) {
        std::cout << e.getErrorMessage() << std::endl;
        return 0;
    }

    auto init = std::chrono::high_resolution_clock::now();

    int device;
    hipDeviceProp_t devattr;
    hipGetDevice(&device);
    hipGetDeviceProperties(&devattr, device);
    const int maxshared = devattr.sharedMemPerBlock;

    int queue_capacity = 1;

    int num_gpus = cli_args.gpu_threads ? cli_args.gpu_threads < 3 : 3;
    int num_frame_buffer = (1 + num_gpus) * 2 + queue_capacity + 1;

    FFMSIndexResult source_index = FFMSIndexResult(cli_args.source_file);
    FFMSIndexResult encode_index = FFMSIndexResult(cli_args.encoded_file);

    VideoManager v1(cli_args.source_file, source_index.index,
                    source_index.selected_video_track);
    int width = v1.reader->frame_width, height = v1.reader->frame_height;

    VideoManager v2(cli_args.encoded_file, encode_index.index,
                    encode_index.selected_video_track, width, height);

    if (cli_args.end_frame == -1) {
        cli_args.end_frame = v2.reader->total_frame_count;
    }

    int num_frames = cli_args.end_frame - cli_args.start_frame;

    std::vector<uint8_t *> frame_buffers(num_frame_buffer);
    for (int i = 0; i < static_cast<int>(frame_buffers.size()); ++i) {
        frame_buffers[i] =
            GpuWorker::allocate_external_rgb_buffer(width, height);
    }

    frame_pool_t frame_buffer_pool(frame_buffers);
    frame_queue_t frame_queue(1);

    std::vector<GpuWorker> gpu_workers;
    gpu_workers.reserve(num_gpus);

    std::generate_n(std::back_inserter(gpu_workers), num_gpus, [&]() {
        return GpuWorker(cli_args.metric, width, height, maxshared);
    });

    std::thread reader_thread(frame_reader_thread, std::ref(v1), std::ref(v2),
                              cli_args.start_frame, cli_args.end_frame,
                              cli_args.every_nth_frame, std::ref(frame_queue),
                              std::ref(frame_buffer_pool));

    score_queue_t score_queue(1);

    std::vector<std::thread> workers;
    for (int i = 0; i < num_gpus; ++i) {
        workers.emplace_back(frame_worker_thread, std::ref(frame_queue),
                             std::ref(frame_buffer_pool),
                             std::ref(gpu_workers[i]), cli_args.metric,
                             cli_args.intensity_target_nits,
                             std::ref(score_queue));
    }

    const int score_vector_size = (cli_args.metric == MetricType::SSIMULACRA2)
                                      ? num_frames
                                      : num_frames * 3;
    std::vector<float> scores(score_vector_size);

    std::thread score_thread(aggregate_scores_function, std::ref(score_queue),
                             std::ref(scores), cli_args.metric);

    reader_thread.join();
    frame_queue.close();

    for (auto &w : workers)
        w.join();

    score_queue.close();
    score_thread.join();

    auto fin = std::chrono::high_resolution_clock::now();

    int millitaken =
        std::chrono::duration_cast<std::chrono::milliseconds>(fin - init)
            .count();

    float fps = num_frames * 1000 / millitaken;

    // posttreatment

    // json output
    if (cli_args.json_output_file != "") {
        std::ofstream jsonfile(cli_args.json_output_file, std::ios_base::out);
        if (!jsonfile) {
            std::cout << "Failed to open output file" << std::endl;
            return 0;
        }
        jsonfile << "[";
        for (int i = 0; i < num_frames; i++) {
            jsonfile << "[";
            switch (cli_args.metric) {
            case MetricType::Butteraugli:
                jsonfile << scores[3 * i] << ", ";
                jsonfile << scores[3 * i + 1] << ", ";
                jsonfile << scores[3 * i + 2];
                break;
            case MetricType::SSIMULACRA2:
                jsonfile << scores[i];
                break;
            case MetricType::Unknown:
                break;
            }
            if (i == num_frames - 1) {
                jsonfile << "]";
            } else {
                jsonfile << "], ";
            }
        }
        jsonfile << "]";
    }

    // console output
    std::cout << (cli_args.metric == MetricType::Butteraugli ? "Butteraugli"
                                                             : "SSIMU2")
              << " Result between " << cli_args.source_file << " and "
              << cli_args.encoded_file << std::endl;
    std::cout << "Computed " << num_frames << " frames at " << fps << " fps\n"
              << std::endl;

    if (cli_args.metric == MetricType::Butteraugli) {
        std::vector<float> norm2(num_frames), norm3(num_frames),
            norminf(num_frames);

        for (int i = 0; i < num_frames; ++i) {
            norm2[i] = scores[3 * i];
            norm3[i] = scores[3 * i + 1];
            norminf[i] = scores[3 * i + 2];
        }

        print_aggergate_metric_statistics(norm2, "2-Norm");
        print_aggergate_metric_statistics(norm3, "3-Norm");
        print_aggergate_metric_statistics(norminf, "INF-Norm");

    } else if (cli_args.metric == MetricType::SSIMULACRA2) {
        print_aggergate_metric_statistics(scores, "SSIMULACRA2");
    }
    return 0;
}
