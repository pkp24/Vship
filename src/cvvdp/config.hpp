#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>\\n#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../../../third_party/rapidjson/include/rapidjson/document.h"
#include "../../../third_party/rapidjson/include/rapidjson/error/en.h"

#include "../util/VshipExceptions.hpp"

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
#include <limits.h>
#endif

namespace cvvdp {

constexpr float kPi = 3.14159265358979323846f;

inline std::string load_file_to_string(const std::filesystem::path& filepath) {
    std::ifstream file(filepath, std::ios::in | std::ios::binary);
    if (!file) {
        std::cerr << "[CVVDP] Failed to open configuration file: " << filepath.string() << std::endl;
        throw VshipError(FileNotFound, __FILE__, __LINE__);
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

inline void parse_json_document(const std::filesystem::path& filepath, rapidjson::Document& out_doc) {
    const std::string payload = load_file_to_string(filepath);
    out_doc.Parse(payload.c_str());

    if (out_doc.HasParseError()) {
        std::cerr << "[CVVDP] JSON parse error in " << filepath.string()
                  << " at offset " << out_doc.GetErrorOffset() << ": "
                  << rapidjson::GetParseError_En(out_doc.GetParseError()) << std::endl;
        throw VshipError(ConfigurationError, __FILE__, __LINE__);
    }
}

inline const rapidjson::Value& require_member(const rapidjson::Value& object,
                                              const char* key,
                                              const std::filesystem::path& filepath) {
    if (!object.IsObject() || !object.HasMember(key)) {
        std::cerr << "[CVVDP] Missing required key '" << key
                  << "' in " << filepath.string() << std::endl;
        throw VshipError(ConfigurationError, __FILE__, __LINE__);
    }
    return object[key];
}

inline std::string get_string(const rapidjson::Value& object,
                              const char* key,
                              const std::filesystem::path& filepath,
                              const std::string& default_value = "",
                              bool required = false) {
    if (!object.HasMember(key)) {
        if (required) {
            std::cerr << "[CVVDP] Missing required string '" << key
                      << "' in " << filepath.string() << std::endl;
            throw VshipError(ConfigurationError, __FILE__, __LINE__);
        }
        return default_value;
    }

    const rapidjson::Value& value = object[key];
    if (!value.IsString()) {
        std::cerr << "[CVVDP] Expected string for '" << key << "' in "
                  << filepath.string() << std::endl;
        throw VshipError(ConfigurationError, __FILE__, __LINE__);
    }

    return value.GetString();
}

inline float get_float(const rapidjson::Value& object,
                       const char* key,
                       const std::filesystem::path& filepath,
                       float default_value = 0.0f,
                       bool required = false) {
    if (!object.HasMember(key)) {
        if (required) {
            std::cerr << "[CVVDP] Missing required float '" << key
                      << "' in " << filepath.string() << std::endl;
            throw VshipError(ConfigurationError, __FILE__, __LINE__);
        }
        return default_value;
    }

    const rapidjson::Value& value = object[key];
    if (!value.IsNumber()) {
        std::cerr << "[CVVDP] Expected numeric value for '" << key
                  << "' in " << filepath.string() << std::endl;
        throw VshipError(ConfigurationError, __FILE__, __LINE__);
    }

    return static_cast<float>(value.GetDouble());
}

inline int get_int(const rapidjson::Value& object,
                   const char* key,
                   const std::filesystem::path& filepath,
                   int default_value = 0,
                   bool required = false) {
    if (!object.HasMember(key)) {
        if (required) {
            std::cerr << "[CVVDP] Missing required integer '" << key
                      << "' in " << filepath.string() << std::endl;
            throw VshipError(ConfigurationError, __FILE__, __LINE__);
        }
        return default_value;
    }

    const rapidjson::Value& value = object[key];
    if (!value.IsInt()) {
        std::cerr << "[CVVDP] Expected integer for '" << key
                  << "' in " << filepath.string() << std::endl;
        throw VshipError(ConfigurationError, __FILE__, __LINE__);
    }

    return value.GetInt();
}

inline std::vector<float> get_float_array(const rapidjson::Value& object,
                                          const char* key,
                                          const std::filesystem::path& filepath,
                                          bool required = false) {
    if (!object.HasMember(key)) {
        if (required) {
            std::cerr << "[CVVDP] Missing required float array '" << key
                      << "' in " << filepath.string() << std::endl;
            throw VshipError(ConfigurationError, __FILE__, __LINE__);
        }
        return {};
    }

    const rapidjson::Value& value = object[key];
    if (!value.IsArray()) {
        std::cerr << "[CVVDP] Expected array for '" << key
                  << "' in " << filepath.string() << std::endl;
        throw VshipError(ConfigurationError, __FILE__, __LINE__);
    }

    std::vector<float> result;
    result.reserve(value.Size());
    for (auto& v : value.GetArray()) {
        if (!v.IsNumber()) {
            std::cerr << "[CVVDP] Non-numeric entry detected in array '" << key
                      << "' in " << filepath.string() << std::endl;
            throw VshipError(ConfigurationError, __FILE__, __LINE__);
        }
        result.push_back(static_cast<float>(v.GetDouble()));
    }
    return result;
}

inline std::vector<int> get_int_array(const rapidjson::Value& object,
                                      const char* key,
                                      const std::filesystem::path& filepath,
                                      bool required = false) {
    if (!object.HasMember(key)) {
        if (required) {
            std::cerr << "[CVVDP] Missing required integer array '" << key
                      << "' in " << filepath.string() << std::endl;
            throw VshipError(ConfigurationError, __FILE__, __LINE__);
        }
        return {};
    }

    const rapidjson::Value& value = object[key];
    if (!value.IsArray()) {
        std::cerr << "[CVVDP] Expected array for '" << key
                  << "' in " << filepath.string() << std::endl;
        throw VshipError(ConfigurationError, __FILE__, __LINE__);
    }

    std::vector<int> result;
    result.reserve(value.Size());
    for (auto& v : value.GetArray()) {
        if (!v.IsInt()) {
            std::cerr << "[CVVDP] Non-integer entry detected in array '" << key
                      << "' in " << filepath.string() << std::endl;
            throw VshipError(ConfigurationError, __FILE__, __LINE__);
        }
        result.push_back(v.GetInt());
    }
    return result;
}

inline std::filesystem::path executable_directory() {
#if defined(_WIN32)
    char buffer[MAX_PATH];
    DWORD length = GetModuleFileNameA(nullptr, buffer, MAX_PATH);
    if (length == 0 || length == MAX_PATH) {
        return std::filesystem::current_path();
    }
    return std::filesystem::path(std::string(buffer, static_cast<size_t>(length))).parent_path();
#elif defined(__linux__)
    char buffer[PATH_MAX];
    const ssize_t length = ::readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (length <= 0) {
        return std::filesystem::current_path();
    }
    buffer[length] = '\0';
    return std::filesystem::path(buffer).parent_path();
#else
    return std::filesystem::current_path();
#endif
}

inline std::filesystem::path resolve_cvvdp_data_root(const std::string& override_path = "") {
    static const std::vector<std::string> required_files = {
        "color_spaces.json",
        "csf_lut_dkl_cone.json",
        "csf_lut_log.json",
        "csf_lut_none.json",
        "csf_lut_weber_fixed_size.json",
        "csf_lut_weber_old.json",
        "csf_lut_weber_supra.json",
        "csf_lut_weber.json",
        "cvvdp_parameters.json",
        "display_models.json"
    };

    std::vector<std::filesystem::path> candidates;
    if (!override_path.empty()) {
        candidates.emplace_back(std::filesystem::absolute(override_path));
    }

    if (const char* env_override = std::getenv("CVVDP_DATA_DIR")) {
        candidates.emplace_back(std::filesystem::absolute(env_override));
    }

    if (const char* ffvship_home = std::getenv("FFVSHIP_HOME")) {
        candidates.emplace_back(std::filesystem::path(ffvship_home) / "cvvdp_data");
        candidates.emplace_back(std::filesystem::path(ffvship_home) / "config" / "cvvdp_data");
    }

    const std::filesystem::path exe_dir = executable_directory();
    candidates.emplace_back(exe_dir / "cvvdp_data");
    candidates.emplace_back(exe_dir / "config" / "cvvdp_data");

    const std::filesystem::path cwd = std::filesystem::current_path();
    candidates.emplace_back(cwd / "cvvdp_data");
    candidates.emplace_back(cwd / "config" / "cvvdp_data");
    candidates.emplace_back(cwd / "Vship" / "config" / "cvvdp_data");

    // Remove duplicate paths while preserving order
    std::vector<std::filesystem::path> unique_candidates;
    for (const auto& candidate : candidates) {
        if (std::find(unique_candidates.begin(), unique_candidates.end(), candidate) == unique_candidates.end()) {
            unique_candidates.push_back(candidate);
        }
    }

    std::ostringstream diagnostics;
    diagnostics << "[CVVDP] Unable to locate cvvdp_data directory. Checked locations:\n";

    for (const auto& candidate : unique_candidates) {
        std::vector<std::string> missing;
        for (const auto& file : required_files) {
            if (!std::filesystem::exists(candidate / file)) {
                missing.push_back(file);
            }
        }

        if (missing.empty()) {
            return candidate;
        }

        diagnostics << " - " << candidate.string() << " (missing ";
        for (size_t i = 0; i < missing.size(); ++i) {
            diagnostics << missing[i];
            if (i + 1 < missing.size()) {
                diagnostics << ", ";
            }
        }
        diagnostics << ")\n";
    }

    std::cerr << diagnostics.str()
              << "[CVVDP] Provide --cvvdp-data-dir or set FFVSHIP_HOME/CVVDP_DATA_DIR to a directory containing these files."
              << std::endl;
    throw VshipError(FileNotFound, __FILE__, __LINE__);
}

struct CVVDPParameters {
    std::string version;
    float mask_p;
    float mask_c;
    int pu_dilate;
    float beta;
    float beta_t;
    float beta_tch;
    float beta_sch;
    float csf_sigma;
    float sensitivity_correction;
    std::string masking_model;
    std::string local_adapt;
    std::string contrast;
    std::string csf;
    float jod_a;
    float jod_exp;
    std::vector<float> mask_q;
    int filter_len;
    float ch_chrom_w;
    float ch_trans_w;
    std::vector<float> sigma_tf;
    std::vector<float> beta_tf;
    std::string xchannel_masking;
    std::vector<float> xcm_weights;
    std::vector<float> baseband_weight;
    std::string dclamp_type;
    float d_max;
    float image_int;
    std::string Bloch_int;
    float bfilt_duration;

    static CVVDPParameters load(const std::filesystem::path& filepath) {
        rapidjson::Document doc;
        parse_json_document(filepath, doc);

        if (!doc.IsObject()) {
            std::cerr << "[CVVDP] Expected top-level object in " << filepath.string() << std::endl;
            throw VshipError(ConfigurationError, __FILE__, __LINE__);
        }

        CVVDPParameters params;
        params.version = get_string(doc, "version", filepath, "", false);
        params.mask_p = get_float(doc, "mask_p", filepath, 0.0f, true);
        params.mask_c = get_float(doc, "mask_c", filepath, 0.0f, true);
        params.pu_dilate = get_int(doc, "pu_dilate", filepath, 0, true);
        params.beta = get_float(doc, "beta", filepath, 0.0f, true);
        params.beta_t = get_float(doc, "beta_t", filepath);
        params.beta_tch = get_float(doc, "beta_tch", filepath);
        params.beta_sch = get_float(doc, "beta_sch", filepath);
        params.csf_sigma = get_float(doc, "csf_sigma", filepath);
        params.sensitivity_correction = get_float(doc, "sensitivity_correction", filepath);
        params.masking_model = get_string(doc, "masking_model", filepath);
        params.local_adapt = get_string(doc, "local_adapt", filepath);
        params.contrast = get_string(doc, "contrast", filepath);
        params.csf = get_string(doc, "csf", filepath, "weber_fixed_size", false);
        params.jod_a = get_float(doc, "jod_a", filepath, 0.0f, true);
        params.jod_exp = get_float(doc, "jod_exp", filepath, 0.0f, true);
        params.mask_q = get_float_array(doc, "mask_q", filepath);
        params.filter_len = get_int(doc, "filter_len", filepath, -1, false);
        params.ch_chrom_w = get_float(doc, "ch_chrom_w", filepath, 1.0f, false);
        params.ch_trans_w = get_float(doc, "ch_trans_w", filepath, 1.0f, false);
        params.sigma_tf = get_float_array(doc, "sigma_tf", filepath);
        params.beta_tf = get_float_array(doc, "beta_tf", filepath);
        params.xchannel_masking = get_string(doc, "xchannel_masking", filepath);
        params.xcm_weights = get_float_array(doc, "xcm_weights", filepath);
        params.baseband_weight = get_float_array(doc, "baseband_weight", filepath);
        params.dclamp_type = get_string(doc, "dclamp_type", filepath, "soft", false);
        params.d_max = get_float(doc, "d_max", filepath, 0.0f, true);
        params.image_int = get_float(doc, "image_int", filepath, 0.0f, false);
        params.Bloch_int = get_string(doc, "Bloch_int", filepath);
        params.bfilt_duration = get_float(doc, "bfilt_duration", filepath);
        return params;
    }
};

struct DisplayModel {
    std::string name;
    std::string colorspace;
    int width = 0;
    int height = 0;
    float viewing_distance_meters = 0.0f;
    float diagonal_size_inches = 0.0f;
    float max_luminance = 0.0f;
    float min_luminance = 0.0f;
    float contrast = 0.0f;
    float E_ambient = 0.0f;
    float fov_diagonal = 0.0f;

    float get_ppd() const {
        const float diagonal_pixels = std::sqrt(static_cast<float>(width * width + height * height));
        const float diagonal_meters = diagonal_size_inches * 0.0254f;
        const float fov_rad = 2.0f * std::atan(diagonal_meters / (2.0f * viewing_distance_meters));
        const float fov_deg = fov_rad * 180.0f / kPi;
        return diagonal_pixels / std::max(fov_deg, std::numeric_limits<float>::epsilon());
    }

    float get_black_level() const {
        float Lblack = (contrast > 0.0f) ? max_luminance / contrast : 0.0f;
        if (E_ambient > 0.0f) {
            constexpr float k_refl = 0.01f;
            Lblack += k_refl * E_ambient / kPi;
        }
        return Lblack;
    }

    static DisplayModel load(const std::filesystem::path& filepath, const std::string& display_name) {
        rapidjson::Document doc;
        parse_json_document(filepath, doc);

        if (!doc.IsObject()) {
            std::cerr << "[CVVDP] Expected top-level object in " << filepath.string() << std::endl;
            throw VshipError(ConfigurationError, __FILE__, __LINE__);
        }

        if (!doc.HasMember(display_name.c_str())) {
            std::cerr << "[CVVDP] Display model '" << display_name
                      << "' not found in " << filepath.string() << std::endl;
            throw VshipError(ConfigurationError, __FILE__, __LINE__);
        }

        const rapidjson::Value& entry = doc[display_name.c_str()];
        if (!entry.IsObject()) {
            std::cerr << "[CVVDP] Display entry for '" << display_name
                      << "' is not an object in " << filepath.string() << std::endl;
            throw VshipError(ConfigurationError, __FILE__, __LINE__);
        }

        DisplayModel model;
        model.name = get_string(entry, "name", filepath, display_name, false);
        model.colorspace = get_string(entry, "colorspace", filepath, "BT.709-sRGB", false);

        const std::vector<int> resolution = get_int_array(entry, "resolution", filepath, false);
        if (resolution.size() >= 2) {
            model.width = resolution[0];
            model.height = resolution[1];
        } else {
            model.width = 3840;
            model.height = 2160;
        }

        model.viewing_distance_meters = get_float(entry, "viewing_distance_meters", filepath, 0.0f, false);
        if (model.viewing_distance_meters <= 0.0f) {
            const float viewing_distance_inches = get_float(entry, "viewing_distance_inches", filepath, 0.0f, false);
            if (viewing_distance_inches > 0.0f) {
                model.viewing_distance_meters = viewing_distance_inches * 0.0254f;
            }
        }

        model.diagonal_size_inches = get_float(entry, "diagonal_size_inches", filepath, 30.0f, false);
        if (model.viewing_distance_meters <= 0.0f) {
            const float aspect = static_cast<float>(model.width) / std::max(static_cast<float>(model.height), 1.0f);
            const float height_inches = model.diagonal_size_inches / std::sqrt(1.0f + aspect * aspect);
            model.viewing_distance_meters = height_inches * 2.0f * 0.0254f;
        }

        model.max_luminance = get_float(entry, "max_luminance", filepath, 200.0f, false);
        model.min_luminance = get_float(entry, "min_luminance", filepath, model.max_luminance / 1000.0f, false);
        model.contrast = get_float(entry, "contrast", filepath, 1000.0f, false);
        model.E_ambient = get_float(entry, "E_ambient", filepath, 0.0f, false);
        model.fov_diagonal = get_float(entry, "fov_diagonal", filepath, 0.0f, false);

        return model;
    }
};

} // namespace cvvdp

