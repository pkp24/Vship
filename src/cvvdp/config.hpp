#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include "../util/VshipExceptions.hpp"

// Simple JSON parser for CVVDP configuration files
namespace cvvdp {

// Utility functions for parsing JSON (simplified - only handles what we need)
inline std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, (last - first + 1));
}

inline std::string extractString(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos = json.find(":", pos);
    if (pos == std::string::npos) return "";

    pos = json.find("\"", pos);
    if (pos == std::string::npos) return "";

    size_t end = json.find("\"", pos + 1);
    if (end == std::string::npos) return "";

    return json.substr(pos + 1, end - pos - 1);
}

inline float extractFloat(const std::string& json, const std::string& key, float defaultValue = 0.0f) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return defaultValue;

    pos = json.find(":", pos);
    if (pos == std::string::npos) return defaultValue;

    // Skip whitespace
    pos++;
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n')) pos++;

    // Find end of number
    size_t end = pos;
    while (end < json.length() && (isdigit(json[end]) || json[end] == '.' || json[end] == '-' || json[end] == 'e' || json[end] == 'E' || json[end] == '+')) end++;

    if (end == pos) return defaultValue;

    return std::stof(json.substr(pos, end - pos));
}

inline int extractInt(const std::string& json, const std::string& key, int defaultValue = 0) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return defaultValue;

    pos = json.find(":", pos);
    if (pos == std::string::npos) return defaultValue;

    // Skip whitespace
    pos++;
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n')) pos++;

    // Find end of number
    size_t end = pos;
    while (end < json.length() && (isdigit(json[end]) || json[end] == '-')) end++;

    if (end == pos) return defaultValue;

    return std::stoi(json.substr(pos, end - pos));
}

inline std::vector<float> extractFloatArray(const std::string& json, const std::string& key) {
    std::vector<float> result;
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return result;

    pos = json.find(":", pos);
    if (pos == std::string::npos) return result;

    pos = json.find("[", pos);
    if (pos == std::string::npos) return result;

    size_t end = json.find("]", pos);
    if (end == std::string::npos) return result;

    std::string arrayContent = json.substr(pos + 1, end - pos - 1);
    size_t numPos = 0;

    while (numPos < arrayContent.length()) {
        // Skip whitespace and commas
        while (numPos < arrayContent.length() && (arrayContent[numPos] == ' ' || arrayContent[numPos] == ',' || arrayContent[numPos] == '\t' || arrayContent[numPos] == '\n')) numPos++;

        if (numPos >= arrayContent.length()) break;

        // Find end of number
        size_t numEnd = numPos;
        while (numEnd < arrayContent.length() && (isdigit(arrayContent[numEnd]) || arrayContent[numEnd] == '.' || arrayContent[numEnd] == '-' || arrayContent[numEnd] == 'e' || arrayContent[numEnd] == 'E' || arrayContent[numEnd] == '+')) numEnd++;

        if (numEnd > numPos) {
            result.push_back(std::stof(arrayContent.substr(numPos, numEnd - numPos)));
            numPos = numEnd;
        } else {
            break;
        }
    }

    return result;
}

inline std::vector<int> extractIntArray(const std::string& json, const std::string& key) {
    std::vector<int> result;
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return result;

    pos = json.find(":", pos);
    if (pos == std::string::npos) return result;

    pos = json.find("[", pos);
    if (pos == std::string::npos) return result;

    size_t end = json.find("]", pos);
    if (end == std::string::npos) return result;

    std::string arrayContent = json.substr(pos + 1, end - pos - 1);
    size_t numPos = 0;

    while (numPos < arrayContent.length()) {
        // Skip whitespace and commas
        while (numPos < arrayContent.length() && (arrayContent[numPos] == ' ' || arrayContent[numPos] == ',' || arrayContent[numPos] == '\t' || arrayContent[numPos] == '\n')) numPos++;

        if (numPos >= arrayContent.length()) break;

        // Find end of number
        size_t numEnd = numPos;
        while (numEnd < arrayContent.length() && (isdigit(arrayContent[numEnd]) || arrayContent[numEnd] == '-')) numEnd++;

        if (numEnd > numPos) {
            result.push_back(std::stoi(arrayContent.substr(numPos, numEnd - numPos)));
            numPos = numEnd;
        } else {
            break;
        }
    }

    return result;
}

inline std::string loadFileToString(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw VshipError(FileNotFound, __FILE__, __LINE__);
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    return content;
}

// CVVDP Parameters structure
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

    static CVVDPParameters load(const std::string& filepath) {
        std::string json = loadFileToString(filepath);
        CVVDPParameters params;

        params.version = extractString(json, "version");
        params.mask_p = extractFloat(json, "mask_p");
        params.mask_c = extractFloat(json, "mask_c");
        params.pu_dilate = extractInt(json, "pu_dilate");
        params.beta = extractFloat(json, "beta");
        params.beta_t = extractFloat(json, "beta_t");
        params.beta_tch = extractFloat(json, "beta_tch");
        params.beta_sch = extractFloat(json, "beta_sch");
        params.csf_sigma = extractFloat(json, "csf_sigma");
        params.sensitivity_correction = extractFloat(json, "sensitivity_correction");
        params.masking_model = extractString(json, "masking_model");
        params.local_adapt = extractString(json, "local_adapt");
        params.contrast = extractString(json, "contrast");
        params.csf = extractString(json, "csf");
        params.jod_a = extractFloat(json, "jod_a");
        params.jod_exp = extractFloat(json, "jod_exp");
        params.mask_q = extractFloatArray(json, "mask_q");
        params.filter_len = extractInt(json, "filter_len", -1);
        params.ch_chrom_w = extractFloat(json, "ch_chrom_w");
        params.ch_trans_w = extractFloat(json, "ch_trans_w");
        params.sigma_tf = extractFloatArray(json, "sigma_tf");
        params.beta_tf = extractFloatArray(json, "beta_tf");
        params.xchannel_masking = extractString(json, "xchannel_masking");
        params.xcm_weights = extractFloatArray(json, "xcm_weights");
        params.baseband_weight = extractFloatArray(json, "baseband_weight");
        params.dclamp_type = extractString(json, "dclamp_type");
        params.d_max = extractFloat(json, "d_max");
        params.image_int = extractFloat(json, "image_int");
        params.Bloch_int = extractString(json, "Bloch_int");
        params.bfilt_duration = extractFloat(json, "bfilt_duration");

        return params;
    }
};

// Display model structure
struct DisplayModel {
    std::string name;
    std::string colorspace;
    int width;
    int height;
    float viewing_distance_meters;
    float diagonal_size_inches;
    float max_luminance;
    float min_luminance;
    float contrast;
    float E_ambient;
    float fov_diagonal;

    float get_ppd() const {
        // Calculate pixels per degree
        float diagonal_pixels = sqrtf(width * width + height * height);
        float diagonal_meters = diagonal_size_inches * 0.0254f; // inches to meters
        float fov_rad = 2.0f * atanf(diagonal_meters / (2.0f * viewing_distance_meters));
        float fov_deg = fov_rad * 180.0f / M_PI;
        return diagonal_pixels / fov_deg;
    }

    float get_black_level() const {
        // Calculate effective black level from contrast and ambient light
        float Lblack = max_luminance / contrast;
        if (E_ambient > 0) {
            float k_refl = 0.01f; // Default reflectance coefficient
            Lblack += k_refl * E_ambient / M_PI;
        }
        return Lblack;
    }

    static DisplayModel load(const std::string& filepath, const std::string& display_name) {
        std::string json = loadFileToString(filepath);

        // Find the display model section
        std::string search = "\"" + display_name + "\"";
        size_t pos = json.find(search);
        if (pos == std::string::npos) {
            throw VshipError(ConfigurationError, __FILE__, __LINE__);
        }

        // Find the opening brace for this display
        size_t start = json.find("{", pos);
        if (start == std::string::npos) {
            throw VshipError(ConfigurationError, __FILE__, __LINE__);
        }

        // Find the matching closing brace (simplified - assumes no nested objects)
        size_t end = json.find("}", start);
        if (end == std::string::npos) {
            throw VshipError(ConfigurationError, __FILE__, __LINE__);
        }

        std::string displayJson = json.substr(start, end - start + 1);

        DisplayModel model;
        model.name = extractString(displayJson, "name");
        model.colorspace = extractString(displayJson, "colorspace");
        if (model.colorspace.empty()) {
            model.colorspace = "BT.709-sRGB"; // Default for SDR
        }

        std::vector<int> resolution = extractIntArray(displayJson, "resolution");
        if (resolution.size() >= 2) {
            model.width = resolution[0];
            model.height = resolution[1];
        } else {
            model.width = 3840;
            model.height = 2160;
        }

        model.viewing_distance_meters = extractFloat(displayJson, "viewing_distance_meters");
        if (model.viewing_distance_meters == 0.0f) {
            // Try viewing_distance_inches
            float viewing_distance_inches = extractFloat(displayJson, "viewing_distance_inches");
            if (viewing_distance_inches > 0.0f) {
                model.viewing_distance_meters = viewing_distance_inches * 0.0254f;
            } else {
                // Calculate from display height (2x height is common)
                model.diagonal_size_inches = extractFloat(displayJson, "diagonal_size_inches", 30.0f);
                float aspect = (float)model.width / (float)model.height;
                float height_inches = model.diagonal_size_inches / sqrtf(1.0f + aspect * aspect);
                model.viewing_distance_meters = 2.0f * height_inches * 0.0254f;
            }
        }

        model.diagonal_size_inches = extractFloat(displayJson, "diagonal_size_inches", 30.0f);
        model.max_luminance = extractFloat(displayJson, "max_luminance", 200.0f);
        model.min_luminance = extractFloat(displayJson, "min_luminance");
        model.contrast = extractFloat(displayJson, "contrast", 1000.0f);
        model.E_ambient = extractFloat(displayJson, "E_ambient");
        model.fov_diagonal = extractFloat(displayJson, "fov_diagonal");

        return model;
    }
};

} // namespace cvvdp
