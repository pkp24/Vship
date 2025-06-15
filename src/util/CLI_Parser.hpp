#pragma once

#include <map>
#include <variant>
#include <iomanip> 
#include <vector>

namespace helper{

struct ArgParser {
    using TargetVariant = std::variant<bool*, int*, std::string*>;

    struct FlagGroup {
        std::vector<std::string> aliases; TargetVariant target; std::string help_text;
    };

    std::vector<FlagGroup> flag_groups;
    std::map<std::string, TargetVariant> alias_map;

    ArgParser() {
        add_flag({"-h", "--help"}, &show_help_flag, "Display this help message");
    }

    template<typename TargetType>
    void add_flag(const std::vector<std::string>& flag_names, TargetType* target_pointer,
                  std::string help_description = "") {
        static_assert(std::is_same_v<TargetType, bool> || std::is_same_v<TargetType, int> ||
            std::is_same_v<TargetType, std::string>, "Unsupported flag type"
        );

        flag_groups.push_back({flag_names, target_pointer, help_description});
        const auto& group = flag_groups.back();
        for (const std::string& name : flag_names) alias_map[name] = group.target;
    }

    // Takes in a vector of string from the cli to parse, returns a 0 if sucessfull
    int parse_cli_args(const std::vector<std::string>& args) const {
        size_t current_arg_index = 0;
        while (current_arg_index < args.size()) {
            if (!parse_flag(args, current_arg_index)) {
                std::cerr << "Failed to parse argument: " << args[current_arg_index] << "\n";
                return 1;
            }
            ++current_arg_index;
        }
        if (show_help_flag || args.size() == 0) { print_help(); return 1; }
        return 0;
    }

private:
    mutable bool show_help_flag = false;

    // Parse a flag at index in 'arguments', returns false if unknown or invalid
    bool parse_flag(const std::vector<std::string>& arguments, size_t& index) const {
        const std::string& flag = arguments[index];
        auto found = alias_map.find(flag);
        if (found == alias_map.end()) {
            std::cerr << "Unknown argument: " << flag << "\n";
            return false;
        }
    
        if (std::holds_alternative<bool*>(found->second)) {
            bool* ptr = std::get<bool*>(found->second);
            *ptr = !*ptr;
            return true;
        }
    
        if (index + 1 >= arguments.size()) {
            std::cerr << flag << " requires an argument\n";
            return false;
        }
    
        if (std::holds_alternative<std::string*>(found->second)) {
            std::string* ptr = std::get<std::string*>(found->second);
            *ptr = arguments[++index];
            return true;
        }
    
        // Assume it is an int
        int* ptr = std::get<int*>(found->second);
        try {
            *ptr = std::stoi(arguments[++index]);
            return true;
        } catch (...) {
            std::cerr << "Invalid integer value: " << arguments[index] << " for arg "
                << flag << "\n";
            return false;
        }
    }

    void print_help() const {
        std::cout << "Available options:\n";

        size_t max_length = 0;
        std::vector<std::string> formatted_aliases;
        formatted_aliases.reserve(flag_groups.size());

        for (const auto& group : flag_groups) {
            std::string joined;
            for (size_t i = 0; i < group.aliases.size(); ++i) {
                joined += group.aliases[i];
                if (i + 1 < group.aliases.size()) joined += ", ";
            }
            max_length = std::max(max_length, joined.size());
            formatted_aliases.push_back(std::move(joined));
        }

        for (size_t i = 0; i < flag_groups.size(); ++i) {
            std::cout << "  " << std::left << std::setw(static_cast<int>(max_length))
                      << formatted_aliases[i] << "  " << flag_groups[i].help_text << "\n";
        }
    }
};

} //namespace