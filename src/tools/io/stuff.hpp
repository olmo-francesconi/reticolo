#pragma once

#include "tools/types.hpp"

#include <string>
#include <format>
#include <array>

namespace reticolo
{
    namespace IO
    {
        inline std::string pretty_welcome()
        {
            // clang-format off
            const std::string welcome_logo =
                R"(________________________________________________________________________________)" "\n"
                R"(                                                                                )" "\n"
                R"(         ██████╗ ███████╗████████╗██╗ ██████╗ ██████╗ ██╗      ██████╗          )" "\n"
                R"(         ██╔══██╗██╔════╝╚══██╔══╝██║██╔════╝██╔═══██╗██║     ██╔═══██╗         )" "\n"
                R"(         ██████╔╝█████╗     ██║   ██║██║     ██║   ██║██║     ██║   ██║         )" "\n"
                R"(         ██╔══██╗██╔══╝     ██║   ██║██║     ██║   ██║██║     ██║   ██║         )" "\n"
                R"(         ██║  ██║███████╗   ██║   ██║╚██████╗╚██████╔╝███████╗╚██████╔╝         )" "\n"
                R"(         ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝ ╚═════╝          )" "\n"
                R"(________________________________________________________________________________)" "\n";                                                                        
            //clang-format on
            return welcome_logo;
        };

        inline std::string pretty_bytes(size_t bytes)
        {
            std::array<std::string, 7> suffixes({"B", "KB", "MB", "GB", "TB", "PB", "EB"});
            uint s = 0; // which suffix to use
            double count = bytes;
            while (count >= 1024 && s < 7)
            {
                s++;
                count /= 1024;
            }
            return std::format("{:>6.2f}", count) + " " + suffixes[s];
        };
    } // namespace IO
} // namespace reticolo