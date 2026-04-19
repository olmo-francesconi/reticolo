/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tools/io_utils.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

******************************************************************************/

#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <format>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "reticolo/core/tools/timer.hpp"
#include "reticolo/core/types/complex.hpp"
#include "reticolo/core/types/real.hpp"

namespace reticolo::IO {

/*--------------------------------------------------------------------------------------------------
    Styling
--------------------------------------------------------------------------------------------------*/

/* Returns the reticolo Welcome screen as a std::string */
inline auto pretty_welcome() -> std::string {
    // clang-format off
    const std::string WelcomeLogo =
        R"(________________________________________________________________________________)" "\n"
        R"(                                                                                )" "\n"
        R"(         ██████╗ ███████╗████████╗██╗ ██████╗ ██████╗ ██╗      ██████╗          )" "\n"
        R"(         ██╔══██╗██╔════╝╚══██╔══╝██║██╔════╝██╔═══██╗██║     ██╔═══██╗         )" "\n"
        R"(         ██████╔╝█████╗     ██║   ██║██║     ██║   ██║██║     ██║   ██║         )" "\n"
        R"(         ██╔══██╗██╔══╝     ██║   ██║██║     ██║   ██║██║     ██║   ██║         )" "\n"
        R"(         ██║  ██║███████╗   ██║   ██║╚██████╗╚██████╔╝███████╗╚██████╔╝         )" "\n"
        R"(         ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝ ╚═════╝          )" "\n"
        R"(________________________________________________________________________________)" "\n";
    // clang-format on
    return WelcomeLogo;
}

/* Returns a std::string representing the byte size in a convenient unit*/
inline auto pretty_bytes(size_t bytes) -> std::string {
    std::array<std::string, 7> Suffixes({" B", "KB", "MB", "GB", "TB", "PB", "EB"});

    std::size_t SuffixIndex = 0;  // Pretty suffix index
    double      Count = bytes;    // Pretty value
    while (Count >= 1024 && (SuffixIndex + 1) < Suffixes.size()) {
        SuffixIndex++;
        Count /= 1024;
    }
    return std::format("{:>7.2f}", Count) + " " + Suffixes[SuffixIndex];
}

/*--------------------------------------------------------------------------------------------------
    Logging helper functions
--------------------------------------------------------------------------------------------------*/

/* Default reticolo log line init with timing */
inline auto LI_time() -> std::string {
    auto        Time = std::chrono::duration<double>(GlobalTimer.elapsed_s());
    std::string Message = "reticolo......." + std::format("{:%T}", Time) + " | ";
    return Message;
}

/* Default reticolo log line init with dots */
inline auto LI_dots() -> std::string {
    std::string Message = "reticolo............... | ";
    return Message;
}

/* Default reticolo log line empty init */
inline auto LI_void() -> std::string {
    std::string Message = "                        | ";
    return Message;
}

/* Default reticolo log error line init */
inline auto LI_erro() -> std::string {
    std::string Message = "reticolo..........ERROR | ";
    return Message;
}

/* Default reticolo log warning line init */
inline auto LI_warn() -> std::string {
    std::string Message = "reticolo........WARNING | ";
    return Message;
}

/*--------------------------------------------------------------------------------------------------
    Generic print() functions for the various types
--------------------------------------------------------------------------------------------------*/

/* Print Reals in standard format (signed 8 digits scientific) */
template <RealValue T>
inline auto print(T val) -> std::string {
    return std::format("{:+8e}", val);
}

/* Print Complex numbers in standard format (dual signed 8 digits scientific) */
template <ComplexValue T>
inline auto print(T val) -> std::string {
    return std::format("{:+8e} {:+8e}I", val.real(), val.imag());
}

inline auto print(std::size_t val) -> std::string { return std::format("{:}", val); }
template <typename T>
    requires std::is_integral_v<T>
inline auto print(T val) -> std::string {
    return std::format("{:}", val);
}

/* Print Vectors in standard format */
template <typename T>
inline auto print(const std::vector<T>& Vect) -> std::string {
    std::stringstream Res;
    Res << "[" << print(Vect[0]);
    for (std::size_t Comp = 1; Comp < Vect.size(); Comp++) {
        Res << ", " << print(Vect[Comp]);
    }
    Res << "]";
    return Res.str();
}

}  // namespace reticolo::IO
