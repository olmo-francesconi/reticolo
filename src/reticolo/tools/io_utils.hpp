/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tools/io_utils.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <array>
#include <cstddef>
#include <format>
#include <sstream>
#include <string>

#include "reticolo/tools/timer.hpp"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"

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

const size_t KILO = 1024;  // Definition of what a Kilo of stuff is (Decimal: 1000, Binary: 1024)
/* Returns a std::string representing the byte size in a convenient unit*/
inline auto pretty_bytes(size_t bytes) -> std::string {
    std::array<std::string, 7> Suffixes({"B", "KB", "MB", "GB", "TB", "PB", "EB"});

    uint   SuffixIndex = 0;  // Pretty suffix index
    double Count = bytes;    // Pretty value
    while (Count >= KILO && SuffixIndex < Suffixes.size()) {
        SuffixIndex++;
        Count /= KILO;
    }
    return std::format("{:>6.2f}", Count) + " " + Suffixes[SuffixIndex];
}

/*--------------------------------------------------------------------------------------------------
    Logging helper functions
--------------------------------------------------------------------------------------------------*/

/* Default reticolo log line init with timing */
inline auto LI_time() -> std::string {
    std::string Message = "reticolo..." + std::format("{:.>10.3f}", GlobalTimer.elapsed_s()) + " s | ";
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
auto print(T val) -> std::string {
    return std::format("{:+8e}", val);
}

/* Print Complex numbers in standard format (dual signed 8 digits scientific) */
template <ComplexValue T>
auto print(T val) -> std::string {
    return std::format("{:+8e}{:+8e}I", val.real(), val.imag());
}

/* Print uint Vectors in standard format */
template <size_t dim>
inline auto print(const uintvect<dim>& Vect) -> std::string {
    std::stringstream Res;
    Res << "[" << Vect[0];
    for (int Comp = 1; Comp < dim; Comp++) {
        Res << " x " << Vect[Comp];
    }
    Res << "]";
    return Res.str();
}

}  // namespace reticolo::IO
