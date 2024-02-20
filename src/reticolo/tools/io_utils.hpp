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

#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"

namespace reticolo::IO {
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
};

const size_t KILO = 1024;
inline auto  pretty_bytes(size_t bytes) -> std::string {
    std::array<std::string, 7> Suffixes({"B", "KB", "MB", "GB", "TB", "PB", "EB"});
    uint                       SuffixIndex = 0;  // which suffix to use
    double                     Count = bytes;
    while (Count >= KILO && SuffixIndex < Suffixes.size()) {
        SuffixIndex++;
        Count /= KILO;
    }
    return std::format("{:>6.2f}", Count) + " " + Suffixes[SuffixIndex];
};

////////////////////////////////////////////////////////////
// Generic print() functions for the various types
////////////////////////////////////////////////////////////

// Print Reals in standard format (signed 8 digits scientific)
template <RealValue T>
auto print(T val) -> std::string {
    return std::format("{:+8e}", val);
}

// Print Complex numbers in standard format (dual signed 8 digits scientific)
template <ComplexValue T>
auto print(T val) -> std::string {
    return std::format("{:+8e}{:+8e}I", val.real(), val.imag());
}

// Print uint Vectors in standard format
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
