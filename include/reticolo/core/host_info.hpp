#pragma once

// Runtime host/environment information for the banner. Unlike
// <reticolo/core/build_info.hpp> (all compile-time), everything here is read
// live from the machine the app is running on — hostname, CPU brand, and the
// number of logical cores the process can see. Host-only; no device code.

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <thread>

#if defined(__APPLE__)
    #include <sys/sysctl.h>
#endif
#if defined(__linux__)
    #include <fstream>
#endif
#if !defined(_WIN32)
    #include <unistd.h>
#endif

namespace reticolo::host {

inline constexpr std::size_t str_buf = 256;

inline std::string name() {
#if defined(_WIN32)
    return "unknown";
#else
    std::array<char, str_buf> buf{};
    if (::gethostname(buf.data(), buf.size()) == 0) {
        return std::string{buf.data()};
    }
    return "unknown";
#endif
}

// Logical cores visible to the process (hyperthreads counted). Never 0 —
// hardware_concurrency() is allowed to fail, in which case we report 1.
inline unsigned logical_cores() {
    unsigned const n = std::thread::hardware_concurrency();
    return n == 0 ? 1U : n;
}

inline std::string cpu_brand() {
#if defined(__APPLE__)
    std::array<char, str_buf> buf{};
    std::size_t sz = buf.size();
    if (::sysctlbyname("machdep.cpu.brand_string", buf.data(), &sz, nullptr, 0) == 0) {
        return std::string{buf.data()};
    }
    return "unknown CPU";
#elif defined(__linux__)
    std::ifstream f{"/proc/cpuinfo"};
    for (std::string line; std::getline(f, line);) {
        if (line.rfind("model name", 0) != 0) {
            continue;
        }
        auto const colon = line.find(':');
        if (colon == std::string::npos) {
            break;
        }
        auto const rhs   = std::string_view{line}.substr(colon + 1);
        auto const start = rhs.find_first_not_of(" \t");
        return start == std::string_view::npos ? std::string{"unknown CPU"}
                                               : std::string{rhs.substr(start)};
    }
    return "unknown CPU";
#else
    return "unknown CPU";
#endif
}

}  // namespace reticolo::host
