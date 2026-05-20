#pragma once

// Compile-time build information. Consumed by `log::banner()` and anything
// else that wants to stamp a run with the build it came from. Values come
// from a mix of CMake-injected `target_compile_definitions` and predefined
// compiler macros — everything resolves at compile time, no runtime IO.

#include <string_view>

#ifndef RETICOLO_VERSION
    #define RETICOLO_VERSION "unknown"
#endif
#ifndef RETICOLO_GIT_COMMIT
    #define RETICOLO_GIT_COMMIT "unknown"
#endif
#ifndef RETICOLO_GIT_BRANCH
    #define RETICOLO_GIT_BRANCH "unknown"
#endif

namespace reticolo::build {

inline constexpr std::string_view version    = RETICOLO_VERSION;
inline constexpr std::string_view git_commit = RETICOLO_GIT_COMMIT;
inline constexpr std::string_view git_branch = RETICOLO_GIT_BRANCH;

inline constexpr std::string_view compiler =
#if defined(__clang__)
    "clang " __clang_version__
#elif defined(__GNUC__)
    "gcc " __VERSION__
#elif defined(_MSC_VER)
    "msvc"
#else
    "unknown"
#endif
    ;

inline constexpr std::string_view build_type =
#ifdef NDEBUG
    "release"
#else
    "debug"
#endif
    ;

inline constexpr std::string_view simd =
#if defined(__AVX512F__)
    "AVX-512"
#elif defined(__AVX2__)
    "AVX2"
#elif defined(__AVX__)
    "AVX"
#elif defined(__SSE4_2__)
    "SSE4.2"
#elif defined(__ARM_FEATURE_SVE)
    "SVE"
#elif defined(__ARM_NEON)
    "NEON"
#else
    "scalar"
#endif
    ;

inline constexpr bool openmp_enabled =
#ifdef _OPENMP
    true
#else
    false
#endif
    ;

}  // namespace reticolo::build
