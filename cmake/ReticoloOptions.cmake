# Build options and the configuration summary.
#
# Every option is OFF by default except RETICOLO_BUILD_IO / _ENABLE_OPENMP, so a
# bare or FetchContent'd configure builds core/io/cli only; the presets in
# CMakePresets.json turn the rest on for the dev/CI workflow.

macro(reticolo_declare_options)
    option(RETICOLO_BUILD_APPS         "Build reference apps"                   OFF)
    option(RETICOLO_BUILD_BENCHMARKS   "Build the bench_* suite (needs apps)"   OFF)
    option(RETICOLO_BUILD_EXAMPLES     "Build standalone example consumers"     OFF)
    option(RETICOLO_BUILD_TESTS        "Build test suite"                       OFF)
    option(RETICOLO_BUILD_IO           "Build the HDF5-backed io subsystem"     ON)
    option(RETICOLO_ENABLE_CUDA        "Build the CUDA backend (experimental)"  OFF)
    option(RETICOLO_ENABLE_OPENMP      "Enable OpenMP threading"                ON)
    option(RETICOLO_WARNINGS_AS_ERRORS "Treat warnings as errors"               OFF)
    option(RETICOLO_TUNE_NATIVE        "Build with -march=native"               ON)
endmacro()

# C++20 here means concepts, <numbers>, <bit>, three-way comparison and
# designated initialisers used throughout. Older toolchains do compile far
# enough to emit hundreds of lines of template diagnostics before failing;
# one clear message up front is cheaper.
function(reticolo_require_compiler)
    set(_min_gnu   10)
    set(_min_clang 13)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS _min_gnu)
        message(FATAL_ERROR
            "reticolo needs GCC >= ${_min_gnu} for C++20 (found ${CMAKE_CXX_COMPILER_VERSION})")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang"
           AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS _min_clang)
        message(FATAL_ERROR
            "reticolo needs Clang >= ${_min_clang} for C++20 (found ${CMAKE_CXX_COMPILER_VERSION})")
    endif()
endfunction()

function(reticolo_print_summary)
    # `-march=native` is not universal: nvcc rejects it on the host pass and it
    # is meaningless on some cross toolchains, which is why linux-nvcc turns
    # RETICOLO_TUNE_NATIVE off. Report what was actually applied, not what was
    # requested.
    message(STATUS "")
    message(STATUS "reticolo ${PROJECT_VERSION}  —  ${CMAKE_BUILD_TYPE}")
    message(STATUS "  compiler      ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} (C++${CMAKE_CXX_STANDARD})")
    message(STATUS "  commit        ${RETICOLO_GIT_COMMIT} (${RETICOLO_GIT_BRANCH})")
    message(STATUS "  native tuning ${RETICOLO_TUNE_NATIVE}")
    message(STATUS "  warn-as-error ${RETICOLO_WARNINGS_AS_ERRORS}")
    if(RETICOLO_BUILD_IO)
        message(STATUS "  HDF5          ${HDF5_VERSION}")
    else()
        message(STATUS "  HDF5          off (RETICOLO_BUILD_IO=OFF — no io, no cli)")
    endif()
    if(RETICOLO_ENABLE_OPENMP)
        message(STATUS "  OpenMP        ${OpenMP_CXX_VERSION}")
    else()
        message(STATUS "  OpenMP        off (serial)")
    endif()
    if(RETICOLO_ENABLE_CUDA)
        message(STATUS "  CUDA arch     ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
    message(STATUS "  components    apps=${RETICOLO_BUILD_APPS} bench=${RETICOLO_BUILD_BENCHMARKS} tests=${RETICOLO_BUILD_TESTS} examples=${RETICOLO_BUILD_EXAMPLES}")
    message(STATUS "")
endfunction()
