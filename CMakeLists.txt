cmake_minimum_required(VERSION 3.25)

project(
    reticolo
    VERSION 0.5.0
    DESCRIPTION "C++20 library for lattice quantum field theory MCMC (OpenMP + optional CUDA)"
    LANGUAGES CXX
)

# This file is a table of contents. The substance lives in cmake/:
#   ReticoloOptions.cmake     options, compiler floor, configuration summary
#   ReticoloDependencies.cmake HDF5 / OpenMP / cxxopts / sleef
#   ReticoloWarnings.cmake    the project warning policy
#   ReticoloBuildStamp.cmake  build-time git stamping for the run record
#
# Four targets, and the dependency direction is one-way:
#   reticolo::core  INTERFACE, header-only        (never sees <hdf5.h> or CUDA)
#   reticolo::io    STATIC, the only HDF5 linker
#   reticolo::cli   INTERFACE, wraps cxxopts
#   reticolo::cuda  INTERFACE, header-only device stack (opt-in)
#   reticolo::reticolo  umbrella over whichever of the above are enabled

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(ReticoloOptions)
include(ReticoloDependencies)
include(ReticoloWarnings)
include(ReticoloBuildStamp)

reticolo_declare_options()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

reticolo_require_compiler()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
if(RETICOLO_BUILD_IO)
    reticolo_find_hdf5()
endif()
if(RETICOLO_ENABLE_OPENMP)
    reticolo_find_openmp()  # may switch RETICOLO_ENABLE_OPENMP off
endif()
reticolo_fetch_cxxopts()
reticolo_fetch_sleef()

# ---------------------------------------------------------------------------
# Build metadata
# ---------------------------------------------------------------------------
# Configure-time capture, used for the console banner and the compile-flags
# stamp. The AUTHORITATIVE per-run git stamp is regenerated at build time — see
# cmake/ReticoloBuildStamp.cmake — because these go stale the moment you commit.
function(_reticolo_git out_var)
    execute_process(
        COMMAND git ${ARGN}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE _out
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _rc
        ERROR_QUIET
    )
    if(NOT _rc EQUAL 0 OR _out STREQUAL "")
        set(_out "unknown")
    endif()
    set(${out_var} "${_out}" PARENT_SCOPE)
endfunction()

_reticolo_git(RETICOLO_GIT_COMMIT rev-parse --short HEAD)
_reticolo_git(RETICOLO_GIT_BRANCH rev-parse --abbrev-ref HEAD)

string(TOUPPER "${CMAKE_BUILD_TYPE}" _reticolo_build_type_upper)
set(_reticolo_compile_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_reticolo_build_type_upper}}")
string(STRIP "${_reticolo_compile_flags}" _reticolo_compile_flags)

# ---------------------------------------------------------------------------
# reticolo::core  (INTERFACE, header-only)
# ---------------------------------------------------------------------------
add_library(reticolo_core INTERFACE)
add_library(reticolo::core ALIAS reticolo_core)

target_compile_features(reticolo_core INTERFACE cxx_std_20)
target_include_directories(reticolo_core INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
# Sleef's exported target doesn't propagate its build-tree include dir to
# consumers (it only sets it on install). Pin the generated <sleef.h> path
# explicitly so the action hot loops can include it during the build.
target_include_directories(reticolo_core SYSTEM INTERFACE
    $<BUILD_INTERFACE:${sleef_BINARY_DIR}/include>
)
target_link_libraries(reticolo_core INTERFACE sleef)
# OpenMP propagates through the umbrella, so any app linking reticolo::core
# picks up `-fopenmp` and libomp without having to opt in per-app.
if(RETICOLO_ENABLE_OPENMP)
    target_link_libraries(reticolo_core INTERFACE reticolo::openmp)
endif()

target_compile_definitions(reticolo_core INTERFACE
    RETICOLO_VERSION="${PROJECT_VERSION}"
    RETICOLO_GIT_COMMIT="${RETICOLO_GIT_COMMIT}"
    RETICOLO_GIT_BRANCH="${RETICOLO_GIT_BRANCH}"
    RETICOLO_COMPILE_FLAGS="${_reticolo_compile_flags}"
)

target_compile_options(reticolo_core INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-fno-math-errno>
    # Pin FP contraction to per-expression (the C FP_CONTRACT default, and Clang's
    # default) so multiply-adds fuse identically across compilers and code paths.
    # GCC otherwise defaults to `fast` (contraction ACROSS statements), which makes
    # the tiled hot-loop kernels round ~1 ULP differently from the scalar gather
    # reference at data-dependent sites — breaking the cross-thread / cross-compiler
    # bit-reproducibility the HMC and LLR checkpoints rely on. Intra-expression FMA
    # (e.g. the SU(N) matrix kernels, written as single expressions) still contracts,
    # so this is perf-neutral for the compute-bound gauge paths. CXX-only: nvcc device
    # code uses -fmad, unaffected.
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=on>
)

# AVX-512 note: compilers default to 256-bit vectors even with -mavx512f
# (prefer-vector-width=256), so the gauge site-batch defaults to 8 lanes. For
# full 512-bit registers on an AVX-512 host, configure with
#   -DCMAKE_CXX_FLAGS="-mprefer-vector-width=512 -DRETICOLO_GAUGE_K_BATCH_MAX=16"
# (see include/reticolo/math/group/base.hpp).
if(RETICOLO_TUNE_NATIVE)
    target_compile_options(reticolo_core INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-march=native>
    )
endif()

# NOTE: the warning policy is deliberately NOT applied to reticolo_core. It is an
# INTERFACE target, so `reticolo_configure_warnings` would attach the flags with
# INTERFACE scope and export -Wall/-Wextra/-Wconversion/-Werror to every consumer
# — the standalone examples and any external project that links reticolo::core
# would be judged by OUR policy. Each compiled target in this repo calls
# reticolo_configure_warnings() on itself instead (src/io, src/lint, apps, tests).

# ---------------------------------------------------------------------------
# reticolo::io (STATIC; owns libhdf5) + reticolo::cli + the lint amalgamation
# ---------------------------------------------------------------------------
if(RETICOLO_BUILD_IO)
    add_subdirectory(src/io)
    add_subdirectory(src/cli)
    add_subdirectory(src/lint)  # CI-only; see src/lint/CMakeLists.txt
endif()

# ---------------------------------------------------------------------------
# reticolo::cuda  (INTERFACE; header-only — the device stack lives entirely in
# include/reticolo/cuda/*.{hpp,cuh}. nvcc compiles only the consumers that
# include it: apps/cuda + tests/cuda. CUDA headers must never leak into
# reticolo::core, mirroring how reticolo::io quarantines <hdf5.h>; here that is
# automatic, since cuda/ headers are only ever pulled into a .cu TU.)
# ---------------------------------------------------------------------------
if(RETICOLO_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 80 90)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(reticolo_cuda INTERFACE)
    add_library(reticolo::cuda ALIAS reticolo_cuda)
    target_include_directories(reticolo_cuda INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    )
    target_link_libraries(reticolo_cuda INTERFACE CUDA::cudart)
endif()

# ---------------------------------------------------------------------------
# reticolo::reticolo  (umbrella)
# ---------------------------------------------------------------------------
add_library(reticolo INTERFACE)
add_library(reticolo::reticolo ALIAS reticolo)
target_link_libraries(reticolo INTERFACE reticolo::core)
if(RETICOLO_BUILD_IO)
    target_link_libraries(reticolo INTERFACE reticolo::io reticolo::cli)
endif()
if(RETICOLO_ENABLE_CUDA)
    target_link_libraries(reticolo INTERFACE reticolo::cuda)
endif()

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
if(PROJECT_IS_TOP_LEVEL AND RETICOLO_BUILD_APPS AND EXISTS "${PROJECT_SOURCE_DIR}/apps/CMakeLists.txt")
    add_subdirectory(apps)
endif()

if(PROJECT_IS_TOP_LEVEL AND RETICOLO_BUILD_TESTS AND EXISTS "${PROJECT_SOURCE_DIR}/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(tests)
endif()

if(PROJECT_IS_TOP_LEVEL AND RETICOLO_BUILD_EXAMPLES AND EXISTS "${PROJECT_SOURCE_DIR}/examples/CMakeLists.txt")
    add_subdirectory(examples)
endif()

# Aggregate `cuda` target: every apps/cuda executable + cuda test exe, collected
# into RETICOLO_CUDA_TARGETS by the cuda helpers (apps/) and test registrations
# (tests/). The linux-nvcc build preset builds just this, so the GPU surface stays
# in sync automatically — no hand-picked target list to maintain.
if(RETICOLO_ENABLE_CUDA)
    get_property(_cuda_targets GLOBAL PROPERTY RETICOLO_CUDA_TARGETS)
    if(_cuda_targets)
        add_custom_target(cuda)
        add_dependencies(cuda ${_cuda_targets})
    endif()
endif()

reticolo_print_summary()
