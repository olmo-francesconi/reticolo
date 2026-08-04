# Build options, the compiler floor, and the configure report sections.
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

function(_reticolo_opt_row name note)
    if(${name})
        set(_v "ON ")
    else()
        set(_v "OFF")
    endif()
    string(APPEND name "                                ")
    string(SUBSTRING "${name}" 0 30 _padded)
    reticolo_note("┃   ${_padded}${_v}    ${note}")
endfunction()

# Section 1 tail: host / toolchain facts, then the resolved option set.
# CMake does not expose the active preset name, but this repo's convention is
# build/<preset>, so the binary-dir basename is it.
function(reticolo_preset_name out_var)
    get_filename_component(_p "${PROJECT_BINARY_DIR}" NAME)
    set(${out_var} "${_p}" PARENT_SCOPE)
endfunction()

function(reticolo_report_system)
    cmake_host_system_information(RESULT _cores QUERY NUMBER_OF_LOGICAL_CORES)
    cmake_host_system_information(RESULT _cpu   QUERY PROCESSOR_DESCRIPTION)
    reticolo_preset_name(preset_name)
    file(RELATIVE_PATH _bin "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")

    reticolo_line("preset   : ${preset_name}")
    reticolo_line("source   : ${PROJECT_SOURCE_DIR}")
    reticolo_line("build    : ${_bin}  ·  ${CMAKE_BUILD_TYPE}")
    # No .git (a source export, or Modal/Kaggle shipping the working tree) makes
    # both fall back to "unknown"; say that once instead of twice.
    if(RETICOLO_GIT_COMMIT STREQUAL "unknown")
        reticolo_line("branch   : (not a git checkout — no commit metadata)")
    else()
        reticolo_line("branch   : ${RETICOLO_GIT_BRANCH} @ ${RETICOLO_GIT_COMMIT}")
    endif()
    reticolo_line("compiler : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}  ·  C++${CMAKE_CXX_STANDARD}")
    # PROCESSOR_DESCRIPTION is empty or "unknown" on some containers (gVisor),
    # which otherwise renders as "24 core unknown · 24 logical cores".
    if(_cpu STREQUAL "" OR _cpu MATCHES "unknown")
        set(_cpu_part "${_cores} logical cores")
    else()
        set(_cpu_part "${_cpu} · ${_cores} logical cores")
    endif()
    reticolo_line("host     : ${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_VERSION} ${CMAKE_HOST_SYSTEM_PROCESSOR}  ·  ${_cpu_part}")
    reticolo_line("cmake    : ${CMAKE_VERSION}  ·  ${CMAKE_GENERATOR}")
    reticolo_rule()
    reticolo_line("options")
    _reticolo_opt_row(RETICOLO_BUILD_IO           "HDF5-backed io subsystem")
    _reticolo_opt_row(RETICOLO_BUILD_APPS         "reference apps")
    _reticolo_opt_row(RETICOLO_BUILD_BENCHMARKS   "bench_* suite")
    _reticolo_opt_row(RETICOLO_BUILD_TESTS        "test suite")
    _reticolo_opt_row(RETICOLO_BUILD_EXAMPLES     "standalone example consumers")
    _reticolo_opt_row(RETICOLO_ENABLE_OPENMP      "OpenMP threading")
    _reticolo_opt_row(RETICOLO_ENABLE_CUDA        "device backend")
    _reticolo_opt_row(RETICOLO_TUNE_NATIVE        "-march=native")
    _reticolo_opt_row(RETICOLO_WARNINGS_AS_ERRORS "-Werror")
endfunction()

# Section 3: what this configure actually produced.
function(reticolo_report_components)
    reticolo_rule()
    reticolo_line("3 · reticolo components")
    reticolo_line("")
    if(RETICOLO_ENABLE_OPENMP)
        set(_core_note "sleef · OpenMP ${OpenMP_CXX_VERSION}")
    else()
        set(_core_note "sleef · serial")
    endif()
    reticolo_line("  reticolo::core      INTERFACE   header-only        ${_core_note}")
    if(RETICOLO_BUILD_IO)
        reticolo_line("  reticolo::io        STATIC      writer + reader    HDF5 ${HDF5_VERSION}")
        reticolo_line("  reticolo::cli       INTERFACE   cxxopts shim")
    else()
        reticolo_line("  reticolo::io        —           disabled (RETICOLO_BUILD_IO=OFF)")
        reticolo_line("  reticolo::cli       —           disabled (needs io)")
    endif()
    if(RETICOLO_ENABLE_CUDA)
        reticolo_line("  reticolo::cuda      INTERFACE   device stack       arch ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        reticolo_line("  reticolo::cuda      —           disabled")
    endif()
    reticolo_line("  reticolo::reticolo  INTERFACE   umbrella")
    reticolo_line("")
    if(RETICOLO_GIT_COMMIT STREQUAL "unknown")
        reticolo_line("  build stamp   no git metadata — /run@commit will read 'unknown'")
    else()
        reticolo_line("  build stamp   ${RETICOLO_GIT_COMMIT} (${RETICOLO_GIT_BRANCH})   → writer.cpp only")
    endif()
    if(RETICOLO_WARNINGS_AS_ERRORS)
        reticolo_line("  warnings      project policy + -Werror   (per-target, not exported)")
    else()
        reticolo_line("  warnings      project policy             (per-target, not exported)")
    endif()
    set(_flags "-ffp-contract=on -fno-math-errno")
    if(RETICOLO_TUNE_NATIVE)
        set(_flags "-march=native ${_flags}")
    endif()
    reticolo_line("  flags         ${_flags}")
endfunction()

# Section 4.
function(reticolo_report_final)
    if(RETICOLO_ENABLE_OPENMP)
        set(_thr "OpenMP ${OpenMP_CXX_VERSION}")
    else()
        set(_thr "serial")
    endif()
    reticolo_rule()
    reticolo_line("configured  reticolo ${PROJECT_VERSION} · ${CMAKE_BUILD_TYPE} · ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} · ${_thr}")
    reticolo_preset_name(_preset)
    reticolo_line("ready       cmake --build --preset ${_preset}")
    reticolo_close()
endfunction()
