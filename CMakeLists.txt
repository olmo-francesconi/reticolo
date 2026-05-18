cmake_minimum_required(VERSION 3.25)

project(
    reticolo
    VERSION 0.3.0
    DESCRIPTION "Lightweight serial C++23 library for lattice quantum field theory MCMC"
    LANGUAGES CXX
)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RETICOLO_BUILD_APPS         "Build reference apps"                ON)
option(RETICOLO_BUILD_TESTS        "Build test suite"                    ON)
option(RETICOLO_BUILD_IO           "Build the HDF5-backed io subsystem"  ON)
option(RETICOLO_WARNINGS_AS_ERRORS "Treat warnings as errors"            OFF)
option(RETICOLO_TUNE_NATIVE        "Build with -march=native"            ON)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(ReticoloWarnings)

if(RETICOLO_BUILD_IO)
    find_package(HDF5 REQUIRED COMPONENTS C)
endif()

include(FetchContent)
FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG        v3.2.1
    GIT_SHALLOW    TRUE
    SYSTEM
)
set(CXXOPTS_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(CXXOPTS_BUILD_TESTS    OFF CACHE INTERNAL "")
set(CXXOPTS_ENABLE_INSTALL OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(cxxopts)

# Sleef: portable vector libm (sin/cos/exp). Used by the transcendental action
# hot loops (SineGordon, XY, CompactU1). We only need the main vector lib —
# disable DFT, quad, GNU ABI shims, scalar-only lib, and tests.
FetchContent_Declare(
    sleef
    GIT_REPOSITORY https://github.com/shibatch/sleef.git
    GIT_TAG        3.6.1
    GIT_SHALLOW    TRUE
    SYSTEM
)
set(SLEEF_BUILD_DFT            OFF CACHE INTERNAL "")
set(SLEEF_BUILD_QUAD           OFF CACHE INTERNAL "")
set(SLEEF_BUILD_GNUABI_LIBS    OFF CACHE INTERNAL "")
set(SLEEF_BUILD_SCALAR_LIB     OFF CACHE INTERNAL "")
set(SLEEF_BUILD_INLINE_HEADERS OFF CACHE INTERNAL "")
set(SLEEF_BUILD_TESTS          OFF CACHE INTERNAL "")
set(SLEEF_BUILD_LIBM           ON  CACHE INTERNAL "")
set(SLEEF_BUILD_SHARED_LIBS    OFF CACHE INTERNAL "")
set(BUILD_SHARED_LIBS          OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(sleef)

execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE  RETICOLO_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE  _reticolo_git_rc
    ERROR_QUIET
)
if(NOT _reticolo_git_rc EQUAL 0)
    set(RETICOLO_GIT_COMMIT "unknown")
endif()

# ---------------------------------------------------------------------------
# reticolo::core  (header-only)
# ---------------------------------------------------------------------------
add_library(reticolo_core INTERFACE)
add_library(reticolo::core ALIAS reticolo_core)

target_compile_features(reticolo_core INTERFACE cxx_std_23)
target_include_directories(reticolo_core INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
# Sleef's exported target doesn't propagate its build-tree include dir to
# consumers (it only sets it on install). Pin the generated <sleef.h> path
# explicitly so the action hot loops can include it during the build.
target_include_directories(reticolo_core SYSTEM INTERFACE
    $<BUILD_INTERFACE:${sleef_BINARY_DIR}/include>
)
target_link_libraries(reticolo_core INTERFACE sleef)

string(TOUPPER "${CMAKE_BUILD_TYPE}" _reticolo_build_type_upper)
set(_reticolo_compile_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_reticolo_build_type_upper}}")
string(STRIP "${_reticolo_compile_flags}" _reticolo_compile_flags)

target_compile_definitions(reticolo_core INTERFACE
    RETICOLO_VERSION="${PROJECT_VERSION}"
    RETICOLO_GIT_COMMIT="${RETICOLO_GIT_COMMIT}"
    RETICOLO_COMPILE_FLAGS="${_reticolo_compile_flags}"
)

target_compile_options(reticolo_core INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-fno-math-errno>
)

if(RETICOLO_TUNE_NATIVE)
    target_compile_options(reticolo_core INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-march=native>
    )
endif()

reticolo_configure_warnings(reticolo_core)

# ---------------------------------------------------------------------------
# reticolo::io  (static; owns libhdf5)
# ---------------------------------------------------------------------------
if(RETICOLO_BUILD_IO)
    add_subdirectory(src/io)
endif()

# ---------------------------------------------------------------------------
# reticolo::cli  (header-only; wraps cxxopts; needs reticolo::io for stamping)
# ---------------------------------------------------------------------------
if(RETICOLO_BUILD_IO)
    add_subdirectory(src/cli)
endif()

# ---------------------------------------------------------------------------
# Lint amalgamation TU (CI-only — see src/lint/CMakeLists.txt for rationale).
# Only registered if we have the umbrella's transitive deps.
# ---------------------------------------------------------------------------
if(RETICOLO_BUILD_IO)
    add_subdirectory(src/lint)
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

message(STATUS "reticolo: version       ${PROJECT_VERSION}")
message(STATUS "reticolo: commit        ${RETICOLO_GIT_COMMIT}")
message(STATUS "reticolo: build type    ${CMAKE_BUILD_TYPE}")
message(STATUS "reticolo: C++ standard  ${CMAKE_CXX_STANDARD}")
message(STATUS "reticolo: native tuning ${RETICOLO_TUNE_NATIVE}")
if(RETICOLO_BUILD_IO)
    message(STATUS "reticolo: HDF5          ${HDF5_VERSION}")
endif()
