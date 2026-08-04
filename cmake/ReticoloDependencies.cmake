# External dependencies, in one place.
#
#   HDF5    — system package, only when RETICOLO_BUILD_IO (reticolo::io owns it)
#   OpenMP  — system package, optional: absent → the build degrades to serial
#   cxxopts — fetched, header-only, used by reticolo::cli
#   sleef   — fetched, vector libm for the transcendental action hot loops
#
# Fetched deps are declared SYSTEM so their headers never trip our warning
# policy, and their own tests/examples/extras are switched off.
#
# These are macro(), not function(), on purpose: find_package() and
# FetchContent_MakeAvailable() define their result variables (HDF5_VERSION,
# OpenMP_CXX_VERSION, sleef_BINARY_DIR, ...) in the calling scope. A function
# would swallow them and the top level would silently see empty strings.

include(FetchContent)

macro(reticolo_find_hdf5)
    find_package(HDF5 REQUIRED COMPONENTS C)
endmacro()

# Optional OpenMP as an INTERFACE target. The `#pragma omp` in core headers is
# inert without -fopenmp, so a toolchain that lacks OpenMP (notably Apple Clang)
# degrades to serial rather than failing the configure — which is what lets a
# standalone consumer build reticolo with whatever compiler it has.
macro(reticolo_find_openmp)
    find_package(OpenMP COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        add_library(reticolo_openmp INTERFACE)
        add_library(reticolo::openmp ALIAS reticolo_openmp)
        target_link_libraries(reticolo_openmp INTERFACE OpenMP::OpenMP_CXX)
    else()
        message(STATUS "reticolo: OpenMP not found — building without it (serial)")
        set(RETICOLO_ENABLE_OPENMP OFF)
    endif()
endmacro()

macro(reticolo_fetch_cxxopts)
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
endmacro()

# Sleef: portable vector libm (sin/cos/exp) for the transcendental action hot
# loops (SineGordon, XY, CompactU1). Only the main vector lib is needed — DFT,
# quad, GNU ABI shims, the scalar-only lib and the tests are all off.
#
# NOTE: this deliberately sets SLEEF_BUILD_SHARED_LIBS only. An earlier version
# also forced the GLOBAL `BUILD_SHARED_LIBS` to OFF as CACHE INTERNAL, which
# silently imposed static linkage on the whole project and on any parent that
# add_subdirectory'd reticolo. Sleef's own switch is sufficient.
macro(reticolo_fetch_sleef)
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
    FetchContent_MakeAvailable(sleef)
endmacro()
