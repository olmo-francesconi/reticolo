# External dependencies — section 2 of the configure report.
#
#   HDF5    — system, only when RETICOLO_BUILD_IO (reticolo::io owns it)
#   OpenMP  — system, optional: absent → the build degrades to serial
#   cxxopts — fetched, header-only, used by reticolo::cli
#   sleef   — fetched, vector libm for the transcendental action hot loops
#   Catch2  — fetched, only when RETICOLO_BUILD_TESTS
#
# Fetched deps are declared SYSTEM so their headers never trip our warning
# policy, and their own tests/examples/extras are switched off. Each one prints
# a heading, lets its own chatter land indented underneath, and closes with a
# one-line verdict that also feeds the section recap.
#
# These are macro(), not function(), on purpose: find_package() and
# FetchContent_MakeAvailable() define their result variables (HDF5_VERSION,
# OpenMP_CXX_VERSION, sleef_BINARY_DIR, ...) in the calling scope. A function
# would swallow them and the top level would silently see empty strings.

include(FetchContent)

macro(reticolo_deps_begin)
    reticolo_rule()
    reticolo_line("2 · external dependencies")
endmacro()

macro(reticolo_dep_hdf5)
    if(RETICOLO_BUILD_IO)
        reticolo_dep_begin("HDF5" "system")
        find_package(HDF5 REQUIRED COMPONENTS C)
        reticolo_dep_end()
        reticolo_dep_record(ok "HDF5" "${HDF5_VERSION}" "system" "reticolo::io" "")
    else()
        reticolo_dep_begin("HDF5" "system")
        reticolo_dep_end()
        reticolo_dep_record(skip "HDF5" "" "system" "" "RETICOLO_BUILD_IO=OFF, no io and no cli")
    endif()
endmacro()

# The `#pragma omp` in core headers is inert without -fopenmp, so a toolchain
# that lacks OpenMP (notably Apple Clang) degrades to serial rather than failing
# the configure — which is what lets a standalone consumer build reticolo with
# whatever compiler it has.
macro(reticolo_dep_openmp)
    reticolo_dep_begin("OpenMP" "system")
    if(RETICOLO_ENABLE_OPENMP)
        find_package(OpenMP COMPONENTS CXX)
        reticolo_dep_end()
        if(OpenMP_CXX_FOUND)
            add_library(reticolo_openmp INTERFACE)
            add_library(reticolo::openmp ALIAS reticolo_openmp)
            target_link_libraries(reticolo_openmp INTERFACE OpenMP::OpenMP_CXX)
            reticolo_dep_record(ok "OpenMP" "${OpenMP_CXX_VERSION}" "system"
                                "reticolo::core" "3 threading layers live")
        else()
            set(RETICOLO_ENABLE_OPENMP OFF)
            reticolo_dep_record(skip "OpenMP" "" "system" ""
                                "not found for this toolchain, building serial")
        endif()
    else()
        reticolo_dep_end()
        reticolo_dep_record(skip "OpenMP" "" "system" ""
                            "RETICOLO_ENABLE_OPENMP=OFF, every omp pragma is inert")
    endif()
endmacro()

macro(reticolo_dep_cxxopts)
    reticolo_dep_begin("cxxopts" "fetched v3.2.1")
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
    reticolo_dep_end()
    reticolo_dep_record(ok "cxxopts" "v3.2.1" "fetched" "reticolo::cli" "")
endmacro()

# Sleef: portable vector libm (sin/cos/exp) for the transcendental action hot
# loops (SineGordon, XY, CompactU1). Only the main vector lib is needed — DFT,
# quad, GNU ABI shims, the scalar-only lib and the tests are all off.
#
# NOTE: this deliberately sets SLEEF_BUILD_SHARED_LIBS only. An earlier version
# also forced the GLOBAL `BUILD_SHARED_LIBS` to OFF as CACHE INTERNAL, which
# silently imposed static linkage on the whole project and on any parent that
# add_subdirectory'd reticolo. Sleef's own switch is sufficient.
macro(reticolo_dep_sleef)
    reticolo_dep_begin("sleef" "fetched 3.6.1")
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
    reticolo_dep_end()
    reticolo_dep_record(ok "sleef" "3.6.1" "fetched" "reticolo::core" "libm only")
endmacro()

# Fetched HERE rather than inside tests/CMakeLists.txt so that every external
# dependency is resolved (and reported) in one place, before any reticolo target
# is defined. tests/ just uses the Catch2::Catch2 target.
macro(reticolo_dep_catch2)
    if(RETICOLO_BUILD_TESTS)
        reticolo_dep_begin("Catch2" "fetched v3.5.4")
        FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG        v3.5.4
            GIT_SHALLOW    TRUE
            SYSTEM
        )
        FetchContent_MakeAvailable(Catch2)
        list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
        reticolo_dep_end()
        reticolo_dep_record(ok "Catch2" "v3.5.4" "fetched" "tests" "")
    endif()
endmacro()

macro(reticolo_resolve_dependencies)
    reticolo_deps_begin()
    reticolo_dep_hdf5()
    reticolo_dep_openmp()
    reticolo_dep_cxxopts()
    reticolo_dep_sleef()
    reticolo_dep_catch2()
    reticolo_dep_recap()
endmacro()
