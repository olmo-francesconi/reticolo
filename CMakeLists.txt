# Works with 3.21 and tested through 4.2
cmake_minimum_required(VERSION 3.21...4.2)

# Project name and a few useful settings. Other commands can pick up the results
project(
  reticolo
  VERSION 0.1
  DESCRIPTION "C++ library for lattice QFT calculations"
  LANGUAGES CXX)

include(FetchContent)
include(GNUInstallDirs)

# Export compilation commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(RETICOLO_BUILD_TESTS "Build the reticolo test suite" ON)
option(RETICOLO_ENABLE_OPENMP "Enable OpenMP support when available" ON)
option(RETICOLO_FETCH_DEPS "Allow CMake to fetch missing third-party dependencies" ON)
option(RETICOLO_WARNINGS_AS_ERRORS "Treat warnings as errors for reticolo targets" OFF)
option(RETICOLO_HDF5_NATIVE_COMPLEX "Use HDF5 2.x native complex datatypes for complex-valued fields and observables" OFF)
set(RETICOLO_COMPILER_PROFILE "AUTO" CACHE STRING "Compiler policy profile: AUTO, APPLE_CLANG, LLVM_CLANG, GCC, MSVC")
set_property(CACHE RETICOLO_COMPILER_PROFILE PROPERTY STRINGS AUTO APPLE_CLANG LLVM_CLANG GCC MSVC)

set(RETICOLO_YAML_CPP_TAG "0.8.0" CACHE STRING "Pinned yaml-cpp version for FetchContent fallback")
set(RETICOLO_CXXOPTS_TAG "v3.3.1" CACHE STRING "Pinned cxxopts version for FetchContent fallback")

function(reticolo_detect_compiler_profile out_var)
    if(RETICOLO_COMPILER_PROFILE AND NOT RETICOLO_COMPILER_PROFILE STREQUAL "AUTO")
        set(${out_var} "${RETICOLO_COMPILER_PROFILE}" PARENT_SCOPE)
        return()
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        set(${out_var} "APPLE_CLANG" PARENT_SCOPE)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(${out_var} "LLVM_CLANG" PARENT_SCOPE)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(${out_var} "GCC" PARENT_SCOPE)
    elseif(MSVC)
        set(${out_var} "MSVC" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Unsupported compiler '${CMAKE_CXX_COMPILER_ID}'. Set RETICOLO_COMPILER_PROFILE explicitly if needed.")
    endif()
endfunction()

function(reticolo_apply_compiler_policy profile_var)
    if(${profile_var} STREQUAL "APPLE_CLANG")
        message(STATUS "Compiler profile: APPLE_CLANG")
        message(STATUS "Apple Clang detected. OpenMP is not expected from the default Xcode toolchain; prefer the 'macos-llvm' preset for OpenMP-enabled builds.")
    elseif(${profile_var} STREQUAL "LLVM_CLANG")
        message(STATUS "Compiler profile: LLVM_CLANG")
        if(APPLE AND NOT DEFINED CMAKE_OSX_SYSROOT)
            execute_process(
                COMMAND xcrun --show-sdk-path
                OUTPUT_VARIABLE RETICOLO_MACOS_SDK_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(RETICOLO_MACOS_SDK_PATH)
                set(CMAKE_OSX_SYSROOT "${RETICOLO_MACOS_SDK_PATH}" CACHE PATH "macOS SDK path for non-Apple Clang toolchains" FORCE)
                message(STATUS "Using macOS SDK for LLVM Clang: ${CMAKE_OSX_SYSROOT}")
            endif()
        endif()
    elseif(${profile_var} STREQUAL "GCC")
        message(STATUS "Compiler profile: GCC")
    elseif(${profile_var} STREQUAL "MSVC")
        message(STATUS "Compiler profile: MSVC")
    else()
        message(FATAL_ERROR "Unknown compiler profile '${${profile_var}}'")
    endif()
endfunction()

reticolo_detect_compiler_profile(RETICOLO_ACTIVE_COMPILER_PROFILE)
reticolo_apply_compiler_policy(RETICOLO_ACTIVE_COMPILER_PROFILE)

if(APPLE AND RETICOLO_ACTIVE_COMPILER_PROFILE STREQUAL "LLVM_CLANG" AND CMAKE_OSX_SYSROOT)
    string(APPEND CMAKE_CXX_FLAGS " -isysroot ${CMAKE_OSX_SYSROOT}")
    string(APPEND CMAKE_C_FLAGS " -isysroot ${CMAKE_OSX_SYSROOT}")
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -isysroot ${CMAKE_OSX_SYSROOT}")
    string(APPEND CMAKE_SHARED_LINKER_FLAGS " -isysroot ${CMAKE_OSX_SYSROOT}")
    string(APPEND CMAKE_MODULE_LINKER_FLAGS " -isysroot ${CMAKE_OSX_SYSROOT}")
endif()

function(reticolo_configure_warnings target_name)
    if(RETICOLO_ACTIVE_COMPILER_PROFILE STREQUAL "MSVC")
        target_compile_options(${target_name} INTERFACE /W4)
        if(RETICOLO_WARNINGS_AS_ERRORS)
            target_compile_options(${target_name} INTERFACE /WX)
        endif()
    else()
        target_compile_options(${target_name} INTERFACE -Wall -Wextra -Wpedantic)
        if(RETICOLO_WARNINGS_AS_ERRORS)
            target_compile_options(${target_name} INTERFACE -Werror)
        endif()
        target_compile_options(${target_name} INTERFACE $<$<CONFIG:Release>:-ffast-math>)
        if(RETICOLO_ACTIVE_COMPILER_PROFILE STREQUAL "APPLE_CLANG" OR
           RETICOLO_ACTIVE_COMPILER_PROFILE STREQUAL "LLVM_CLANG")
            target_compile_options(${target_name} INTERFACE $<$<CONFIG:Release>:-Wno-nan-infinity-disabled>)
        endif()
    endif()
endfunction()

function(reticolo_find_or_fetch_yaml_cpp)
    find_package(yaml-cpp CONFIG QUIET)
    if(TARGET yaml-cpp::yaml-cpp)
        return()
    endif()

    if(EXISTS "${PROJECT_SOURCE_DIR}/build/_deps/yaml-cpp-src/CMakeLists.txt")
        message(STATUS "Using cached yaml-cpp sources from build/_deps/yaml-cpp-src")
        FetchContent_Declare(
            yaml-cpp
            SOURCE_DIR "${PROJECT_SOURCE_DIR}/build/_deps/yaml-cpp-src"
        )
        FetchContent_MakeAvailable(yaml-cpp)
        return()
    endif()

    if(NOT RETICOLO_FETCH_DEPS)
        message(FATAL_ERROR "yaml-cpp was not found. Install it or configure with -DRETICOLO_FETCH_DEPS=ON.")
    endif()

    FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG ${RETICOLO_YAML_CPP_TAG}
        OVERRIDE_FIND_PACKAGE
    )
    FetchContent_MakeAvailable(yaml-cpp)
endfunction()

function(reticolo_find_or_fetch_cxxopts)
    find_package(cxxopts CONFIG QUIET)
    if(TARGET cxxopts::cxxopts)
        return()
    endif()

    if(EXISTS "${PROJECT_SOURCE_DIR}/build/_deps/cxxopts-src/CMakeLists.txt")
        message(STATUS "Using cached cxxopts sources from build/_deps/cxxopts-src")
        FetchContent_Declare(
            cxxopts
            SOURCE_DIR "${PROJECT_SOURCE_DIR}/build/_deps/cxxopts-src"
        )
        FetchContent_MakeAvailable(cxxopts)
        return()
    endif()

    if(NOT RETICOLO_FETCH_DEPS)
        message(FATAL_ERROR "cxxopts was not found. Install it or configure with -DRETICOLO_FETCH_DEPS=ON.")
    endif()

    FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG ${RETICOLO_CXXOPTS_TAG}
        OVERRIDE_FIND_PACKAGE
    )
    FetchContent_MakeAvailable(cxxopts)
endfunction()

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # Let's ensure -std=c++xx instead of -std=g++xx
    set(CMAKE_CXX_EXTENSIONS OFF)

    # Let's nicely support folders in IDEs
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    # Testing only available if this is the main app
    # Note this needs to be done in the main CMakeLists
    # since it calls enable_testing, which must be in the
    # main CMakeLists.
    include(CTest)

    # Docs only available if this is the main app
    # find_package(Doxygen)
    # if(Doxygen_FOUND)
    #     add_subdirectory(docs)
    # else()
    #     message(STATUS "Doxygen not found, not building docs")
    # endif()
endif()

find_package(HDF5 REQUIRED COMPONENTS C)

if(DEFINED HDF5_VERSION)
    string(REPLACE "." ";" RETICOLO_HDF5_VERSION_PARTS "${HDF5_VERSION}")
    list(LENGTH RETICOLO_HDF5_VERSION_PARTS RETICOLO_HDF5_VERSION_PARTS_LEN)
    if(RETICOLO_HDF5_VERSION_PARTS_LEN GREATER 0)
        list(GET RETICOLO_HDF5_VERSION_PARTS 0 RETICOLO_HDF5_VERSION_MAJOR)
    endif()
    if(RETICOLO_HDF5_VERSION_PARTS_LEN GREATER 1)
        list(GET RETICOLO_HDF5_VERSION_PARTS 1 RETICOLO_HDF5_VERSION_MINOR)
    endif()
    if(RETICOLO_HDF5_VERSION_PARTS_LEN GREATER 2)
        list(GET RETICOLO_HDF5_VERSION_PARTS 2 RETICOLO_HDF5_VERSION_PATCH)
    endif()
    message(STATUS "HDF5 version: ${HDF5_VERSION}")
else()
    message(STATUS "HDF5 version: unknown")
endif()

if(RETICOLO_HDF5_NATIVE_COMPLEX)
    if(NOT DEFINED HDF5_VERSION_MAJOR OR HDF5_VERSION_MAJOR LESS 2)
        message(FATAL_ERROR "RETICOLO_HDF5_NATIVE_COMPLEX requires HDF5 2.x or newer.")
    endif()
    message(STATUS "HDF5 native complex mode enabled. This changes the on-disk complex datatype representation and is not backward-compatible with HDF5 1.x readers.")
else()
    message(STATUS "HDF5 native complex mode disabled. Reticolo will keep the legacy compound complex schema for compatibility.")
endif()

if(RETICOLO_ENABLE_OPENMP)
    find_package(OpenMP COMPONENTS CXX QUIET)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: enabling parallel support")
    else()
        if(RETICOLO_ACTIVE_COMPILER_PROFILE STREQUAL "APPLE_CLANG")
            message(STATUS "OpenMP not found for Apple Clang: building without OpenMP support. Use the 'macos-llvm' preset to target Homebrew LLVM/libomp.")
        else()
            message(STATUS "OpenMP not found: building without OpenMP support")
        endif()
    endif()
endif()

reticolo_find_or_fetch_yaml_cpp()
reticolo_find_or_fetch_cxxopts()

# The compiled library code is here
add_subdirectory(src)

# The executables code is here
add_subdirectory(apps)

# Testing only available if this is the main app
# Emergency override MODERN_CMAKE_BUILD_TESTING provided as well
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR MODERN_CMAKE_BUILD_TESTING)
   AND BUILD_TESTING
   AND RETICOLO_BUILD_TESTS)
    add_subdirectory(tests)
endif()
