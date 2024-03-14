# Works with 3.14 and tested through 3.28
cmake_minimum_required(VERSION 3.14...3.28)

# Compiler
set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")

# set(CMAKE_C_COMPILER "gcc-13")
# set(CMAKE_CXX_COMPILER "g++-13")
# set(CMAKE_CXX_FLAGS "-Wall -Wl,-ld_classic")

# Export compilation commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# setup build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -Ofast")

# IF(CMAKE_BUILD_TYPE EQUAL Debug)
#   set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -lprofiler -ltcmalloc")
#   include_directories(/opt/homebrew/Cellar/gperftools/2.15/include)
#   link_directories(/opt/homebrew/Cellar/gperftools/2.15/lib)
#   link_libraries(profiler)
#   link_libraries(tcmalloc)
# endif()

# Project name and a few useful settings. Other commands can pick up the results
project(
  reticolo
  VERSION 0.1
  DESCRIPTION "C++ library for lattice QFT calculations"
  LANGUAGES CXX)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

  # Optionally set things like CMAKE_CXX_STANDARD, CMAKE_POSITION_INDEPENDENT_CODE here
  set(CMAKE_CXX_STANDARD 20)

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
  find_package(Doxygen)
  if(Doxygen_FOUND)
    add_subdirectory(docs)
  else()
    message(STATUS "Doxygen not found, not building docs")
  endif()
endif()

# HDF5
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(HDF5_ROOT /opt/hdf5-llvm)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(HDF5_ROOT /opt/hdf5-gnu)
endif()
set(HDF5_NO_FIND_PACKAGE_CONFIG_FILE true)
find_package(HDF5 REQUIRED CXX HL)

# OpenMP
FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  ENDIF()

# The compiled library code is here
add_subdirectory(src)

# The executables code is here
add_subdirectory(apps)

# Testing only available if this is the main app
# Emergency override MODERN_CMAKE_BUILD_TESTING provided as well
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR MODERN_CMAKE_BUILD_TESTING)
   AND BUILD_TESTING)
  add_subdirectory(tests)
endif()
