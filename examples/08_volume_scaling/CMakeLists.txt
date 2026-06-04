cmake_minimum_required(VERSION 3.25)
project(example_08_volume_scaling CXX)

# --- find-or-fetch reticolo -------------------------------------------------
# Reuse the target when built inside the reticolo tree; otherwise build against
# a sibling checkout if present, else fetch a pinned release from git.
if(NOT TARGET reticolo::reticolo)
    set(_root "${CMAKE_CURRENT_SOURCE_DIR}/../..")
    if(EXISTS "${_root}/include/reticolo/reticolo.hpp")
        add_subdirectory("${_root}" "${CMAKE_BINARY_DIR}/_reticolo" EXCLUDE_FROM_ALL)
    else()
        include(FetchContent)
        FetchContent_Declare(reticolo
            GIT_REPOSITORY https://github.com/olmo-francesconi/reticolo.git
            GIT_TAG        main    # pin a release tag/SHA for reproducible builds
            GIT_SHALLOW    TRUE)
        FetchContent_MakeAvailable(reticolo)
    endif()
endif()
# ----------------------------------------------------------------------------

add_executable(ex08_bench_volume_scaling bench_volume_scaling.cpp)
set_target_properties(ex08_bench_volume_scaling PROPERTIES OUTPUT_NAME bench_volume_scaling)
target_link_libraries(ex08_bench_volume_scaling PRIVATE reticolo::reticolo)
