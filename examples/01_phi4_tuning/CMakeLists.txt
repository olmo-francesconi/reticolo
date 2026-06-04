cmake_minimum_required(VERSION 3.25)
project(example_01_phi4_tuning CXX)

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

add_executable(ex01_phi4_hmc phi4_hmc.cpp)
set_target_properties(ex01_phi4_hmc PROPERTIES OUTPUT_NAME phi4_hmc)
target_link_libraries(ex01_phi4_hmc PRIVATE reticolo::reticolo)
