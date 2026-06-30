cmake_minimum_required(VERSION 3.25)
project(example_08_u1_llr_smoothed CXX)

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

foreach(_app u1_llr u1_llr_smoothed)
    add_executable(ex08_${_app} ${_app}.cpp)
    set_target_properties(ex08_${_app} PROPERTIES OUTPUT_NAME ${_app})
    target_link_libraries(ex08_${_app} PRIVATE reticolo::reticolo)
endforeach()
