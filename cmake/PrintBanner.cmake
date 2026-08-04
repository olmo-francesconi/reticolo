# Print the reticolo logo for a non-configure phase. Run via `cmake -P`:
#
#   cmake -DRETICOLO_LOGO_VERSION=0.5.0 [-DRETICOLO_LOGO_TAG="t e s t"] \
#         -P cmake/PrintBanner.cmake
#
# Used by the reticolo_banner custom target (build phase) and by
# tools/check.sh (test phase). Kept a separate script because a custom
# target's COMMAND is a fresh cmake process with none of the project's scope.
cmake_minimum_required(VERSION 3.25)
include(${CMAKE_CURRENT_LIST_DIR}/ReticoloBanner.cmake)
if(NOT DEFINED RETICOLO_LOGO_TAG)
    set(RETICOLO_LOGO_TAG "b u i l d")
endif()
reticolo_logo("${RETICOLO_LOGO_VERSION}" "${RETICOLO_LOGO_TAG}")
