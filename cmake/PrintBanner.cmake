# Print the reticolo logo for the BUILD phase. Run via `cmake -P` from the
# reticolo_banner custom target (see reticolo_attach_build_banner). Kept a
# separate script because a custom target's COMMAND is a fresh cmake process
# with none of the project's scope.
cmake_minimum_required(VERSION 3.25)
include(${CMAKE_CURRENT_LIST_DIR}/ReticoloBanner.cmake)
reticolo_logo("${RETICOLO_LOGO_VERSION}" "b u i l d")
