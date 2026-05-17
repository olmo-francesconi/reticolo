add_library(reticolo_io STATIC writer.cpp)
add_library(reticolo::io ALIAS reticolo_io)

target_link_libraries(reticolo_io
    PUBLIC  reticolo::core
            HDF5::HDF5
)

target_include_directories(reticolo_io PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(reticolo_io PUBLIC cxx_std_23)

# The /run@compile_flags stamp uses the same RETICOLO_COMPILE_FLAGS macro
# the core target carries.
target_compile_definitions(reticolo_io PRIVATE
    RETICOLO_VERSION="${PROJECT_VERSION}"
    RETICOLO_GIT_COMMIT="${RETICOLO_GIT_COMMIT}"
    RETICOLO_COMPILE_FLAGS="${_reticolo_compile_flags}"
    RETICOLO_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
)

reticolo_configure_warnings(reticolo_io)
