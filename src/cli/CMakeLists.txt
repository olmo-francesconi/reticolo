add_library(reticolo_cli INTERFACE)
add_library(reticolo::cli ALIAS reticolo_cli)

target_link_libraries(reticolo_cli INTERFACE
    reticolo::core
    reticolo::io
    cxxopts::cxxopts
)

target_include_directories(reticolo_cli INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(reticolo_cli INTERFACE cxx_std_23)
