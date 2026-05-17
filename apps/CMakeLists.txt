# Helper: declare a reference simulation app. Apps are single-TU executables
# that link the full umbrella (reticolo::core + io + cli).
function(reticolo_add_app name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE reticolo::reticolo)
    reticolo_configure_warnings(${name})
endfunction()

reticolo_add_app(phi4_hmc phi4_hmc.cpp)
