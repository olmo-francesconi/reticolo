# Phase 2 lint gate: "zero integrator-specific CUDA code." The integrator is a
# reused type parameter (alg::integ::*), never reimplemented in a kernel. This
# greps the CUDA kernel sources for integrator names and fails if any appear.
#
# hmc_probe.cu is excluded by name: it legitimately NAMES the integrator tags
# to instantiate the generic driver and prove reuse (the test harness, not
# kernel code).
#
# Invoked as: cmake -DSRC_DIR=<src/cuda> -P check_no_integrator_kernels.cmake

file(GLOB cu_files "${SRC_DIR}/*.cu")
set(offenders "")
foreach(f ${cu_files})
    get_filename_component(name "${f}" NAME)
    if(name STREQUAL "hmc_probe.cu")
        continue()
    endif()
    file(READ "${f}" content)
    if(content MATCHES "Leapfrog|Omelyan|integ::")
        list(APPEND offenders "${name}")
    endif()
endforeach()

if(offenders)
    message(FATAL_ERROR
        "integrator-specific code found in CUDA kernel sources: ${offenders}")
endif()
message(STATUS "no integrator-specific code in CUDA kernel sources")
