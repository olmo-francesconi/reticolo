# Phase 2 lint gate: "zero integrator-specific CUDA code." The integrator is a
# reused type parameter (alg::integ::*), never reimplemented in a kernel. This
# greps the CUDA kernel sources for integrator names and fails if any appear.
#
# hmc_probe.cu / f32_probe.cu / u1_probe.cu / su2_probe.cu / su3_probe.cu (test
# harnesses) and bench_hmc.cu (perf harness) are excluded by name: they
# legitimately *instantiate* the generic integrator (alg::integ::Leapfrog) as a
# type parameter — that is reuse, not a reimplemented integrator-specific kernel.
#
# Invoked as: cmake -DSRC_DIR=<src/cuda> -P check_no_integrator_kernels.cmake

file(GLOB cu_files "${SRC_DIR}/*.cu")
set(offenders "")
foreach(f ${cu_files})
    get_filename_component(name "${f}" NAME)
    if(name STREQUAL "hmc_probe.cu" OR name STREQUAL "f32_probe.cu" OR
       name STREQUAL "u1_probe.cu" OR name STREQUAL "su2_probe.cu" OR
       name STREQUAL "su3_probe.cu" OR name STREQUAL "bench_hmc.cu")
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
