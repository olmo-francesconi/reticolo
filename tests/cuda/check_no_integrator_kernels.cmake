# Lint gate: "zero integrator-specific CUDA code." The integrator is a reused
# type parameter (alg::integ::*), never reimplemented in a kernel. This greps
# the device kernel headers for integrator names and fails if any appear.
#
# Scanned: SRC_DIR/**/*.cuh — the nvcc-only kernel headers (stencil, reduce,
# reduce_fwd, gauge_*, device_action, gauge/*). hmc.cuh is the one exception: it
# is the integrator-orchestration layer and legitimately names the reused
# alg::integ::* tag as a type parameter (default + MD loop), which is reuse, not
# a reimplemented integrator-specific kernel. The host-callable API headers
# (*.hpp, e.g. integ_ops.hpp) and the validation tests (tests/cuda/*.cu) are not
# kernel sources and are out of scope; the perf drivers live under apps/cuda/.
#
# Invoked as: cmake -DSRC_DIR=<include/reticolo/cuda> -P check_no_integrator_kernels.cmake

file(GLOB_RECURSE cuh_files "${SRC_DIR}/*.cuh")
set(offenders "")
foreach(f ${cuh_files})
    get_filename_component(name "${f}" NAME)
    if(name STREQUAL "hmc.cuh")
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
