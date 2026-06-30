# Lint gate: "zero integrator-specific CUDA code." The integrator is a reused
# type parameter (alg::integ::*), never reimplemented in a kernel. This greps
# the real CUDA kernel sources for integrator names and fails if any appear.
#
# Only the backend kernel TUs are scanned: SRC_DIR/*.cu (i.e. reduce.cu). The
# validation harnesses under SRC_DIR/probes/ are deliberately NOT scanned — they
# legitimately *instantiate* the generic integrator (alg::integ::Leapfrog) as a
# type parameter, which is reuse, not a reimplemented integrator-specific kernel.
# The perf / profiling drivers live under apps/cuda/ and are likewise outside
# SRC_DIR.
#
# Invoked as: cmake -DSRC_DIR=<src/cuda> -P check_no_integrator_kernels.cmake

file(GLOB cu_files "${SRC_DIR}/*.cu")
set(offenders "")
foreach(f ${cu_files})
    get_filename_component(name "${f}" NAME)
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
