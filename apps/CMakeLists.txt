# Helper: declare a reference simulation app. Apps are single-TU executables
# that link the full umbrella (reticolo::core + io + cli).
function(reticolo_add_app name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE reticolo::reticolo)
    reticolo_configure_warnings(${name})
endfunction()

# Benchmarks are dev-only perf tools — never run by ctest. Gated behind their own
# flag (OFF by default; ON in the dev presets) so CI can skip ~10 umbrella-heavy
# TUs it would only compile and discard.
function(reticolo_add_bench name)
    if(NOT RETICOLO_BUILD_BENCHMARKS)
        return()
    endif()
    reticolo_add_app(${name} ${ARGN})
endfunction()

reticolo_add_app(phi4_hmc            phi4_hmc.cpp)
reticolo_add_app(phi4_hmc_f32        phi4_hmc_f32.cpp)
reticolo_add_app(phi4_llr            phi4_llr.cpp)
reticolo_add_app(bose_gas_hmc        bose_gas_hmc.cpp)
reticolo_add_app(bose_gas_llr        bose_gas_llr.cpp)
reticolo_add_app(u1_hmc              u1_hmc.cpp)
reticolo_add_app(su2_hmc             su2_hmc.cpp)
reticolo_add_app(su2_hmc_f32         su2_hmc_f32.cpp)
reticolo_add_app(su3_hmc             su3_hmc.cpp)
reticolo_add_app(su3_hmc_f32         su3_hmc_f32.cpp)
reticolo_add_app(su2_llr             su2_llr.cpp)
reticolo_add_bench(bench_actions      bench_actions.cpp)
reticolo_add_bench(bench_rng          bench_rng.cpp)
reticolo_add_bench(bench_readme       bench_readme.cpp)
reticolo_add_bench(bench_integrators  bench_integrators.cpp)
reticolo_add_bench(bench_integrator_efficiency bench_integrator_efficiency.cpp)
reticolo_add_app(u1_llr              u1_llr.cpp)
reticolo_add_app(u1_llr_smoothed       u1_llr_smoothed.cpp)
reticolo_add_app(phi6_hmc            phi6_hmc.cpp)
reticolo_add_app(sine_gordon_hmc     sine_gordon_hmc.cpp)
reticolo_add_bench(bench_gauge_vs_scalar bench_gauge_vs_scalar.cpp)
reticolo_add_bench(bench_scaling_all     bench_scaling_all.cpp)
reticolo_add_bench(bench_hmc_breakdown   bench_hmc_breakdown.cpp)
reticolo_add_bench(bench_gauge_pad       bench_gauge_pad.cpp)
reticolo_add_bench(bench_phase_quenched  bench_phase_quenched.cpp)
reticolo_add_bench(bench_precision_hmc   bench_precision_hmc.cpp)

# CUDA-backed apps: native .cu twins of the CPU reference sims, compiled by
# nvcc. They use cuda::Hmc / DeviceField directly and io::Writer directly —
# io::Writer PIMPLs HDF5 so nvcc never sees <hdf5.h>, it just links the prebuilt
# reticolo::io archive. Linking the umbrella is safe now that the host warning
# flags are $<COMPILE_LANGUAGE:CXX>-guarded (cmake/ReticoloWarnings.cmake), so
# nvcc does not inherit -Werror on the .cu compile.
if(RETICOLO_ENABLE_CUDA)
    # Shared .cu target setup. reticolo_configure_warnings is $<COMPILE_LANGUAGE:CXX>
    # -guarded (cmake/ReticoloWarnings.cmake), so the host -Wall/-Werror never reach
    # nvcc — that is what lets a .cu app link the umbrella cleanly.
    function(reticolo_cuda_props name)
        reticolo_configure_warnings(${name})  # CXX-guarded; inert on .cu sources
        set_target_properties(${name} PROPERTIES
            CUDA_STANDARD              20
            CUDA_STANDARD_REQUIRED     ON
            CUDA_SEPARABLE_COMPILATION ON)
    endfunction()

    # Every cuda executable self-registers into the global RETICOLO_CUDA_TARGETS
    # list; the top-level CMakeLists turns it into the aggregate `cuda` target that
    # the linux-nvcc preset builds — so a new app joins with no list to maintain.

    # Lean GPU drivers (bench / Nsight profiling): link ONLY reticolo::cuda — no
    # io/cli, no HDF5 — so they build in a lean profiling config. Header-only core
    # + the device stack reach them through reticolo::cuda's INTERFACE include dir
    # (it carries the whole include/ tree); cudart comes via the same target.
    function(reticolo_add_cuda_tool name)
        add_executable(${name} ${ARGN})
        target_link_libraries(${name} PRIVATE reticolo::cuda)
        reticolo_cuda_props(${name})
        set_property(GLOBAL APPEND PROPERTY RETICOLO_CUDA_TARGETS ${name})
    endfunction()

    # Reference sims: link the umbrella for io::Writer (HDF5, PIMPL'd) + cli::Parser.
    function(reticolo_add_cuda_app name)
        add_executable(${name} ${ARGN})
        target_link_libraries(${name} PRIVATE reticolo::reticolo)
        reticolo_cuda_props(${name})
        set_property(GLOBAL APPEND PROPERTY RETICOLO_CUDA_TARGETS ${name})
    endfunction()

    reticolo_add_cuda_tool(bench_hmc_cuda   cuda/bench_hmc_cuda.cu)
    reticolo_add_cuda_tool(profile_hmc_cuda cuda/profile_hmc_cuda.cu)
    reticolo_add_cuda_app(phi4_hmc_cuda     cuda/phi4_hmc_cuda.cu)
    reticolo_add_cuda_app(phi4_llr_cuda     cuda/phi4_llr_cuda.cu)
    reticolo_add_cuda_app(u1_llr_cuda       cuda/u1_llr_cuda.cu)
    reticolo_add_cuda_app(su2_hmc_cuda      cuda/su2_hmc_cuda.cu)
    reticolo_add_cuda_app(su2_llr_cuda      cuda/su2_llr_cuda.cu)
    reticolo_add_cuda_app(su3_llr_cuda      cuda/su3_llr_cuda.cu)
    reticolo_add_cuda_app(bose_gas_hmc_cuda cuda/bose_gas_hmc_cuda.cu)
    reticolo_add_cuda_app(bose_gas_llr_cuda cuda/bose_gas_llr_cuda.cu)
    # Nightly physics harness: runs CPU alg::Hmc and GPU cuda::Hmc side-by-side
    # and asserts the [nightly] identities (<exp(-dH)>=1, CPU-vs-GPU equivalence,
    # plaquette at beta). App-level, not a ctest gate (it runs full simulations).
    reticolo_add_cuda_app(nightly_cuda      cuda/nightly_cuda.cu)
endif()

# OpenMP is inherited transitively through reticolo::core — no per-app link
# needed. Pragmas in headers/apps are inert when -fopenmp is off.
