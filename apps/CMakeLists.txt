# Helper: declare a reference simulation app. Apps are single-TU executables
# that link the full umbrella (reticolo::core + io + cli).
function(reticolo_add_app name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE reticolo::reticolo)
    reticolo_configure_warnings(${name})
endfunction()

reticolo_add_app(phi4_hmc            phi4_hmc.cpp)
reticolo_add_app(phi4_hmc_f32        phi4_hmc_f32.cpp)
reticolo_add_app(phi4_llr            phi4_llr.cpp)
reticolo_add_app(bose_gas_hmc        bose_gas_hmc.cpp)
reticolo_add_app(bose_gas_llr        bose_gas_llr.cpp)
reticolo_add_app(u1_metropolis       u1_metropolis.cpp)
reticolo_add_app(u1_hmc              u1_hmc.cpp)
reticolo_add_app(su2_hmc             su2_hmc.cpp)
reticolo_add_app(su2_hmc_f32         su2_hmc_f32.cpp)
reticolo_add_app(su3_hmc             su3_hmc.cpp)
reticolo_add_app(su3_hmc_f32         su3_hmc_f32.cpp)
reticolo_add_app(su2_llr             su2_llr.cpp)
reticolo_add_app(bench_wilson_vs_compact_u1 bench_wilson_vs_compact_u1.cpp)
reticolo_add_app(bench_actions      bench_actions.cpp)
reticolo_add_app(bench_rng          bench_rng.cpp)
reticolo_add_app(bench_readme       bench_readme.cpp)
reticolo_add_app(bench_integrators  bench_integrators.cpp)
reticolo_add_app(u1_llr              u1_llr.cpp)
reticolo_add_app(u1_llr_smoothed       u1_llr_smoothed.cpp)
reticolo_add_app(phi6_hmc            phi6_hmc.cpp)
reticolo_add_app(sine_gordon_hmc     sine_gordon_hmc.cpp)
reticolo_add_app(xy_wolff            xy_wolff.cpp)
reticolo_add_app(on_sigma_metropolis on_sigma_metropolis.cpp)
reticolo_add_app(on_sigma_wolff      on_sigma_wolff.cpp)
reticolo_add_app(bench_scalars       bench_scalars.cpp)
reticolo_add_app(bench_gauge_vs_scalar bench_gauge_vs_scalar.cpp)
reticolo_add_app(bench_phase_quenched  bench_phase_quenched.cpp)
reticolo_add_app(bench_metropolis_sweep bench_metropolis_sweep.cpp)
reticolo_add_app(bench_precision_hmc   bench_precision_hmc.cpp)

# CUDA-backed apps: native .cu twins of the CPU reference sims, compiled by
# nvcc. They use cuda::Hmc / DeviceField directly and io::Writer directly —
# io::Writer PIMPLs HDF5 so nvcc never sees <hdf5.h>, it just links the prebuilt
# reticolo::io archive. Linking the umbrella is safe now that the host warning
# flags are $<COMPILE_LANGUAGE:CXX>-guarded (cmake/ReticoloWarnings.cmake), so
# nvcc does not inherit -Werror on the .cu compile.
if(RETICOLO_ENABLE_CUDA)
    function(reticolo_add_cuda_app name)
        add_executable(${name} ${ARGN})
        target_link_libraries(${name} PRIVATE reticolo::reticolo)
        reticolo_configure_warnings(${name})  # CXX-guarded; inert on .cu sources
        set_target_properties(${name} PROPERTIES
            CUDA_STANDARD              20
            CUDA_STANDARD_REQUIRED     ON
            CUDA_SEPARABLE_COMPILATION ON)
    endfunction()
    reticolo_add_cuda_app(phi4_cuda_hmc  phi4_cuda_hmc.cu)
    reticolo_add_cuda_app(su2_cuda_hmc   su2_cuda_hmc.cu)
endif()

# OpenMP is inherited transitively through reticolo::core — no per-app link
# needed. Pragmas in headers/apps are inert when -fopenmp is off.
