# Helper: declare a reference simulation app. Apps are single-TU executables
# that link the full umbrella (reticolo::core + io + cli).
function(reticolo_add_app name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE reticolo::reticolo)
    reticolo_configure_warnings(${name})
endfunction()

reticolo_add_app(phi4_hmc            phi4_hmc.cpp)
reticolo_add_app(phi4_llr            phi4_llr.cpp)
reticolo_add_app(bose_gas_hmc        bose_gas_hmc.cpp)
reticolo_add_app(bose_gas_llr        bose_gas_llr.cpp)
reticolo_add_app(u1_metropolis       u1_metropolis.cpp)
reticolo_add_app(u1_hmc              u1_hmc.cpp)
reticolo_add_app(su2_hmc             su2_hmc.cpp)
reticolo_add_app(u1_llr              u1_llr.cpp)
reticolo_add_app(phi6_hmc            phi6_hmc.cpp)
reticolo_add_app(sine_gordon_hmc     sine_gordon_hmc.cpp)
reticolo_add_app(xy_wolff            xy_wolff.cpp)
reticolo_add_app(on_sigma_metropolis on_sigma_metropolis.cpp)
reticolo_add_app(on_sigma_wolff      on_sigma_wolff.cpp)
reticolo_add_app(bench_scalars       bench_scalars.cpp)
reticolo_add_app(bench_gauge_vs_scalar bench_gauge_vs_scalar.cpp)
reticolo_add_app(bench_phase_quenched  bench_phase_quenched.cpp)
reticolo_add_app(tune_phi4           tune_phi4.cpp)

# OpenMP is inherited transitively through reticolo::core — no per-app link
# needed. Pragmas in headers/apps are inert when -fopenmp is off.
