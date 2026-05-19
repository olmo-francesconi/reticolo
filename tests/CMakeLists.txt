include(FetchContent)

set(CATCH2_VERSION v3.5.4)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        ${CATCH2_VERSION}
    GIT_SHALLOW    TRUE
    SYSTEM
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)

function(reticolo_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE reticolo::reticolo Catch2::Catch2WithMain)
    reticolo_configure_warnings(${name})
    catch_discover_tests(${name})
endfunction()

reticolo_add_test(test_site            unit/test_site.cpp)
reticolo_add_test(test_indexing        unit/test_indexing.cpp)
reticolo_add_test(test_indexing_pool   unit/test_indexing_pool.cpp)
reticolo_add_test(test_lattice         unit/test_lattice.cpp)
reticolo_add_test(test_hot_loop        unit/test_hot_loop.cpp)
reticolo_add_test(test_rng             unit/test_rng.cpp)
reticolo_add_test(test_log             unit/test_log.cpp)
reticolo_add_test(test_observers       unit/test_observers.cpp)
reticolo_add_test(test_su2_ops         unit/test_su2_ops.cpp)
reticolo_add_test(test_su3_ops         unit/test_su3_ops.cpp)
reticolo_add_test(test_obs_analysis    unit/test_obs_analysis.cpp)
reticolo_add_test(test_phi4_force_consistency       physics/test_phi4_force_consistency.cpp)
reticolo_add_test(test_compact_u1_force_consistency physics/test_compact_u1_force_consistency.cpp)
reticolo_add_test(test_u1_hmc_reversibility         physics/test_u1_hmc_reversibility.cpp)
reticolo_add_test(test_wilson_u1_vs_compact         physics/test_wilson_u1_vs_compact.cpp)
reticolo_add_test(test_su2_hmc_reversibility        physics/test_su2_hmc_reversibility.cpp)
reticolo_add_test(test_wilson_su2_weak_coupling     physics/test_wilson_su2_weak_coupling.cpp)
reticolo_add_test(test_su3_hmc_reversibility        physics/test_su3_hmc_reversibility.cpp)
reticolo_add_test(test_wilson_su3_weak_coupling     physics/test_wilson_su3_weak_coupling.cpp)
reticolo_add_test(test_phi6_force_consistency       physics/test_phi6_force_consistency.cpp)
reticolo_add_test(test_sine_gordon_force_consistency physics/test_sine_gordon_force_consistency.cpp)
reticolo_add_test(test_xy_force_consistency         physics/test_xy_force_consistency.cpp)
reticolo_add_test(test_on_sigma_metropolis          physics/test_on_sigma_metropolis.cpp)
reticolo_add_test(test_hmc_reversibility            physics/test_hmc_reversibility.cpp)
reticolo_add_test(test_hmc_integrator_order         physics/test_hmc_integrator_order.cpp)
reticolo_add_test(test_metropolis_gaussian_limit    physics/test_metropolis_gaussian_limit.cpp)
reticolo_add_test(test_wolff_cluster                physics/test_wolff_cluster.cpp)

if(RETICOLO_BUILD_IO)
    reticolo_add_test(test_writer_roundtrip         io/test_writer_roundtrip.cpp)
    reticolo_add_test(test_writer_metadata          io/test_writer_metadata.cpp)
    reticolo_add_test(test_writer_phase_collision   io/test_writer_phase_collision.cpp)
    reticolo_add_test(test_writer_vars_stamping     io/test_writer_vars_stamping.cpp)
    reticolo_add_test(test_cli_parser               unit/test_cli_parser.cpp)
endif()

if(RETICOLO_BUILD_IO AND RETICOLO_BUILD_APPS)
    function(reticolo_add_app_smoke test_name app_target define_name source)
        reticolo_add_test(${test_name} ${source})
        target_compile_definitions(${test_name} PRIVATE
            ${define_name}="$<TARGET_FILE:${app_target}>"
        )
        add_dependencies(${test_name} ${app_target})
    endfunction()

    reticolo_add_app_smoke(test_phi4_hmc_smoke          phi4_hmc
        PHI4_HMC_BINARY            apps/test_phi4_hmc_smoke.cpp)
    reticolo_add_app_smoke(test_phi6_hmc_smoke          phi6_hmc
        PHI6_HMC_BINARY            apps/test_phi6_hmc_smoke.cpp)
    reticolo_add_app_smoke(test_sine_gordon_hmc_smoke   sine_gordon_hmc
        SINE_GORDON_HMC_BINARY     apps/test_sine_gordon_hmc_smoke.cpp)
    reticolo_add_app_smoke(test_xy_wolff_smoke          xy_wolff
        XY_WOLFF_BINARY            apps/test_xy_wolff_smoke.cpp)
    reticolo_add_app_smoke(test_on_sigma_metropolis_smoke on_sigma_metropolis
        ON_SIGMA_METROPOLIS_BINARY apps/test_on_sigma_metropolis_smoke.cpp)
    reticolo_add_app_smoke(test_on_sigma_wolff_smoke      on_sigma_wolff
        ON_SIGMA_WOLFF_BINARY      apps/test_on_sigma_wolff_smoke.cpp)
    reticolo_add_app_smoke(test_u1_metropolis_smoke       u1_metropolis
        U1_METROPOLIS_BINARY       apps/test_u1_metropolis_smoke.cpp)
    reticolo_add_app_smoke(test_u1_hmc_smoke              u1_hmc
        U1_HMC_BINARY              apps/test_u1_hmc_smoke.cpp)
    reticolo_add_app_smoke(test_u1_llr_smoke              u1_llr
        U1_LLR_BINARY              apps/test_u1_llr_smoke.cpp)
endif()
