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
    add_executable(${name} ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp ${ARGN})
    target_link_libraries(${name} PRIVATE reticolo::reticolo Catch2::Catch2)
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
reticolo_add_test(test_gauge_batched_kernels unit/test_gauge_batched_kernels.cpp)
reticolo_add_test(test_obs_analysis    unit/test_obs_analysis.cpp)
reticolo_add_test(test_fused_action_force unit/test_fused_action_force.cpp)
reticolo_add_test(test_llr_exchange    unit/test_llr_exchange.cpp)
reticolo_add_test(test_phi4_force_consistency       physics/test_phi4_force_consistency.cpp)
reticolo_add_test(test_compact_u1_force_consistency physics/test_compact_u1_force_consistency.cpp)
reticolo_add_test(test_u1_hmc_reversibility         physics/test_u1_hmc_reversibility.cpp)
reticolo_add_test(test_wilson_u1_vs_compact         physics/test_wilson_u1_vs_compact.cpp)
reticolo_add_test(test_su2_hmc_reversibility        physics/test_su2_hmc_reversibility.cpp)
reticolo_add_test(test_su2_hmc_mixed_precision      physics/test_su2_hmc_mixed_precision.cpp)
reticolo_add_test(test_su3_hmc_reversibility        physics/test_su3_hmc_reversibility.cpp)
reticolo_add_test(test_su3_hmc_mixed_precision      physics/test_su3_hmc_mixed_precision.cpp)
reticolo_add_test(test_phi6_force_consistency       physics/test_phi6_force_consistency.cpp)
reticolo_add_test(test_sine_gordon_force_consistency physics/test_sine_gordon_force_consistency.cpp)
reticolo_add_test(test_xy_force_consistency         physics/test_xy_force_consistency.cpp)
reticolo_add_test(test_bose_gas_force_consistency   physics/test_bose_gas_force_consistency.cpp)
reticolo_add_test(test_hmc_reversibility            physics/test_hmc_reversibility.cpp)
reticolo_add_test(test_hmc_integrator_order         physics/test_hmc_integrator_order.cpp)
reticolo_add_test(test_phi4_hmc_mixed_precision      physics/test_phi4_hmc_mixed_precision.cpp)
reticolo_add_test(test_api_uniformity               unit/test_api_uniformity.cpp)
reticolo_add_test(test_philox                        unit/test_philox.cpp)

if(RETICOLO_ENABLE_CUDA)
    # Native .cu Catch2 tests: each device-validation gate is its own TU, like
    # the CPU tests. nvcc compiles the .cu; the host warning flags applied by
    # reticolo_add_test are CXX-guarded, so they stay inert on the device source.
    function(reticolo_add_cuda_test name source)
        reticolo_add_test(${name} ${source})
        set_target_properties(${name} PROPERTIES
            CUDA_STANDARD              20
            CUDA_STANDARD_REQUIRED     ON
            CUDA_SEPARABLE_COMPILATION ON)
        set_property(GLOBAL APPEND PROPERTY RETICOLO_CUDA_TARGETS ${name})
    endfunction()

    reticolo_add_cuda_test(test_cuda_selftest    cuda/test_cuda_selftest.cu)
    reticolo_add_cuda_test(test_cuda_rng         cuda/test_cuda_rng.cu)
    reticolo_add_cuda_test(test_cuda_stencil     cuda/test_cuda_stencil.cu)
    reticolo_add_cuda_test(test_cuda_phi4        cuda/test_cuda_phi4.cu)
    reticolo_add_cuda_test(test_cuda_scalar      cuda/test_cuda_scalar.cu)
    reticolo_add_cuda_test(test_cuda_f32         cuda/test_cuda_f32.cu)
    reticolo_add_cuda_test(test_cuda_hmc         cuda/test_cuda_hmc.cu)
    reticolo_add_cuda_test(test_cuda_u1          cuda/test_cuda_u1.cu)
    reticolo_add_cuda_test(test_cuda_wilson_u1   cuda/test_cuda_wilson_u1.cu)
    reticolo_add_cuda_test(test_cuda_su2         cuda/test_cuda_su2.cu)
    reticolo_add_cuda_test(test_cuda_su3         cuda/test_cuda_su3.cu)
    reticolo_add_cuda_test(test_cuda_bose_gas    cuda/test_cuda_bose_gas.cu)

    if(RETICOLO_BUILD_IO)
        # Native .cu test: orchestrates cuda::Hmc + io::checkpoint in one TU.
        reticolo_add_cuda_test(test_cuda_checkpoint cuda/test_cuda_checkpoint.cu)
    endif()

    # Lint gate: no integrator reimplemented as a CUDA kernel. Scans the device
    # kernel headers (hmc.cuh excepted — it legitimately names the reused
    # alg::integ::* type parameter).
    add_test(NAME cuda_no_integrator_kernels
             COMMAND ${CMAKE_COMMAND} -DSRC_DIR=${PROJECT_SOURCE_DIR}/include/reticolo/cuda
                     -P ${CMAKE_CURRENT_SOURCE_DIR}/cuda/check_no_integrator_kernels.cmake)
endif()

if(RETICOLO_BUILD_IO)
    reticolo_add_test(test_writer_roundtrip         io/test_writer_roundtrip.cpp)
    reticolo_add_test(test_writer_metadata          io/test_writer_metadata.cpp)
    reticolo_add_test(test_writer_phase_collision   io/test_writer_phase_collision.cpp)
    reticolo_add_test(test_writer_vars_stamping     io/test_writer_vars_stamping.cpp)
    reticolo_add_test(test_checkpoint_roundtrip     io/test_checkpoint_roundtrip.cpp)
    reticolo_add_test(test_cli_parser               unit/test_cli_parser.cpp)
endif()

if(RETICOLO_BUILD_IO AND RETICOLO_BUILD_APPS)
    function(reticolo_add_app_smoke test_name app_target source)
        reticolo_add_test(${test_name} ${source})
        string(TOUPPER "${app_target}" _u)
        target_compile_definitions(${test_name} PRIVATE
            ${_u}_BINARY="$<TARGET_FILE:${app_target}>"
        )
        add_dependencies(${test_name} ${app_target})
    endfunction()

    reticolo_add_app_smoke(test_phi4_hmc_smoke             phi4_hmc             apps/test_phi4_hmc_smoke.cpp)
    reticolo_add_app_smoke(test_phi4_hmc_resume_equivalence phi4_hmc            apps/test_phi4_hmc_resume_equivalence.cpp)
    reticolo_add_app_smoke(test_phi6_hmc_smoke             phi6_hmc             apps/test_phi6_hmc_smoke.cpp)
    reticolo_add_app_smoke(test_sine_gordon_hmc_smoke      sine_gordon_hmc      apps/test_sine_gordon_hmc_smoke.cpp)
    reticolo_add_app_smoke(test_u1_hmc_smoke               u1_hmc               apps/test_u1_hmc_smoke.cpp)
    reticolo_add_app_smoke(test_u1_llr_smoke               u1_llr               apps/test_u1_llr_smoke.cpp)
    reticolo_add_app_smoke(test_su2_llr_smoke              su2_llr              apps/test_su2_llr_smoke.cpp)
    reticolo_add_app_smoke(test_bose_gas_hmc_smoke         bose_gas_hmc         apps/test_bose_gas_hmc_smoke.cpp)
    reticolo_add_app_smoke(test_bose_gas_llr_smoke         bose_gas_llr         apps/test_bose_gas_llr_smoke.cpp)

    # CUDA app smoke tests need the device binary (built only when CUDA is on)
    # and a GPU at run time.
    if(RETICOLO_ENABLE_CUDA)
        reticolo_add_app_smoke(test_phi4_cuda_hmc_smoke    phi4_cuda_hmc        apps/test_phi4_cuda_hmc_smoke.cpp)
        set_property(GLOBAL APPEND PROPERTY RETICOLO_CUDA_TARGETS test_phi4_cuda_hmc_smoke)
        reticolo_add_app_smoke(test_phi4_llr_cuda_smoke    phi4_llr_cuda        apps/test_phi4_llr_cuda_smoke.cpp)
        set_property(GLOBAL APPEND PROPERTY RETICOLO_CUDA_TARGETS test_phi4_llr_cuda_smoke)
        reticolo_add_app_smoke(test_u1_llr_cuda_smoke      u1_llr_cuda          apps/test_u1_llr_cuda_smoke.cpp)
        set_property(GLOBAL APPEND PROPERTY RETICOLO_CUDA_TARGETS test_u1_llr_cuda_smoke)
        reticolo_add_app_smoke(test_su2_llr_cuda_smoke     su2_llr_cuda         apps/test_su2_llr_cuda_smoke.cpp)
        set_property(GLOBAL APPEND PROPERTY RETICOLO_CUDA_TARGETS test_su2_llr_cuda_smoke)
        reticolo_add_app_smoke(test_su3_llr_cuda_smoke     su3_llr_cuda         apps/test_su3_llr_cuda_smoke.cpp)
        set_property(GLOBAL APPEND PROPERTY RETICOLO_CUDA_TARGETS test_su3_llr_cuda_smoke)
        reticolo_add_app_smoke(test_bose_gas_cuda_hmc_smoke bose_gas_cuda_hmc   apps/test_bose_gas_cuda_hmc_smoke.cpp)
        set_property(GLOBAL APPEND PROPERTY RETICOLO_CUDA_TARGETS test_bose_gas_cuda_hmc_smoke)
        reticolo_add_app_smoke(test_bose_gas_llr_cuda_smoke bose_gas_llr_cuda   apps/test_bose_gas_llr_cuda_smoke.cpp)
        set_property(GLOBAL APPEND PROPERTY RETICOLO_CUDA_TARGETS test_bose_gas_llr_cuda_smoke)
    endif()
endif()
