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
reticolo_add_test(test_bc              unit/test_bc.cpp)
reticolo_add_test(test_indexing        unit/test_indexing.cpp)
reticolo_add_test(test_indexing_pool   unit/test_indexing_pool.cpp)
reticolo_add_test(test_lattice         unit/test_lattice.cpp)
reticolo_add_test(test_rng             unit/test_rng.cpp)
reticolo_add_test(test_log             unit/test_log.cpp)
reticolo_add_test(test_phi4_force_consistency       physics/test_phi4_force_consistency.cpp)
reticolo_add_test(test_hmc_reversibility            physics/test_hmc_reversibility.cpp)
reticolo_add_test(test_hmc_integrator_order         physics/test_hmc_integrator_order.cpp)
reticolo_add_test(test_metropolis_gaussian_limit    physics/test_metropolis_gaussian_limit.cpp)

if(RETICOLO_BUILD_IO)
    reticolo_add_test(test_writer_roundtrip         io/test_writer_roundtrip.cpp)
    reticolo_add_test(test_writer_metadata          io/test_writer_metadata.cpp)
    reticolo_add_test(test_writer_phase_collision   io/test_writer_phase_collision.cpp)
endif()
