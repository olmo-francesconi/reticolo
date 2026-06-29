#include <reticolo/cuda/selftest.hpp>

#include <catch2/catch_test_macros.hpp>

// Phase 0 exit gate: the CUDA toolchain builds, a kernel launches, and a
// host→device→host round-trip preserves data. Only registered when
// RETICOLO_ENABLE_CUDA is on (see tests/CMakeLists.txt); requires a GPU at
// run time.
TEST_CASE("cuda backend self-test round-trips through the device", "[cuda]") {
    REQUIRE(reticolo::cuda::selftest());
}
</content>
