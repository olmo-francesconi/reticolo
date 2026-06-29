#include <reticolo/core/indexing.hpp>
#include <reticolo/cuda/device_buffer.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/gauge_probe.hpp>
#include <reticolo/cuda/reduce.hpp>
#include <reticolo/cuda/selftest.hpp>
#include <reticolo/cuda/stencil_probe.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <vector>

// CUDA tests. Only registered when RETICOLO_ENABLE_CUDA is on (see
// tests/CMakeLists.txt); all require a GPU at run time.

// Phase 0 exit gate: the toolchain builds, a kernel launches, and a
// host→device→host round-trip preserves data.
TEST_CASE("cuda backend self-test round-trips through the device", "[cuda]") {
    REQUIRE(reticolo::cuda::selftest());
}

// Phase 1: the device's closed-form periodic indexing must reproduce the CPU
// reference neighbour table exactly — checked on the host (next/prev are
// __host__ __device__). Covers a 4D cube and a non-cube, non-power-of-two shape.
TEST_CASE("cuda DeviceTopology matches the reference Indexing", "[cuda]") {
    for (std::vector<std::size_t> shape : {std::vector<std::size_t>{4, 4, 4, 4},
                                           std::vector<std::size_t>{6, 4, 5}}) {
        auto const idx = reticolo::Indexing::acquire(shape);
        auto const topo = reticolo::cuda::make_device_topology(shape);
        REQUIRE(topo.nsites == static_cast<long>(idx->nsites()));

        for (std::size_t s = 0; s < idx->nsites(); ++s) {
            for (std::size_t mu = 0; mu < shape.size(); ++mu) {
                REQUIRE(topo.next(static_cast<long>(s), static_cast<int>(mu)) ==
                        static_cast<long>(idx->next(reticolo::Site{s}, mu).value()));
                REQUIRE(topo.prev(static_cast<long>(s), static_cast<int>(mu)) ==
                        static_cast<long>(idx->prev(reticolo::Site{s}, mu).value()));
            }
        }
    }
}

// Phase 1: deterministic f64 reduction + axpy on the device.
TEST_CASE("cuda reduce_sum_f64 and axpy_f64 are correct", "[cuda]") {
    using namespace reticolo::cuda;
    constexpr long n = 10000;

    std::vector<double> hx(n);
    std::vector<double> hy(n);
    double host_sum = 0.0;
    for (long i = 0; i < n; ++i) {
        hx[static_cast<std::size_t>(i)] = 0.5 * static_cast<double>(i) - 3.0;
        hy[static_cast<std::size_t>(i)] = 1.0;
        host_sum += hx[static_cast<std::size_t>(i)];
    }

    DeviceBuffer<double> dx{static_cast<std::size_t>(n)};
    DeviceBuffer<double> dy{static_cast<std::size_t>(n)};
    dx.copy_from_host(hx.data());
    dy.copy_from_host(hy.data());

    // reduce: Σ x
    double const dev_sum = reduce_sum_f64(dx.data(), n);
    REQUIRE(dev_sum == Catch::Approx(host_sum).epsilon(1e-12));

    // determinism: same inputs → identical bits across calls
    REQUIRE(reduce_sum_f64(dx.data(), n) == dev_sum);

    // axpy: y += 2*x, then reduce y == Σ(1 + 2x) = n + 2·Σx
    axpy_f64(2.0, dx.data(), dy.data(), n);
    double const dev_y_sum = reduce_sum_f64(dy.data(), n);
    REQUIRE(dev_y_sum == Catch::Approx(static_cast<double>(n) + 2.0 * host_sum).epsilon(1e-12));
}

// Phase 1 (M1): the gauge action/drift headers compile under nvcc (Sleef now
// guarded out of vec_libm.hpp) and produce finite output. That this links at
// all means src/cuda/gauge_probe.cu compiled the transcendental paths.
TEST_CASE("cuda gauge headers compile and run under nvcc", "[cuda]") {
    REQUIRE(reticolo::cuda::gauge_headers_compile());
}

// Phase 1 exit gate: the generic stencil + reduce_fwd device skeletons, driven
// by a dummy Phi4-shaped functor pair, satisfy force == -dS/dphi by central
// finite differences. Validates the scalar device action protocol end-to-end.
TEST_CASE("cuda stencil force matches finite-difference of reduce_fwd action", "[cuda]") {
    REQUIRE(reticolo::cuda::stencil_force_matches_fd());
}
