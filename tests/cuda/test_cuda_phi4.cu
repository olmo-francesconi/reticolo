#include <reticolo/action/site/phi4.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/cuda/actions/site/phi4.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_action.cuh>
#include <reticolo/cuda/device_field.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

// Scalar device field + generic DeviceAction over the real Phi4 functor pair.
// The CPU action::Phi4 and the device path call the same shared HD per-site
// formula (action::formula::phi4_*), so they must agree to roundoff — this TU
// runs both (CPU on the host, device on the GPU) and compares.

namespace reticolo::cuda {

namespace {

constexpr double kKappa  = 0.18;
constexpr double kLambda = 0.55;
std::vector<std::size_t> const kShape{6, 4, 5};

Lattice<double> make_field() {
    Lattice<double> l{kShape};
    double* const d = l.data();
    for (std::size_t i = 0; i < l.nsites(); ++i) {
        d[i] = 0.5 * std::sin(0.3 * static_cast<double>(i) + 1.0);
    }
    return l;
}

}  // namespace

bool phi4_roundtrip_ok() {
    Lattice<double> const host = make_field();
    DeviceField<double> dev{kShape};
    dev.copy_from_host(host);

    Lattice<double> back{kShape};
    dev.copy_to_host(back);
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    for (std::size_t i = 0; i < host.nsites(); ++i) {
        if (host.data()[i] != back.data()[i]) {
            return false;
        }
    }
    return true;
}

bool phi4_cpu_matches_device() {
    Lattice<double> const host = make_field();

    action::Phi4<double> cpu{};
    cpu.kappa          = kKappa;
    cpu.lambda         = kLambda;
    double const s_cpu = cpu.s_full(host);
    Lattice<double> f_cpu{kShape};
    cpu.compute_force(host, f_cpu);

    DeviceField<double> dfield{kShape};
    DeviceField<double> dforce{dfield.topology()};
    dfield.copy_from_host(host);

    DeviceAction<action::Phi4<double>, DeviceField<double>> const act{cpu, dfield.topology()};
    double const s_dev = act.s_full(dfield);
    act.compute_force(dfield, dforce);

    Lattice<double> f_dev{kShape};
    dforce.copy_to_host(f_dev);
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    if (std::abs(s_cpu - s_dev) > 1e-10 * (1.0 + std::abs(s_cpu))) {
        return false;
    }
    for (std::size_t i = 0; i < host.nsites(); ++i) {
        double const a = f_cpu.data()[i];
        double const b = f_dev.data()[i];
        if (std::abs(a - b) > 1e-10 * (1.0 + std::abs(a))) {
            return false;
        }
    }
    return true;
}

}  // namespace reticolo::cuda

// A scalar field survives the host<->device round-trip unchanged.
TEST_CASE("cuda DeviceField round-trips a scalar field", "[cuda]") {
    REQUIRE(reticolo::cuda::phi4_roundtrip_ok());
}

// The generic DeviceAction over the real Phi4 functor pair reproduces the CPU
// action::Phi4 s_full and force to roundoff.
TEST_CASE("cuda DeviceAction<Phi4> matches CPU action::Phi4 to roundoff", "[cuda]") {
    REQUIRE(reticolo::cuda::phi4_cpu_matches_device());
}
