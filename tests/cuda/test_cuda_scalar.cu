#include <reticolo/action/site/phi6.hpp>
#include <reticolo/action/site/sine_gordon.hpp>
#include <reticolo/action/site/xy.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/cuda/actions/phi6.hpp>
#include <reticolo/cuda/actions/sine_gordon.hpp>
#include <reticolo/cuda/actions/xy.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_action.cuh>
#include <reticolo/cuda/device_field.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

// Scalar-action coverage. Each action's CPU s_full + force and the generic device
// path call the SAME shared HD formula (action::detail::*), so they must agree —
// to roundoff for the polynomial Phi6, to a bounded tolerance for the
// transcendental SineGordon / XY (device sin/cos vs Sleef/libm). This TU runs
// both sides and compares, mirroring phi4_probe.cu.

namespace reticolo::cuda {

namespace {

std::vector<std::size_t> const kShape{6, 4, 5};

Lattice<double> make_field(double scale, double bias) {
    Lattice<double> l{kShape};
    double* const d = l.data();
    for (std::size_t i = 0; i < l.nsites(); ++i) {
        d[i] = (scale * std::sin(0.3 * static_cast<double>(i) + 1.0)) + bias;
    }
    return l;
}

// s_full + force, CPU vs generic DeviceAction, within relative tolerances.
template <class HostAction>
bool matches(HostAction const& cpu, Lattice<double> const& host, double s_tol, double f_tol) {
    double const s_cpu = cpu.s_full(host);
    Lattice<double> f_cpu{kShape};
    cpu.compute_force(host, f_cpu);

    DeviceField<double> dfield{kShape};
    DeviceField<double> dforce{dfield.topology()};
    dfield.copy_from_host(host);

    DeviceAction<HostAction, DeviceField<double>> const act{cpu, dfield.topology()};
    double const s_dev = act.s_full(dfield);
    act.compute_force(dfield, dforce);

    Lattice<double> f_dev{kShape};
    dforce.copy_to_host(f_dev);
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    if (std::abs(s_cpu - s_dev) > s_tol * (1.0 + std::abs(s_cpu))) {
        return false;
    }
    for (std::size_t i = 0; i < host.nsites(); ++i) {
        double const a = f_cpu.data()[i];
        double const b = f_dev.data()[i];
        if (std::abs(a - b) > f_tol * (1.0 + std::abs(a))) {
            return false;
        }
    }
    return true;
}

}  // namespace

bool phi6_cpu_matches_device() {
    action::Phi6<double> cpu{};
    cpu.kappa  = 0.18;
    cpu.lambda = 0.55;
    cpu.g6     = 0.20;
    return matches(cpu, make_field(0.5, 0.0), 1e-10, 1e-10);
}

bool sine_gordon_cpu_matches_device() {
    action::SineGordon<double> cpu{};
    cpu.kappa = 0.18;
    cpu.alpha = 0.75;
    return matches(cpu, make_field(0.9, 0.0), 1e-9, 1e-9);
}

bool xy_cpu_matches_device() {
    action::Xy<double> cpu{};
    cpu.beta = 0.62;
    return matches(cpu, make_field(2.5, 1.0), 1e-9, 1e-9);
}

}  // namespace reticolo::cuda

// Each scalar action's device path reproduces the CPU action's s_full + force
// via the shared HD formula (one source of truth).
TEST_CASE("cuda DeviceAction<Phi6> matches CPU action::Phi6", "[cuda]") {
    REQUIRE(reticolo::cuda::phi6_cpu_matches_device());
}

TEST_CASE("cuda DeviceAction<SineGordon> matches CPU action::SineGordon", "[cuda]") {
    REQUIRE(reticolo::cuda::sine_gordon_cpu_matches_device());
}

TEST_CASE("cuda DeviceAction<Xy> matches CPU action::Xy", "[cuda]") {
    REQUIRE(reticolo::cuda::xy_cpu_matches_device());
}
