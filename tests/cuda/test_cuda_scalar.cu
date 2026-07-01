#include <reticolo/action/site/phi4.hpp>
#include <reticolo/action/site/phi6.hpp>
#include <reticolo/action/site/sine_gordon.hpp>
#include <reticolo/action/site/xy.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/cuda/actions/phi4.hpp>
#include <reticolo/cuda/actions/phi6.hpp>
#include <reticolo/cuda/actions/sine_gordon.hpp>
#include <reticolo/cuda/actions/xy.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_action.cuh>
#include <reticolo/cuda/device_buffer.hpp>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/reduce.cuh>

#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

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

// The fused s_full_and_force (dual-output stencil, used by the LLR WindowedAction)
// must reproduce the two-pass path — compute_force for the force, s_full for the
// action — since both call the same shared formula on the same neighbour sums.
// This is also the ONLY thing that instantiates each action's *ForceEnergyFunctor,
// so it doubles as the compile check for the fused device path.
template <class HostAction>
bool fused_matches_twopass(HostAction const& a, Lattice<double> const& host, double tol) {
    DeviceField<double> dfield{kShape};
    dfield.copy_from_host(host);
    DeviceAction<HostAction, DeviceField<double>> const act{a, dfield.topology()};

    DeviceField<double> f_two{dfield.topology()};
    act.compute_force(dfield, f_two);
    double const s_ref = act.s_full(dfield);

    DeviceBuffer<double> partials{static_cast<std::size_t>(k_reduce_max_grid)};
    DeviceBuffer<double> s_dev{1};
    DeviceField<double> f_fused{dfield.topology()};
    act.s_full_and_force(s_dev.data(), dfield, f_fused, partials.data(), nullptr);
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    double s_fused = 0.0;
    s_dev.copy_to_host(&s_fused);
    Lattice<double> f_two_h{kShape};
    Lattice<double> f_fused_h{kShape};
    f_two.copy_to_host(f_two_h);
    f_fused.copy_to_host(f_fused_h);
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    if (std::abs(s_ref - s_fused) > tol * (1.0 + std::abs(s_ref))) {
        return false;
    }
    for (std::size_t i = 0; i < host.nsites(); ++i) {
        double const x = f_two_h.data()[i];
        double const y = f_fused_h.data()[i];
        if (std::abs(x - y) > tol * (1.0 + std::abs(x))) {
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

bool phi4_fused_ok() {
    action::Phi4<double> const a{.kappa = 0.18, .lambda = 1.0};
    return fused_matches_twopass(a, make_field(0.5, 0.0), 1e-10);
}

bool phi6_fused_ok() {
    action::Phi6<double> a{};
    a.kappa  = 0.18;
    a.lambda = 0.55;
    a.g6     = 0.20;
    return fused_matches_twopass(a, make_field(0.5, 0.0), 1e-10);
}

bool sine_gordon_fused_ok() {
    action::SineGordon<double> a{};
    a.kappa = 0.18;
    a.alpha = 0.75;
    return fused_matches_twopass(a, make_field(0.9, 0.0), 1e-10);
}

bool xy_fused_ok() {
    action::Xy<double> a{};
    a.beta = 0.62;
    return fused_matches_twopass(a, make_field(2.5, 1.0), 1e-10);
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

// The fused force+action path (dual-output stencil) reproduces the two-pass
// compute_force + s_full for every real-scalar site action — the LLR
// WindowedAction depends on this, and it's what compile-checks each functor.
TEST_CASE("cuda fused s_full_and_force matches two-pass (Phi4)", "[cuda]") {
    REQUIRE(reticolo::cuda::phi4_fused_ok());
}

TEST_CASE("cuda fused s_full_and_force matches two-pass (Phi6)", "[cuda]") {
    REQUIRE(reticolo::cuda::phi6_fused_ok());
}

TEST_CASE("cuda fused s_full_and_force matches two-pass (SineGordon)", "[cuda]") {
    REQUIRE(reticolo::cuda::sine_gordon_fused_ok());
}

TEST_CASE("cuda fused s_full_and_force matches two-pass (Xy)", "[cuda]") {
    REQUIRE(reticolo::cuda::xy_fused_ok());
}
