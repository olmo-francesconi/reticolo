#include <reticolo/core/sys/hd.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/obs_reduce.cuh>
#include <reticolo/obs/reduce.hpp>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

// Gate for the device fused observable reduction. `cuda::reduce` folds a pack of
// per-site kernels in ONE pass over a DeviceField, one double lane each; the
// kernels are the SAME RETICOLO_HD objects `obs::reduce` folds on the host
// (obs::kernel::*), so this checks that one definition produces the same physics
// on both backends.
//
// Agreement is to tolerance, not bit-exact: the two folds have different
// reduction trees (device block-partials vs the host's fixed 8-lane
// item-order sum). That is the project's standing position — CPU↔GPU is the
// same ensemble, not the same chain.

namespace reticolo::cuda {

namespace {

// Deterministic smooth field, no RNG, so the test is reproducible and the sums
// are well-conditioned.
std::vector<double> host_field(std::size_t n) {
    std::vector<double> h(n);
    for (std::size_t i = 0; i < n; ++i) {
        h[i] = 0.5 * std::sin((0.3 * static_cast<double>(i)) + 1.0);
    }
    return h;
}

}  // namespace

// Namespace scope, not local to the test: CUDA forbids a type defined inside a
// host function as a template argument of a __global__ template instantiation.
struct CubeKernel {
    RETICOLO_HD double operator()(double self) const noexcept { return self * self * self; }
};

bool device_reduce_matches_host() {
    std::vector<std::size_t> const shape{6, 4, 5};
    auto const topo = make_device_topology(shape);
    auto const n    = static_cast<std::size_t>(topo.nsites);
    auto const h    = host_field(n);

    DeviceField<double> f{topo};
    f.copy_from_host(h.data());
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    auto const [d_phi, d_sq, d_quartic] =
        reduce(f, obs::kernel::phi, obs::kernel::phi_sq, obs::kernel::phi_quartic);

    // Host reference: plain serial sums over the same values.
    double r_phi     = 0.0;
    double r_sq      = 0.0;
    double r_quartic = 0.0;
    for (double const v : h) {
        r_phi += v;
        r_sq += v * v;
        r_quartic += v * v * v * v;
    }

    return d_phi == Catch::Approx(r_phi).epsilon(1e-12).margin(1e-12) &&
           d_sq == Catch::Approx(r_sq).epsilon(1e-12).margin(1e-12) &&
           d_quartic == Catch::Approx(r_quartic).epsilon(1e-12).margin(1e-12);
}

// A single lane and a pack must agree — fusing kernels must not perturb any
// individual lane's value.
bool device_reduce_fusion_is_consistent() {
    std::vector<std::size_t> const shape{8, 8, 4};
    auto const topo = make_device_topology(shape);
    auto const n    = static_cast<std::size_t>(topo.nsites);
    auto const h    = host_field(n);

    DeviceField<double> f{topo};
    f.copy_from_host(h.data());
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    auto const [solo_phi]            = reduce(f, obs::kernel::phi);
    auto const [solo_sq]             = reduce(f, obs::kernel::phi_sq);
    auto const [fused_phi, fused_sq] = reduce(f, obs::kernel::phi, obs::kernel::phi_sq);

    return solo_phi == fused_phi && solo_sq == fused_sq;
}

// Repeated runs on the same field must give bit-identical sums: the launch
// config is fixed, so the reduction tree cannot vary run to run.
bool device_reduce_is_deterministic() {
    std::vector<std::size_t> const shape{10, 6, 6};
    auto const topo = make_device_topology(shape);
    auto const n    = static_cast<std::size_t>(topo.nsites);
    auto const h    = host_field(n);

    DeviceField<double> f{topo};
    f.copy_from_host(h.data());
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    auto const [a_phi, a_sq] = reduce(f, obs::kernel::phi, obs::kernel::phi_sq);
    for (int rep = 0; rep < 4; ++rep) {
        auto const [b_phi, b_sq] = reduce(f, obs::kernel::phi, obs::kernel::phi_sq);
        if (a_phi != b_phi || a_sq != b_sq) {
            return false;
        }
    }
    return true;
}

// A lambda kernel must fuse in exactly like the builtins — the point of the
// pack is that the device is no longer limited to the two hardcoded reductions.
bool device_reduce_accepts_a_custom_kernel() {
    std::vector<std::size_t> const shape{4, 4, 4};
    auto const topo = make_device_topology(shape);
    auto const n    = static_cast<std::size_t>(topo.nsites);
    auto const h    = host_field(n);

    DeviceField<double> f{topo};
    f.copy_from_host(h.data());
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    auto const [cube] = reduce(f, CubeKernel{});

    double ref = 0.0;
    for (double const v : h) {
        ref += v * v * v;
    }
    return cube == Catch::Approx(ref).epsilon(1e-12).margin(1e-12);
}

}  // namespace reticolo::cuda

TEST_CASE("cuda obs::reduce matches the host sums", "[cuda][obs]") {
    REQUIRE(reticolo::cuda::device_reduce_matches_host());
}

TEST_CASE("cuda obs::reduce lanes are unaffected by fusion", "[cuda][obs]") {
    REQUIRE(reticolo::cuda::device_reduce_fusion_is_consistent());
}

TEST_CASE("cuda obs::reduce is run-to-run deterministic", "[cuda][obs]") {
    REQUIRE(reticolo::cuda::device_reduce_is_deterministic());
}

TEST_CASE("cuda obs::reduce fuses a custom kernel", "[cuda][obs]") {
    REQUIRE(reticolo::cuda::device_reduce_accepts_a_custom_kernel());
}
