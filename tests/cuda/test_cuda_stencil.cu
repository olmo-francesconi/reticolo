#include <reticolo/cuda/actions/nn/phi4.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_buffer.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/reduce_fwd.cuh>
#include <reticolo/cuda/stencil.cuh>

#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

// Force-vs-finite-difference gate for the scalar device protocol. Instantiates
// stencil<Phi4ForceFunctor> and reduce_fwd<Phi4EnergyFunctor> over a shared
// (kappa, lambda) and checks that the stencil force equals -dS/dphi computed by
// central differences of the reduce_fwd action. Validates BOTH skeletons and
// that their access policies (all-2d force vs forward-only action) are mutually
// consistent — a missing or double-counted neighbour breaks the identity by
// O(kappa), far above the FD tolerance. The functors call the SHARED HD per-site
// formula (action::formula::phi4_*), so this also exercises the real device Phi4
// math, not a stand-in.

namespace reticolo::cuda {

bool stencil_force_matches_fd() {
    std::vector<std::size_t> const shape{6, 4, 5};
    auto const topo = make_device_topology(shape);
    auto const n    = static_cast<std::size_t>(topo.nsites);

    // Deterministic smooth field in roughly (-0.5, 0.5); no RNG so the test is
    // reproducible and the FD step stays well-conditioned.
    std::vector<double> h(n);
    for (std::size_t i = 0; i < n; ++i) {
        h[i] = 0.5 * std::sin(0.3 * static_cast<double>(i) + 1.0);
    }

    Phi4ForceFunctor<double> const fforce{0.12, 0.7};
    Phi4EnergyFunctor<double> const fen{0.12, 0.7};

    DeviceBuffer<double> dfield{n};
    DeviceBuffer<double> dforce{n};
    DeviceBuffer<double> dscratch{n};

    dfield.copy_from_host(h.data());
    stencil_launch(fforce, dfield.data(), dforce.data(), topo);
    std::vector<double> force(n);
    dforce.copy_to_host(force.data());
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    double const eps    = 1e-4;
    long const probes[] = {0, 7, 31, 53, topo.nsites - 1};
    for (long const j : probes) {
        auto const jj = static_cast<std::size_t>(j);

        std::vector<double> hp = h;
        hp[jj]                 = h[jj] + eps;
        dfield.copy_from_host(hp.data());
        double const s_plus = reduce_fwd_launch(fen, dfield.data(), dscratch.data(), topo);

        hp[jj] = h[jj] - eps;
        dfield.copy_from_host(hp.data());
        double const s_minus = reduce_fwd_launch(fen, dfield.data(), dscratch.data(), topo);

        double const fd  = (s_minus - s_plus) / (2.0 * eps);  // -dS/dphi
        double const tol = 1e-5 * (1.0 + std::abs(force[jj])) + 1e-7;
        if (std::abs(fd - force[jj]) > tol) {
            return false;
        }
    }
    return true;
}

}  // namespace reticolo::cuda

#include <reticolo/action/gauge/wilson.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/math/group/su2.hpp>
#include <reticolo/math/group/su3.hpp>
#include <reticolo/math/group/u1.hpp>

#include <cmath>
#include <cstddef>

// Proves the gauge math headers parse and their hot bodies compile under nvcc
// once <sleef.h>/SIMD intrinsics are guarded out of vec_libm.hpp (the
// #ifndef __CUDACC__ split). Exercising s_full / compute_force / expi_lmul_slab
// forces instantiation of the transcendental paths (U(1) cos/sin batch, SU(3)
// acos/sincos batch) that now route through the scalar std fallback rather than
// Sleef. No device kernels.

namespace reticolo::cuda {

namespace {

template <class G>
[[nodiscard]] bool exercise() {
    MatrixLinkLattice<G> u{{4, 4, 4, 4}};
    MatrixLinkLattice<G> mom{u.indexing()};
    MatrixLinkLattice<G> force{u.indexing()};

    action::Wilson<G> w{};
    w.beta = 1.0;

    double const s = w.s_full(u);  // plaquette ReTr  (U(1): cos batch)
    w.compute_force(u, force);     // staple force    (U(1): sin batch)
    G::expi_lmul_slab(u.mu_block_data(0),
                      mom.mu_block_data(0),
                      0.01,
                      u.nsites());  // group exp        (SU(3): acos/sincos batch)

    return std::isfinite(s);
}

}  // namespace

bool gauge_headers_compile() {
    return exercise<math::group::SU3>() && exercise<math::group::SU2>() &&
           exercise<math::group::U1>();
}

}  // namespace reticolo::cuda

// The generic stencil + reduce_fwd device skeletons, driven by a dummy
// Phi4-shaped functor pair, satisfy force == -dS/dphi by central finite
// differences. Validates the scalar device action protocol end-to-end.
TEST_CASE("cuda stencil force matches finite-difference of reduce_fwd action", "[cuda]") {
    REQUIRE(reticolo::cuda::stencil_force_matches_fd());
}

// The gauge action/drift headers compile under nvcc (Sleef now guarded out of
// vec_libm.hpp) and produce finite output.
TEST_CASE("cuda gauge headers compile and run under nvcc", "[cuda]") {
    REQUIRE(reticolo::cuda::gauge_headers_compile());
}
