#include <reticolo/cuda/actions/phi4.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_buffer.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/probes/stencil_probe.hpp>
#include <reticolo/cuda/reduce_fwd.cuh>
#include <reticolo/cuda/stencil.cuh>

#include <cmath>
#include <cstddef>
#include <vector>

// Force-vs-finite-difference gate for the scalar device protocol. Instantiates
// stencil<Phi4ForceFunctor> and reduce_fwd<Phi4EnergyFunctor> over a shared
// (kappa, lambda) and checks that the stencil force equals -dS/dphi computed by
// central differences of the reduce_fwd action. Validates BOTH skeletons and
// that their access policies (all-2d force vs forward-only action) are mutually
// consistent — a missing or double-counted neighbour breaks the identity by
// O(kappa), far above the FD tolerance. The functors call the SHARED HD per-site
// formula (action::detail::phi4_*), so this also exercises the real device Phi4
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
