#include <reticolo/action/detail/gauge_group/su2.hpp>
#include <reticolo/action/detail/gauge_group/su3.hpp>
#include <reticolo/action/detail/gauge_group/u1.hpp>
#include <reticolo/action/wilson.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/cuda/probes/gauge_probe.hpp>

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
    return exercise<gauge_group::SU3>() && exercise<gauge_group::SU2>() &&
           exercise<gauge_group::U1>();
}

}  // namespace reticolo::cuda
