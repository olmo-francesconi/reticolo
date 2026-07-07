// Wilson SU(2)/SU(3) force-vs-action consistency by finite difference.
//
// The scalar/U(1) force tests central-difference s_full against a single field
// DOF because those fields carry free real coordinates. A matrix gauge field
// does not: the DOF is a group element and the force lives in the Lie algebra,
// so there is no per-component correspondence to difference against.
//
// Instead we use the relation that makes HMC's own Hamiltonian conserved.
// With H = kinetic_slab(P) + s_full(U), kick P += dt·F and the drift map
// U ← exp(dt·P̂)·U, energy conservation forces, for any algebra element X,
//
//     d/dε S(exp(ε·X̂)·U) |_0  =  -Σ_k (∂K/∂P_k) X_k  =  -2·g(X, F)
//
// where g is the symmetric bilinear form with g(P,P) = kinetic_slab(P). We get
// g by polarization of the code's OWN kinetic_slab — so the check uses the
// exact kinetic metric the integrator uses and never re-encodes a generator
// normalization:
//
//     -2·g(X, F)  =  -(1/2)·[ kinetic_slab(X + F) − kinetic_slab(X − F) ].
//
// LHS is the finite difference of s_full under the group's own expi drift;
// RHS is built from compute_force. Agreement validates F = −grad S for the
// SU(N) staple force independently of the reversibility/order tests (which pin
// the integrator to itself, not to s_full).

#include <reticolo/action/concepts.hpp>
#include <reticolo/action/gauge/wilson.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/math/gauge_group/su2.hpp>
#include <reticolo/math/gauge_group/su3.hpp>
#include <reticolo/math/gauge_group/u1.hpp>

#include <cstddef>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::MatrixLinkLattice;
using reticolo::action::HmcAction;
using reticolo::action::Wilson;
using reticolo::gauge_group::SU2;
using reticolo::gauge_group::SU3;
using reticolo::gauge_group::U1;

namespace {

// Set every link to the identity element, then left-multiply by exp(scale·X0)
// with a random algebra X0 so the config is a generic (exact) group element.
// Matrix groups (SU(2)/SU(3)): identity is diag(1) in the NxN complex matrix.
// U(1) (nc == 1, storage IS the angle theta): identity is theta = 0, already
// set by the zero-fill above — there is no separate diagonal to write.
template <class G>
void hot_start(MatrixLinkLattice<G, double>& u, FastRng& rng, double scale) {
    constexpr std::size_t nc                 = G::n_real_components;
    [[maybe_unused]] constexpr std::size_t n = G::n_color;
    std::size_t const d                      = u.ndims();
    std::size_t const ns                     = u.nsites();

    double* const data      = u.data();
    std::size_t const total = d * nc * ns;
    for (std::size_t i = 0; i < total; ++i) {
        data[i] = 0.0;
    }
    if constexpr (nc > 1) {
        // Diagonal real parts of each mu-block → identity. Full NxN complex,
        // row-major: Re(i,i) sits at storage index 2*i*(n+1).
        for (std::size_t mu = 0; mu < d; ++mu) {
            double* const blk = u.mu_block_data(mu);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t s = 0; s < ns; ++s) {
                    blk[((2 * i * (n + 1)) * ns) + s] = 1.0;
                }
            }
        }
    }
    MatrixLinkLattice<G, double> x0{u.indexing()};
    for (std::size_t mu = 0; mu < d; ++mu) {
        G::sample_algebra_slab(x0.mu_block_data(mu), rng, ns);
        G::expi_lmul_slab(u.mu_block_data(mu), x0.mu_block_data(mu), scale, ns);
    }
}

template <class G>
void check_force_fd(double beta, unsigned seed) {
    using Field = MatrixLinkLattice<G, double>;
    static_assert(HmcAction<Wilson<G>, Field>);

    Wilson<G, double> const action{.beta = beta};
    Field u{{4, 4, 4, 4}};
    FastRng rng{seed};
    hot_start<G>(u, rng, 0.8);

    std::size_t const d  = u.ndims();
    std::size_t const ns = u.nsites();

    Field force{u.indexing()};
    action.compute_force(u, force);

    // Random algebra perturbation direction X (momentum-shaped).
    Field x{u.indexing()};
    for (std::size_t mu = 0; mu < d; ++mu) {
        G::sample_algebra_slab(x.mu_block_data(mu), rng, ns);
    }

    // predicted = -(1/2)[ kinetic_slab(X+F) - kinetic_slab(X-F) ], per mu-block.
    Field xpf{u.indexing()};
    Field xmf{u.indexing()};
    {
        double const* const xd  = x.data();
        double const* const fd  = force.data();
        double* const pd        = xpf.data();
        double* const md        = xmf.data();
        std::size_t const nflat = d * G::n_real_components * ns;
        for (std::size_t i = 0; i < nflat; ++i) {
            pd[i] = xd[i] + fd[i];
            md[i] = xd[i] - fd[i];
        }
    }
    double k_plus  = 0.0;
    double k_minus = 0.0;
    for (std::size_t mu = 0; mu < d; ++mu) {
        k_plus += G::kinetic_slab(xpf.mu_block_data(mu), ns);
        k_minus += G::kinetic_slab(xmf.mu_block_data(mu), ns);
    }
    double const predicted = -0.5 * (k_plus - k_minus);

    // Finite difference of s_full under the group's own expi drift.
    constexpr double k_eps = 1e-4;
    Field u_plus           = u;
    Field u_minus          = u;
    for (std::size_t mu = 0; mu < d; ++mu) {
        G::expi_lmul_slab(u_plus.mu_block_data(mu), x.mu_block_data(mu), k_eps, ns);
        G::expi_lmul_slab(u_minus.mu_block_data(mu), x.mu_block_data(mu), -k_eps, ns);
    }
    double const s_plus  = action.s_full(u_plus);
    double const s_minus = action.s_full(u_minus);
    double const ds_de   = (s_plus - s_minus) / (2.0 * k_eps);

    REQUIRE(ds_de == Catch::Approx(predicted).epsilon(1e-5));
}

}  // namespace

TEST_CASE("Wilson<SU2>: compute_force matches finite-difference of s_full",
          "[physics][gauge][su2][force]") {
    check_force_fd<SU2>(2.4, 20240703);
    check_force_fd<SU2>(1.7, 917);
}

TEST_CASE("Wilson<SU3>: compute_force matches finite-difference of s_full",
          "[physics][gauge][su3][force]") {
    check_force_fd<SU3>(6.0, 20240703);
    check_force_fd<SU3>(5.4, 917);
}

TEST_CASE("Wilson<U1>: compute_force matches finite-difference of s_full",
          "[physics][gauge][u1][force]") {
    check_force_fd<U1>(1.0, 20240703);
    check_force_fd<U1>(0.7, 917);
}
