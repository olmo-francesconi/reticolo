#include <reticolo/action/gauge/wilson.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/rng/rng.hpp>
#include <reticolo/math/group/su2.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::MatrixLinkLattice;
using reticolo::action::Wilson;
using SU2 = reticolo::math::group::SU2;

// The Wilson S-reduction returns double whatever the link precision — the
// mixed-precision invariant (the β·n_plaq − (β/N)·Σ combine would cancel
// catastrophically in float otherwise).
static_assert(std::is_same_v<decltype(std::declval<Wilson<SU2, float> const&>().s_full(
                                 std::declval<MatrixLinkLattice<SU2, float> const&>())),
                             double>);
static_assert(std::is_same_v<decltype(std::declval<Wilson<SU2, double> const&>().s_full(
                                 std::declval<MatrixLinkLattice<SU2, double> const&>())),
                             double>);

// On a single random gauge configuration (no Monte Carlo), the float gauge
// kernels must reproduce the double kernels to within single precision. This
// validates the precision-generic force / kinetic / s_full directly.
TEST_CASE("SU(2): float kernels reproduce double on one configuration",
          "[physics][su2][mixed-precision]") {
    constexpr double k_beta = 2.3;
    MatrixLinkLattice<SU2, double> ud{{4, 4, 4, 4}};
    std::size_t const ns = ud.nsites();
    std::size_t const d  = ud.ndims();

    // Random config: cold (identity) then exp(0.5·P) per direction.
    for (std::size_t mu = 0; mu < d; ++mu) {
        double* const blk = ud.mu_block_data(mu);
        for (std::size_t s = 0; s < ns; ++s) {
            blk[(0 * ns) + s] = 1.0;
            blk[(6 * ns) + s] = 1.0;
        }
    }
    FastRng rng{7};
    std::vector<double> p(SU2::n_real_components * ns);
    for (std::size_t mu = 0; mu < d; ++mu) {
        SU2::sample_algebra_slab(p.data(), rng, ns);
        SU2::expi_lmul_slab(ud.mu_block_data(mu), p.data(), 0.5, ns);
    }

    // Narrow to a float copy.
    MatrixLinkLattice<SU2, float> uf{ud.shape()};
    for (std::size_t i = 0; i < ud.ncomponents(); ++i) {
        uf.data()[i] = static_cast<float>(ud.data()[i]);
    }

    // s_full agreement.
    Wilson<SU2, double> const ad{.beta = k_beta};
    Wilson<SU2, float> const af{.beta = static_cast<float>(k_beta)};
    REQUIRE(af.s_full(uf) == Catch::Approx(ad.s_full(ud)).epsilon(1.0e-4));

    // Force agreement, elementwise.
    MatrixLinkLattice<SU2, double> fd{ud.indexing()};
    MatrixLinkLattice<SU2, float> ff{uf.indexing()};
    reticolo::action::formula::wilson_kernels<SU2>::compute_force(ud, fd, k_beta / 2.0);
    reticolo::action::formula::wilson_kernels<SU2>::compute_force(uf, ff, k_beta / 2.0);
    for (std::size_t i = 0; i < fd.ncomponents(); ++i) {
        REQUIRE(static_cast<double>(ff.data()[i]) ==
                Catch::Approx(fd.data()[i]).epsilon(2.0e-3).margin(1.0e-4));
    }

    // Kinetic agreement (double-accumulated either way).
    SU2::sample_algebra_slab(p.data(), rng, ns);
    std::vector<float> pf(p.size());
    for (std::size_t i = 0; i < p.size(); ++i) {
        pf[i] = static_cast<float>(p[i]);
    }
    double const kd = SU2::kinetic_slab(p.data(), ns);
    double const kf = SU2::kinetic_slab(pf.data(), ns);
    REQUIRE(kf == Catch::Approx(kd).epsilon(1.0e-5));
}
