#include <reticolo/action/detail/gauge/gauge_group/su3.hpp>
#include <reticolo/action/gauge/wilson.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/rng.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::MatrixLinkLattice;
using reticolo::action::Wilson;
using SU3 = reticolo::gauge_group::SU3;

static_assert(std::is_same_v<decltype(std::declval<Wilson<SU3, float> const&>().s_full(
                                 std::declval<MatrixLinkLattice<SU3, float> const&>())),
                             double>);

// On a single random gauge configuration (no Monte Carlo), the float SU(3)
// kernels must reproduce the double kernels to within single precision. The
// SU(3) exp is double-internal on float storage, so the agreement is tight.
TEST_CASE("SU(3): float kernels reproduce double on one configuration",
          "[physics][su3][mixed-precision]") {
    constexpr double k_beta = 5.7;
    MatrixLinkLattice<SU3, double> ud{{4, 4, 4, 4}};
    std::size_t const ns = ud.nsites();
    std::size_t const d  = ud.ndims();

    // Random config: cold (identity) then exp(0.5·P) per direction.
    for (std::size_t mu = 0; mu < d; ++mu) {
        double* const blk = ud.mu_block_data(mu);
        for (std::size_t s = 0; s < ns; ++s) {
            blk[(0 * ns) + s]  = 1.0;
            blk[(8 * ns) + s]  = 1.0;
            blk[(16 * ns) + s] = 1.0;
        }
    }
    FastRng rng{11};
    std::vector<double> p(SU3::n_real_components * ns);
    for (std::size_t mu = 0; mu < d; ++mu) {
        SU3::sample_algebra_slab(p.data(), rng, ns);
        SU3::expi_lmul_slab(ud.mu_block_data(mu), p.data(), 0.5, ns);
    }

    MatrixLinkLattice<SU3, float> uf{ud.shape()};
    for (std::size_t i = 0; i < ud.ncomponents(); ++i) {
        uf.data()[i] = static_cast<float>(ud.data()[i]);
    }

    Wilson<SU3, double> const ad{.beta = k_beta};
    Wilson<SU3, float> const af{.beta = static_cast<float>(k_beta)};
    REQUIRE(af.s_full(uf) == Catch::Approx(ad.s_full(ud)).epsilon(1.0e-4));

    MatrixLinkLattice<SU3, double> fd{ud.indexing()};
    MatrixLinkLattice<SU3, float> ff{uf.indexing()};
    SU3::compute_force(ud, fd, k_beta / 3.0);
    SU3::compute_force(uf, ff, k_beta / 3.0);
    for (std::size_t i = 0; i < fd.ncomponents(); ++i) {
        REQUIRE(static_cast<double>(ff.data()[i]) ==
                Catch::Approx(fd.data()[i]).epsilon(2.0e-3).margin(1.0e-4));
    }

    SU3::sample_algebra_slab(p.data(), rng, ns);
    std::vector<float> pf(p.size());
    for (std::size_t i = 0; i < p.size(); ++i) {
        pf[i] = static_cast<float>(p[i]);
    }
    double const kd = SU3::kinetic_slab(p.data(), ns);
    double const kf = SU3::kinetic_slab(pf.data(), ns);
    REQUIRE(kf == Catch::Approx(kd).epsilon(1.0e-5));
}
