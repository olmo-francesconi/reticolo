#include <reticolo/action/gauge/wilson.hpp>
#include <reticolo/core/field/matrix_link_lattice.hpp>
#include <reticolo/core/rng/fast_rng.hpp>
#include <reticolo/math/group/su3.hpp>

#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>

// Opt-in site-axis cache padding (g_gauge_link_padding) only RELOCATES a gauge
// field's per-component storage — it must not change a single computed value.
// Build the identical config + momentum on a PACKED lattice (link_span ==
// nsites) and a PADDED one (link_span > nsites), then check the Wilson action,
// the kinetic energy, and the full staple force are bit-for-bit equal. Any
// mismatch means a kernel is still keying a component stride off nsites instead
// of link_span.

using reticolo::FastRng;
using reticolo::MatrixLinkLattice;
using reticolo::action::Wilson;
using SU3 = reticolo::math::group::SU3;

namespace {

struct Result {
    std::size_t span;
    double s_full;
    double kinetic;
    std::vector<double> force;  // logical [mu][k][site], gathered off link_span
};

// Fill a field's buffer with `count` links of arbitrary (but seed-reproducible)
// real data at the field's own component stride. Not a valid group element —
// irrelevant here: we are testing that the kernels are stride-correct, not the
// physics, so identical logical data in → identical numbers out.
Result run(std::vector<std::size_t> const& shape, bool pad, unsigned seed) {
    reticolo::g_gauge_link_padding = pad;
    MatrixLinkLattice<SU3, double> u{shape};
    MatrixLinkLattice<SU3, double> p{shape};
    MatrixLinkLattice<SU3, double> f{shape};
    reticolo::g_gauge_link_padding = false;  // restore immediately after construction

    std::size_t const ns   = u.nsites();
    std::size_t const span = u.link_span();
    std::size_t const d    = u.ndims();

    FastRng rng_u{seed};
    for (std::size_t mu = 0; mu < d; ++mu) {
        SU3::sample_algebra_slab(u.mu_block_data(mu), rng_u, span, ns);
    }
    FastRng rng_p{seed + 1U};
    for (std::size_t mu = 0; mu < d; ++mu) {
        SU3::sample_algebra_slab(p.mu_block_data(mu), rng_p, span, ns);
    }

    Wilson<SU3, double> const action{.beta = 2.0};
    double const s = action.s_full_uncached(u);
    double kin     = 0.0;
    for (std::size_t mu = 0; mu < d; ++mu) {
        kin += SU3::kinetic_slab(p.mu_block_data(mu), span, ns);
    }
    action.force_into(u, f);

    std::vector<double> force;
    force.reserve(d * SU3::n_real_components * ns);
    for (std::size_t mu = 0; mu < d; ++mu) {
        double const* const blk = f.mu_block_data(mu);
        for (std::size_t k = 0; k < SU3::n_real_components; ++k) {
            for (std::size_t site = 0; site < ns; ++site) {
                force.push_back(blk[(k * span) + site]);
            }
        }
    }
    return {span, s, kin, std::move(force)};
}

}  // namespace

TEST_CASE("gauge link padding relocates storage without changing any value (SU3)",
          "[gauge][padding]") {
    for (auto const& shape :
         {std::vector<std::size_t>{4, 4, 4, 4}, std::vector<std::size_t>{8, 8}}) {
        auto const packed = run(shape, false, 12345);
        auto const padded = run(shape, true, 12345);

        // The test is only meaningful if padding actually widened the stride.
        REQUIRE(padded.span > packed.span);

        REQUIRE(padded.s_full == packed.s_full);
        REQUIRE(padded.kinetic == packed.kinetic);
        REQUIRE(padded.force == packed.force);
    }
}

// `set_cold_identity` must key its component stride off link_span(), not
// nsites(). Every gauge app used to open-code this loop with nsites, which is
// correct only while padding is off — exactly the configuration this file
// exists to test the other side of. With padding ON the two differ, and the
// nsites version writes 1.0 into the wrong components, so the "identity" is not
// even a group element. Nothing covered cold start under padding before, which
// is why the bug stayed latent.
TEST_CASE("set_cold_identity writes a real identity under link padding", "[gauge][padding]") {
    reticolo::MatrixLinkLattice<reticolo::math::group::SU3, double>::SizeVec const shape{
        4, 4, 4, 4};

    auto check_identity = [&](bool padded) {
        bool const saved               = reticolo::g_gauge_link_padding;
        reticolo::g_gauge_link_padding = padded;
        MatrixLinkLattice<reticolo::math::group::SU3, double> u{shape};
        reticolo::set_cold_identity(u);
        reticolo::g_gauge_link_padding = saved;

        std::size_t const span = u.link_span();
        std::size_t const ns   = u.nsites();
        INFO("padded=" << padded << " span=" << span << " nsites=" << ns);
        REQUIRE(padded == (span > ns));

        for (std::size_t mu = 0; mu < u.ndims(); ++mu) {
            double const* const blk = u.mu_block_data(mu);
            for (std::size_t k = 0; k < u.n_real_components; ++k) {
                bool const is_diag_re = (k == 0) || (k == 8) || (k == 16);
                double const want     = is_diag_re ? 1.0 : 0.0;
                for (std::size_t s = 0; s < ns; ++s) {
                    REQUIRE(blk[(k * span) + s] == want);
                }
            }
        }
    };

    check_identity(false);
    check_identity(true);
}
