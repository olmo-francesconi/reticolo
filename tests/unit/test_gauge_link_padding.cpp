#include <reticolo/action/gauge/wilson.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/rng/rng.hpp>
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
