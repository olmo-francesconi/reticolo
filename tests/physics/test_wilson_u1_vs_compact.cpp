// Bit-identity check: Wilson<U1>::{s_full, compute_force} must reproduce the
// matching CompactU1 methods on the same physical configuration. Establishes
// that the generic Wilson<G> action over MatrixLinkLattice<G,T> reduces to the
// hand-tuned CompactU1 path at N = 1.

#include <reticolo/action/compact_u1.hpp>
#include <reticolo/action/detail/gauge_group/u1.hpp>
#include <reticolo/action/wilson.hpp>
#include <reticolo/core/link_lattice.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <cmath>
#include <cstddef>

#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::LinkLattice;
using reticolo::MatrixLinkLattice;
using reticolo::Site;
using reticolo::action::CompactU1;
using reticolo::action::Wilson;
using reticolo::gauge_group::U1;

namespace {

// Fill both fields with the same random angles so they describe the same
// physical configuration. MatrixLinkLattice<U1, double> has nc = 1, so its
// flat buffer matches LinkLattice<double> element-for-element.
void fill_matched(LinkLattice<double>& a, MatrixLinkLattice<U1, double>& b, FastRng& rng) {
    std::size_t const n = a.nlinks();
    REQUIRE(n == b.nlinks());
    for (std::size_t i = 0; i < n; ++i) {
        double const v = rng.normal();
        a.data()[i]    = v;
        b.data()[i]    = v;
    }
}

}  // namespace

TEST_CASE("Wilson<U1>::s_full matches CompactU1::s_full on 4D L=6",
          "[physics][gauge][wilson][u1]") {
    LinkLattice<double>::SizeVec const shape{6, 6, 6, 6};
    LinkLattice<double> theta{shape};
    MatrixLinkLattice<U1, double> u{shape};
    FastRng rng{31415};
    fill_matched(theta, u, rng);

    CompactU1<double> const compact{.beta = 1.2};
    Wilson<U1, double> const wilson{.beta = 1.2};

    double const s_compact = compact.s_full(theta);
    double const s_wilson  = wilson.s_full(u);
    REQUIRE(std::abs(s_compact - s_wilson) < 1e-9 * std::max(1.0, std::abs(s_compact)));
}

TEST_CASE("Wilson<U1>::compute_force matches CompactU1::compute_force on 4D L=4",
          "[physics][gauge][wilson][u1]") {
    LinkLattice<double>::SizeVec const shape{4, 4, 4, 4};
    LinkLattice<double> theta{shape};
    MatrixLinkLattice<U1, double> u{shape};
    FastRng rng{27182};
    fill_matched(theta, u, rng);

    CompactU1<double> const compact{.beta = 0.9};
    Wilson<U1, double> const wilson{.beta = 0.9};

    LinkLattice<double> f_compact{shape};
    MatrixLinkLattice<U1, double> f_wilson{shape};
    compact.compute_force(theta, f_compact);
    wilson.compute_force(u, f_wilson);

    std::size_t const n = f_compact.nlinks();
    double max_err      = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        max_err = std::max(max_err, std::abs(f_compact.data()[i] - f_wilson.data()[i]));
    }
    REQUIRE(max_err < 1e-12);
}
