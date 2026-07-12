#include <reticolo/core/exec/nn_site.hpp>
#include <reticolo/core/field/lattice.hpp>
#include <reticolo/core/rng/fast_rng.hpp>
#include <reticolo/core/field/site.hpp>

#include "nn_reference.hpp"

#include <cstddef>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::Lattice;
using reticolo::exec::nn_reduce_fwd;
using reticolo::exec::nn_visit_all;

namespace {

template <class Shape>
Lattice<double> make_random_lattice(Shape shape, std::uint64_t seed) {
    Lattice<double> l{Lattice<double>::SizeVec(shape.begin(), shape.end())};
    FastRng rng{seed};
    double* data = l.data();
    for (std::size_t i = 0; i < l.nsites(); ++i) {
        data[i] = rng.normal();
    }
    return l;
}

}  // namespace

TEST_CASE("nn_visit_all dispatch matches gather fallback in 1D/2D/3D/4D", "[hot_loop]") {
    std::vector<std::vector<std::size_t>> const shapes = {
        {8},           // 1D
        {6, 5},        // 2D
        {4, 5, 6},     // 3D
        {3, 4, 5, 3},  // 4D
    };

    for (auto const& shape : shapes) {
        auto l = make_random_lattice(shape, /*seed=*/42);
        std::vector<double> out_dispatch(l.nsites(), 0.0);
        std::vector<double> out_fallback(l.nsites(), 0.0);

        // Dispatch (uses the per-ndim specialisation).
        nn_visit_all<double>(l, [&out_dispatch](std::size_t i, double phi, double nbrs) {
            out_dispatch[i] = phi * 7.0 + nbrs;  // arbitrary linear body
        });
        // Reference (gather through the computed Indexing::next/prev).
        reticolo::test::visit_nn_ref<double>(
            l, [&out_fallback](std::size_t i, double phi, double nbrs) {
                out_fallback[i] = phi * 7.0 + nbrs;
            });

        for (std::size_t i = 0; i < l.nsites(); ++i) {
            INFO("ndims=" << shape.size() << " i=" << i);
            REQUIRE(out_dispatch[i] == out_fallback[i]);
        }
    }
}

TEST_CASE("nn_reduce_fwd dispatch matches gather fallback in 1D/2D/3D/4D", "[hot_loop]") {
    std::vector<std::vector<std::size_t>> const shapes = {
        {8},
        {6, 5},
        {4, 5, 6},
        {3, 4, 5, 3},
    };

    for (auto const& shape : shapes) {
        auto l = make_random_lattice(shape, /*seed=*/123);
        // Arbitrary non-trivial body that touches both phi and fwd_sum.
        auto body = [](double phi, double fwd) { return (phi * phi) - phi * fwd; };

        double const a = nn_reduce_fwd<double>(l, body);
        double const b = reticolo::test::reduce_fwd_ref<double>(l, body);

        // Both paths visit every (site, mu) pair with the same body; the
        // dispatch path accumulates in vector lanes and the fallback in a
        // single scalar accumulator, so the reduction order differs and the
        // sums can disagree in the last ULP across ISAs. Match at machine
        // precision instead of bit identity.
        INFO("ndims=" << shape.size());
        REQUIRE(a == Catch::Approx(b).margin(1e-12));
    }
}

TEST_CASE("nn_visit_all nbrs sums all 2*ndims neighbours (3D constant field)", "[hot_loop]") {
    Lattice<double> l{{4, 4, 4}, /*fill=*/3.5};
    // Write-disjoint capture (the map now threads): assert after the sweep, not
    // from inside the body — Catch2 REQUIRE is not thread-safe in a parallel region.
    std::vector<double> phi_seen(l.nsites(), 0.0);
    std::vector<double> nbr_seen(l.nsites(), 0.0);

    nn_visit_all<double>(l, [&](std::size_t i, double phi, double nbrs) {
        phi_seen[i] = phi;
        nbr_seen[i] = nbrs;
    });
    for (std::size_t i = 0; i < l.nsites(); ++i) {
        INFO("site=" << i);
        REQUIRE(phi_seen[i] == 3.5);
        REQUIRE(nbr_seen[i] == 21.0);  // 2 * 3 neighbours, each 3.5 => 6 * 3.5
    }
}

TEST_CASE("nn_reduce_fwd fwd_sum sums ndims forward neighbours (2D constant field)", "[hot_loop]") {
    Lattice<double> l{{5, 7}, /*fill=*/2.0};
    // body returns fwd_sum directly => total = sum_x (2 forward nbrs) = nsites * 4.0
    double const total = nn_reduce_fwd<double>(l, [](double /*phi*/, double fwd) { return fwd; });
    REQUIRE(total == static_cast<double>(l.nsites()) * 4.0);
}
