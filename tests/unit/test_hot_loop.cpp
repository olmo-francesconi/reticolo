#include <reticolo/action/site/detail/traversal.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <cstddef>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::Lattice;
using reticolo::action::detail::reduce_fwd;
using reticolo::action::detail::reduce_fwd_fallback_;
using reticolo::action::detail::visit_nn;
using reticolo::action::detail::visit_nn_fallback_;

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

TEST_CASE("visit_nn dispatch matches gather fallback in 1D/2D/3D/4D", "[hot_loop]") {
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
        visit_nn<double>(l, [&out_dispatch](std::size_t i, double phi, double nbrs) {
            out_dispatch[i] = phi * 7.0 + nbrs;  // arbitrary linear body
        });
        // Fallback (gather through Indexing).
        visit_nn_fallback_<double>(
            l, std::size_t{0}, l.nsites(), [&out_fallback](std::size_t i, double phi, double nbrs) {
                out_fallback[i] = phi * 7.0 + nbrs;
            });

        for (std::size_t i = 0; i < l.nsites(); ++i) {
            INFO("ndims=" << shape.size() << " i=" << i);
            REQUIRE(out_dispatch[i] == out_fallback[i]);
        }
    }
}

TEST_CASE("reduce_fwd dispatch matches gather fallback in 1D/2D/3D/4D", "[hot_loop]") {
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

        double const a = reduce_fwd<double>(l, body);
        double const b = reduce_fwd_fallback_<double>(l, std::size_t{0}, l.nsites(), body);

        // Both paths visit every (site, mu) pair with the same body; the
        // dispatch path accumulates in vector lanes and the fallback in a
        // single scalar accumulator, so the reduction order differs and the
        // sums can disagree in the last ULP across ISAs. Match at machine
        // precision instead of bit identity.
        INFO("ndims=" << shape.size());
        REQUIRE(a == Catch::Approx(b).margin(1e-12));
    }
}

TEST_CASE("visit_nn nbrs sums all 2*ndims neighbours (3D constant field)", "[hot_loop]") {
    Lattice<double> l{{4, 4, 4}, /*fill=*/3.5};
    std::size_t visited = 0;
    double max_err      = 0.0;

    visit_nn<double>(l, [&](std::size_t /*i*/, double phi, double nbrs) {
        ++visited;
        REQUIRE(phi == 3.5);
        // 2 * 3 neighbours, each equal to 3.5 => sum = 6 * 3.5 = 21.0
        max_err = std::max(max_err, std::abs(nbrs - 21.0));
    });
    REQUIRE(visited == l.nsites());
    REQUIRE(max_err == 0.0);
}

TEST_CASE("reduce_fwd fwd_sum sums ndims forward neighbours (2D constant field)", "[hot_loop]") {
    Lattice<double> l{{5, 7}, /*fill=*/2.0};
    // body returns fwd_sum directly => total = sum_x (2 forward nbrs) = nsites * 4.0
    double const total = reduce_fwd<double>(l, [](double /*phi*/, double fwd) { return fwd; });
    REQUIRE(total == static_cast<double>(l.nsites()) * 4.0);
}
