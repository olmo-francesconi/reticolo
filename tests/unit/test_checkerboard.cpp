#include <reticolo/action/nn.hpp>
#include <reticolo/core/exec/nn_checkerboard.hpp>
#include <reticolo/core/field/lattice.hpp>
#include <reticolo/core/rng/fast_rng.hpp>
#include <reticolo/updater/metropolis/metropolis.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

// Invariants of the checkerboard traversal, independent of any physics.
//
// The local sweep threads with no halo and no lock on the strength of ONE claim:
// the two colours partition the lattice, and every neighbour of a site is of the
// other colour. If that fails the sweep reads data it is concurrently writing,
// which no action-level test would catch — the result would just be subtly wrong
// and thread-count dependent.

namespace {

// Tag every site with its own index, so a neighbour's value identifies it.
reticolo::Lattice<double> indexed_lattice(reticolo::Lattice<double>::SizeVec const& shape) {
    reticolo::Lattice<double> l{shape};
    for (std::size_t i = 0; i < l.nsites(); ++i) {
        l.data()[i] = static_cast<double>(i);
    }
    return l;
}

}  // namespace

TEST_CASE("the two colours partition the lattice", "[core][checkerboard]") {
    using namespace reticolo;

    auto const shapes =
        std::vector<Lattice<double>::SizeVec>{{8}, {8, 8}, {6, 8}, {4, 6, 8}, {4, 4, 4, 4}};

    for (auto const& shape : shapes) {
        Lattice<double> l = indexed_lattice(shape);
        // A per-site counter, NOT a shared container: the traversal is threaded,
        // and the claim under test is exactly that each site is touched once, so
        // every thread writes a distinct slot. A std::set here would be a data
        // race — and one invisible under a toolchain without OpenMP.
        std::vector<int> visits(l.nsites(), 0);

        for (int color = 0; color < 2; ++color) {
            auto const n = exec::nn_color_reduce<std::size_t>(
                l, color, [&visits](std::size_t i, double, auto const&) {
                    ++visits[i];
                    return std::size_t{1};
                });
            // Each colour is exactly half the sites.
            REQUIRE(n == l.nsites() / 2);
        }
        INFO("ndims=" << shape.size());
        auto const once = std::count(visits.begin(), visits.end(), 1);
        REQUIRE(static_cast<std::size_t>(once) == l.nsites());
    }
}

TEST_CASE("every neighbour of a colour is of the other colour", "[core][checkerboard]") {
    using namespace reticolo;

    // This is the race-freedom argument, checked directly: for each visited site
    // the traversal must hand back neighbours whose own parity is the opposite.
    auto const shapes = std::vector<Lattice<double>::SizeVec>{{8, 8}, {4, 6, 8}, {4, 4, 4, 4}};

    for (auto const& shape : shapes) {
        Lattice<double> l   = indexed_lattice(shape);
        std::size_t const d = shape.size();
        std::vector<std::size_t> stride(d, 1);
        for (std::size_t k = 1; k < d; ++k) {
            stride[k] = stride[k - 1] * shape[k - 1];
        }
        auto const parity_of = [&](std::size_t s) {
            std::size_t p = 0;
            for (std::size_t k = 0; k < d; ++k) {
                p += (s / stride[k]) % shape[k];
            }
            return p & 1U;
        };

        for (int color = 0; color < 2; ++color) {
            std::size_t const bad = exec::nn_color_reduce<std::size_t>(
                l, color, [&](std::size_t i, double, auto const& gather) {
                    std::size_t wrong = 0;
                    if (parity_of(i) != static_cast<std::size_t>(color)) {
                        ++wrong;  // the site itself is the wrong colour
                    }
                    gather([&](std::size_t, double nbr) {
                        auto const ns = static_cast<std::size_t>(nbr);
                        if (parity_of(ns) == static_cast<std::size_t>(color)) {
                            ++wrong;
                        }
                    });
                    return wrong;
                });
            INFO("ndims=" << d << " color=" << color);
            REQUIRE(bad == 0);
        }
    }
}

TEST_CASE("the gather yields exactly 2*ndims neighbours per site", "[core][checkerboard]") {
    using namespace reticolo;
    Lattice<double> l          = indexed_lattice({4, 6, 8});
    std::size_t const expected = 2 * l.ndims();

    for (int color = 0; color < 2; ++color) {
        std::size_t const bad = exec::nn_color_reduce<std::size_t>(
            l, color, [&](std::size_t, double, auto const& gather) {
                std::size_t n = 0;
                gather([&](std::size_t, double) { ++n; });
                return n == expected ? std::size_t{0} : std::size_t{1};
            });
        REQUIRE(bad == 0);
    }
}

TEST_CASE("an odd extent is refused, not silently miscoloured", "[core][checkerboard]") {
    using namespace reticolo;

    // An odd extent wraps a colour onto itself under periodic BCs, so the
    // bipartite argument fails and the sweep would read what it writes. The
    // updater must refuse at construction rather than run a wrong chain.
    REQUIRE(exec::checkerboard_ok(Lattice<double>{{8, 8, 8}}));
    REQUIRE_FALSE(exec::checkerboard_ok(Lattice<double>{{8, 7, 8}}));
    REQUIRE_FALSE(exec::checkerboard_ok(Lattice<double>{{5}}));

    action::Phi4<double> const a{.kappa = 0.1, .lambda = 0.5};
    Lattice<double> odd{{8, 7}};
    REQUIRE_THROWS_AS((updater::Metropolis{a, odd, FastRng{1}, {.sigma = 0.4}, log::Mode::silent}),
                      std::invalid_argument);
}

TEST_CASE("the sequential nest visits every site exactly once", "[core][checkerboard]") {
    using namespace reticolo;

    // `nn_sequential_reduce` is the single-colour fallback for an action whose ΔS
    // reads a global scalar (the LLR window). It drops the parity split, so what
    // has to hold instead is that it covers the lattice — and that it still hands
    // back the full 2·ndims neighbour set, since the local energy is the same
    // all-directions fold either way. Odd extents are explicitly in the list:
    // unlike the colour walk, this one has no parity precondition.
    auto const shapes =
        std::vector<Lattice<double>::SizeVec>{{8}, {7}, {8, 8}, {4, 6, 8}, {5, 6, 3}, {4, 4, 4, 4}};

    for (auto const& shape : shapes) {
        Lattice<double> l = indexed_lattice(shape);
        std::vector<int> visits(l.nsites(), 0);
        std::size_t const expected = 2 * l.ndims();

        auto const bad_degree = exec::nn_sequential_reduce<std::size_t>(
            l, [&](std::size_t i, double, auto const& gather) {
                ++visits[i];
                std::size_t n = 0;
                gather([&](std::size_t, double) { ++n; });
                return n == expected ? std::size_t{0} : std::size_t{1};
            });

        INFO("ndims=" << shape.size());
        REQUIRE(bad_degree == 0);
        auto const once = std::count(visits.begin(), visits.end(), 1);
        REQUIRE(static_cast<std::size_t>(once) == l.nsites());
    }
}

TEST_CASE("the sequential nest reads its own writes", "[core][checkerboard]") {
    using namespace reticolo;

    // The defining difference from the colour walk: a site sees the UPDATED value
    // of an earlier-visited neighbour, which is exactly what lets a running global
    // scalar stay consistent within one pass. Zero the lattice as we go and count
    // how many sites still see a non-zero backward x-neighbour — under
    // lexicographic order that must be nobody except the wrap at x = 0.
    Lattice<double> l = indexed_lattice({8, 8});
    for (std::size_t i = 0; i < l.nsites(); ++i) {
        l.data()[i] = 1.0;  // 0 marks "already updated"
    }
    double* const data      = l.data();
    std::size_t const stale = exec::nn_sequential_reduce<std::size_t>(
        l, [data](std::size_t i, double, auto const& gather) {
            std::size_t updated = 0;
            gather([&](std::size_t, double nbr) { updated += (nbr == 0.0) ? 1 : 0; });
            data[i] = 0.0;
            // From site 1 on, the backward x-neighbour was written earlier in this
            // very pass, so it must read back as updated.
            return (i > 0 && updated == 0) ? std::size_t{1} : std::size_t{0};
        });
    REQUIRE(stale == 0);
}

TEST_CASE("colour indices cover the sites for the link-field walk", "[core][checkerboard]") {
    using namespace reticolo;

    // `color_index_reduce` is the index-only walk the gauge sweep uses (its
    // staple reaches two hops away, so the NN gather is the wrong shape). Same
    // partition guarantee, checked the same way.
    Lattice<double> l = indexed_lattice({4, 6, 8});
    std::vector<int> visits(l.nsites(), 0);  // per-site, see above
    for (int color = 0; color < 2; ++color) {
        auto const n = exec::color_index_reduce<std::size_t>(l, color, [&visits](std::size_t i) {
            ++visits[i];
            return std::size_t{1};
        });
        REQUIRE(n == l.nsites() / 2);
    }
    auto const once = std::count(visits.begin(), visits.end(), 1);
    REQUIRE(static_cast<std::size_t>(once) == l.nsites());
}
