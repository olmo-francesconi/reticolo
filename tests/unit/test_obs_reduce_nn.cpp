#include <reticolo/action/nn/phi4.hpp>
#include <reticolo/core/field/lattice.hpp>
#include <reticolo/core/field/site.hpp>
#include <reticolo/obs/reduce.hpp>

#include <cmath>
#include <cstddef>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using reticolo::Lattice;
using reticolo::Site;
namespace obs  = reticolo::obs;
namespace exec = reticolo::exec;

// obs::reduce_nn hands each kernel (self, agg): agg is the Policy-selected
// neighbour aggregate. These check the aggregate is what the action families see,
// and that a kernel shaped like an action leaf reproduces s_full exactly.

TEST_CASE("reduce_nn FwdOnly aggregate = Σ forward neighbours", "[obs][reduce_nn]") {
    double const c = 0.5;
    Lattice<double> phi{{4, 4, 4}, c};  // 3 dims → 3 forward neighbours, each = c
    auto const [s_agg] = obs::reduce_nn(phi, [](double /*self*/, double agg) { return agg; });
    REQUIRE(s_agg == Catch::Approx(static_cast<double>(phi.nsites()) * 3.0 * c).margin(1e-12));
}

TEST_CASE("reduce_nn AllDirs aggregate = Σ all 2·d neighbours", "[obs][reduce_nn]") {
    double const c = 0.5;
    Lattice<double> phi{{4, 4}, c};  // 2 dims → 4 neighbours, each = c
    auto const [s_agg] =
        obs::reduce_nn<exec::AllDirs>(phi, [](double /*self*/, double agg) { return agg; });
    REQUIRE(s_agg == Catch::Approx(static_cast<double>(phi.nsites()) * 4.0 * c).margin(1e-12));
}

TEST_CASE("reduce_nn reproduces phi4 s_full (the action as a fused observable)",
          "[obs][reduce_nn]") {
    Lattice<double> phi{{6, 6}};
    for (Site const x : phi.sites()) {
        phi[x] = 0.1 * std::sin(0.3 * static_cast<double>(x.value()));
    }
    double const kappa = 0.18;
    double const lambda = 1.0;
    reticolo::action::Phi4<double> phi4{.kappa = kappa, .lambda = lambda};

    // Same per-site formula the leaf uses: S_site = -2κ φ·fwd + φ² + λ(φ²-1)².
    auto const [s] = obs::reduce_nn(phi, [=](double self, double fwd) {
        double const p2  = self * self;
        double const dev = p2 - 1.0;
        return (-2.0 * kappa * self * fwd) + p2 + (lambda * dev * dev);
    });
    REQUIRE(s == Catch::Approx(phi4.s_full(phi)).margin(1e-10));
}

TEST_CASE("reduce_nn fuses a neighbour kernel with a pure per-site kernel", "[obs][reduce_nn]") {
    double const c = 0.5;
    Lattice<double> phi{{4, 4}, c};
    auto const [s_agg, s_sq] = obs::reduce_nn(
        phi,
        [](double /*self*/, double agg) { return agg; },        // neighbour lane
        [](double self, double /*agg*/) { return self * self; });  // per-site lane, ignores agg
    auto const n = static_cast<double>(phi.nsites());
    REQUIRE(s_agg == Catch::Approx(n * 2.0 * c).margin(1e-12));  // FwdOnly, 2 dims
    REQUIRE(s_sq == Catch::Approx(n * c * c).margin(1e-12));
}
