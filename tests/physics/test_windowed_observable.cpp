#include <reticolo/reticolo.hpp>

#include <cstddef>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace reticolo;

namespace {

// A custom window observable, defined the SAME way as any action: an NNAction
// leaf that supplies only per-site formula kernels. Its `s_full`/`compute_force`
// run through the shared parallel sweep engine — no hand-rolled lattice loop.
// Q = Σ_x φ(x) (unnormalised magnetization): the per-site value is φ (the
// neighbour sum is ignored) and −dQ/dφ = −1 at every site.
template <class T = double>
struct MagSum : action::NNAction<MagSum<T>, T> {
    using value_type = T;
    [[nodiscard]] auto action_kernel() const noexcept {
        return [](T self, T /*agg*/) { return self; };
    }
    [[nodiscard]] auto force_kernel() const noexcept {
        return [](std::size_t /*i*/, T /*self*/, T /*agg*/) { return T{-1}; };
    }
};

}  // namespace

// The windowed force must be the analytic −dS_win/dφ, where the window is on an
// arbitrary observable Q (not the action, not its imaginary part). Checked
// against a central finite difference of `s_full`.
TEST_CASE("WindowedAction: force matches FD of s_full for a custom-observable window",
          "[action][window]") {
    Lattice<double> phi{{6, 6}};
    FastRng rng{7};
    double* const d = phi.data();
    for (std::size_t i = 0; i < phi.nsites(); ++i) {
        d[i] = 0.3 * rng.normal();
    }

    using WA = action::WindowedAction<act::Phi4<double>,
                                      double,
                                      Lattice<double>,
                                      action::ObservableConstraint<MagSum<double>>>;
    WA wa{.base  = act::Phi4<double>{.kappa = 0.18, .lambda = 1.0},
          .a     = 0.3,
          .E_n   = 2.0,
          .delta = 5.0};

    // WindowedAction is itself an HmcAction — it drives HMC directly.
    STATIC_REQUIRE(action::HmcAction<WA, Lattice<double>>);

    Lattice<double> force{phi.indexing()};
    wa.compute_force(phi, force);

    constexpr double eps = 1e-6;
    for (std::size_t i = 0; i < phi.nsites(); i += 5) {
        double const saved = d[i];
        d[i]               = saved + eps;
        double const sp    = wa.s_full(phi);
        d[i]               = saved - eps;
        double const sm    = wa.s_full(phi);
        d[i]               = saved;
        double const fd    = -(sp - sm) / (2.0 * eps);
        REQUIRE(force.data()[i] == Catch::Approx(fd).epsilon(1e-5).margin(1e-6));
    }
}
