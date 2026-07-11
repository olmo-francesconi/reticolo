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

// The custom constraint threads through the full LLR replica: an orch::llr::Replica
// parameterised on ObservableConstraint<MagSum> samples phi^4 while its window /
// exchange / adaptation observable is the magnetization. `energy()` must read the
// custom observable of the current config (not the action).
TEST_CASE("orch::llr::Replica: window / exchange observable is a custom observable",
          "[llr][window]") {
    using Obs = action::ObservableConstraint<MagSum<double>>;
    using Rep = orch::llr::
        Replica<act::Phi4<double>, FastRng, updater::integ::Omelyan2, double, Lattice<double>, Obs>;

    act::Phi4<double> const base{.kappa = 0.18, .lambda = 1.0};
    Rep r{base,
          FastRng{5},
          Rep::Spec{.id = "r0", .shape = {4, 4, 4}, .e_n = 0.0, .delta = 10.0},
          updater::HmcSpec{.tau = 0.5, .n_md = 4}};

    r.thermalize(3, log::Mode::silent);
    (void)r.sample(1, log::Mode::silent);  // run a trajectory → populates the constraint cache

    // energy() returns the cached custom-observable value; it must equal a fresh
    // magnetization Σφ of the replica's current config.
    double q_direct         = 0.0;
    double const* const fld = r.field().data();
    for (std::size_t i = 0; i < r.field().nsites(); ++i) {
        q_direct += fld[i];
    }
    REQUIRE(static_cast<double>(r.energy()) == Catch::Approx(q_direct).epsilon(1e-12));
}
