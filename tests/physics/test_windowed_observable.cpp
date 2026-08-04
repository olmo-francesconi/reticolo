#include <reticolo/reticolo.hpp>

#include <complex>
#include <cstddef>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace reticolo;
using C = std::complex<double>;

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

// The same observable, but exposing the fused `s_full_and_force` fast-path.
// WindowedAction probes for it and, when present, gets Q and −dQ/dφ from ONE
// sweep instead of two. Exact here because both kernels ignore the neighbour
// aggregate, so the fused pass evaluates identical per-site expressions.
template <class T = double>
struct MagSumFused : action::NNAction<MagSumFused<T>, T> {
    using value_type = T;
    [[nodiscard]] auto action_kernel() const noexcept {
        return [](T self, T /*agg*/) { return self; };
    }
    [[nodiscard]] auto force_kernel() const noexcept {
        return [](std::size_t /*i*/, T /*self*/, T /*agg*/) { return T{-1}; };
    }
    [[nodiscard]] double s_full_and_force(Lattice<T> const& l, Lattice<T>& force) const noexcept {
        return this->staged_force_energy(l, force, [](std::size_t /*i*/, T self, T /*agg*/) {
            return std::pair<T, T>{T{-1}, self};
        });
    }
};

template <class Obs>
constexpr bool k_constraint_fuses = requires(action::ObservableConstraint<Obs> const& c,
                                             act::Phi4<double> const& b,
                                             Lattice<double> const& l,
                                             Lattice<double>& f) { c.value_and_force(b, l, f); };

}  // namespace

// The fused path must be selected only when the observable offers it — the
// branch is compile-time, so a silent mis-detection would route every custom
// observable through the slow two-sweep form (or worse, the reverse).
TEST_CASE("ObservableConstraint fuses Q value+gradient only when the observable can",
          "[action][window][fused]") {
    STATIC_REQUIRE(k_constraint_fuses<MagSumFused<double>>);
    STATIC_REQUIRE_FALSE(k_constraint_fuses<MagSum<double>>);
}

// Fusing must not change the answer: the fused and unfused constraints wrap the
// same base action and window, so both force paths must agree to roundoff.
TEST_CASE("WindowedAction: fused observable force equals the two-sweep force",
          "[action][window][fused]") {
    Lattice<double> phi{{6, 6}};
    FastRng rng{31337};
    double* const d = phi.data();
    for (std::size_t i = 0; i < phi.nsites(); ++i) {
        d[i] = 0.3 * rng.normal();
    }

    act::Phi4<double> const base{.kappa = 0.18, .lambda = 1.0};
    action::WindowedAction<act::Phi4<double>,
                           double,
                           Lattice<double>,
                           action::ObservableConstraint<MagSumFused<double>>>
        fused{.base = base, .a = 0.3, .E_n = 2.0, .delta = 5.0};
    action::WindowedAction<act::Phi4<double>,
                           double,
                           Lattice<double>,
                           action::ObservableConstraint<MagSum<double>>>
        plain{.base = base, .a = 0.3, .E_n = 2.0, .delta = 5.0};

    Lattice<double> f_fused{phi.indexing()};
    Lattice<double> f_plain{phi.indexing()};
    fused.compute_force(phi, f_fused);
    plain.compute_force(phi, f_plain);

    // The fused kick is the path an LLR MD step actually runs.
    constexpr double k_dt = 0.05;
    Lattice<double> m_fused{phi.indexing()};
    Lattice<double> m_plain{phi.indexing()};
    fused.compute_force_and_kick(phi, m_fused, k_dt);
    plain.compute_force_and_kick(phi, m_plain, k_dt);

    for (std::size_t i = 0; i < phi.nsites(); ++i) {
        INFO("site " << i);
        REQUIRE(f_fused.data()[i] == Catch::Approx(f_plain.data()[i]).margin(1e-12));
        REQUIRE(m_fused.data()[i] == Catch::Approx(m_plain.data()[i]).margin(1e-13));
    }

    // And the fused window still agrees with a central FD of its own s_full.
    constexpr double eps = 1e-6;
    for (std::size_t i = 0; i < phi.nsites(); i += 7) {
        double const saved = d[i];
        d[i]               = saved + eps;
        double const sp    = fused.s_full(phi);
        d[i]               = saved - eps;
        double const sm    = fused.s_full(phi);
        d[i]               = saved;
        INFO("site " << i);
        REQUIRE(f_fused.data()[i] ==
                Catch::Approx(-(sp - sm) / (2.0 * eps)).epsilon(1e-5).margin(1e-6));
    }
}

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
    REQUIRE(r.energy() == Catch::Approx(q_direct).epsilon(1e-12));
}

// SelfConstraint windows the base action itself (Q = S_base) — the mode every
// shipped real-action LLR app uses (phi4_llr, su2_llr, u1_llr, ...). Same FD
// check as the custom-observable case above, through the fused self-force path.
TEST_CASE("WindowedAction: force matches FD of s_full for SelfConstraint (Phi4)",
          "[action][window]") {
    Lattice<double> phi{{6, 6}};
    FastRng rng{7};
    double* const d = phi.data();
    for (std::size_t i = 0; i < phi.nsites(); ++i) {
        d[i] = 0.3 * rng.normal();
    }

    using WA =
        action::WindowedAction<act::Phi4<double>, double, Lattice<double>, action::SelfConstraint>;
    WA wa{.base  = act::Phi4<double>{.kappa = 0.18, .lambda = 1.0},
          .a     = 0.3,
          .E_n   = 2.0,
          .delta = 5.0};

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

// The fused kick (`compute_force_and_kick`) is the path every LLR app actually
// runs each MD step; it must agree with the generic two-pass compute_force +
// kick_add. This is the branch a plain force-vs-FD test never exercises.
TEST_CASE("WindowedAction: compute_force_and_kick matches two-pass force+kick (SelfConstraint)",
          "[action][window]") {
    Lattice<double> phi{{6, 6}};
    FastRng rng{11};
    double* const d = phi.data();
    for (std::size_t i = 0; i < phi.nsites(); ++i) {
        d[i] = 0.3 * rng.normal();
    }

    using WA =
        action::WindowedAction<act::Phi4<double>, double, Lattice<double>, action::SelfConstraint>;
    WA wa{.base  = act::Phi4<double>{.kappa = 0.18, .lambda = 1.0},
          .a     = 0.3,
          .E_n   = 2.0,
          .delta = 5.0};
    STATIC_REQUIRE(action::HasFusedKick<WA, Lattice<double>>);

    constexpr double k_dt = 0.05;

    Lattice<double> force{phi.indexing()};
    wa.compute_force(phi, force);
    Lattice<double> mom_twopass{phi.indexing()};
    exec::kick_add(mom_twopass, force, k_dt);

    Lattice<double> mom_fused{phi.indexing()};
    wa.compute_force_and_kick(phi, mom_fused, k_dt);

    double const* const mt = mom_twopass.data();
    double const* const mf = mom_fused.data();
    for (std::size_t i = 0; i < phi.nsites(); ++i) {
        REQUIRE(mf[i] == Catch::Approx(mt[i]).epsilon(1e-10).margin(1e-12));
    }
}

// ImagConstraint windows the imaginary observable S_I — the mode every shipped
// sign-problem LLR app uses (bose_gas_llr). Couplings/shape/tolerances mirror
// tests/physics/test_bose_gas_force_consistency.cpp; force is the combined
// F_R + scale*F_I packaged as a single complex per site.
TEST_CASE("WindowedAction: force matches FD of s_full for ImagConstraint (BoseGas)",
          "[action][window]") {
    act::BoseGas<double> const base{.mass = 1.0, .lambda = 0.5, .mu = 0.4};
    Lattice<C> phi{{4, 4, 4, 4}};
    FastRng rng{321};
    C* const d = phi.data();
    for (std::size_t i = 0; i < phi.nsites(); ++i) {
        d[i] = C{rng.normal(), rng.normal()};
    }

    using WA =
        action::WindowedAction<act::BoseGas<double>, double, Lattice<C>, action::ImagConstraint>;
    WA wa{.base = base, .a = 0.3, .E_n = 0.5, .delta = 2.0};

    STATIC_REQUIRE(action::HmcAction<WA, Lattice<C>>);

    Lattice<C> force{phi.indexing()};
    wa.compute_force(phi, force);

    constexpr double k_eps = 1e-4;
    constexpr double k_tol = 1e-6;
    for (std::size_t i = 0; i < phi.nsites(); i += 20) {
        C const saved = d[i];

        d[i]              = saved + C{k_eps, 0.0};
        double const sr_p = wa.s_full(phi);
        d[i]              = saved - C{k_eps, 0.0};
        double const sr_m = wa.s_full(phi);

        d[i]              = saved + C{0.0, k_eps};
        double const si_p = wa.s_full(phi);
        d[i]              = saved - C{0.0, k_eps};
        double const si_m = wa.s_full(phi);

        d[i] = saved;

        REQUIRE(force.data()[i].real() ==
                Catch::Approx(-(sr_p - sr_m) / (2.0 * k_eps)).margin(k_tol));
        REQUIRE(force.data()[i].imag() ==
                Catch::Approx(-(si_p - si_m) / (2.0 * k_eps)).margin(k_tol));
    }
}

// The fused kick for ImagConstraint routes through ImagPart's
// `compute_force_combined_and_kick` (a single neighbour pass for F_R + scale*F_I)
// — the exact path bose_gas_llr executes every MD step. Must agree with the
// generic two-pass compute_force + kick_add.
TEST_CASE("WindowedAction: compute_force_and_kick matches two-pass force+kick (ImagConstraint)",
          "[action][window]") {
    act::BoseGas<double> const base{.mass = 1.0, .lambda = 0.5, .mu = 0.4};
    Lattice<C> phi{{4, 4, 4, 4}};
    FastRng rng{654};
    C* const d = phi.data();
    for (std::size_t i = 0; i < phi.nsites(); ++i) {
        d[i] = C{rng.normal(), rng.normal()};
    }

    using WA =
        action::WindowedAction<act::BoseGas<double>, double, Lattice<C>, action::ImagConstraint>;
    WA wa{.base = base, .a = 0.3, .E_n = 0.5, .delta = 2.0};
    STATIC_REQUIRE(action::HasFusedKick<WA, Lattice<C>>);

    constexpr double k_dt = 0.05;

    Lattice<C> force{phi.indexing()};
    wa.compute_force(phi, force);
    Lattice<C> mom_twopass{phi.indexing()};
    exec::kick_add(mom_twopass, force, k_dt);

    Lattice<C> mom_fused{phi.indexing()};
    wa.compute_force_and_kick(phi, mom_fused, k_dt);

    C const* const mt = mom_twopass.data();
    C const* const mf = mom_fused.data();
    for (std::size_t i = 0; i < phi.nsites(); ++i) {
        REQUIRE(mf[i].real() == Catch::Approx(mt[i].real()).epsilon(1e-10).margin(1e-12));
        REQUIRE(mf[i].imag() == Catch::Approx(mt[i].imag()).epsilon(1e-10).margin(1e-12));
    }
}
