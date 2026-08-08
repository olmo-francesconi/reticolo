// Contract checks for the API-uniformity changes. All static / O(1) numeric —
// no Monte Carlo. See docs/architecture.md and the action concepts header.

#include <reticolo/reticolo.hpp>

#include <array>
#include <complex>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace reticolo;

// ---- real_scalar_t: the trait underpinning real MD kick coefficients (#3) ----
TEST_CASE("real_scalar_t strips complex and passes real through", "[api][traits]") {
    STATIC_REQUIRE(std::is_same_v<real_scalar_t<double>, double>);
    STATIC_REQUIRE(std::is_same_v<real_scalar_t<float>, float>);
    STATIC_REQUIRE(std::is_same_v<real_scalar_t<std::complex<double>>, double>);
    STATIC_REQUIRE(std::is_same_v<real_scalar_t<std::complex<float>>, float>);
    // The LLR alias is the same trait.
    STATIC_REQUIRE(std::is_same_v<action::scalar_of_t<std::complex<double>>, double>);
}

// ---- #1: s_full / s_imag and their caches return `double` for every action,
// regardless of the field scalar type (volume sums accumulate in double). ----
namespace {
template <class A, class F>
constexpr bool s_full_returns_double =
    std::is_same_v<decltype(std::declval<A const&>().s_full(std::declval<Lattice<F> const&>())),
                   double>;
template <class A>
constexpr bool cache_returns_double =
    std::is_same_v<decltype(std::declval<A const&>().last_s_full()), double>;
}  // namespace

TEST_CASE("s_full and its cache return double across actions", "[api][sfull]") {
    STATIC_REQUIRE(s_full_returns_double<act::Phi4<double>, double>);
    STATIC_REQUIRE(s_full_returns_double<act::Phi6<double>, double>);
    STATIC_REQUIRE(s_full_returns_double<act::Xy<double>, double>);
    STATIC_REQUIRE(s_full_returns_double<act::SineGordon<double>, double>);

    STATIC_REQUIRE(cache_returns_double<act::Phi6<double>>);
    STATIC_REQUIRE(cache_returns_double<act::Xy<double>>);
    STATIC_REQUIRE(cache_returns_double<act::SineGordon<double>>);

    // float fields too — the sum still widens to double.
    STATIC_REQUIRE(s_full_returns_double<act::Phi6<float>, float>);
    STATIC_REQUIRE(s_full_returns_double<act::SineGordon<float>, float>);

    // Gauge link action.
    STATIC_REQUIRE(
        std::is_same_v<decltype(std::declval<act::Wilson<math::group::U1, double> const&>().s_full(
                           std::declval<MatrixLinkLattice<math::group::U1, double> const&>())),
                       double>);
}

TEST_CASE("BoseGas s_full / s_imag and caches return double", "[api][sfull][bose]") {
    using C = std::complex<double>;
    STATIC_REQUIRE(std::is_same_v<decltype(std::declval<act::BoseGas<double> const&>().s_full(
                                      std::declval<Lattice<C> const&>())),
                                  double>);
    STATIC_REQUIRE(std::is_same_v<decltype(std::declval<act::BoseGas<double> const&>().s_imag(
                                      std::declval<Lattice<C> const&>())),
                                  double>);
    STATIC_REQUIRE(
        std::is_same_v<decltype(std::declval<act::BoseGas<double> const&>().last_s_imag()),
                       double>);
}

// ---- #3: the fused MD kick takes a REAL scalar coefficient everywhere, so a
// complex-field action no longer carries a complex k_dt. ----
// Per-action `HasFusedKick` assertions live at the top of each action's
// force-consistency TU (phi4, phi6, sine_gordon, bose_gas), so they are not
// repeated here. What IS specific to this file: a COMPLEX-field action still
// takes a REAL kick coefficient — the contract that lets one integrator drive
// both real and complex fields.
TEST_CASE("BoseGas fused kick takes a real coefficient", "[api][kick]") {
    using C = std::complex<double>;
    // Gauge is the one family with no force-consistency static_assert of its own.
    STATIC_REQUIRE(action::HasFusedKick<act::Wilson<math::group::U1, double>,
                                        MatrixLinkLattice<math::group::U1, double>>);

    // The kick coefficient parameter is a real double, not std::complex.
    using KickArg = double;
    STATIC_REQUIRE(
        requires(act::BoseGas<double> const& a, Lattice<C> const& l, Lattice<C>& m, KickArg k) {
            a.compute_force_and_kick(l, m, k);
        });
}

TEST_CASE("BoseGas fused kick equals force-then-kick", "[api][kick][bose]") {
    using C = std::complex<double>;
    act::BoseGas<double> const bg{.mass = 1.0, .lambda = 0.5, .mu = 0.3};
    Lattice<C> field{Lattice<C>::SizeVec{4, 4, 4}};
    FastRng rng{7};
    for (C& v : field) {
        v = C{rng.normal(), rng.normal()};
    }
    double const k_dt = 0.123;

    Lattice<C> mom_fused{field.indexing()};
    Lattice<C> mom_ref{field.indexing()};
    for (std::size_t i = 0; i < field.nsites(); ++i) {
        mom_fused.data()[i] = C{1.0, -2.0};
        mom_ref.data()[i]   = C{1.0, -2.0};
    }

    bg.compute_force_and_kick(field, mom_fused, k_dt);

    Lattice<C> force{field.indexing()};
    bg.compute_force(field, force);
    for (std::size_t i = 0; i < field.nsites(); ++i) {
        mom_ref.data()[i] += k_dt * force.data()[i];
    }

    for (std::size_t i = 0; i < field.nsites(); ++i) {
        REQUIRE(mom_fused.data()[i].real() == Catch::Approx(mom_ref.data()[i].real()));
        REQUIRE(mom_fused.data()[i].imag() == Catch::Approx(mom_ref.data()[i].imag()));
    }
}

// ---- HmcResult exposes acceptance() as the blessed accessor. ----
TEST_CASE("HmcResult exposes acceptance()", "[api][hmc]") {
    updater::HmcResult const h{.dH = -0.1, .accepted = true};
    REQUIRE(h.acceptance() == 1.0);

    updater::HmcResult const rej{.dH = 2.0, .accepted = false};
    REQUIRE(rej.acceptance() == 0.0);
}

// ---- #7: the shared app setup helpers register the universal flags and join
// the output path. ----
TEST_CASE("app::common_flags registers shared flags; out_path joins", "[api][app]") {
    cli::Parser p{"test_app", "api uniformity app helper"};
    auto const cf    = app::common_flags(p, {.L = 6, .out = "foo.h5"});
    auto const& ndim = p.opt<int>("ndim", 3, "spatial dimensions");  // app-specific flag still ok

    std::array<char const*, 3> argv{"test_app", "--size=12", "--workspace=/tmp/ws"};
    p.parse(static_cast<int>(argv.size()), argv.data());

    REQUIRE(cf.L == 12);          // overridden on the command line
    REQUIRE(cf.seed == 42ULL);    // helper default
    REQUIRE(cf.out == "foo.h5");  // per-app default, not overridden
    REQUIRE(cf.workspace == "/tmp/ws");
    REQUIRE(ndim == 3);
    REQUIRE(app::out_path(cf) == "/tmp/ws/foo.h5");
}

// ---- one parameter shape for everything that carries an action ---------------
//
// The rule, pinned here so it cannot drift back: `<Action, Rng, Sampler[, …]>`,
// with the element and field types DERIVED (the action names both) and the MD
// integrator riding on the sampler tag rather than sitting beside it. These are
// all compile-time; the point is that a shape regression fails to build here
// rather than in fifteen apps.
TEST_CASE("orchestration templates share one parameter shape", "[api][orch]") {
    using Phi4 = act::Phi4<double>;
    using Su3  = act::Wilson<math::group::SU3, double>;

    using SpanHmc   = orch::span::Chain<Phi4, FastRng, updater::HmcSampler<>>;
    using SpanMetro = orch::span::Chain<Phi4, FastRng, updater::MetropolisSampler>;
    using SpanGauge = orch::span::Chain<Su3, FastRng, updater::HmcSampler<>>;
    using ReplHmc   = orch::llr::Replica<Phi4, FastRng, updater::HmcSampler<>>;
    using ReplMetro = orch::llr::Replica<Phi4, FastRng, updater::MetropolisSampler>;
    using ReplGauge = orch::llr::Replica<Su3, FastRng, updater::HmcSampler<>>;

    // The field comes from the ACTION, so gauge and scalar spell the same shape.
    STATIC_REQUIRE(std::is_same_v<SpanHmc::field_type, Lattice<double>>);
    STATIC_REQUIRE(std::is_same_v<ReplHmc::field_type, Lattice<double>>);
    STATIC_REQUIRE(
        std::is_same_v<SpanGauge::field_type, MatrixLinkLattice<math::group::SU3, double>>);
    STATIC_REQUIRE(
        std::is_same_v<ReplGauge::field_type, MatrixLinkLattice<math::group::SU3, double>>);

    // The tag names the updater; the integrator rides on it.
    STATIC_REQUIRE(std::is_same_v<SpanHmc::sampler_type, updater::Hmc<Phi4, FastRng>>);
    STATIC_REQUIRE(std::is_same_v<SpanMetro::sampler_type, updater::Metropolis<Phi4, FastRng>>);
    STATIC_REQUIRE(std::is_same_v<
                   orch::span::Chain<Phi4, FastRng, updater::HmcSampler<updater::integ::Omelyan4>>::
                       sampler_type,
                   updater::Hmc<Phi4, FastRng, updater::integ::Omelyan4>>);

    // Both worker kinds model the sampler contract, whichever sampler they hold.
    STATIC_REQUIRE(updater::Updater<SpanMetro::sampler_type>);
    STATIC_REQUIRE(updater::Updater<ReplMetro::sampler_type>);

    // A Spec carries ONLY the chosen sampler's knobs — never a field the sampler
    // in play would ignore.
    using SpanOrchHmc   = orch::span::Orchestrator<Phi4, FastRng, updater::HmcSampler<>>;
    using SpanOrchMetro = orch::span::Orchestrator<Phi4, FastRng, updater::MetropolisSampler>;
    using LlrOrchHmc    = orch::llr::Orchestrator<Phi4, FastRng, updater::HmcSampler<>>;
    using LlrOrchMetro  = orch::llr::Orchestrator<Phi4, FastRng, updater::MetropolisSampler>;

    STATIC_REQUIRE(std::is_same_v<decltype(SpanOrchHmc::Spec::sampler), updater::HmcSpec>);
    STATIC_REQUIRE(std::is_same_v<decltype(SpanOrchMetro::Spec::sampler), updater::MetropolisSpec>);
    STATIC_REQUIRE(std::is_same_v<decltype(LlrOrchHmc::Spec::sampler), updater::HmcSpec>);
    STATIC_REQUIRE(std::is_same_v<decltype(LlrOrchMetro::Spec::sampler), updater::MetropolisSpec>);
}

// `span::scan` is the sugar that fills Spec::points for the common one-knob
// linear grid — the span analogue of an LLR ladder. Endpoints must be exact
// (a sweep that misses kappa_max by a rounding step is a silently wrong study),
// and n = 1 must degenerate to the low end rather than divide by zero.
TEST_CASE("span::scan sweeps one member with exact endpoints", "[api][orch][span]") {
    using Phi4 = act::Phi4<double>;

    auto const pts = orch::span::scan(Phi4{.lambda = 1.5}, &Phi4::kappa, 0.10, 0.26, 5);
    REQUIRE(pts.size() == 5);
    REQUIRE(pts.front().kappa == Catch::Approx(0.10));
    REQUIRE(pts.back().kappa == Catch::Approx(0.26));
    REQUIRE(pts[2].kappa == Catch::Approx(0.18));
    for (auto const& a : pts) {
        REQUIRE(a.lambda == Catch::Approx(1.5));  // untouched members ride along
    }

    auto const one = orch::span::scan(Phi4{.lambda = 1.5}, &Phi4::kappa, 0.10, 0.26, 1);
    REQUIRE(one.size() == 1);
    REQUIRE(one.front().kappa == Catch::Approx(0.10));
}
