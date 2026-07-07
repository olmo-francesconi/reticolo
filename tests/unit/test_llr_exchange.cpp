#include <reticolo/action/site/phi4.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng/rng.hpp>
#include <reticolo/llr/exchange.hpp>
#include <reticolo/llr/replica.hpp>

#include <cstddef>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::Lattice;
using reticolo::action::Phi4;

namespace alg = reticolo::alg;
namespace llr = reticolo::llr;

using ReplicaT = llr::Replica<Phi4<double>, FastRng>;

namespace {

ReplicaT make_replica(char const* id, unsigned long long seed, double e_n) {
    Phi4<double> const base{.kappa = 0.18, .lambda = 1.0};
    return {base,
            FastRng{seed},
            ReplicaT::Spec{.id = id, .shape = {4, 4, 4}, .e_n = e_n, .delta = 10.0},
            alg::HmcSpec{.tau = 0.1, .n_md = 2}};
}

std::vector<double> snapshot(Lattice<double> const& phi) {
    return {phi.data(), phi.data() + phi.nsites()};
}

// Populate the base action's last_s_full cache the way a trajectory would
// (HMC's h1 s_full call), without running any MC.
double prime_energy_cache(ReplicaT const& r) {
    return r.windowed_action().base.s_full(r.phi());
}

}  // namespace

TEST_CASE("llr::try_exchange: accepted swap moves fields and energy caches",
          "[unit][llr][exchange]") {
    auto r0 = make_replica("r000", 11, 0.0);
    auto r1 = make_replica("r001", 22, 0.0);
    r0.hot_start(0.5);
    r1.hot_start(0.5);

    double const e0 = prime_energy_cache(r0);
    double const e1 = prime_energy_cache(r1);
    REQUIRE(r0.energy() == e0);
    REQUIRE(r1.energy() == e1);
    REQUIRE(e0 != e1);

    auto const phi0 = snapshot(r0.phi());
    auto const phi1 = snapshot(r1.phi());

    // (a0 - a1) * (e0 - e1) > 0 => log_p > 0 => deterministic accept.
    if (e0 > e1) {
        r0.set_a(1.0);
        r1.set_a(0.0);
    } else {
        r0.set_a(0.0);
        r1.set_a(1.0);
    }

    FastRng rng{3};
    REQUIRE(llr::try_exchange(r0, r1, rng));

    REQUIRE(snapshot(r0.phi()) == phi1);
    REQUIRE(snapshot(r1.phi()) == phi0);

    // The caches followed the configs: energy() must report the energy of
    // the config each replica now holds, and agree with a fresh sweep.
    REQUIRE(r0.energy() == e1);
    REQUIRE(r1.energy() == e0);
    Phi4<double> const probe{.kappa = 0.18, .lambda = 1.0};
    REQUIRE(probe.s_full(r0.phi()) == Catch::Approx(e1).epsilon(1e-12));
    REQUIRE(probe.s_full(r1.phi()) == Catch::Approx(e0).epsilon(1e-12));
}

TEST_CASE("llr::try_exchange: acceptance includes the Gaussian-window term",
          "[unit][llr][exchange]") {
    // Engineer a swap whose FULL log_p >= 0 (deterministic accept, no rng draw)
    // while the linear-only part is hugely negative (would essentially always
    // reject). Only passes if the window quadratic term is in the acceptance.
    auto r0 = make_replica("r000", 77, 0.0);
    auto r1 = make_replica("r001", 88, 0.0);
    r0.hot_start(0.5);
    r1.hot_start(0.5);

    double const e0 = prime_energy_cache(r0);
    double const e1 = prime_energy_cache(r1);
    REQUIRE(e0 != e1);

    // δ_0 = δ_1 = 1 ⇒ window term = (e0 − e1)(E_n1 − E_n0). Pick E_n so it = +60,
    // and a so the linear term = −50 ⇒ full log_p = +10 ≥ 0 (accept), but the
    // linear-only value −50 would reject with probability 1 − e^{−50} ≈ 1.
    r0.set_delta(1.0);
    r1.set_delta(1.0);
    r0.set_E_n(0.0);
    r1.set_E_n(60.0 / (e0 - e1));
    r0.set_a(-50.0 / (e0 - e1));
    r1.set_a(0.0);

    FastRng rng{9};
    REQUIRE(llr::try_exchange(r0, r1, rng));
}

TEST_CASE("llr::try_exchange: rejected swap leaves fields and energies untouched",
          "[unit][llr][exchange]") {
    auto r0 = make_replica("r000", 33, 0.0);
    auto r1 = make_replica("r001", 44, 0.0);
    r0.hot_start(0.5);
    r1.hot_start(0.5);

    double const e0 = prime_energy_cache(r0);
    double const e1 = prime_energy_cache(r1);
    REQUIRE(e0 != e1);

    auto const phi0 = snapshot(r0.phi());
    auto const phi1 = snapshot(r1.phi());

    // log_p = (a0 - a1)(e0 - e1) = -1000 => exp underflows => deterministic reject.
    r0.set_a(-1000.0 / (e0 - e1));
    r1.set_a(0.0);

    FastRng rng{5};
    REQUIRE_FALSE(llr::try_exchange(r0, r1, rng));

    REQUIRE(snapshot(r0.phi()) == phi0);
    REQUIRE(snapshot(r1.phi()) == phi1);
    REQUIRE(r0.energy() == e0);
    REQUIRE(r1.energy() == e1);
}
