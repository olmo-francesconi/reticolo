// StreamSet parallel-stream suite: per-family stream independence
// (xoshiro jump / Philox counter subspace / SplitMix-decorrelated seeds),
// driver isolation, bit-exact state round-trip, and the Hmc-level guarantees
// that rest on it (same-seed determinism, frozen threading config) — the
// properties the multi-stream checkpoint layout and Hmc's owned RNG depend on.

#include <reticolo/action/nn/phi4.hpp>
#include <reticolo/core/field/lattice.hpp>
#include <reticolo/core/log/log.hpp>
#include <reticolo/core/rng/fast_rng.hpp>
#include <reticolo/core/rng/mt19937_rng.hpp>
#include <reticolo/core/rng/philox_rng.hpp>
#include <reticolo/core/rng/ranlxd_rng.hpp>
#include <reticolo/core/rng/stream_set.hpp>
#include <reticolo/updater/hmc/hmc.hpp>
#include <reticolo/updater/hmc/integrators.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::JumpStream;
using reticolo::KeyedStream;
using reticolo::Mt19937Rng;
using reticolo::PhiloxRng;
using reticolo::RanlxdRng;
using reticolo::Rng;
using reticolo::StreamSet;

// Cross-family concept coverage: stream_set.hpp itself is family-agnostic
// (no concrete generator includes), so the family × concept matrix is
// pinned here instead.
static_assert(JumpStream<FastRng>);
static_assert(KeyedStream<PhiloxRng>);
static_assert(KeyedStream<RanlxdRng>);
static_assert(KeyedStream<Mt19937Rng>);
static_assert(Rng<StreamSet<FastRng>>);
static_assert(Rng<StreamSet<PhiloxRng>>);
static_assert(Rng<StreamSet<RanlxdRng>>);
static_assert(Rng<StreamSet<Mt19937Rng>>);

namespace {
// Odd count and a non-dividing stream count on purpose: blocks are uneven and
// some streams end a fill holding a cached Box-Muller spare — the states the
// round-trip below must capture.
constexpr std::size_t k_fill    = 501;
constexpr std::size_t k_streams = 4;

// StreamSet no longer holds a fill-splitting helper (normal_fill_sites /
// block() / visit_blocks() were removed — that's the owner's job now, see
// updater::Hmc::sample_momenta_). Mirror the same even/remainder split here so
// the round-trip test still exercises every site stream, in a fixed,
// reproducible order.
template <class R>
void fill_all_streams(StreamSet<R>& s, double* out, std::size_t n) {
    std::size_t const ns       = s.n_streams();
    std::size_t const base_cnt = n / ns;
    std::size_t const rem      = n % ns;
    std::size_t off            = 0;
    for (std::size_t k = 0; k < ns; ++k) {
        std::size_t const cnt = base_cnt + (k < rem ? 1 : 0);
        s.site_stream(k).normal_fill(out + off, cnt);
        off += cnt;
    }
}
}  // namespace

TEST_CASE("splitmix64 matches the reference sequence", "[rng][stream]") {
    REQUIRE(reticolo::splitmix64(0) == 0xE220A8397B1DCDAFULL);
}

TEST_CASE("FastRng::jump diverges from the origin and is deterministic", "[rng][stream]") {
    FastRng a{42};
    FastRng b = a;
    FastRng c = a;
    b.jump();
    c.jump();
    REQUIRE(b.uniform_u64() == c.uniform_u64());  // jump is a pure state map
    REQUIRE(a.uniform_u64() != b.uniform_u64());  // 2^128 apart
    FastRng d = b;
    d.jump();
    REQUIRE(d.uniform_u64() != b.uniform_u64());  // successive jumps differ
}

TEST_CASE("PhiloxRng stream 0 is bit-identical to a plain PhiloxRng", "[rng][stream]") {
    PhiloxRng plain{7};
    PhiloxRng s0 = PhiloxRng::stream(7, 0);
    PhiloxRng s1 = PhiloxRng::stream(7, 1);
    for (int i = 0; i < 64; ++i) {
        REQUIRE(s0.uniform_u64() == plain.uniform_u64());
    }
    REQUIRE(s1.uniform_u64() != PhiloxRng::stream(7, 0).uniform_u64());
}

TEMPLATE_TEST_CASE("StreamSet streams are mutually distinct",
                   "[rng][stream]",
                   FastRng,
                   PhiloxRng,
                   RanlxdRng,
                   Mt19937Rng) {
    StreamSet<TestType> s{42, k_streams};
    REQUIRE(s.driver().uniform_u64() != s.site_stream(0).uniform_u64());
    for (std::size_t k = 0; k + 1 < s.n_streams(); ++k) {
        REQUIRE(s.site_stream(k).uniform_u64() != s.site_stream(k + 1).uniform_u64());
    }
}

TEMPLATE_TEST_CASE("StreamSet state words round-trip bit-exact",
                   "[rng][stream]",
                   FastRng,
                   PhiloxRng,
                   RanlxdRng,
                   Mt19937Rng) {
    StreamSet<TestType> s{42, k_streams};
    std::vector<double> f(k_fill);
    fill_all_streams(s, f.data(), k_fill);  // odd blocks → live cached spares
    std::vector<std::uint64_t> const snap = s.state_words();
    REQUIRE(snap.size() == (k_streams + 1) * StreamSet<TestType>::words_per_stream);

    std::vector<double> cont(k_fill);
    fill_all_streams(s, cont.data(), k_fill);
    double const serial_cont = s.uniform();

    StreamSet<TestType> t{777, k_streams};  // restore must override the seed
    t.restore_state_words(snap);
    std::vector<double> resumed(k_fill);
    fill_all_streams(t, resumed.data(), k_fill);
    REQUIRE(resumed == cont);
    REQUIRE(t.uniform() == serial_cont);
}

TEMPLATE_TEST_CASE("StreamSet rejects a wrong-length state restore",
                   "[rng][stream]",
                   FastRng,
                   PhiloxRng) {
    StreamSet<TestType> a{1, 2};
    StreamSet<TestType> b{1, 4};
    REQUIRE_THROWS(b.restore_state_words(a.state_words()));
}

// ---- Hmc-level guarantees built on top of the owned StreamSet ----

TEST_CASE("Hmc constructed twice with the same seed is bit-identical", "[rng][stream][hmc]") {
    using reticolo::Lattice;
    using reticolo::action::Phi4;
    using reticolo::updater::Hmc;
    using reticolo::updater::HmcSpec;

    auto run = [] {
        Phi4<double> const action{.kappa = 0.18, .lambda = 1.0};
        Lattice<double> phi{{4, 4, 4}};
        FastRng seed_rng{123};
        for (double& v : phi) {
            v = seed_rng.normal();
        }
        Hmc hmc{action,
                phi,
                FastRng{99},
                HmcSpec{.tau = 0.3, .n_md = 4},
                reticolo::updater::integ::leapfrog,
                reticolo::log::Mode::silent};
        for (int t = 0; t < 3; ++t) {
            hmc.step(reticolo::log::Mode::silent);
        }
        return std::vector<double>{phi.data(), phi.data() + phi.nsites()};
    };

    auto const f1 = run();
    auto const f2 = run();
    REQUIRE(f1 == f2);
}

TEST_CASE("Hmc::set_spec throws on a threading change but not on tau/n_md", "[rng][stream][hmc]") {
    using reticolo::Lattice;
    using reticolo::action::Phi4;
    using reticolo::updater::Hmc;
    using reticolo::updater::HmcSpec;

    Phi4<double> const action{.kappa = 0.18, .lambda = 1.0};
    Lattice<double> phi{{4, 4, 4}};
    Hmc hmc{action,
            phi,
            FastRng{7},
            HmcSpec{.tau = 0.3, .n_md = 4, .n_threads = 1, .slabs_per_thread = 1},
            reticolo::updater::integ::leapfrog,
            reticolo::log::Mode::silent};

    REQUIRE_NOTHROW(hmc.set_spec({.tau = 0.7, .n_md = 8, .n_threads = 1, .slabs_per_thread = 1}));
    REQUIRE_THROWS_AS(hmc.set_spec({.tau = 0.7, .n_md = 8, .n_threads = 1, .slabs_per_thread = 2}),
                      std::logic_error);
}
