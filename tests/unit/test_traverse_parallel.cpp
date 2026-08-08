#include <reticolo/action/complex/bose_gas.hpp>
#include <reticolo/action/gauge/wilson.hpp>
#include <reticolo/action/nn/phi4.hpp>
#include <reticolo/action/nn/sine_gordon.hpp>
#include <reticolo/core/exec/nn_site.hpp>
#include <reticolo/core/exec/parallel.hpp>
#include <reticolo/core/field/lattice.hpp>
#include <reticolo/core/field/matrix_link_lattice.hpp>
#include <reticolo/core/log/log.hpp>
#include <reticolo/core/rng/fast_rng.hpp>
#include <reticolo/math/group/su2.hpp>
#include <reticolo/math/group/su3.hpp>
#include <reticolo/math/group/u1.hpp>
#include <reticolo/updater/hmc/hmc.hpp>

#include "nn_reference.hpp"

#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#ifdef _OPENMP
    #include <omp.h>
#endif

using reticolo::FastRng;
using reticolo::Lattice;
using reticolo::MatrixLinkLattice;
using reticolo::action::BoseGas;
using reticolo::action::Phi4;
using reticolo::action::SineGordon;
using reticolo::action::Wilson;
using reticolo::math::group::SU2;
using reticolo::math::group::SU3;
using reticolo::math::group::U1;

namespace {

// A lattice whose canonical partition has many slabs, so compute_force / s_full
// exercise the threaded traversal path. One shape per dimensionality (2D/3D/4D)
// so all the per-dim parallel paths are covered.
Lattice<double> hot_lattice(Lattice<double>::SizeVec const& shape) {
    Lattice<double> phi{shape};
    FastRng rng{2024};
    double* const d = phi.data();
    for (std::size_t i = 0; i < phi.nsites(); ++i) {
        d[i] = rng.normal();
    }
    return phi;
}

std::vector<double> force_vec(Phi4<double> const& action, Lattice<double> const& phi) {
    Lattice<double> f{phi.shape()};
    action.compute_force(phi, f);
    return {f.data(), f.data() + f.nsites()};
}

// {2D 256², 3D 42³, 4D 16⁴} — each yields ≫64 partition slabs (the threaded path).
std::vector<Lattice<double>::SizeVec> hot_shapes() {
    return {{256, 256}, {42, 42, 42}, {16, 16, 16, 16}};
}

// Set every link to the identity element, then left-multiply by exp(scale·X0)
// with a random algebra X0, so the config is a generic (exact) group element —
// same recipe as tests/physics/test_wilson_force_consistency.cpp's hot_start,
// duplicated locally since this file may only touch itself.
template <class G>
void gauge_hot_start(MatrixLinkLattice<G, double>& u, FastRng& rng, double scale) {
    constexpr std::size_t nc = G::n_real_components;
    std::size_t const d      = u.ndims();
    std::size_t const ns     = u.nsites();

    double* const data      = u.data();
    std::size_t const total = d * nc * ns;
    for (std::size_t i = 0; i < total; ++i) {
        data[i] = 0.0;
    }
    if constexpr (nc > 1) {
        constexpr std::size_t n_color = G::n_color;
        for (std::size_t mu = 0; mu < d; ++mu) {
            double* const blk = u.mu_block_data(mu);
            for (std::size_t i = 0; i < n_color; ++i) {
                for (std::size_t s = 0; s < ns; ++s) {
                    blk[((2 * i * (n_color + 1)) * ns) + s] = 1.0;
                }
            }
        }
    }
    MatrixLinkLattice<G, double> x0{u.shape()};
    for (std::size_t mu = 0; mu < d; ++mu) {
        G::sample_algebra_slab(x0.mu_block_data(mu), rng, ns);
        G::expi_lmul_slab(u.mu_block_data(mu), x0.mu_block_data(mu), scale, ns);
    }
}

template <class G>
std::vector<double> gauge_force_vec(Wilson<G, double> const& action,
                                    MatrixLinkLattice<G, double> const& u) {
    MatrixLinkLattice<G, double> f{u.shape()};
    action.compute_force(u, f);
    return {f.data(), f.data() + f.ncomponents()};
}

std::vector<std::complex<double>> bose_force_vec(BoseGas<double> const& action,
                                                 Lattice<std::complex<double>> const& phi) {
    Lattice<std::complex<double>> f{phi.shape()};
    action.compute_force(phi, f);
    return {f.data(), f.data() + f.nsites()};
}

}  // namespace

// The threaded per-dim force pass must land on the identical answer as the plain
// gather over computed next/prev (no neighbour table anymore) — each site is
// written exactly once and its neighbour sum is in the same order, so the
// tiling/slab decomposition and thread count are bit-for-bit irrelevant.
TEST_CASE("threaded compute_force equals the gather fallback, every dimension",
          "[hot_loop][parallel]") {
    Phi4<double> const action{.kappa = 0.18, .lambda = 1.0};
    for (auto const& shape : hot_shapes()) {
        auto const phi = hot_lattice(shape);

        auto kern = action.force_kernel();
        std::vector<double> ref(phi.nsites(), 0.0);
        reticolo::test::visit_nn_ref<double>(
            phi, [&](std::size_t i, double p, double nb) { ref[i] = kern(i, p, nb); });

        auto const got = force_vec(action, phi);
        for (std::size_t i = 0; i < ref.size(); ++i) {
            INFO("ndims=" << shape.size() << " site=" << i);
            REQUIRE(got[i] == ref[i]);
        }
    }
}

// The force is a write-disjoint map — each site's value is independent of how the
// lattice is slabbed — so it stays bit-identical across thread counts (the slab
// count now tracks the team). s_full is a reduction: deterministic for a FIXED team
// (no race), but it re-folds when the team changes, so it is NOT compared across
// thread counts here — only run-to-run at a fixed team.
TEST_CASE("threaded force is thread-count invariant; s_full is fixed-team deterministic",
          "[hot_loop][parallel]") {
    Phi4<double> const action{.kappa = 0.18, .lambda = 1.0};
    for (auto const& shape : hot_shapes()) {
        auto const phi = hot_lattice(shape);

        auto set_threads = [](int nthr) {
#ifdef _OPENMP
            omp_set_num_threads(nthr);
#else
            (void)nthr;
#endif
        };

        set_threads(1);
        auto const f_ref = force_vec(action, phi);
        for (int nthr : {1, 2, 4, 8}) {
            set_threads(nthr);
            auto const f = force_vec(action, phi);
            INFO("ndims=" << shape.size() << " threads=" << nthr);
            for (std::size_t i = 0; i < f.size(); ++i) {
                REQUIRE(f[i] == f_ref[i]);  // map: bit-identical for any team
            }
            double const s0 = action.s_full(phi);  // reduce: deterministic at this team
            REQUIRE(action.s_full(phi) == s0);
        }
    }
}

// exec::slab_scope raises the partition target from `team` items to
// `team × slabs_per_thread` — finer slicing, more (smaller) items. No test in
// the tree has ever run above 1 slab/thread: the single largest coverage gap
// here. Force stays a write-disjoint map regardless of how finely the field
// is sliced, so it must be bit-identical across slab counts; s_full is only
// checked run-to-run at each FIXED slab setting (a reduce re-folds when the
// partition changes). partition() reads g_slabs_per_thread unconditionally —
// this exercises the threaded path even serial, so no #ifdef _OPENMP guard.
TEST_CASE(
    "threaded compute_force is slabs_per_thread invariant; s_full is fixed-slab deterministic",
    "[hot_loop][parallel]") {
    Phi4<double> const action{.kappa = 0.18, .lambda = 1.0};
    for (auto const& shape : hot_shapes()) {
        auto const phi = hot_lattice(shape);

        std::vector<double> f_ref;
        {
            reticolo::exec::slab_scope const slabs{1};
            f_ref = force_vec(action, phi);
        }
        for (int n_slabs : {1, 2, 4}) {
            reticolo::exec::slab_scope const slabs{n_slabs};
            auto const f = force_vec(action, phi);
            INFO("ndims=" << shape.size() << " slabs_per_thread=" << n_slabs);
            for (std::size_t i = 0; i < f.size(); ++i) {
                REQUIRE(f[i] == f_ref[i]);  // map: bit-identical for any slab count
            }
            double const s0 = action.s_full(phi);  // reduce: deterministic at this slab count
            REQUIRE(action.s_full(phi) == s0);
        }
    }
}

// SineGordon exercises the extra per-site transcendental passes: prep() sin-batches
// the force scratch, s_full cos-batches + reduces. The force stays bit-identical
// across thread counts (map); s_full is checked run-to-run at a fixed team (reduce).
TEST_CASE("SineGordon force is thread-count invariant; s_full is fixed-team deterministic",
          "[hot_loop][parallel]") {
    auto const phi = hot_lattice({16, 16, 16, 16});
    SineGordon<double> const action{.kappa = 0.18, .alpha = 0.7};

    auto force = [&] {
        Lattice<double> f{phi.shape()};
        action.compute_force(phi, f);  // triggers prep() sin-batch + nn_visit_all
        return std::vector<double>{f.data(), f.data() + f.nsites()};
    };
    auto set_threads = [](int nthr) {
#ifdef _OPENMP
        omp_set_num_threads(nthr);
#else
        (void)nthr;
#endif
    };

    set_threads(1);
    auto const f_ref = force();
    for (int nthr : {1, 2, 4, 8}) {
        set_threads(nthr);
        auto const f = force();
        INFO("threads=" << nthr);
        for (std::size_t i = 0; i < f.size(); ++i) {
            REQUIRE(f[i] == f_ref[i]);
        }
        double const s0 = action.s_full(phi);
        REQUIRE(action.s_full(phi) == s0);
    }
}

// The FULL trajectory — momentum fill, snapshot copy, kinetic, s_full, MD drifts
// and fused kicks — is one deterministic function of (field, seed) for a FIXED team
// × slab config (every per-site op runs on the same partition, every reduce folds
// the same partials, no races). Two steps run twice at the same thread count must
// match ΔH and the evolved field bit-for-bit. (Across thread counts the reduces
// re-fold, so a chain reproduces only at the same threading — by design.)
TEST_CASE("full hmc.step is deterministic for a fixed team", "[hot_loop][parallel]") {
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    auto run = [&] {
        auto phi = hot_lattice({20, 20, 20, 20});
        Phi4<double> const action{.kappa = 0.18, .lambda = 1.0};
        FastRng rng{2026};
        reticolo::updater::Hmc hmc{action,
                                   phi,
                                   rng,
                                   {.tau = 0.5, .n_md = 6},
                                   reticolo::updater::integ::leapfrog,
                                   reticolo::log::Mode::silent};
        double dh_sum = 0.0;
        for (int t = 0; t < 2; ++t) {
            dh_sum += hmc.step(reticolo::log::Mode::silent).dH;
        }
        return std::pair{dh_sum, std::vector<double>{phi.data(), phi.data() + phi.nsites()}};
    };

    auto const [dh_ref, f_ref] = run();
    auto const [dh, f]         = run();
    REQUIRE(dh == dh_ref);
    for (std::size_t i = 0; i < f.size(); ++i) {
        REQUIRE(f[i] == f_ref[i]);
    }
}

// The hand-rolled batched gauge kernels (Wilson<SU2/SU3/U1>) and the split-last
// complex engine (BoseGas, core/exec/nn_split.hpp — otherwise zero direct test
// coverage) have never run above one thread anywhere in the tree. Same map
// argument as Phi4/SineGordon above: compute_force is write-disjoint, so it
// stays bit-identical across thread counts. Lattices stay small — unit-test
// budget, not a benchmark.
TEST_CASE("gauge and complex actions: compute_force is thread-count invariant",
          "[hot_loop][parallel]") {
    auto set_threads = [](int nthr) {
#ifdef _OPENMP
        omp_set_num_threads(nthr);
#else
        (void)nthr;
#endif
    };

    SECTION("Wilson<SU2>") {
        Wilson<SU2, double> const action{.beta = 2.4};
        MatrixLinkLattice<SU2, double> u{{4, 4, 4, 4}};
        FastRng rng{4242};
        gauge_hot_start<SU2>(u, rng, 0.8);

        set_threads(1);
        auto const f_ref = gauge_force_vec<SU2>(action, u);
        for (int nthr : {1, 2, 4}) {
            set_threads(nthr);
            auto const f = gauge_force_vec<SU2>(action, u);
            INFO("threads=" << nthr);
            for (std::size_t i = 0; i < f.size(); ++i) {
                REQUIRE(f[i] == f_ref[i]);
            }
        }
    }

    SECTION("Wilson<SU3>") {
        Wilson<SU3, double> const action{.beta = 6.0};
        MatrixLinkLattice<SU3, double> u{{4, 4, 4, 4}};
        FastRng rng{4343};
        gauge_hot_start<SU3>(u, rng, 0.8);

        set_threads(1);
        auto const f_ref = gauge_force_vec<SU3>(action, u);
        for (int nthr : {1, 2, 4}) {
            set_threads(nthr);
            auto const f = gauge_force_vec<SU3>(action, u);
            INFO("threads=" << nthr);
            for (std::size_t i = 0; i < f.size(); ++i) {
                REQUIRE(f[i] == f_ref[i]);
            }
        }
    }

    SECTION("Wilson<U1>") {
        Wilson<U1, double> const action{.beta = 1.0};
        MatrixLinkLattice<U1, double> u{{4, 4, 4, 4}};
        FastRng rng{4444};
        gauge_hot_start<U1>(u, rng, 0.8);

        set_threads(1);
        auto const f_ref = gauge_force_vec<U1>(action, u);
        for (int nthr : {1, 2, 4}) {
            set_threads(nthr);
            auto const f = gauge_force_vec<U1>(action, u);
            INFO("threads=" << nthr);
            for (std::size_t i = 0; i < f.size(); ++i) {
                REQUIRE(f[i] == f_ref[i]);
            }
        }
    }

    SECTION("BoseGas") {
        BoseGas<double> const action{.mass = 1.0, .lambda = 0.5, .mu = 0.4};
        Lattice<std::complex<double>> phi{{4, 4, 4, 4}};
        FastRng rng{4545};
        std::complex<double>* const d = phi.data();
        for (std::size_t i = 0; i < phi.nsites(); ++i) {
            d[i] = {rng.normal(), rng.normal()};
        }

        set_threads(1);
        auto const f_ref = bose_force_vec(action, phi);
        for (int nthr : {1, 2, 4}) {
            set_threads(nthr);
            auto const f = bose_force_vec(action, phi);
            INFO("threads=" << nthr);
            for (std::size_t i = 0; i < f.size(); ++i) {
                REQUIRE(f[i] == f_ref[i]);
            }
        }
    }
}

// check_streams_() (hmc.hpp) exists to catch a field partition that no longer
// matches the RNG stream count frozen at construction. But n_threads=0 only
// resolves the ambient ONCE, at construction (see the ctor's comment), and
// every step() rebinds exec::team_scope{n_threads_} to that frozen value —
// so a later OMP_NUM_THREADS change never reaches the partition
// check_streams_() re-derives inside sample_momenta_(). Verified directly
// (construct at various ambients, switch to various other ambients, step()):
// it never throws. The freeze the ctor documents already makes this specific
// drift scenario unreachable through the public API — check_streams_ is a
// backstop for a mismatch the current code has no path left to produce, not
// a guard this scenario exercises. Asserting REQUIRE_NOTHROW documents that
// finding instead of manufacturing a throw the real code doesn't raise.
// Guarded because macos-appleclang builds OpenMP OFF, where traverse_threads
// always returns 1 and the partition cannot drift regardless.
#ifdef _OPENMP
TEST_CASE("Hmc's frozen thread count survives an OMP_NUM_THREADS change after construction",
          "[hot_loop][parallel]") {
    // Construct at the same ambient in both runs, then move the ambient only in
    // the second: the frozen team must hold the chain bit-for-bit, not merely
    // avoid throwing.
    auto run = [](int ambient_after) {
        omp_set_num_threads(2);
        auto phi = hot_lattice({16, 16, 16, 16});
        Phi4<double> const action{.kappa = 0.18, .lambda = 1.0};
        FastRng rng{77};
        reticolo::updater::Hmc hmc{action,
                                   phi,
                                   rng,
                                   {.tau = 0.3, .n_md = 2},
                                   reticolo::updater::integ::leapfrog,
                                   reticolo::log::Mode::silent};
        omp_set_num_threads(ambient_after);
        double dh = 0.0;
        REQUIRE_NOTHROW(dh = hmc.step(reticolo::log::Mode::silent).dH);
        return std::pair{dh, std::vector<double>{phi.data(), phi.data() + phi.nsites()}};
    };

    auto const [dh_ref, f_ref] = run(2);
    for (int ambient : {1, 6, 8}) {
        auto const [dh, f] = run(ambient);
        INFO("ambient after construction=" << ambient);
        REQUIRE(dh == dh_ref);
        for (std::size_t i = 0; i < f.size(); ++i) {
            REQUIRE(f[i] == f_ref[i]);
        }
    }
    omp_set_num_threads(2);
}
#endif
