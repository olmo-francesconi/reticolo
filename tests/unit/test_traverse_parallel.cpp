#include <reticolo/action/nn/phi4.hpp>
#include <reticolo/action/nn/sine_gordon.hpp>
#include <reticolo/action/sweep/site.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/parallel.hpp>
#include <reticolo/core/rng/fast_rng.hpp>
#include <reticolo/updater/hmc/hmc.hpp>

#include "nn_reference.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#ifdef _OPENMP
    #include <omp.h>
#endif

using reticolo::FastRng;
using reticolo::Lattice;
using reticolo::action::Phi4;
using reticolo::action::SineGordon;

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
    Lattice<double> f{phi.indexing()};
    action.compute_force(phi, f);
    return {f.data(), f.data() + f.nsites()};
}

// {2D 256², 3D 42³, 4D 16⁴} — each yields ≫64 partition slabs (the threaded path).
std::vector<Lattice<double>::SizeVec> hot_shapes() {
    return {{256, 256}, {42, 42, 42}, {16, 16, 16, 16}};
}

}  // namespace

// The threaded per-dim force pass must land on the identical answer as the plain
// gather through the neighbour table — each site is written exactly once and its
// neighbour sum is in the same order, so the tiling/slab decomposition and thread
// count are bit-for-bit irrelevant.
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

// SineGordon exercises the extra per-site transcendental passes: prep() sin-batches
// the force scratch, s_full cos-batches + reduces. The force stays bit-identical
// across thread counts (map); s_full is checked run-to-run at a fixed team (reduce).
TEST_CASE("SineGordon force is thread-count invariant; s_full is fixed-team deterministic",
          "[hot_loop][parallel]") {
    auto const phi = hot_lattice({16, 16, 16, 16});
    SineGordon<double> const action{.kappa = 0.18, .alpha = 0.7};

    auto force = [&] {
        Lattice<double> f{phi.indexing()};
        action.compute_force(phi, f);  // triggers prep() sin-batch + visit_nn
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
