#include <reticolo/action/site/phi4.hpp>
#include <reticolo/action/site/sine_gordon.hpp>
#include <reticolo/action/sweep/site.hpp>
#include <reticolo/algorithm/hmc.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/parallel.hpp>
#include <reticolo/core/rng/rng.hpp>

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
using reticolo::action::sweep::visit_nn_fallback_;

namespace {

// A lattice above the byte threshold, so compute_force / s_full take the
// threaded traversal path rather than the small-lattice serial one. One shape per
// dimensionality (2D/3D/4D) so all the per-dim parallel paths are exercised.
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

// {2D 256², 3D 42³, 4D 16⁴} — each above the 512 KB threshold (65536 sites @ 8B).
std::vector<Lattice<double>::SizeVec> hot_shapes() {
    return {{256, 256}, {42, 42, 42}, {16, 16, 16, 16}};
}

}  // namespace

// The threaded per-dim force pass must land on the identical answer as the plain
// gather through the neighbour table — each site is written exactly once and its
// neighbour sum is in the same order, so the tiling/row decomposition and thread
// count are bit-for-bit irrelevant.
TEST_CASE("threaded compute_force equals the gather fallback, every dimension",
          "[hot_loop][parallel]") {
    Phi4<double> const action{.kappa = 0.18, .lambda = 1.0};
    for (auto const& shape : hot_shapes()) {
        auto const phi = hot_lattice(shape);
        REQUIRE(reticolo::exec::want_threads(phi.nsites(), phi.bytes_per_site()));

        auto kern = action.force_kernel();
        std::vector<double> ref(phi.nsites(), 0.0);
        visit_nn_fallback_<double>(
            phi, std::size_t{0}, phi.nsites(), [&](std::size_t i, double p, double nb) {
                ref[i] = kern(i, p, nb);
            });

        auto const got = force_vec(action, phi);
        for (std::size_t i = 0; i < ref.size(); ++i) {
            INFO("ndims=" << shape.size() << " site=" << i);
            REQUIRE(got[i] == ref[i]);
        }
    }
}

// Force output is order-independent (each site written once) and the s_full
// reduction is a fixed work-item partition summed in canonical order, so both are
// identical for any thread count — in every dimension. On a serial build this runs
// once and trivially holds; on an OpenMP build it varies the team size for real.
TEST_CASE("threaded force + s_full are thread-count invariant, every dimension",
          "[hot_loop][parallel]") {
    Phi4<double> const action{.kappa = 0.18, .lambda = 1.0};
    for (auto const& shape : hot_shapes()) {
        auto const phi = hot_lattice(shape);
        REQUIRE(reticolo::exec::want_threads(phi.nsites(), phi.bytes_per_site()));

        auto at = [&](int nthr) {
#ifdef _OPENMP
            omp_set_num_threads(nthr);
#else
            (void)nthr;
#endif
            return std::pair{force_vec(action, phi), action.s_full(phi)};
        };

        auto const [f_ref, s_ref] = at(1);
        for (int nthr : {1, 2, 4, 8}) {
            auto const [f, s] = at(nthr);
            INFO("ndims=" << shape.size() << " threads=" << nthr);
            REQUIRE(s == s_ref);  // deterministic partials -> thread-invariant
            for (std::size_t i = 0; i < f.size(); ++i) {
                REQUIRE(f[i] == f_ref[i]);  // bit-identical force
            }
        }
    }
}

// SineGordon exercises the extra per-site transcendental passes: prep() sin-batches
// the force scratch, and s_full cos-batches + reduces. Both are now worksplit, so
// force + s_full must stay bit-identical for any thread count (chunks are a SIMD-
// width multiple, so the Sleef batch takes the same vector path regardless).
TEST_CASE("SineGordon force + s_full are thread-count invariant", "[hot_loop][parallel]") {
    auto const phi = hot_lattice({16, 16, 16, 16});
    REQUIRE(reticolo::exec::want_threads(phi.nsites(), phi.bytes_per_site()));
    SineGordon<double> const action{.kappa = 0.18, .alpha = 0.7};

    auto at = [&](int nthr) {
#ifdef _OPENMP
        omp_set_num_threads(nthr);
#else
        (void)nthr;
#endif
        Lattice<double> f{phi.indexing()};
        action.compute_force(phi, f);  // triggers prep() sin-batch + visit_nn
        return std::pair{std::vector<double>{f.data(), f.data() + f.nsites()}, action.s_full(phi)};
    };

    auto const [f_ref, s_ref] = at(1);
    for (int nthr : {1, 2, 4, 8}) {
        auto const [f, s] = at(nthr);
        INFO("threads=" << nthr);
        REQUIRE(s == s_ref);
        for (std::size_t i = 0; i < f.size(); ++i) {
            REQUIRE(f[i] == f_ref[i]);
        }
    }
}

// The FULL trajectory — momentum fill, snapshot copy, kinetic, s_full, MD drifts
// and fused kicks — must be one deterministic function of (field, seed) at any
// team size: every per-site op runs on the canonical field partition and every
// reduce folds fixed per-item partials. Two steps (a determinism check, not a
// simulation) on a >threshold lattice; ΔH and the evolved field are compared
// bit-for-bit across thread counts.
TEST_CASE("full hmc.step is thread-count invariant", "[hot_loop][parallel]") {
    auto run = [&](int nthr) {
#ifdef _OPENMP
        omp_set_num_threads(nthr);
#else
        (void)nthr;
#endif
        auto phi = hot_lattice({20, 20, 20, 20});
        REQUIRE(reticolo::exec::want_threads(phi.nsites(), phi.bytes_per_site()));
        Phi4<double> const action{.kappa = 0.18, .lambda = 1.0};
        FastRng rng{2026};
        reticolo::alg::Hmc hmc{action,
                               phi,
                               rng,
                               {.tau = 0.5, .n_md = 6},
                               reticolo::alg::integ::leapfrog,
                               reticolo::log::Mode::silent};
        double dh_sum = 0.0;
        for (int t = 0; t < 2; ++t) {
            dh_sum += hmc.step(reticolo::log::Mode::silent).dH;
        }
        return std::pair{dh_sum, std::vector<double>{phi.data(), phi.data() + phi.nsites()}};
    };

    auto const [dh_ref, f_ref] = run(1);
    for (int nthr : {1, 2, 4, 8}) {
        auto const [dh, f] = run(nthr);
        INFO("threads=" << nthr);
        REQUIRE(dh == dh_ref);  // ΔH bit-identical → every op thread-invariant
        for (std::size_t i = 0; i < f.size(); ++i) {
            REQUIRE(f[i] == f_ref[i]);
        }
    }
}
