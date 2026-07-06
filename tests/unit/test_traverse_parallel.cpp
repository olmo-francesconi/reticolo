#include <reticolo/action/site/detail/traversal.hpp>
#include <reticolo/action/site/phi4.hpp>
#include <reticolo/action/site/sine_gordon.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/parallel.hpp>
#include <reticolo/core/rng.hpp>

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
using reticolo::action::detail::visit_nn_fallback_;

namespace {

// A lattice above k_traverse_min_sites, so compute_force / s_full take the
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

// {2D 160², 3D 26³, 4D 16⁴} — each above 16384 sites.
std::vector<Lattice<double>::SizeVec> hot_shapes() {
    return {{160, 160}, {26, 26, 26}, {16, 16, 16, 16}};
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
        REQUIRE(phi.nsites() > reticolo::detail::k_traverse_min_sites);

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
        REQUIRE(phi.nsites() > reticolo::detail::k_traverse_min_sites);

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
    REQUIRE(phi.nsites() > reticolo::detail::k_traverse_min_sites);
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
