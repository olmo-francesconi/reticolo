#include <reticolo/action/site/detail/traversal.hpp>
#include <reticolo/action/site/phi4.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/parallel.hpp>
#include <reticolo/core/rng.hpp>

#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#ifdef _OPENMP
    #include <omp.h>
#endif

using reticolo::FastRng;
using reticolo::Lattice;
using reticolo::action::Phi4;
using reticolo::action::detail::visit_nn_fallback_;

namespace {

// A 4D lattice above k_traverse_min_sites, so compute_force / s_full take the
// cache-tiled (and, on an OpenMP build, threaded) traversal path rather than the
// small-lattice serial one.
Lattice<double> hot_4d(std::size_t side) {
    Lattice<double> phi{Lattice<double>::SizeVec(4, side)};
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

}  // namespace

// The tiled force pass must land on the identical answer as the plain gather
// through the neighbour table — each site is written exactly once, so the tiling
// and thread decomposition are bit-for-bit irrelevant to the result.
TEST_CASE("tiled compute_force equals the gather fallback (above threshold)", "[hot_loop][parallel]") {
    auto const phi = hot_4d(16);  // 16^4 = 65536 > 16384
    REQUIRE(phi.nsites() > reticolo::detail::k_traverse_min_sites);
    Phi4<double> const action{.kappa = 0.18, .lambda = 1.0};

    auto kern = action.force_kernel();
    std::vector<double> ref(phi.nsites(), 0.0);
    visit_nn_fallback_<double>(
        phi, [&](std::size_t i, double p, double nb) { ref[i] = kern(i, p, nb); });

    auto const got = force_vec(action, phi);
    for (std::size_t i = 0; i < ref.size(); ++i) {
        INFO("site " << i);
        REQUIRE(got[i] == ref[i]);
    }
}

// The force output is order-independent (each site written once) and the s_full
// reduction is decomposed into a fixed tile grid summed in canonical order, so
// both are identical for any thread count. On a serial build this runs once and
// trivially holds; on an OpenMP build it varies the team size for real.
TEST_CASE("tiled force + s_full are thread-count invariant", "[hot_loop][parallel]") {
    auto const phi = hot_4d(16);
    REQUIRE(phi.nsites() > reticolo::detail::k_traverse_min_sites);
    Phi4<double> const action{.kappa = 0.18, .lambda = 1.0};

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
        REQUIRE(s == s_ref);  // deterministic partials -> thread-invariant
        for (std::size_t i = 0; i < f.size(); ++i) {
            INFO("threads=" << nthr << " site=" << i);
            REQUIRE(f[i] == f_ref[i]);  // bit-identical force
        }
    }
}
