#pragma once

// Shared force-vs-finite-difference harness. Every action must satisfy
// F = −dS/dfield; the check is the same central-difference sweep for all of
// them, differing only in the field type, how a site is perturbed, and the
// tolerances. Single-sourced here so adding an action means writing the action's
// couplings and calling one function, not copying a 30-line loop (which is how
// six near-identical harnesses accumulated).
//
// Each action keeps its OWN TEST_CASE in its own TU, so `ctest -R <action>` and
// Catch2 tag filters stay granular — this consolidates the harness, not the
// cases.

#include <reticolo/core/field/lattice.hpp>
#include <reticolo/core/field/site.hpp>
#include <reticolo/core/rng/fast_rng.hpp>

#include <complex>
#include <cstddef>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace reticolo::test {

struct FdSpec {
    double eps   = 1e-4;
    double tol   = 1e-7;
    int n_probes = 25;
    // Sites are probed at a stride rather than at random when `stride > 0`, so a
    // failure reproduces without depending on the RNG draw order.
    std::size_t stride = 0;
};

// Fill a real scalar lattice with iid N(0, 1) (scaled by `amp`).
template <class T>
inline void randomize(Lattice<T>& phi, FastRng& rng, double amp = 1.0) {
    for (Site const x : phi.sites()) {
        phi[x] = static_cast<T>(amp * rng.normal());
    }
}

// Fill a complex lattice with iid N(0, 1) on each component.
template <class T>
inline void randomize(Lattice<std::complex<T>>& phi, FastRng& rng, double amp = 1.0) {
    for (Site const x : phi.sites()) {
        phi[x] =
            std::complex<T>{static_cast<T>(amp * rng.normal()), static_cast<T>(amp * rng.normal())};
    }
}

// F(x) == −(S(φ+ε e_x) − S(φ−ε e_x)) / 2ε for a sample of sites, for a REAL
// scalar field. `force` must already hold `action.compute_force(phi, ·)`.
template <class Action, class T>
inline void require_force_matches_fd(Action const& action,
                                     Lattice<T>& phi,
                                     Lattice<T> const& force,
                                     FastRng& rng,
                                     FdSpec const& spec = {}) {
    auto const eps = static_cast<T>(spec.eps);
    for (int probe = 0; probe < spec.n_probes; ++probe) {
        Site const x  = spec.stride > 0
                            ? Site{(static_cast<std::size_t>(probe) * spec.stride) % phi.nsites()}
                            : Site{rng.uniform_int(phi.nsites())};
        T const saved = phi[x];

        phi[x]               = saved + eps;
        double const s_plus  = action.s_full(phi);
        phi[x]               = saved - eps;
        double const s_minus = action.s_full(phi);
        phi[x]               = saved;

        double const fd = -(s_plus - s_minus) / (2.0 * static_cast<double>(eps));
        INFO("site " << x.value());
        REQUIRE(static_cast<double>(force[x]) == Catch::Approx(fd).margin(spec.tol));
    }
}

// Complex twin: Re and Im are independent directions, so each site contributes
// two finite differences and the packed complex force carries one in each part.
template <class Action, class T>
inline void require_force_matches_fd(Action const& action,
                                     Lattice<std::complex<T>>& phi,
                                     Lattice<std::complex<T>> const& force,
                                     FastRng& rng,
                                     FdSpec const& spec = {}) {
    using C      = std::complex<T>;
    auto const e = static_cast<T>(spec.eps);
    for (int probe = 0; probe < spec.n_probes; ++probe) {
        Site const x  = spec.stride > 0
                            ? Site{(static_cast<std::size_t>(probe) * spec.stride) % phi.nsites()}
                            : Site{rng.uniform_int(phi.nsites())};
        C const saved = phi[x];

        phi[x]              = saved + C{e, T{0}};
        double const s_re_p = action.s_full(phi);
        phi[x]              = saved - C{e, T{0}};
        double const s_re_m = action.s_full(phi);
        phi[x]              = saved + C{T{0}, e};
        double const s_im_p = action.s_full(phi);
        phi[x]              = saved - C{T{0}, e};
        double const s_im_m = action.s_full(phi);
        phi[x]              = saved;

        double const den = 2.0 * static_cast<double>(e);
        INFO("site " << x.value());
        REQUIRE(static_cast<double>(force[x].real()) ==
                Catch::Approx(-(s_re_p - s_re_m) / den).margin(spec.tol));
        REQUIRE(static_cast<double>(force[x].imag()) ==
                Catch::Approx(-(s_im_p - s_im_m) / den).margin(spec.tol));
    }
}

// Convenience: build the force buffer, run the sweep. The common case.
template <class Action, class Field>
inline void
require_force_consistent(Action const& action, Field& phi, FastRng& rng, FdSpec const& spec = {}) {
    Field force{phi.indexing()};
    action.compute_force(phi, force);
    require_force_matches_fd(action, phi, force, rng, spec);
}

}  // namespace reticolo::test
