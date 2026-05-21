# Writing a test

Every action, algorithm, and IO component in reticolo has tests under
`tests/`. The patterns are uniform — once you've seen one of each kind,
adding a new test takes ~10 minutes. This document is the catalogue.

## Where tests live

```
tests/
  CMakeLists.txt        # one line per test target via reticolo_add_test()
  test_main.cpp         # shared Catch entry — silences the logger before tests
  unit/                 # types in isolation: Site, Indexing, Lattice, FastRng, Parser, Series, log, …
  physics/              # action / algorithm correctness via FD checks, reversibility, limits
  io/                   # Writer roundtrip + metadata + phase collision tests
  apps/                 # end-to-end: run the binary, open the HDF5, assert the schema
```

Catch2 v3 (vendored via FetchContent). One file per test target;
`catch_discover_tests` registers each `TEST_CASE` as its own ctest entry.

## Adding a test — the CMake line

```cmake
# tests/CMakeLists.txt
reticolo_add_test(test_my_thing  unit/test_my_thing.cpp)
```

`reticolo_add_test()` is defined at the top of `tests/CMakeLists.txt`. It
adds `tests/test_main.cpp` to the sources automatically, links
`reticolo::reticolo + Catch2::Catch2`, configures warnings, and runs
`catch_discover_tests`. You touch one file in `tests/`, one line in
`CMakeLists.txt`, and that's it.

## The shared test main (silences the logger)

`tests/test_main.cpp`:

```cpp
#include <reticolo/core/log.hpp>
#include <catch2/catch_session.hpp>

int main(int argc, char* argv[]) {
    reticolo::log::off();
    return Catch::Session().run(argc, argv);
}
```

Every test target links this file, so logs are suppressed by default.
Tests that *need* to inspect log output (only `tests/unit/test_log.cpp`
so far) flip the switch back on inside an RAII helper — see
[`docs/logging.md`](logging.md).

## The five canonical patterns

### Pattern 1 — concept static_assert

Every action header has one. It's a unit test that the concept refinement
lattice didn't slip:

```cpp
#include <reticolo/action/phi4.hpp>
#include <reticolo/action/detail/concepts.hpp>

using reticolo::action::HasForce;
using reticolo::action::HasSEff;
using reticolo::action::LocalAction;
using reticolo::action::Phi4;

static_assert(LocalAction<Phi4<double>, double>);
static_assert(HasSEff<Phi4<double>, double>);
static_assert(HasForce<Phi4<double>, double>);
```

These cost zero runtime; they're failures-at-compile-time. Put them at
the top of every action's force-consistency test.

### Pattern 2 — `ds_local` matches finite difference of `s_full`

The Metropolis-only contract. If `ds_local(φ, x, nv) ≠ s_full(φ_new) − s_full(φ)`,
the Metropolis update is wrong by definition. Random spot check:

```cpp
TEST_CASE("Phi4: ds_local matches finite difference of s_full", "[physics][phi4]") {
    Phi4<double> const action{.kappa = 0.13, .lambda = 0.05};

    Lattice<double> phi{{6, 6, 6}};
    FastRng rng{1234};
    randomize(phi, rng);

    for (std::size_t trial = 0; trial < 20; ++trial) {
        Site const x     = Site{rng.uniform_int(phi.nsites())};
        double const old = phi[x];
        double const nv  = old + rng.normal();

        double const ds_predicted = action.ds_local(phi, x, nv);

        double const s_old = action.s_full(phi);
        phi[x]             = nv;
        double const s_new = action.s_full(phi);
        phi[x]             = old;

        REQUIRE(std::abs(ds_predicted - (s_new - s_old)) < 1e-9);
    }
}
```

Tolerance 1e-9 is for a tight exact-arithmetic identity. Loosen to 1e-6
only if you have a numerical reason to.

### Pattern 3 — `compute_force` matches central finite difference of `s_full`

The HMC contract: `F(x) = −∂S/∂φ(x)`. Central difference is O(ε²)
accurate; with ε = 1e-4 you should see <1e-7 disagreement.

```cpp
TEST_CASE("Phi4: compute_force matches central FD of s_full", "[physics][phi4]") {
    Phi4<double> const action{.kappa = 0.18, .lambda = 0.04};

    Lattice<double> phi{{6, 6, 6}};
    Lattice<double> force{phi.indexing()};            // sibling — shares Indexing
    FastRng rng{56789};
    randomize(phi, rng);

    action.compute_force(phi, force);

    constexpr double k_eps = 1e-4;
    constexpr double k_tol = 1e-7;

    for (std::size_t trial = 0; trial < 25; ++trial) {
        Site const x     = Site{rng.uniform_int(phi.nsites())};
        double const old = phi[x];

        phi[x]              = old + k_eps;
        double const s_plus = action.s_full(phi);
        phi[x]              = old - k_eps;
        double const s_minus = action.s_full(phi);
        phi[x] = old;

        double const force_predicted = force[x];
        double const force_numeric   = -(s_plus - s_minus) / (2.0 * k_eps);

        REQUIRE(std::abs(force_predicted - force_numeric) < k_tol);
    }
}
```

Every HMC-capable action has this pattern in
`tests/physics/test_<action>_force_consistency.cpp`. Copy one and edit
the action type + couplings.

### Pattern 4 — algorithm reversibility

For HMC: integrating forward then back must return to the starting field
(modulo round-off). This is the integrator's symplectic contract:

```cpp
TEST_CASE("HMC: integrating forward + back returns the initial field", "[hmc]") {
    Phi4<double> const action{.kappa = 0.13, .lambda = 0.05};
    Lattice<double> phi0{{6, 6, 6}};
    FastRng rng{42};
    randomize(phi0, rng);

    auto phi = phi0;              // deep copy (Lattice value semantics)
    alg::Hmc<Phi4<double>, FastRng> hmc{action, phi, rng, {.tau = 1.0, .n_md = 20}};

    hmc.integrate_only(/*tau=*/+1.0, /*n_md=*/20);
    /* flip momentum sign on hmc.momentum() */
    hmc.integrate_only(/*tau=*/+1.0, /*n_md=*/20);

    for (Site x : phi.sites()) {
        REQUIRE(std::abs(phi[x] - phi0[x]) < 1e-9);
    }
}
```

`hmc.integrate_only(tau, n_md)` exposes the bare integrator (no
momentum resampling, no Metropolis accept/reject). See the actual
implementations in `tests/physics/test_hmc_reversibility.cpp`,
`test_u1_hmc_reversibility.cpp`, `test_su2_hmc_reversibility.cpp`.

### Pattern 5 — Limiting-regime sanity

The cheapest correctness signal: pick a limit where the answer is known
analytically and check.

- **Free theory (λ=0)** for Phi4: `force(x) = 2κ·Σ_NN φ_nn − 2φ(x)`.
- **Gaussian Metropolis at β=0** for OnSigma: acceptance → 1.
- **β=0 Wolff** for XY: every cluster is a singleton (no link probabilities
  fire).
- **Large β Wolff**: clusters span the lattice.
- **Integrator order** for HMC: `⟨|ΔH|⟩` should scale as `Δt²` (Leapfrog),
  `Δt²` (Omelyan2 — same order, smaller prefactor), or `Δt⁴` (Omelyan4).

See `tests/physics/test_hmc_integrator_order.cpp` for the cleanest
example — sweep `Δt`, fit log-log slope, REQUIRE the slope is within
~0.3 of the theoretical value.

## App smoke tests

For every reference app, there's a smoke test in `tests/apps/` that:

1. Runs the binary with tiny `--L --n_therm --n_prod` arguments.
2. Opens the produced HDF5 and asserts the schema (every series exists,
   shape is right, dtype is right).
3. (Optionally) Checks one or two physics-grounded invariants.

The boilerplate is factored into `tests/test_helpers.hpp`. A new
app gets a smoke test by copying
`tests/apps/test_phi4_hmc_smoke.cpp` and changing the binary name + the
expected paths. ~30 LOC each.

## Random data, deterministic seeds

Always seed RNGs explicitly. Most tests use `FastRng{42}` or some other
small literal. Catch's "random" features are not used — the tests are
deterministic.

A `randomize(phi, rng)` helper appears in many physics tests; not in a
shared header (no template-shared-helper file), just copied. Three
lines, fine to duplicate.

## What's NOT tested

- **Numerical accuracy of the action kernels beyond force-vs-FD.** If a
  Wilson SU(3) plaquette is off by 1e-12, the FD check will catch it; we
  don't separately benchmark the matrix products.
- **Performance.** Benchmarks live in `apps/bench_*.cpp` — they're not
  in `tests/` because they're noisy on CI and have soft pass/fail.
- **Logger output content beyond `tests/unit/test_log.cpp`.** Other
  tests run with `log::off()`; they don't assert on log lines.

## Common mistakes

- **Forgetting the `force` lattice's sibling construction.** Use
  `Lattice<T> force{phi.indexing()};` (share the Indexing), not
  `Lattice<T> force{phi.shape()};` (allocates a fresh Indexing — works
  but defeats the design). The smoke test won't catch this.
- **Using too-tight tolerances on FD.** Central diff is O(ε²); with
  ε = 1e-4 the noise floor is ~1e-8 not 1e-12. The shipped tests use
  1e-7 with room.
- **Leaving the field at a perturbed value.** Always reset
  `phi[x] = old;` after a single-site FD probe inside a loop.
