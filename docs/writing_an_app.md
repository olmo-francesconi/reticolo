# Writing an app (and a test for it)

This is the canonical walkthrough. We'll build `apps/phi4_hmc.cpp`: 4D φ⁴
scalar field driven by HMC, full reproducibility metadata in the HDF5
output, susceptibility + magnetisation observers recorded as time series.
The completed app is ~55 LOC.

## Builtin apps vs standalone examples

**Builtin app** (`apps/`): one canonical sim per action/algorithm variant,
built in-tree only. Drop a `.cpp` here and register it in
`apps/CMakeLists.txt` with `reticolo_add_app(my_app my_app.cpp)`.

**Standalone example** (`examples/NN_short_name/`): an end-to-end study
that a user can copy out of the repo and build independently. Each example
is its own CMake project that links `reticolo::reticolo` via a find-or-fetch
block (see the `apps/ and examples/` section in [architecture.md](architecture.md)).
Details on writing one are at the end of this file.

---

## Skeleton (builtin app)

Apps live in [`apps/`](../apps/) and get registered through one CMake line:

```cmake
# apps/CMakeLists.txt
reticolo_add_app(my_app my_app.cpp)
```

`reticolo_add_app` is a tiny helper that adds an executable and links it
against `reticolo::reticolo`.

## Step 1 — the umbrella include

```cpp
#include <reticolo/reticolo.hpp>

int main(int argc, char** argv) {
    using namespace reticolo;
    // ...
}
```

One include gets you `Lattice`, `FastRng`, `act::Phi4`, `alg::Hmc`,
`io::Writer`, `cli::Parser`, `obs::*`.

## Step 2 — declare the CLI flags

```cpp
cli::Parser p{"phi4_hmc", "Hybrid Monte Carlo for the phi^4 scalar field"};
auto const& L       = p.req<int>("L,size",    "linear lattice extent");
auto const& kappa   = p.req<double>("kappa",  "hopping parameter");
auto const& lambda  = p.req<double>("lambda", "quartic coupling");
auto const& ndim    = p.opt<int>("ndim",      4,    "spatial dimensions");
auto const& tau     = p.opt<double>("tau",    1.0,  "HMC trajectory length");
auto const& n_md    = p.opt<int>("n_md",      20,   "MD steps per trajectory");
auto const& n_therm = p.opt<int>("n_therm",   200,  "thermalisation trajectories");
auto const& n_prod  = p.opt<int>("n_prod",    1000, "production trajectories");
auto const& seed    = p.opt<unsigned long long>("seed", 42ULL, "RNG seed");
auto const& workspace =
    p.opt<std::string>("workspace", std::string{"."}, "workspace folder (output + logs)");
auto const& outfile =
    p.opt<std::string>("out", std::string{"phi4.h5"}, "HDF5 output file name, inside workspace");
p.parse(argc, argv);
```

`req<T>` returns `T const&` to stable parser-owned storage; the reference
is meaningful only after `parse()` returns. `opt<T>` is the same with a
default. `"L,size"` is cxxopts' "short + long" spec — `-L 16` and
`--size=16` both work.

## Step 3 — field, RNG, action

```cpp
Lattice<double>::SizeVec shape(static_cast<std::size_t>(ndim),
                               static_cast<std::size_t>(L));
Lattice<double>   phi{shape};
FastRng           rng{seed};
act::Phi4<double> phi4{.kappa = kappa, .lambda = lambda};
```

`act::Phi4` is a plain struct with two fields — designated initialisers
make the construction self-documenting.

## Step 4 — open the writer

```cpp
std::string const outpath = (std::filesystem::path{workspace} / outfile).string();
io::Writer out{outpath, argc, argv, &p};
```

Truncate-or-fail. The constructor stamps every reproducibility attribute
in `/run@*` and every resolved CLI flag in `/vars@<name>`.

## Step 5 — phases and series

```cpp
out.start_phase("therm");
out.start_phase("prod");

auto s_therm  = out.series<double>("/therm/stats/s");
auto d_h      = out.series<double>("/prod/stats/dH");
auto accepted = out.series<int>("/prod/stats/accepted");
auto s_prod   = out.series<double>("/prod/obs/s");
auto mag      = out.series<double>("/prod/obs/mag");
auto mag_sq   = out.series<double>("/prod/obs/mag_sq");
```

`start_phase` rejects re-use (typo'd phases raise instead of silently
overwriting). `series<T>` returns a move-only handle that owns a 4096-row
buffer and flushes when full or on dtor. Convention:
`/<phase>/stats/<name>` for integrator diagnostics, `/<phase>/obs/<name>`
for physics observables.

## Step 6 — the updater

```cpp
alg::Hmc hmc{phi4, phi, rng, {.tau = tau, .n_md = n_md}};
```

Action, RNG, and field types deduce (CTAD); the default integrator is
Leapfrog. A trailing tag value selects another — `alg::integ::omelyan2`
or `alg::integ::omelyan4`, e.g.
`alg::Hmc hmc{phi4, phi, rng, spec, alg::integ::omelyan2}`. The tag's
type is what picks the integrator at compile time — it's a type, not a
runtime flag. The `Hmc` ctor pre-allocates momentum, force, and rollback
as sibling lattices of `phi`.

## Step 7 — own the for loop

```cpp
for (int i = 0; i < n_therm; ++i) {
    (void)hmc.step();
    s_therm.append(phi4.s_full(phi));
}
for (int i = 0; i < n_prod; ++i) {
    auto const step = hmc.step();
    d_h.append(step.dH);
    accepted.append(step.accepted ? 1 : 0);
    s_prod.append(phi4.s_full(phi));
    mag.append(obs::magnetization(phi));
    mag_sq.append(obs::mean_sq(phi));
}
```

Apps own the for loop. No closure-based driver, no `simulate(action,
n_steps)` helper. Trajectory cadence, measurement cadence, conditional
logic — all visible in plain `main`. `hmc.step()` returns
`HmcResult { dH, accepted }`.

## Step 8 — build and run

```sh
cmake --build --preset macos-appleclang --target phi4_hmc
build/macos-appleclang/apps/phi4_hmc --size 8 --kappa 0.13 --lambda 0.02 --out phi4.h5
```

Inspect with `h5ls -r phi4.h5` and `h5dump -A phi4.h5 | grep '@'`. The
HDF5 is fully self-describing — push it through h5py for analysis. See
[`examples/01_phi4_tuning/analyze.py`](../examples/01_phi4_tuning/analyze.py).

## When to start a new app

When you'd otherwise be adding a `--mode` flag to an existing one. The
cost is one file in `apps/`, one line in `apps/CMakeLists.txt`, and one
smoke test in `tests/apps/`. Cheaper than the conditional logic + docs a
multi-mode app accumulates.

## Logging — automatic

Action / algorithm / Replica constructors emit their own `init`, `act`,
`hmc` lines; `step()` logs each completed update. One call at the top of
`main` enables the file logger:

```cpp
log::start(workspace, outfile);   // call once, before constructing anything
```

`start` creates the workspace folder, opens the main log file
`<workspace>/<stem>.log` (stem = the `--out` name minus extension — sweeps
sharing a workspace don't clobber each other) and mirrors every entry into
it, then prints the banner. LLR apps pass `/*replicas=*/true` to add the
run-id column and per-replica `<stem>.<rNNN>.log` files.

Suppress everything with `log::off()`; opt out of a single algorithm step
with `log::Mode::silent` (e.g. `hmc.step(log::Mode::silent)` during
thermalisation).

---

# Adding an action

Three steps: **pick a family**, **write the formula**, **wire it up**. The
family base supplies all the machinery (`s_full` / `compute_force` / the fused
kick / the `last_s_full` cache); the leaf carries only the physics. The compiler
is the only registry — there is nothing to "register" beyond an `#include` in
the family aggregator.

Pick the family base by the shape of the interaction:

| family | base / handle | field | when |
| ------ | ------------- | ----- | ---- |
| site   | `detail::SiteAction<D,T>`    | `Lattice<T>`          | self + nearest-neighbour sum (Phi4/Phi6/SineGordon) |
| bond   | `detail::BondAction<D,T>`    | `Lattice<T>`          | transcendental of the endpoint *difference* (XY) |
| complex| `detail::ComplexAction<D,T>` | `Lattice<complex<T>>` | S = S_R + i·S_I sign problem (BoseGas) |
| gauge  | `detail::GaugeAction<D>`     | link field            | plaquette gauge theory — usually just write a group model (see below) |

## A new site action

Copy `site/phi4.hpp` + `site/formula/phi4_formula.hpp` and swap the formula body,
the couplings, and `describe`.

1. **Formula** — `include/reticolo/action/<family>/formula/<name>_formula.hpp`. Two
   `RETICOLO_HD inline` free templates in `namespace reticolo::action::detail`,
   taking pre-gathered neighbour sums. **This is all the physics**, and the
   single source of truth — the CUDA functors call the *same* functions:

   ```cpp
   template <class T>
   [[nodiscard]] RETICOLO_HD inline T <name>_action_site(T phi, T fwd, /*params*/);
   //   `fwd`  = sum over the ndim FORWARD neighbours (each bond once) → for s_full
   template <class T>
   [[nodiscard]] RETICOLO_HD inline T <name>_force_site(T phi, T nbrs, /*params*/);
   //   `nbrs` = sum over all 2·ndim neighbours; returns the force -dS/dphi(x)
   ```

2. **Struct** — `include/reticolo/action/site/<name>.hpp`. Derive from the base
   and bind the couplings into the formula with two kernels. It stays a
   designated-init aggregate:

   ```cpp
   template <class T = double>
   struct MyAction : detail::SiteAction<MyAction<T>, T> {
       using value_type = T;
       T c1 = T{0};                       // couplings
       void describe(log::Entry& e) const { e.line("MyAction<{}>", scalar_name<T>()); e.param("c1={:.3f}", c1); }

       auto force_kernel()  const { return [c = c1](std::size_t /*i*/, T phi, T nbrs)
                                           { return detail::myaction_force_site<T>(phi, nbrs, c); }; }
       auto action_kernel() const { return [c = c1](T phi, T fwd)
                                           { return detail::myaction_action_site<T>(phi, fwd, c); }; }
   };
   ```

   The base gives you `HmcAction` + `HasFusedKick` from these two kernels. The
   couplings are captured by value, so the hot loop is byte-for-byte what a
   hand-written action would emit. Two opt-in refinements the base picks up by
   concept when the leaf provides them: a `prep(l)` hook (called before each
   force pass — `SineGordon` uses it to Sleef-batch `sin` into `this->scratch_`),
   and an LLR fast-path `s_full_and_force` built from `this->staged_force_energy`
   (see `phi4.hpp`).

   Other shapes: for a bond action derive `detail::BondAction` and provide
   `action_bond_kernel` / `force_bond_kernel` / `bond_scale` (copy `site/xy.hpp`);
   for a complex sign-problem action derive `detail::ComplexAction` and add the
   `imag_*` kernels (copy `site/bose_gas.hpp`).

3. **Register** — add `#include <reticolo/action/site/<name>.hpp>` to
   `include/reticolo/action/site.hpp`. `alg::Hmc<MyAction<double>, FastRng>`
   now compiles.

4. **App + test** — copy `apps/phi4_hmc.cpp`; add a force-vs-FD test (pattern
   #2 below) under `tests/physics/`.

## A new gauge group

A gauge group `G` is the template parameter to the generic `Wilson<G>` plaquette
action over `MatrixLinkLattice<G, T>`. The work splits in two, along the
core-math / action-physics line:

- **the group model** — `include/reticolo/math/gauge_group/<g>.hpp`: the *core
  group operations only* — the constants (`n_color`, `n_real_components`, `name`)
  and the HMC algebra hooks (`sample_algebra_slab` / `kinetic_slab` /
  `expi_lmul_slab`) the integrator calls. This is pure group math; it lives under
  `math/` and knows nothing about the Wilson action.
- **the Wilson kernels** — `include/reticolo/action/gauge/formula/wilson_<g>.hpp`:
  a `detail::wilson_kernels<G>` specialization holding the plaquette *physics*
  (`plaq_re_tr`, the `s_full_plane_re_tr_sum` fast-path, `compute_force`,
  `compute_force_and_kick`). This is action-specific and lives in the action layer.

`Wilson<G>` (and the hand-tuned U(1) `CompactU1`) derive from the
`detail::GaugeAction<D>` base, which owns the `last_s_full` cache and the concept
surface; `Wilson<G>` dispatches its plaquette work to `wilson_kernels<G>`. (A
genuinely new gauge *action* — improved, rectangle — derives from
`detail::GaugeAction` and writes its own `s_full_uncached` / `force_into`.)
**Copy `SU2`** (`math/gauge_group/su2.hpp` + `action/gauge/formula/wilson_su2.hpp`
+ `math/su2_ops.hpp`) — the simplest matrix group.

1. **Group model: concept members** (the `GaugeGroup` concept lives in
   `math/gauge_group/base.hpp`):

   ```cpp
   struct MyGroup {
       using scalar_t = double;
       static constexpr std::size_t n_real_components = ...;  // doubles per link (8 = SU(2) 4-complex, 18 = SU(3) 9-complex)
       static constexpr std::size_t n_color           = N;    // the N in SU(N): Wilson 1/N prefactor + the N·n_plaq offset
       static constexpr std::string_view name         = "MyGroup";
   };
   ```

2. **HMC algebra** — `include/reticolo/math/<g>_ops.hpp`, exposed as static
   members on `G`. The HMC integrator's momentum sample / kinetic / drift route
   through these for matrix links (see `algorithm/integ_ops.hpp`,
   `core/matrix_link_lattice.hpp`):

   ```cpp
   template <class Rng> static void sample_algebra_slab(T* p_blk, Rng&, std::size_t n);  // draw P in the Lie algebra
   static double            kinetic_slab (T const* p_blk, std::size_t n);                // ½ Tr P²
   static void              expi_lmul_slab(T* u_blk, T const* p_blk, double dt, std::size_t n); // U ← exp(i·dt·P)·U
   ```

   `expi_lmul` (the matrix-exponential drift) is the genuinely group-specific
   piece — the SU(N) analogue of the scalar `phi += dt·mom`, and the one device
   exception to the shared-formula rule.

3. **Wilson kernels** — a `template <> struct detail::wilson_kernels<MyGroup>`
   in `action/gauge/formula/wilson_<g>.hpp` with the statics `Wilson<G>`
   dispatches to:

   ```cpp
   static double plaq_re_tr(mb, nb, s, s_pmu, s_pnu, stride);                    // Re Tr U_p
   template <class T> static double s_full_plane_re_tr_sum(MatrixLinkLattice<G,T> const&, mu, nu); // optional fast path
   static void compute_force         (MatrixLinkLattice<G,T> const&, MatrixLinkLattice<G,T>&, double beta_over_N);
   static void compute_force_and_kick(... , double beta_over_N, double k_dt);   // optional → HasFusedKick
   ```

   Component `k` of the link at site `s` in direction `μ`'s block is at
   `block_ptr[k·stride + s]` (stride = `nsites`). SU(2)/SU(3) batch these over
   `gauge_group::k_gauge_batch<T>` sites — start unbatched (a plain per-site loop,
   like the Abelian `CompactU1`) and add batching only if it's a measured
   bottleneck.

4. **Register + use** — `#include` the new `wilson_<g>.hpp` from
   `action/gauge/wilson.hpp` (it bundles the shipped groups so `Wilson<G>`
   resolves from a single include). `Wilson<MyGroup, double>` is then an
   `HmcAction`. Copy `apps/su2_hmc.cpp`, and add a reversibility test (pattern #3)
   plus an equivalence check against a known limit.

The Abelian case (`U(1)`) is degenerate — `n_real_components = 1` (just the
angle), no matrix exponential — and already ships both the hand-tuned
`CompactU1` action and a `Wilson<gauge_group::U1>` path that reduces to it
bit-for-bit (`test_wilson_u1_vs_compact`).

---

# Writing a test

Every action, algorithm, and IO component has tests under `tests/`. The
patterns are uniform — once you've seen one of each kind, adding one
takes ~10 minutes.

## Where tests live

```
tests/
  CMakeLists.txt        # one line per test target via reticolo_add_test()
  test_main.cpp         # shared Catch entry; calls log::off()
  unit/                 # types in isolation
  physics/              # action / algorithm correctness
  io/                   # Writer roundtrip + metadata + phase collision
  apps/                 # end-to-end: run the binary, assert HDF5 schema
```

Catch2 v3, vendored via FetchContent. One file per target;
`catch_discover_tests` registers each `TEST_CASE` as its own ctest entry.

## Adding a test

```cmake
# tests/CMakeLists.txt
reticolo_add_test(test_my_thing  unit/test_my_thing.cpp)
```

`reticolo_add_test()` adds `tests/test_main.cpp` to the sources, links
`reticolo::reticolo + Catch2::Catch2`, configures warnings, and runs
`catch_discover_tests`. One file + one line; done.

## The four canonical patterns

### 1. Concept static_assert

Every action header has one. Zero-runtime; failures at compile time:

```cpp
static_assert(HmcAction   <Phi4<double>, Lattice<double>>);
static_assert(HasFusedKick<Phi4<double>, Lattice<double>>);  // if the action fuses the kick
```

Put at the top of every action's force-consistency test. Gauge actions name
their link field instead, e.g. `HmcAction<CompactU1<double>, LinkLattice<double>>`.

### 2. `compute_force` matches central FD of `s_full`

The HMC contract: `F(x) = −∂S/∂φ(x)`. Central diff is O(ε²); with
ε = 1e-4 you should see <1e-7 disagreement:

```cpp
action.compute_force(phi, force);

phi[x]               = old + k_eps;
double const s_plus  = action.s_full(phi);
phi[x]               = old - k_eps;
double const s_minus = action.s_full(phi);
phi[x] = old;

double const force_numeric = -(s_plus - s_minus) / (2.0 * k_eps);
REQUIRE(std::abs(force[x] - force_numeric) < 1e-7);
```

Every action has this pattern in
`tests/physics/test_<action>_force_consistency.cpp`. Copy one, edit the
action type + couplings.

### 3. HMC reversibility

Integrating forward then back must return to the starting field (modulo
round-off). The integrator's symplectic contract:

```cpp
auto phi = phi0;                          // deep copy
alg::Hmc hmc{action, phi, rng, {.tau = 1.0, .n_md = 20}};

hmc.integrate_only(/*tau=*/+1.0, /*n_md=*/20);
/* flip momentum sign on hmc.momentum() */
hmc.integrate_only(/*tau=*/+1.0, /*n_md=*/20);

for (Site x : phi.sites()) REQUIRE(std::abs(phi[x] - phi0[x]) < 1e-9);
```

`integrate_only` exposes the bare integrator (no momentum resampling, no
accept/reject). See `tests/physics/test_hmc_reversibility.cpp`,
`test_u1_hmc_reversibility.cpp`, `test_su2_hmc_reversibility.cpp`.

### 4. Limiting regime

Pick a limit where the answer is known analytically:

- **Free theory (λ=0)** for Phi4: `force(x) = 2κ·Σ_NN φ_nn − 2φ(x)`.
- **Aligned config (θ=0)** for CompactU1: every plaquette contributes 0, so `S = 0`.
- **Wilson<U1> vs CompactU1**: the generic matrix-group action must match the
  hand-tuned Abelian path bit-for-bit at N=1.
- **Integrator order** for HMC: `⟨|ΔH|⟩` scales as `Δt²` (Leapfrog,
  Omelyan2) or `Δt⁴` (Omelyan4). See
  `tests/physics/test_hmc_integrator_order.cpp` for the cleanest example
  — sweep `Δt`, fit log-log slope, REQUIRE within ~0.3 of theory.

## App smoke tests

For every reference app, a smoke test in `tests/apps/`:

1. Runs the binary with tiny `--L --n_therm --n_prod`.
2. Opens the HDF5 and asserts the schema.
3. Optionally checks one or two physics-grounded invariants.

Boilerplate is factored into `tests/test_helpers.hpp`. A new app's
smoke test = copy `tests/apps/test_phi4_hmc_smoke.cpp` and change the
binary name + expected paths.

## Conventions

- Seed RNGs explicitly. Most tests use `FastRng{42}` or similar.
- Use `Lattice<T> force{phi.indexing()}` (sibling), not
  `Lattice<T> force{phi.shape()}` (allocates a fresh `Indexing`).
- Reset `phi[x] = old;` after a single-site FD probe inside a loop.
- Central diff noise floor with ε = 1e-4 is ~1e-8, not 1e-12 — don't
  over-tighten the tolerance.

---

# Writing a standalone example

A standalone example lives in `examples/NN_short_name/`, carries its own
`CMakeLists.txt` (find-or-fetch), a `run.sh` sweep, and an `analyze.py`.
It must build from that directory alone — no dependency on the in-tree
`apps/` build.

## Minimal layout (single binary — model: `examples/01_phi4_tuning/`)

```
examples/01_phi4_tuning/
  CMakeLists.txt     # find-or-fetch + one add_executable
  phi4_hmc.cpp       # driver source (copy of or inspired by apps/phi4_hmc.cpp)
  run.sh             # bash sweep using build_example from _common/preset.sh
  analyze.py         # python post-processing
  README.md          # physics write-up + knobs table + "Building & running"
```

## Two-binary template (`examples/04_phi4_llr/`)

Same layout; `CMakeLists.txt` registers two targets via a `foreach`:

```cmake
foreach(_app phi4_hmc phi4_llr)
    add_executable(ex04_${_app} ${_app}.cpp)
    set_target_properties(ex04_${_app} PROPERTIES OUTPUT_NAME ${_app})
    target_link_libraries(ex04_${_app} PRIVATE reticolo::reticolo)
endforeach()
```

The `exNN_` prefix keeps target names unique across the in-tree aggregate
build while `OUTPUT_NAME` keeps the binary name clean.

## Step 1 — CMakeLists.txt

Copy the find-or-fetch block from an existing example verbatim, then
replace the project name and executable registration. Minimum:

```cmake
cmake_minimum_required(VERSION 3.25)
project(example_NN_<name> CXX)

if(NOT TARGET reticolo::reticolo)
    set(_root "${CMAKE_CURRENT_SOURCE_DIR}/../..")
    if(EXISTS "${_root}/include/reticolo/reticolo.hpp")
        add_subdirectory("${_root}" "${CMAKE_BINARY_DIR}/_reticolo" EXCLUDE_FROM_ALL)
    else()
        include(FetchContent)
        FetchContent_Declare(reticolo
            GIT_REPOSITORY https://github.com/olmo-francesconi/reticolo.git
            GIT_TAG        main    # pin a release tag/SHA for reproducible builds
            GIT_SHALLOW    TRUE)
        FetchContent_MakeAvailable(reticolo)
    endif()
endif()

add_executable(exNN_<binary> <binary>.cpp)
set_target_properties(exNN_<binary> PROPERTIES OUTPUT_NAME <binary>)
target_link_libraries(exNN_<binary> PRIVATE reticolo::reticolo)
```

## Step 2 — run.sh

Source `_common/preset.sh`, call `build_example`, then use `$example_bin`
as the directory holding the built binaries:

```bash
#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common/preset.sh" "$@"
build_example
binary="$example_bin/<binary>"
# ... sweep logic ...
```

`build_example` configures+builds the example's `CMakeLists.txt` in
`$here/build/$preset` via the find-or-fetch path. The user can override
the preset with `RETICOLO_PRESET=macos-llvm ./run.sh` or pass `--preset`.

## Step 3 — register in examples/CMakeLists.txt

```cmake
add_subdirectory(NN_<name>)
```

This is only reached when `RETICOLO_BUILD_EXAMPLES=ON` (set by all
presets); it compile-checks the example in CI without affecting library
consumers.

## Step 4 — README.md

Include a **Building & running** section (see the template in each
example's README). One-liner for standalone build:

```bash
cd examples/NN_<name>
cmake -S . -B build && cmake --build build
```

And note that `./run.sh` does this automatically.
