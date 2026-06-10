# Architecture

The pieces of reticolo a user actually touches, and why each one is shaped
the way it is.

## Three CMake targets

- `reticolo::core` — INTERFACE, header-only. Actions, algorithms, observers,
  lattice, RNG, LLR.
- `reticolo::io` — STATIC. The only TU that `#include <hdf5.h>`; HDF5's
  `hid_t` never leaves `src/io/writer.cpp` thanks to a PIMPL'd `io::Writer`
  and `extern template`'d `Series<T>`.
- `reticolo::cli` — INTERFACE wrapper around cxxopts.

The umbrella `reticolo::reticolo` aggregates all three. Apps link only the
umbrella and include only `<reticolo/reticolo.hpp>`.

Header-only is ideal for the core (cheap compile, perfect inlining into the
templated updaters), but `<hdf5.h>` is 50+ KLOC of C headers that would
balloon every TU touching IO. The split keeps both.

## The umbrella header

```cpp
#include <reticolo/reticolo.hpp>
```

re-exports every public header. The library uses short namespaces directly
(`reticolo::alg`, `reticolo::obs`, `reticolo::io`, `reticolo::cli`) and the
umbrella adds one alias inside `namespace reticolo`:

```cpp
namespace act = action;
```

so `using namespace reticolo;` in an app yields `Lattice<double>`, `FastRng`,
`act::Phi4`, `alg::Hmc`, `io::Writer`, `cli::Parser`, `obs::magnetization`
all in scope.

## Core types

### `Lattice<T>`

Value semantics: copy = deep-copy the field, share the `Indexing`. Move is
cheap. The element type `T` is whatever the action needs — `double` for
real scalar fields, `std::array<double, N>` for O(N) sigma models,
`std::complex<double>` for charged scalars.

```cpp
Lattice<double>                phi{{16, 16, 16}};        // 3D, fresh Indexing from pool
Lattice<double>                momentum{phi.indexing()}; // sibling: share Indexing
Lattice<std::array<double, 3>> on3{{20, 20, 20}};        // O(3)
```

The sibling constructor lets HMC stash momentum, force, and rollback
buffers with one neighbour table instead of three. It is the most
performance-sensitive shortcut in the library.

### `Indexing` + pool

`Indexing` holds the neighbour table (`next(s, mu)`, `prev(s, mu)`), parity
labels, and the shape it was built from. It's expensive to construct
(precomputes everything once) and immutable, so the library keeps a
process-wide pool keyed on `shape`:

```cpp
auto idx = Indexing::acquire(shape);   // hits pool, builds once
```

The pool holds `weak_ptr`s; entries die when nobody holds them. Lattices
with the same shape automatically share.

Boundary conditions are always periodic. `next` / `prev` are unbranched.
Open / antiperiodic BCs are out of scope.

### `FastRng`

xoshiro256++ with SplitMix64 seeding, Box-Muller polar normal with the
second sample cached, Lemire's debiased rejection for `uniform_int(n)`.
Copies are independent — each subsequent draw on a copy diverges. There's
also an `Rng` concept; algorithms typecheck a different generator against
it.

## Actions

Plain structs satisfying C++23 concepts in
[`<reticolo/action/concepts.hpp>`](../include/reticolo/action/concepts.hpp).
No base class, no virtual, no `register_action`. An action just *has* the
right member functions; the updater concept-checks at the call site.

### The concept refinement

- **`LocalAction`** — Metropolis baseline. `s_local(l, x)` returns the
  contribution to S touching site x; `ds_local(l, x, new_v)` returns
  `s_local`'s change if `phi(x)` is replaced. The Metropolis sweep accepts
  with `min(1, exp(-ds_local))`.
- **`HasSEff`** — `s_full(l)` returns total S. Needed by HMC (for ΔH) and
  any diagnostic series logging S.
- **`HasForce`** — `compute_force(l, force)` writes `-dS/dphi` into `force`.
  Called once per MD step. `HasForce + HasSEff + Rng` is the `alg::Hmc`
  requirement.
- **`HasProposal`** — `propose(l, x, rng)` returns a candidate `phi(x)`. If
  present, `alg::Metropolis` uses it; otherwise it falls back to a Gaussian
  random walk. Selection happens at instantiation via `if constexpr`. Used
  by `OnSigma<N>` for uniform-on-sphere draws (Marsaglia).
- **`WolffEmbeddable`** — `axis_type`, `wolff_random_axis(rng)`,
  `wolff_reflect(v, axis)`, `wolff_link_p(v_x, v_y, axis)`. The cluster
  updater stays action-agnostic by deferring the bond-activation
  probability to the action — XY uses
  `1 - exp(min(0, -2β sin(θ_x-r) sin(θ_y-r)))`, OnSigma uses
  `1 - exp(min(0, -2β (r·φ_x)(r·φ_y)))`, etc.

### Example: `act::Phi4`

```cpp
template <class T = double>
struct Phi4 {
    using value_type = T;
    T kappa  = 0;
    T lambda = 0;

    T s_local (Lattice<T> const&, Site) const noexcept;
    T ds_local(Lattice<T> const&, Site, T new_v) const noexcept;
    T s_full  (Lattice<T> const&) const noexcept;
    void compute_force(Lattice<T> const&, Lattice<T>& force) const noexcept;
};
```

Satisfies `LocalAction + HasSEff + HasForce`. Works with both
`alg::Metropolis<Phi4<double>, FastRng>` and
`alg::Hmc<Phi4<double>, FastRng>`. A new action is one header file added
to the umbrella include; nothing else changes.

Concept failures at the updater instantiation site point at the missing
member — read the compiler error.

## Updaters

| updater                | needs                                     |
| ---------------------- | ----------------------------------------- |
| `alg::Metropolis<A,R>` | `LocalAction` (+ optionally `HasProposal`) |
| `alg::Hmc<A,R,Integ>`  | `HasSEff` + `HasForce`                     |
| `alg::Wolff<A,R>`      | `WolffEmbeddable`                          |

The HMC integrator is a **type parameter**, not a runtime switch. Three
ship: `alg::integ::Leapfrog` (2nd-order, cheap default),
`alg::integ::Omelyan2` (2nd-order minimum-norm, ~1.4× speedup at the same
acceptance), `alg::integ::Omelyan4` (4th-order, large τ). Select one at
the `Hmc` ctor with a brace-free tag value (`alg::integ::omelyan2`, CTAD
deduces the type) or the explicit `Hmc<A, R, Omelyan2>` form. Adding one =
struct with a static `run(...)`; the matching `inline constexpr` tag is a
one-liner.

HMC keeps its momentum, force, and rollback buffers as sibling lattices
of the field — one neighbour table, three lattices.

## IO

`io::Writer` opens with truncate-or-fail and stamps a self-describing
`/run@*` block: `cmdline`, `version`, `commit`, `compile_flags`, `hostname`,
`started_utc`, `hdf5_complex_schema`, `hdf5_library_version`. Passing a
`cli::Parser*` also stamps each resolved CLI flag to `/vars@<name>`. There
is no separate config-file format — argv + the git SHA *is* the run record.

`Series<T>` is the only way to write a time series:

```cpp
auto s = writer.series<double>("/prod/obs/s");
s.append(action.s_full(phi));   // buffered, flushed on full chunk or dtor
```

Move-only, returned by value (writer-state mutations cannot invalidate a
handle), buffered with chunked flush, `extern template`'d in `writer.cpp`
for the supported scalar set.

## CLI

`cli::Parser` wraps cxxopts. `req<T>(name, desc)` and
`opt<T>(name, default, desc)` return `T const&` to `unique_ptr`-backed
storage inside the parser:

```cpp
cli::Parser p{"phi4_hmc", "..."};
auto const& kappa = p.req<double>("kappa", "hopping");
p.parse(argc, argv);
// kappa is now meaningful
```

`io::Writer{path, argc, argv, &parser}` calls `parser.stamp_into(writer)`
after init — apps never write the stamping loop.

## Observers

Two namespaces:

- `obs::` — per-configuration scalars: `mean`, `sq`, `magnetization`, `m2`,
  `mean_sq`, `m4`, `two_point`, `vector_magnetization_sq<N>`,
  `xy_magnetization_sq`. All return `double`.
- `obs::analysis::` — ensemble reductions over a `std::span<double const>`:
  `mean`, `susceptibility`, `binder`.

Apps call per-configuration observers in the trajectory loop and stash the
results in a `Series<double>`. Reductions live downstream (Python, or in
C++ over a span).

## Logger

`reticolo::log` is a thread-safe (mutex around `std::cout` / `std::cerr`),
OpenMP-aware logger. Severity = sigil (`·` debug, `┃` info, `⚠` warn,
`✖` error); each line carries elapsed `HHH:MM:SS.mmm` + a 4-char tag.

`log::start(workspace, out_name[, replicas])` is the single init: it
creates the workspace folder, opens the main log file
`<workspace>/<stem>.log` (stem = out_name minus extension, so sweeps
sharing a workspace don't collide) and mirrors every entry into it, then
prints the banner. With `replicas = true` each scoped run id also gets its
own `<workspace>/<stem>.<rNNN>.log`.

`log::scope("rNNN")` (RAII) binds a thread-local replica tag —
`llr::Replica` binds its own id inside its public methods, so apps and
drivers only bind a scope when they run their own logging code inside a
parallel region; transitively-called code picks the tag up automatically.

`Lattice`, `FastRng`, `Writer`, action, algorithm, and `llr::Replica`
constructors auto-announce, so most apps don't call `log::info` at all.
`log::off()` short-circuits before any formatting — tests link a shared
`tests/test_main.cpp` that calls it; benches and `tune_*` apps too.

## LLR replicas

LLR reconstructs the density of states ρ(S) by running N replicas, each
pinned by a Gaussian-window penalty to a different region of action space.
Replicas sample in parallel (OpenMP), each by its own HMC. A
Newton-Raphson + Robbins-Monro loop adapts a per-replica reweighting
parameter `a`. Periodic even/odd replica exchange improves mixing.

`llr::Replica` wraps a `Base` action in `llr::WindowedAction` (adds the
`a·S + (S − E_n)²/2δ²` shift) and holds its own RNG / lattice / HMC. LLR
is the one place in the library where OpenMP shows up.

Two convention watch-outs:

- The Newton/RM step is `C·<dE>/δ²` with `C=12` for the hard window and
  `C=1` for the Gaussian window. Mixing them diverges geometrically.
- In the paper convention (`s_full = -a·S - window`), `a` flips sign at
  the natural peak, so `a_init = -1` and DoS reconstruction uses `(1+a)`.
  Energy convention keeps `a_init = 0` and integrates `a` directly.

## apps/ and examples/

### Builtin apps (`apps/`)

`apps/` is the canonical reference set: one sim per action
(`phi4_hmc`, `phi6_hmc`, `sine_gordon_hmc`, `xy_wolff`,
`on_sigma_metropolis`/`wolff`, `bose_gas_hmc`, `u1_metropolis`/`hmc`,
`su2_hmc`, `su3_hmc`), the LLR sims (`phi4_llr`, `u1_llr`,
`u1_llr_smoothed`, `bose_gas_llr`, `su2_llr`), `f32` variants, and the
`bench_*` suite. Built in-tree only; registered with `reticolo_add_app`
in `apps/CMakeLists.txt`.

### Standalone examples (`examples/`)

`examples/NN_short_name/` are **standalone consumer projects**. Each
directory can be copied out of the repo and still build: it carries its
own driver source(s), an inline find-or-fetch `CMakeLists.txt`, a `run.sh`
bash sweep, and an `analyze.py`.

The find-or-fetch block near the top of each example's `CMakeLists.txt`
resolves `reticolo::reticolo` in three stages:

1. Reuse the target if already configured (in-tree aggregate build).
2. `add_subdirectory("../../")` if the sibling checkout is present
   (`../../include/reticolo/reticolo.hpp` exists).
3. `FetchContent_Declare` a pinned git tag otherwise.

```cmake
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
```

Targets are named `exNN_<binary>` with `OUTPUT_NAME` set to the clean
binary name so the in-tree aggregate build has no target-name collisions.

### Build flags

`RETICOLO_BUILD_APPS`, `RETICOLO_BUILD_EXAMPLES`, and
`RETICOLO_BUILD_TESTS` all default `OFF`. A bare `cmake` or a
`FetchContent`/`add_subdirectory` consumer builds **core/io/cli only**.
Every preset in `CMakePresets.json` sets all three `ON`, so
`cmake --preset … && ctest --preset …` behaves exactly as before.

OpenMP is not `REQUIRED`; if the toolchain lacks it (e.g. Apple Clang)
the configure degrades gracefully to serial. The `macos-appleclang` preset
sets `RETICOLO_ENABLE_OPENMP=OFF` explicitly.

## Tests

- `tests/unit/` — types in isolation: `Site`, `Indexing`, pool, `Lattice`,
  `FastRng`, `Parser`, observers, analysis.
- `tests/physics/` — force-vs-FD consistency for every HMC-capable action,
  HMC reversibility & integrator-order, Metropolis acceptance in the
  Gaussian limit, Wolff invariants.
- `tests/io/` — Writer round-trips, metadata stamping, phase collision
  rejection, `/vars` stamping.
- `tests/apps/` — every reference app smoke-tested end-to-end: run the
  binary on tiny inputs, open the HDF5, assert the schema.

CI matrix: `macos-appleclang`, `macos-llvm`, `linux-gcc`, `linux-clang`,
plus `clang-format` (pinned 20.1.7) and `clang-tidy` (via the amalgamation
pattern in `src/lint/`).
