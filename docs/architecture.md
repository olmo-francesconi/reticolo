# Architecture

This document describes the pieces of reticolo a user actually touches and
why each one is shaped the way it is. The full v3 design rationale (including
the audit of the previous code lines that motivated each constraint) lives in
[`proposals/rewrite_plan_v3.md`](../proposals/rewrite_plan_v3.md).

## Three CMake targets

```
                  ┌─────────────────────────────┐
                  │  reticolo::reticolo         │  <reticolo/reticolo.hpp>
                  │  (INTERFACE umbrella)       │  one include for users
                  └────────────┬────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
┌───────▼──────────┐  ┌────────▼────────┐  ┌─────────▼─────────┐
│ reticolo::core   │  │ reticolo::io    │  │ reticolo::cli     │
│ (INTERFACE,      │  │ (STATIC,        │  │ (INTERFACE,       │
│  header-only)    │  │  PIMPL'd HDF5)  │  │  wraps cxxopts)   │
└──────────────────┘  └─────────────────┘  └───────────────────┘
        │                      │                      │
   actions/                src/io/                  cxxopts
   algorithms/            writer.cpp               (FetchContent)
   observers/             (the ONLY TU
   lattice                 that #includes
   rng                     <hdf5.h>)
```

**Why three targets instead of one header-only library?** Header-only is the
ideal for the core (cheap compile, no link-time surprises, perfect inlining
into the templated updaters), but `<hdf5.h>` is 50+ KLOC of C headers that
would balloon every TU that touched IO. The PIMPL split puts HDF5 inside a
single `.cpp` and exposes a small typed surface (`io::Writer`, `Series<T>`).
HDF5's `hid_t` never leaves `src/io/writer.cpp`.

## The umbrella header

```cpp
#include <reticolo/reticolo.hpp>
```

re-exports every public header (core, actions, algorithms, observers, IO,
CLI). The library uses short namespace names directly — `reticolo::alg`,
`reticolo::obs`, `reticolo::io`, `reticolo::cli` — and the umbrella adds
exactly one alias inside `namespace reticolo` so the longer-named action
namespace can be referenced as `act`:

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
`std::complex<double>` for charged scalars (used by `act::BoseGas`).

```cpp
Lattice<double>            phi{{16, 16, 16}};        // 3D, fresh Indexing from pool
Lattice<double>            momentum{phi.indexing()}; // sibling: same shape, share Indexing
Lattice<std::array<double, 3>> on3{{20, 20, 20}};    // O(3)
```

Two ctors that take a `shared_ptr<Indexing const>` ("sibling" lattices) let
the HMC stash a momentum and force buffer with one neighbour table instead of
three. This is the most performance-sensitive shortcut in the library.

### `Indexing` + pool

`Indexing` holds the neighbour table (`next(s, mu)`, `prev(s, mu)`), parity
labels, and the `shape` it was built from. It's expensive to construct
(precomputes everything once) and immutable once built, so the library
keeps a process-wide pool of them keyed on `shape`:

```cpp
auto idx = Indexing::acquire(shape);   // hits pool, builds once
```

`acquire` returns a `shared_ptr<Indexing const>`; the pool itself holds
`weak_ptr`s, so once nobody references an entry it dies. Lattices with the
same shape automatically share — no manual cache management.

Boundary conditions are always periodic. The current scope of the library
is bulk-spectrum physics where periodic BCs are universal; non-periodic
BCs and the associated bulk/skin partitioning have been removed in favour
of a single tight hot loop with no per-site BC branching.

### `Site`, `Parity`

`Site` is a strong typedef around a flat index. `Parity` labels each site
as `Even` or `Odd` based on the sum of its coordinates — used by checker-
board Metropolis sweeps.

### `FastRng`

xoshiro256++ (Blackman & Vigna 2018) with SplitMix64 seeding, Box-Muller
polar normal with the second sample cached, Lemire's debiased rejection
for `uniform_int(n)`. State is 4 × `uint64_t`. Copies are independent
(each subsequent draw on a copy diverges from the original; useful for
reproducible chains-of-chains).

There's also an `Rng` concept; you can plug in a different generator and
the algorithms typecheck against the concept (`Metropolis`, `Hmc`, `Wolff`
all take an `R` template parameter constrained by `Rng<R>`).

## Actions

Plain structs. Members are doubles or other PODs (the couplings) and a few
member functions providing the local evaluations the updaters need. The
exact set is controlled by the **concept refinement lattice** —
see [`action_concepts.md`](action_concepts.md).

```cpp
template <class T = double>
struct Phi4 {
    using value_type = T;
    T kappa  = 0;
    T lambda = 0;

    T s_local(Lattice<T> const&, Site) const noexcept;
    T ds_local(Lattice<T> const&, Site, T new_v) const noexcept;
    T s_full (Lattice<T> const&) const noexcept;
    void compute_force(Lattice<T> const&, Lattice<T>& force) const noexcept;
};
```

That's it. No virtual table, no inheritance from a base class. Concept
matching at the updater instantiation site picks the right code path.

## Updaters

| updater                 | needs                        | header                                    |
| ----------------------- | ---------------------------- | ----------------------------------------- |
| `alg::Metropolis<A,R>`  | `LocalAction` (+ optionally `HasProposal`)        | `<reticolo/algorithm/metropolis.hpp>` |
| `alg::Hmc<A,R,Integ>`   | `HasSEff` + `HasForce`                            | `<reticolo/algorithm/hmc.hpp>`        |
| `alg::Wolff<A,R>`       | `WolffEmbeddable`                                 | `<reticolo/algorithm/wolff.hpp>`      |

The HMC integrator is a **type parameter**, not a runtime switch. Three
ship: `alg::integ::Leapfrog` (2nd-order, the cheap default),
`alg::integ::Omelyan2` (2nd-order minimum-norm, ~1.4× speedup at the same
acceptance), `alg::integ::Omelyan4` (4th-order, large τ regime). Adding a
new one means writing a struct with a static `run(...)` method and
instantiating `Hmc<A, R, NewInteg>` — no central registry, no
per-trajectory `switch`.

The HMC also keeps its momentum, force, and rollback buffers as **sibling
lattices** (three `Lattice<F>` constructed from the field's
`shared_ptr<Indexing const>`). Three lattices, one neighbour table.

## IO

`io::Writer` opens with truncate-or-fail and stamps a self-describing
`/run@*` block on every output:

```
/run@cmdline               argv joined
/run@version               library version
/run@commit                git SHA
/run@compile_flags         comma-separated flags
/run@hostname              uname -n
/run@started_utc           ISO-8601
/run@hdf5_complex_schema   "legacy_compound"
/run@hdf5_library_version  HDF5 lib version string
```

Passing the constructor a `cli::Parser*` also stamps each resolved CLI flag
to `/vars@<name>` so the file is fully self-describing. There is no separate
config-file format — the argv + the git SHA in `/run@` *is* the run record.

`Series<T>` is the only way to write a time series:

```cpp
auto s = writer.series<double>("/prod/obs/s");
s.append(action.s_full(phi));   // buffered, flushed on full chunk or dtor
```

It's `extern template`-instantiated for the supported scalar set (`int`,
`long`, `long long`, `float`, `double`, signed/unsigned variants,
`std::complex<float/double>`, `std::string` separately) so the heavy HDF5
specialisations compile once in `writer.cpp`. Returned by value (`Series<T>`
is move-only) so writer-state mutations cannot invalidate a handle held by
the caller.

## CLI

`cli::Parser` wraps cxxopts. The novel piece is the lifetime:

```cpp
cli::Parser p{"phi4_hmc", "..."};
auto const& kappa = p.req<double>("kappa", "hopping");  // T const& to stable storage
p.parse(argc, argv);
// kappa is now meaningful
```

`req<T>` and `opt<T>` return `T const&` to a `unique_ptr`-backed slot inside
the parser. Slots live as long as the parser does, so an app can plant
references at registration time and read them after `parse()`.

`io::Writer`'s constructor takes a `cli::Parser const*` and, when non-null,
calls `parser.stamp_into(writer)` after the file is initialised — every
registered var stamps to `/vars@<name>`. Apps therefore never write the
stamping loop themselves.

## Observers

Two namespaces in [`<reticolo/obs/*.hpp>`](../include/reticolo/obs/):

- `obs::` — per-configuration scalars: `mean`, `sq`, `magnetization`, `m2`,
  `mean_sq`, `m4`, `two_point`, `vector_magnetization_sq<N>`,
  `xy_magnetization_sq`. Every observer returns a `double`.
- `obs::analysis::` — ensemble reductions over a `std::span<double const>`:
  `mean`, `susceptibility`, `binder`.

Apps call the per-configuration observers in the trajectory loop and stash
the results in a `Series<double>`. Reductions live downstream (post-process
in Python, or in C++ over a span if you prefer).

## Logger

`reticolo::log` is a threadsafe, OpenMP-aware logger sitting on top of a
single mutex around `std::cout` / `std::cerr`. The surface:

- Severity = sigil — `·` debug, `┃` info, `⚠` warn, `✖` error. Each line
  carries elapsed time (`HHH:MM:SS.mmm`), a 4-char tag, and the message.
- Run column — `r000`, `r001`, … inside a `log::scope("rNNN")` (RAII);
  `main` outside any scope. LLR apps bind the scope per replica inside the
  parallel-for body; everything called transitively from there picks up the
  tag automatically via a thread-local stack.
- Auto-announce — `Lattice`, `RNG`, `Writer`, action, algorithm, and
  `llr::Replica` constructors emit their own descriptor lines, so most apps
  don't call `log::info` at all.
- Global off switch — `log::off()` short-circuits before any formatting.
  Tests link a shared `tests/test_main.cpp` that calls it; benches and the
  `tune_phi4_*` apps call it too.

See [`docs/logging.md`](logging.md) for the full surface.

## LLR replicas

LLR (Logarithmic Linear Relaxation) is an alternative sampling strategy
that reconstructs the density of states ρ(S) by running N replicas, each
pinned by a Gaussian window penalty to a different region of action space.
The replicas are sampled in parallel (OpenMP), each by its own HMC kernel,
and a Newton-Raphson + Robbins-Monro loop adapts a per-replica reweighting
parameter `a` so each replica sits centered on its window. Periodic
even/odd replica exchange improves mixing.

`llr::Replica` wraps a `Base` action in `llr::WindowedAction` (which adds
the `a·S + (S − E_n)²/2δ²` shift) and holds its own RNG / lattice / HMC.
LLR is the one place in the library where OpenMP shows up.

See [`docs/llr.md`](llr.md) for theory + an annotated app walkthrough.

## Tests

Over a hundred Catch2 cases live in `tests/`:

- `tests/unit/` — `Site`, `Indexing`, pool, `Lattice`, `FastRng`,
  `Parser`, observers, analysis.
- `tests/physics/` — force-vs-FD consistency for every action with HMC, HMC
  reversibility & integrator-order tests, Metropolis acceptance in the
  Gaussian limit, Wolff invariants (R²=id, β=0 → singleton cluster,
  large-β → near-full clusters).
- `tests/io/` — Writer round-trips, metadata stamping, phase collision
  rejection, `/vars` stamping.
- `tests/apps/` — every reference app smoke-tested end-to-end: run the
  binary on tiny inputs, open the HDF5, assert the schema. Boilerplate is
  factored into `tests/test_helpers.hpp`.

CI matrix: `macos-appleclang`, `macos-llvm`, `linux-gcc`, `linux-clang`,
plus `clang-format` (pinned 20.1.7 via pip) and `clang-tidy` (scoped to
production TUs via the amalgamation pattern — see `src/lint/`).
