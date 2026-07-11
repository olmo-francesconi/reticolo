# Architecture

The pieces of reticolo a user actually touches, and why each one is shaped
the way it is.

## Three CMake targets

- `reticolo::core` — INTERFACE, header-only. Actions, updaters, observers,
  lattice, RNG, LLR.
- `reticolo::io` — STATIC. The only TU that `#include <hdf5.h>`; HDF5's
  `hid_t` never leaves `src/io/writer.cpp` thanks to a PIMPL'd `io::Writer`
  and `extern template`'d `Series<T>`.
- `reticolo::cli` — INTERFACE wrapper around cxxopts.
- `reticolo::cuda` — INTERFACE, header-only, **optional**
  (`RETICOLO_ENABLE_CUDA=ON`). The device stack lives entirely in
  `include/reticolo/cuda/*.{hpp,cuh}`; nvcc compiles only the `.cu` consumers
  (`apps/cuda/` + `tests/cuda/`). CUDA never leaks into `reticolo::core` —
  automatic here, since `cuda/` headers only ever reach a `.cu` TU. See
  [§ CUDA backend](#cuda-backend).

The umbrella `reticolo::reticolo` aggregates the first three (plus `reticolo::cuda`
when enabled). Apps link only the umbrella and include only
`<reticolo/reticolo.hpp>` — a GPU app adds `<reticolo/cuda/cuda.hpp>`.

Header-only is ideal for the core (cheap compile, perfect inlining into the
templated updaters), but `<hdf5.h>` is 50+ KLOC of C headers that would
balloon every TU touching IO. The split keeps both.

## The umbrella header

```cpp
#include <reticolo/reticolo.hpp>
```

re-exports every public header. The library uses short namespaces directly
(`reticolo::updater`, `reticolo::obs`, `reticolo::io`, `reticolo::cli`) and the
umbrella adds one alias inside `namespace reticolo`:

```cpp
namespace act = action;
```

so `using namespace reticolo;` in an app yields `Lattice<double>`, `FastRng`,
`act::Phi4`, `updater::Hmc`, `io::Writer`, `cli::Parser`, `obs::magnetization`
all in scope.

## Core types

### `Lattice<T>`

Value semantics: copy = deep-copy the field, share the `Indexing`. Move is
cheap. The element type `T` is whatever the action needs — `double` for
real scalar fields, `std::complex<double>` for charged scalars (BoseGas).

```cpp
Lattice<double>                phi{{16, 16, 16}};        // 3D, fresh Indexing from pool
Lattice<double>                momentum{phi.indexing()}; // sibling: share Indexing
Lattice<std::array<double, 3>> on3{{20, 20, 20}};        // O(3)
```

The sibling constructor lets HMC stash momentum, force, and rollback
buffers sharing one `Indexing` instead of three. It is the most
performance-sensitive shortcut in the library.

### `Indexing` + pool

`Indexing` holds the shape it was built from and the corresponding strides, and
computes the neighbours `next(s, mu)` / `prev(s, mu)` on the fly from those
strides — no stored table (the periodic wrap is closed-form, `ndims ≤ 4`). It's
immutable and cheaply shared, so the library keeps a process-wide pool keyed on
`shape`:

```cpp
auto idx = Indexing::acquire(shape);   // hits pool, builds once
```

The pool holds `weak_ptr`s; entries die when nobody holds them. Lattices
with the same shape automatically share.

Boundary conditions are always periodic. `next` / `prev` are unbranched.
Open / antiperiodic BCs are out of scope.

### `FastRng` and the RNG families

xoshiro256++ with SplitMix64 seeding, Box-Muller polar normal with the
second sample cached, Lemire's debiased rejection for `uniform_int(n)`.
Copies are independent — each subsequent draw on a copy diverges. There's
also an `Rng` concept; algorithms typecheck a different generator against
it. Three siblings share the surface: `PhiloxRng` (counter-based, shared
core with the CUDA sampler), `RanlxdRng` (a clean-room, bit-compatible
reimplementation of Lüscher's `ranlxd` — provably distinct streams per seed),
`Mt19937Rng` (`std::mt19937_64`).

### `StreamSet<R>` — per-slab parallel streams

`core/rng/stream_set.hpp`. A pool of independent streams over one family:
one dedicated **driver** stream (serial draws: Metropolis accept, LLR
exchange, hot starts) + n **site** streams, each padded to its own cache
line (`exec::k_cache_line_bytes` — engine state is written per draw, packed
streams false-share). Stream independence is per-family: `FastRng::jump()`
(the published 2^128 xoshiro jump polynomial, provably disjoint),
`PhiloxRng::stream(seed, k)` (disjoint counter subspaces),
`RanlxdRng::stream(seed, k)` (sequential 31-bit seeds — Lüscher's seeding
proves distinct trajectories), and SplitMix-decorrelated seeds for the one
remaining std engine, `Mt19937Rng` (`JumpStream` / `KeyedStream` concepts).
The pool
holds **no geometry** — who draws from which stream is the owning
algorithm's contract.

`updater::Hmc` **owns** its randomness: the ctor takes a freshly-seeded family
generator **by value**, draws one `uniform_u64()` from it, and builds a
`StreamSet` with exactly one site stream per item of the canonical field
partition (`exec::partition` under the Hmc's frozen thread/slab config).
Momentum fills bind slab i to site stream i through
`exec::field_visit_indexed` — the binding is the partition item index, never
the executing thread — and every fill re-derives the partition and throws
`std::logic_error` on a count mismatch, so an environment change mid-run
refuses to sample instead of silently re-binding streams. Consequences:
`n_threads = 0` resolves against the ambient once at construction and is
frozen (`set_spec` refuses threading changes); a chain's identity is
(seed, shape, n_threads, slabs_per_thread), matching the contract the
`s_full`/kinetic reduce folds already impose. Checkpointing goes through
`hmc.rng()`: `io::save_config`/`load_config` write and validate every
stream's state words (`/rng@kind`, `@n_streams`, `@n_words` — a resume with
a different threading config fails loudly).

## Actions

Plain structs satisfying C++20 concepts in
[`<reticolo/action/concepts.hpp>`](../include/reticolo/action/concepts.hpp).
No base class, no virtual, no `register_action`. An action just *has* the
right member functions; the HMC updater concept-checks at the call site.
(The "no virtual" invariant is about the HMC hot loop — actions, integrators,
the updater. It is not a blanket ban: `cli::Parser`'s `VarSlotBase` type-erasure
is one small virtual hierarchy off the hot path, in the CLI layer, and that's
fine. A bare `grep virtual` will hit it; that's not an invariant break.)
Actions split into three families by interaction shape, one folder each:
`action/nn/` (scalar `Lattice<T>` nearest-neighbour — both the self+NN-sum
Phi4/Phi6/SineGordon and the endpoint-difference XY, unified under one
`NNAction`), `action/complex/` (`Lattice<complex<T>>` sign problem: BoseGas'
real part, with the imaginary part bolted on as a mixin), and `action/gauge/`
(`MatrixLinkLattice<G,T>`: Wilson). All satisfy the **same** field-agnostic
concepts; only the `Field` they name differs. Each family folder holds its leaf
structs and its family base (`<family>_action.hpp`) at the top and its per-action
physics in `formula/`; the shared dimension-generic traversal engine lives once
in `action/sweep/`.

### The concepts

- **`HmcAction<A, Field>`** — the baseline. `s_full(l)` returns total S (for
  ΔH and any diagnostic series logging S); `compute_force(l, force)` writes
  `-dS/dfield` into `force`, called once per MD step. `HmcAction + Rng` is the
  entire `updater::Hmc` requirement.
- **`HasFusedKick<A, Field>`** — refines `HmcAction` with
  `compute_force_and_kick(l, mom, k)`, which computes the force and applies
  `mom += k·F` in a single pass without materialising the force lattice. The
  integrator prefers this path when present and falls back to plain
  `compute_force` otherwise.
- **`HasImagPart<A, Field>`** — refines `HmcAction` for complex actions with a
  sign problem: `s_imag(l)` is the imaginary part (the LLR constraint
  observable in the phase-quenched ensemble) and `compute_force_imag(l, force)`
  its gradient. Only `BoseGas` uses it today; `orch::llr::WindowedAction` switches to
  its complex mode when the base satisfies it.

Earlier revisions carried a `LocalAction` Metropolis baseline
(`s_local`/`ds_local`) and a parallel `gauge::` concept family keyed on the
`(Site, mu)` elementary-update arity. With Metropolis and Wolff removed, both
are gone: HMC's `s_full`/`compute_force` are field-agnostic, so one concept set
covers site and gauge alike.

### Family bases — the physics is the only per-action code

The concepts say *what members* an action needs; the **family bases** supply
them so a leaf action carries only physics. Each base is a stateless CRTP mixin
at its family's top (`action/<family>/<family>_action.hpp`) that
owns the loop shells, the fused kick, and the `last_s_full` cache, and calls back
into the leaf's coupling-hoisting kernels:

| base | field | leaf kernels (bind the shared `<family>/formula/*`) |
| ---- | ----- | ---------------------------------------------------- |
| `NNAction<D,T>`      | `Lattice<T>`              | finalizers `action_kernel()`/`force_kernel()` + optional per-bond `action_combine`/`force_combine` (default identity) + `action_scale` (Phi4/Phi6/SineGordon: identity combine; XY: cos/sin bond combine + `−β`) |
| `ComplexAction<D,T>` + `ImagPart<D,T>` mixin | `Lattice<complex<T>>`     | real `action_kernel`/`force_kernel` (split-last) + `imag_action_kernel`/`imag_force_kernel` (`HasImagPart`: BoseGas) |
| `GaugeAction<D>`     | link field               | `s_full_uncached()`, `force_into()` (Wilson`<G>`) |

**One NN traversal, two knobs.** `NNAction` unifies the former SiteAction and
BondAction: every scalar NN action is the same sweep — fold a per-bond COMBINE
over the neighbours into `agg`, then FINALIZE `(self, agg)`. A site action's
combine is the identity (`agg = Σ neighbours`, the finalize does the physics); a
bond action's combine is the per-bond function (`cos(self−nbr)`) with a passthrough
finalize and an overall `action_scale` applied post-reduce (so `s_full` is
bit-identical to the old BondAction). Complex actions decompose the other way: the
real (phase-quenched) part is `ComplexAction` and the imaginary observable is the
orthogonal `ImagPart` mixin the leaf *also* derives — the decorator spirit of
`WindowedAction`, so `struct BoseGas : ComplexAction<…>, ImagPart<…>`.

The kernels return a lambda that captures the couplings by value, so the base
hoists them into the hot loop exactly as a hand-written action would — the
vectorised inner loop is unchanged. The gauge family is the odd one out: the
plaquette *traversal* can't be shared (U(1) batches four signed angles through a
Sleef cos/sin scratch on a `MatrixLinkLattice<U1,T>`; SU(N) batches complex matrix
products on a `MatrixLinkLattice<G,T>`), so `GaugeAction` only owns the
cache + the concept surface and the leaf provides `s_full_uncached`/`force_into`
(delegating the per-plaquette physics to `formula::wilson_kernels<G>` in the action layer; the group model `G` under `math/group/` holds only the core group ops).

### Example: `act::Phi4`

```cpp
template <class T = double>
struct Phi4 : NNAction<Phi4<T>, T> {   // base supplies s_full / compute_force /
    using value_type = T;                        //   compute_force_and_kick / caches
    T kappa  = 0;
    T lambda = 0;
    void describe(log::Entry&) const;

    auto force_kernel()  const { return [k=kappa,lam=lambda](std::size_t, T phi, T nbrs)
                                        { return formula::phi4_force_site<T>(phi,nbrs,k,lam); }; }
    auto action_kernel() const { return [k=kappa,lam=lambda](T phi, T fwd)
                                        { return formula::phi4_action_site<T>(phi,fwd,k,lam); }; }
    // optional LLR fast-path: double s_full_and_force(...) via this->staged_force_energy(...)
};
```

Deriving from `NNAction` still leaves `Phi4` an aggregate (the base is
stateless-by-designated-init), so `act::Phi4<double>{.kappa=…, .lambda=…}` is
unchanged. It satisfies `HmcAction` + `HasFusedKick` and drives
`updater::Hmc<Phi4<double>, FastRng>` directly. A new action is a formula file plus
this ~10-line struct, added to the family aggregator (`action/nn.hpp` /
`action/gauge.hpp`); nothing else changes.

Concept failures at the `updater::Hmc` instantiation site point at the missing
member — read the compiler error.

## Updaters

| updater                | models                | needs                                     |
| ---------------------- | --------------------- | ----------------------------------------- |
| `updater::Hmc<A,R,Integ>`  | `updater::Updater`        | `HmcAction` (+ optionally `HasFusedKick`) |

`updater::Hmc` is the only update algorithm. The **`updater::Updater`** concept
([`updater/concepts.hpp`](../include/reticolo/updater/concepts.hpp)) is the
updater-level analogue of `action::HmcAction` — the contract apps and the
orchestration layer rely on: `step()` (returns `{dH, accepted}`),
`last_s_full()`, `rng()`. It is duck-typed and checked at the use site (both
orchestration workers `static_assert` their sampler against it), so a new
updater is just a class modelling it — no base class, reuse the same
`updater::integ::*` type-parameters. `orch::llr::Replica` and `orch::span::Chain`
each own an `updater::Hmc` and drive it through this surface.

The HMC integrator is a **type parameter**, not a runtime switch. Three
ship: `updater::integ::Omelyan2` (2nd-order minimum-norm, ~1.4× speedup at the
same acceptance — the **default**), `updater::integ::Leapfrog` (2nd-order,
cheapest per MD step), `updater::integ::Omelyan4` (4th-order, large τ). Select
one at the `Hmc` ctor with a brace-free tag value (`updater::integ::leapfrog`, CTAD
deduces the type) or the explicit `Hmc<A, R, Omelyan2>` form. Adding one =
struct with a static `run(...)`; the matching `inline constexpr` tag is a
one-liner.

HMC keeps its momentum, force, and rollback buffers as sibling lattices
of the field — one shared `Indexing`, three lattices.

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

`io::Writer{path, argc, argv, &parser}` calls `parser.stamp(writer)`
after init — apps never write the stamping loop.

### App setup helpers (`app::`)

The pre-loop scaffolding shared by every reference app — the universal flag
block and the workspace/writer open — lives in `reticolo/app/setup.hpp` so it
can't drift between apps:

```cpp
cli::Parser p{"u1_hmc", "..."};
auto const f = app::common_flags(p, {.L = 4, .ndim = 4, .n_prod = 2000, .out = "u1_hmc.h5"});
auto const& beta = p.opt<double>("beta", 1.0, "Wilson coupling");   // physics flags: app's own
if (!p.parse(argc, argv)) return 0;
io::Writer out = app::open_writer(p, f, argc, argv);                // log::start + Writer
```

`common_flags` single-sources the flag *names / types / help text* (`L,size`,
`ndim`, `n_therm`, `n_prod`, `meas_every`, `seed`, `workspace`, `out`); the
per-app *defaults* are passed in, because they legitimately differ (lattice
extent, trajectory counts). This is setup only — the app still builds its own
lattice / rng / action / updater and **owns its trajectory `for` loop**; there
is no generic driver.

### Canonical output schema

So downstream Python reads the same paths across apps, reference apps follow one
HDF5 layout:

| Path                   | Meaning                                                   |
| ---------------------- | -------------------------------------------------------- |
| `/run@*`, `/vars@*`    | reproducibility metadata + every resolved CLI flag       |
| `/<phase>/stats/dH`    | per-trajectory ΔH (HMC); `/stats/s` for S during `therm` |
| `/<phase>/stats/accepted` | 0/1 acceptance flag (HMC), `int` series               |
| `/<phase>/obs/<name>`  | per-config observables (`s`, `mag`, `plaq`, …)           |

Phases are `therm` then `prod` for single-replica apps, `llr` for LLR apps.
Plaquette normalisation is reported as `<cos θ_p>` everywhere: U(1) writes
`S / (β · n_plaq)`; SU(N) writes `1 − S / (β · n_plaq)` — the constant offset is
the action convention, the *stored observable* is the mean plaquette in both.

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
`orch::llr::Replica` binds its own id inside its public methods, so apps and
drivers only bind a scope when they run their own logging code inside a
parallel region; transitively-called code picks the tag up automatically.

`Lattice`, `FastRng`, `Writer`, action, updater, and `orch::llr::Replica`
constructors auto-announce, so most apps don't call `log::info` at all.
`log::off()` short-circuits before any formatting — tests link a shared
`tests/test_main.cpp` that calls it; benches and `tune_*` apps too.

## Orchestration

An orchestration runs **many concurrent simulations** and reduces their output.
`orch/` is that subsystem: a physics-free **spine** at the top, with concrete
orchestrators in subfolders (`orch/span/`, `orch/llr/`). The dependency is
one-way `orch/<name> → orch → core` — the spine never mentions a window, a tilt,
or an exchange.

The spine (`reticolo::orch`) is four small pieces:

- **`Worker`** concept ([`orch/concepts.hpp`](../include/reticolo/orch/concepts.hpp))
  — a self-contained simulation unit; just `id()`. The `Checkpointable`
  refinement adds `field()` + `rng()`.
- **`ThreadPlan` / `plan_threads`** ([`orch/thread_plan.hpp`](../include/reticolo/orch/thread_plan.hpp))
  — split the ambient OpenMP budget into an outer worker team (`concurrency`)
  and inner per-worker HMC teams (`m`); saturate workers first.
- **`parallel_workers(workers, plan, body)`** ([`orch/ensemble.hpp`](../include/reticolo/orch/ensemble.hpp))
  — the one concurrent primitive: run `body(i, worker)` over every worker at
  once under the plan. Concurrent-only — the thread-unsafe drain (HDF5, logging)
  stays a serial loop in the caller, so *the orchestration owns its loop*.
- **`save_ensemble` / `load_ensemble`** ([`orch/checkpoint.hpp`](../include/reticolo/orch/checkpoint.hpp))
  — snapshot every `Checkpointable` worker's field + rng + `OrchState`; opt-in
  `save_extra`/`load_extra` per worker and an orchestrator-level extra hook keep
  it physics-free.

Two orchestrators ship as clients:

### Parameter span (`orch::span`)

`orch::span::Chain<Action,…>` is a generic HMC worker (any scalar/gauge action);
`orch::span::run` builds one per point of a parameter grid, runs them
concurrently, and records `/worker_NNN/…` time series. `apps/param_span_hmc`
sweeps `kappa` in one binary — an in-process replacement for an outer bash sweep.

### LLR (`orch::llr`)

LLR reconstructs the density of states ρ(S) by running N replicas, each pinned by
a Gaussian-window penalty to a different region of action space, each sampled by
its own HMC. A Newton-Raphson + Robbins-Monro loop adapts a per-replica
reweighting parameter `a`; periodic even/odd replica exchange improves mixing.
`orch::llr::Replica` wraps a `Base` action in `orch::llr::WindowedAction` (the
`a·S + (S − E_n)²/2δ²` shift) and is an `orch::Worker`; `orch::llr::run` /
`smoothed::run` are clients of the spine (they drive `parallel_workers` and add
exchange as a serial coupling between waves).

Two convention watch-outs:

- The Newton/RM step is `C·<dE>/δ²` with `C=12` for the hard window and
  `C=1` for the Gaussian window. Mixing them diverges geometrically.
- In the paper convention (`s_full = -a·S - window`), `a` flips sign at
  the natural peak, so `a_init = -1` and DoS reconstruction uses `(1+a)`.
  Energy convention keeps `a_init = 0` and integrates `a` directly.

## CUDA backend

Optional GPU backend (`RETICOLO_ENABLE_CUDA=ON`, default OFF). The design rule
is **one source of truth = the per-site formula**: the CPU and GPU never carry
two copies of the physics.

### The CPU/GPU boundary

GPU code is quarantined under a `cuda/` subfolder at every level — that folder
*is* the boundary:

```
include/reticolo/cuda/          the whole device stack — header-only (.hpp/.cuh).
                                reticolo::cuda is an INTERFACE target; there is no
                                compiled backend TU.
tests/cuda/                     GPU tests: one native .cu Catch2 file per action /
                                concern (test_cuda_<name>.cu), the device-vs-CPU
                                validation gates
apps/cuda/                      all runnable .cu binaries: the reference sims
                                (umbrella-linked) + bench/profile (reticolo::cuda
                                only); apps/ itself stays CPU-only
```

The dependency is strictly one-way: **`cuda → core`, never `core → cuda`**. The
only GPU-awareness in `core` is two deliberate shims:

- `core/hd.hpp` — the `RETICOLO_HD` macro (`__host__ __device__` under nvcc,
  expands to nothing otherwise), so one annotated function compiles for both.
- `core/rng.hpp` — a `__CUDACC__` guard exposing the shared counter-based Philox
  on the device.

Header naming inside `cuda/`: **`.cuh` = contains device kernels** (`__global__`/
`__device__` bodies, nvcc-mandatory); **`.hpp` = host-callable API** (may use the
CUDA runtime like `cudaMalloc`, but defines no kernels). Both are GPU-side —
neither is includable from a pure-host TU.

### How a formula reaches the GPU

The seam is a four-hop chain (Phi4 shown):

1. `action/nn/formula/phi4_formula.hpp` — the per-site formula, `RETICOLO_HD`,
   the single source of truth.
2. `action::Phi4` (CPU) and `cuda::Phi4ForceFunctor` (GPU) **both call that same
   formula** — they cannot silently diverge.
3. `cuda/actions/site/phi4.hpp` — the `device_functors<action::Phi4<T>>` trait adapts
   a *host* action struct into device launchers (force / s_full / sample_momenta).
4. `cuda::DeviceAction<HostAction, Field>` + `cuda::Hmc<DAct, Integ, Field>` — the
   generic device HMC, reusing the *same* `updater::integ::*` integrator tags as the
   CPU `updater::Hmc`. No virtual dispatch, no `switch(action)` — the trait resolves
   at compile time (a lint gate forbids integrator-specific kernel code).

The *loop structure* may legitimately differ CPU↔device (U(1) scatter→gather;
SU(N) expi slab→fused), but each such divergence has a device-vs-host math test.

### Writing a GPU app

A `.cu` app is the GPU twin of a CPU app — same CLI, same HDF5 schema, the
trajectory `for` plainly in `main()`. It includes both umbrellas:

```cpp
#include <reticolo/reticolo.hpp>     // core + io + cli (host)
#include <reticolo/cuda/cuda.hpp>    // the device stack (nvcc-only)
```

`io::Writer` PIMPLs HDF5, so nvcc never sees `<hdf5.h>`; the app just links the
prebuilt `reticolo::io` archive. This works because the host `-Wall`/`-Werror`
flags are `$<COMPILE_LANGUAGE:CXX>`-guarded (`cmake/ReticoloWarnings.cmake`) and
OpenMP is off, so neither reaches the nvcc compile. See
[`writing_a_cuda_app.md`](writing_a_cuda_app.md) for the full walk-through.

## apps/ and examples/

### Builtin apps (`apps/`)

`apps/` is the canonical reference set: one HMC sim per action
(`phi4_hmc`, `phi6_hmc`, `sine_gordon_hmc`, `bose_gas_hmc`, `u1_hmc`,
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
- `tests/physics/` — force-vs-FD consistency for every action,
  HMC reversibility & integrator-order across the gauge groups.
- `tests/io/` — Writer round-trips, metadata stamping, phase collision
  rejection, `/vars` stamping.
- `tests/apps/` — every reference app smoke-tested end-to-end: run the
  binary on tiny inputs, open the HDF5, assert the schema.

CI matrix: `macos-appleclang`, `macos-llvm`, `linux-gcc`, `linux-clang`,
plus `clang-format` (pinned 20.1.7) and `clang-tidy` (via the amalgamation
pattern in `src/lint/`).
