# Architecture

The pieces of reticolo a user actually touches, and why each one is shaped
the way it is.

## Three CMake targets

- `reticolo::core` — INTERFACE, header-only. Actions, updaters, observers,
  lattice, RNG, LLR.
- `reticolo::io` — STATIC. The only TUs that `#include <hdf5.h>` are
  `src/io/writer.cpp` and `src/io/reader.cpp`; HDF5's `hid_t` never leaves
  `src/io/` thanks to a PIMPL'd `io::Writer` and `extern template`'d
  `Series<T>`.
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
`act::Phi4`, `updater::Hmc`, `io::Writer`, `cli::Parser`, `obs::reduce`
all in scope.

## Core types

### `Lattice<T>`

Value semantics: copy = deep-copy, both data and geometry. Move is cheap. The
element type `T` is whatever the action needs — `double` for real scalar fields,
`std::complex<double>` for charged scalars (BoseGas).

```cpp
Lattice<double>                phi{{16, 16, 16}};   // 3D
Lattice<double>                momentum{phi.shape()};  // sibling: same shape, fresh storage
Lattice<std::array<double, 3>> on3{{20, 20, 20}};   // O(3)
```

### Geometry lives in the field

Shape, packed strides and site count are the field's own state, through the
private base `impl::LatticeGeometry` — so `shape()`, `ndims()`, `nsites()`,
`next(s, mu)` and `prev(s, mu)` are members of the field, and no geometry object
appears in any signature. `shape()` returns a `std::span<std::size_t const>` over
extents that live inline (build a shape with `std::vector<std::size_t>`; read one
back through the span).

This used to be a separate `Indexing` class, immutable and shared between
same-shape fields through a `shared_ptr` out of a process-wide `weak_ptr` pool.
That was worth it while `Indexing` carried neighbour **tables** — `nsites·d·2·8`
bytes, tens of MB at production volumes. Those are gone: the periodic wrap is
closed-form on the strides (`ndims ≤ 4`) and the hot nests roll their own
row-nested offsets. What remained was 64 bytes reached through a mutex, an
atomic refcount and a pointer chase, saving 240 bytes across HMC's four buffers
against 40 MB of payload each — so the pool, the sharing and the class went, and
a sibling buffer is now four multiplies:

```cpp
Lattice<double> mom{phi.shape()};   // HMC's mom / force / old_field
```

Two consequences worth knowing. Sibling fields have no lifetime coupling at all
(nothing is shared, so nothing can dangle). And "same geometry" is now a value
question — `a.same_geometry(b)` — where it used to be answered by pointer
identity on the pooled object, which was only ever a proxy for it.

Boundary conditions are always periodic. `next` / `prev` are unbranched.
Open / antiperiodic BCs are out of scope, and would land as a separate geometry
base rather than a runtime flag.

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
in `core/exec/` (namespace `exec`, the `nn_*` family — `nn_visit`/`nn_reduce`
and friends), alongside the per-site `exec` primitives it builds on.

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
  its gradient. Only `BoseGas` uses it today; `action::WindowedAction` switches to
  its complex mode when the base satisfies it.

(Historical note, now partly superseded — a local updater is back, but with the
elementary move kept inside the action so the arity never crosses the
interface; see `action::LocalAction`.) Earlier revisions carried a `LocalAction` Metropolis baseline
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
vectorised inner loop is unchanged.

`NNAction` also exposes the per-site **local energy** as a callable —
`local_energy_kernel()` (a `(cand, gather) -> T` over the all-directions gather)
plus `local_energy_scale()`. `metropolis_stencil` is written in terms of it, but
the reason it is a public member is that a *decorator* action needs to evaluate
several actions' local energies against one traversal: `WindowedAction` wants the
base's ΔS and the constraint observable's ΔQ at the same site, and neither can
drive its own sweep of the other's lattice. Kernel and scale stay separate and
un-premultiplied because a local energy is only ever used as a difference of two
calls — applying the scale to the difference is what keeps this bit-identical to
a hand-written ΔS, which is not a formality in f32.

The gauge family is the odd one out: the
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

### One parameter shape, everywhere

Every class template that carries an action follows the same rule, so there is
one thing to learn rather than six:

```cpp
updater::Hmc          <Action, Rng, Integrator = integ::Omelyan2>
updater::Metropolis   <Action, Rng>
action::WindowedAction<Action, Constraint = void>
orch::span::Chain     <Action, Rng, Sampler>
orch::llr::Replica    <Action, Rng, Sampler, Constraint = void>
orch::llr::Orchestrator<Action, Rng, Sampler, Constraint = void>
cuda::Hmc             <Action, Integ = integ::Omelyan2>
cuda::Metropolis      <Action>
```

- **The element and field types are never parameters.** Every action names both
  (`value_type`, `field_type` — the family bases declare the latter, as does
  `cuda::DeviceAction`), so a parameter for either could only be set to what the
  action already says, or to something wrong. Gauge and scalar therefore read
  identically: `Replica<Wilson<SU3>, FastRng, updater::HmcSampler<>>`.
- **The MD integrator is never a sibling parameter of a sampler choice.** Where
  more than one updater can drive the thing, it rides on the sampler tag
  (`updater::HmcSampler<integ::Omelyan4>`), so it does not exist when the sampler
  is local. Where HMC is the only option (`updater::Hmc` itself, `cuda::`) it is
  a plain parameter.
- **The sampler has no default** in the orchestration workers: which updater
  drives a run is worth reading at the call site. The integrator keeps one
  (`HmcSampler<>` = Omelyan2) — that is a knob inside an already-named sampler.
- **A `Spec` carries only the chosen sampler's knobs.** Both orchestrators nest
  their `Spec` so it is parameterised by their own arguments, and the sampler
  block is `Sampler::spec_type` — `{.tau, .n_md}` under HMC, `{.sigma}` under
  Metropolis. No Spec advertises a field the sampler in play would ignore, and a
  future sampler needs no edit to either. `setup()` resolves the ensemble thread
  request into it (`if constexpr (requires { s.n_threads; })`), so the outer Spec
  keeps only `worker_threads` / `replica_threads`.
- Parameter names are the same words in the same order: `Action`, `Rng`,
  `Sampler`/`Integrator`, `Constraint`.

The one deliberate exception is `cuda::llr::Replica<HostAction, Integ, Field>`,
which keeps both. It takes a *host* action — which names a host lattice, not a
device field — so the device field's precision is a genuine choice; and it is not
sampler-parameterised because a windowed sweep must be sequential with a running
Q, which on a GPU is a serial walk over V sites. Offering that choice would be
offering a trap.


| updater                                    | models             | needs                                     |
| ------------------------------------------ | ------------------ | ----------------------------------------- |
| `updater::Hmc<Action, Rng, Integrator>`    | `updater::Updater` | `HmcAction` (+ optionally `HasFusedKick`) |
| `updater::Metropolis<Action, Rng>`         | `updater::Updater` | `action::LocalAction`                     |

`updater::Hmc` (global) and `updater::Metropolis` (local checkerboard sweep) are the update algorithms. The **`updater::Updater`** concept
([`updater/concepts.hpp`](../include/reticolo/updater/concepts.hpp)) is the
updater-level analogue of `action::HmcAction` — the contract apps and the
orchestration layer rely on: `step().acceptance()`, `last_s_full()`, `rng()`.
It keys on `acceptance()` rather than on `dH`/`accepted` because those are not
shared: one HMC trajectory is accepted or not, while one Metropolis sweep is
V independent accept tests with no meaningful boolean. Algorithm-specific
fields live on the concrete result (`HmcResult::dH`, `action::LocalStats::n_accepted`). It is duck-typed and checked at the use site (both
orchestration workers `static_assert` their sampler against it), so a new
updater is just a class modelling it — no base class, reuse the same
`updater::integ::*` type-parameters. `orch::llr::Replica` and `orch::span::Chain`
each own an `updater::Hmc` and drive it through this surface.

`updater::Metropolis` is the local counterpart: a checkerboard sweep in place of
a trajectory. It owns no momentum, force or rollback buffer — a rejected local
move is discarded before it is written — and carries `S` incrementally from the
ΔS the sweep already knew, so a measurement costs no extra pass. What it needs
from the action is `action::LocalAction`: `n_colors(l)` and
`metropolis_stencil(l, color, noise, logu, sigma)`. Both randomness fields are
filled by the UPDATER (`updater/random_fill.hpp` — the same per-slab Gaussian
fill HMC samples momenta with) and handed down as plain fields, so the action
stays physics-only and the traversal stays in the action family. The colour
index is opaque to the updater — 2 for a site field, 2·ndim for a link field —
which is why one signature covers scalar, complex and gauge without
reintroducing a `(Site, mu)` arity split. It also lets an action opt OUT of the
checkerboard entirely: `action::WindowedAction` declares **one** colour and runs a
sequential pass, because its ΔS reads a global scalar (see [LLR](#llr-orchllr)).
The updater cannot tell, and the even-extent precondition — which belongs to the
checkerboard, not to Metropolis — is skipped when `n_colors == 1`.

The HMC integrator is a **type parameter**, not a runtime switch. Three
ship: `updater::integ::Omelyan2` (2nd-order minimum-norm, ~1.4× speedup at the
same acceptance — the **default**), `updater::integ::Leapfrog` (2nd-order,
cheapest per MD step), `updater::integ::Omelyan4` (4th-order, large τ). Select
one at the `Hmc` ctor with a brace-free tag value (`updater::integ::leapfrog`, CTAD
deduces the type) or the explicit `Hmc<Action, Rng, Omelyan2>` form. Adding one =
struct with a static `run(...)`; the matching `inline constexpr` tag is a
one-liner.

HMC keeps its momentum, force, and rollback buffers as sibling lattices
of the field — same shape, three lattices.

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
auto const cf    = app::common_flags(p, {.L = 4, .out = "u1_hmc.h5"});  // every app
auto const& beta = p.opt<double>("beta", 1.0, "Wilson coupling");       // physics: app's own
auto const& ndim = p.opt<int>("ndim", 4, "spatial dimensions");         // app's own
auto const rf    = app::hmc_run_flags(p, {.n_prod = 2000});             // the HMC family
if (!p.parse(argc, argv)) return 0;
io::Writer out = app::open_writer(p, cf, argc, argv);                   // log::start + Writer
...
long long const start_i = app::resume_or_start(rf, field, hmc, shape);  // 0, or the resume point
```

Two tiers, and a family helper is **all-or-nothing**:

- `common_flags` + `open_writer` — the block *every* app shares regardless of
  algorithm (`L,size`, `seed`, `workspace`, `out`, then the workspace log +
  Writer).
- one helper per algorithm family for the run-control block that family shares
  verbatim: `hmc_run_flags` + `resume_or_start` (`tau`, `n_md`, `n_therm`,
  `n_prod`, `meas_every`, `checkpoint_every`, `resume`) for the 16 HMC apps,
  `llr_run_flags` for the 10 LLR apps. Either every app in the family uses it or
  it does not exist — HMC ran 18 apps without one, which is how `--resume` came
  to be implemented in exactly one of them.

Only the *defaults* are per-app, because they legitimately differ; the flag
names, types and help text are single-sourced. This is setup only — the app
still builds its own lattice / rng / action / updater and **owns its trajectory
`for` loop**; there is no generic driver.

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

- `obs::` — per-configuration reductions: `reduce` / `reduce_nn` fold one pass
  over the lattice through any number of `obs::kernel::*` kernels (`phi`,
  `phi_sq`, `phi_quartic` — `RETICOLO_HD`, shared with `cuda/obs_reduce.cuh`),
  and the finalizers `mean_of`, `sq_of_mean_of`, `mag_abs_of` turn a raw lane
  sum plus `nsites` into the reported quantity. `two_point` is separate. All
  return `double`.
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
prints the banner. With `replicas = true` each scoped line also carries its
run id in a dedicated column — everything still lands in the one main log,
there are no separate per-replica files.

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

`orch::span::Orchestrator<Action, Rng, Sampler>` is two-phase exactly like the
LLR one: `setup(out)` plans threads, **builds the worker grid**, and opens the
`/worker_NNN/…` series; `run(Schedule)` runs the workers concurrently and records
their time series. The worker is `orch::span::Chain<Action, Rng, Sampler>` — any
scalar/gauge action, HMC or local Metropolis.

The one thing the orchestrator cannot synthesise is the **grid**. An LLR ladder
is always `E_n = E_min + n·dE`, so three numbers describe it; a parameter span is
arbitrary — log-spaced, hand-picked, multi-dimensional — and choosing it is the
app's reason to exist. So `Spec::points` *is* the grid, one action per worker,
and everything else about worker construction (ids, per-worker seeds, the thread
plan) is derived rather than re-written per app. `span::scan(base, &Action::knob,
lo, hi, n)` covers the common one-knob linear case in a line; anything else is a
loop that fills `points`, and there is no second mechanism.

```cpp
using Span = orch::span::Orchestrator<act::Phi4<double>, FastRng, updater::HmcSampler<>>;
Span runner{Span::Spec{.shape   = shape,
                       .seed    = seed,
                       .points  = orch::span::scan(base, &Action::kappa, k_min, k_max, n),
                       .sampler = {.tau = tau, .n_md = n_md},
                       .worker_threads = worker_threads},
            std::move(obs)};
```

`apps/param_span_hmc` sweeps `kappa` in one binary — an in-process replacement
for an outer bash sweep.

### LLR (`orch::llr`)

LLR reconstructs the density of states ρ(S) by running N replicas, each pinned by
a Gaussian-window penalty to a different region of action space, each sampled by
its own updater. A Newton-Raphson + Robbins-Monro loop adapts a per-replica
reweighting parameter `a`; periodic even/odd replica exchange improves mixing.
`orch::llr::Replica` wraps a `Base` action in `action::WindowedAction` and is an
`orch::Worker`; `orch::llr::Orchestrator` is the client of the spine — a two-phase
object where `setup(out)` plans threads, builds the replica ladder (a uniform E_n
sweep the orchestrator synthesises from `Spec`), resumes, and opens the series,
and `run()` / `run_smoothed(SmoothConfig)` drive `parallel_workers` and add
exchange as a serial coupling between waves. App-specific field initialisation
(cold-to-identity for gauge, hot-start for U(1)/complex) happens between `setup()`
and `run()` via `replicas()`, guarded by `resuming()`.
`WindowedAction` itself is a general **decorator action** (in `action/`, not
LLR): `S_win = S_base + a·Q + (Q−E_n)²/2δ²` where the constrained observable `Q`
is a `Constraint` policy — `SelfConstraint` (Q = the action, real LLR),
`ImagConstraint` (Q = `s_imag`, sign-problem LLR), or `ObservableConstraint<Obs>`
for any custom observable (window on magnetization, plaquette, topological
charge…). The self/imag defaults reproduce the previous hardcoded modes exactly.

**One orchestrator; the sampler is a template parameter.** There is a single
`Replica` and a single `Orchestrator`, and the sampler occupies the slot the MD
integrator used to hold — because the integrator belongs to the HMC sampler,
where it means something, and a Metropolis-driven run has none:

```cpp
orch::llr::Orchestrator<Action, FastRng, updater::HmcSampler<>>       // HMC, Omelyan2
orch::llr::Orchestrator<Action, FastRng, updater::MetropolisSampler>  // local sweeps
orch::llr::Orchestrator<Action, FastRng, updater::HmcSampler<updater::integ::Omelyan4>>
orch::llr::Orchestrator<Action, FastRng, updater::HmcSampler<>, Constraint>  // window on Q
```

The sampler has **no default** — which updater drives a run is a decision worth
reading at the call site. The integrator does keep one (`HmcSampler<>` is
Omelyan2), since it is a knob *inside* an already-named sampler.

Gauge LLR takes the **same** shape as scalar —
`Orchestrator<Wilson<SU3>, FastRng, updater::HmcSampler<>>`.
The full parameter list is `<Base, Rng, Sampler, Constraint>`, and that is all of
it: the element and field types are **not** parameters, because the action
already names both (`value_type` and `field_type` — every family base declares
the latter). A parameter for either could only ever be set to what the action
says, or to something wrong.

**The two worker kinds share this structure**, because they are the same kind of
thing — an `orch::Worker` wrapping one action and one updater:

```cpp
orch::span::Chain  <Action, Rng, Sampler>
orch::llr::Replica <Action, Rng, Sampler, Constraint = void>
```

Neither takes an integrator parameter: it rides on the HMC tag, so it does not
exist when the sampler is local. Both build the updater through the identical
`make_sampler_` (tag's optional `integrator` alias → does the ctor take one).
`span::Chain` reports `acceptance()` always and dH/accepted only when the sampler
has them (`Chain::k_hmc_stats`), so a span records `/worker_NNN/stats/dH` +
`/stats/accepted` under HMC and `/stats/acceptance` under Metropolis — the same
split the standalone apps make.

The sampler parameter is a **tag** (`updater/samplers.hpp`), not the updater
type: the updater is parameterised on the `WindowedAction` the replica builds
internally, which a caller has no business spelling, so the tag names it via
`Sampler::template type<Action, Rng, Field, T>`. `Sampler::spec_type` gives the
ctor arguments, and an optional `Sampler::integrator` alias is how `Replica`
decides whether that ctor takes an integrator.

Nothing in the LLR layer reaches past `updater::Updater`, so the ladder, the
warm-up ramp, NR/RM, exchange and checkpointing are all shared — which is what
keying that concept on `acceptance()` rather than on an `accepted` flag bought.
Only `Spec` carries both knob sets (`tau`/`n_md` vs `sigma`). Counts are **not**
comparable across the two: a trajectory is `n_md` lattice passes, a sweep is one.

**A windowed sweep is sequential, not checkerboarded.** `a·Q` is linear in Q, so
its per-site increment `a·ΔQ` is purely local; the Gaussian penalty is quadratic
in a **global** scalar, so a single-site move contributes
`[2(Q−E_n)ΔQ + ΔQ²]/2δ²`, which reads the *current* Q. Accepting a move at one
site therefore changes ΔS at every other site of the same colour — exactly the
independence the checkerboard rests on. Freezing Q across a colour would restore
the parallelism and break detailed balance, so `WindowedAction::metropolis_stencil`
declares `n_colors() = 1` and runs one lexicographic pass carrying Q as a running
scalar (`exec::nn_sequential_reduce`). The site-level threading this gives up is
threading LLR never had: a replica runs its lattice passes serially inside the
replica-parallel team. It also drops the even-extent requirement, which belonged
to the checkerboard rather than to the algorithm.

Coverage today: a real nearest-neighbour base with `SelfConstraint` or an
`ObservableConstraint<Obs>` over another NN observable. Gauge bases and
`ImagConstraint` (Q = S_I) have no local windowed form yet and stay on HMC — the
`requires` guard on `metropolis_stencil` makes that a failed `action::LocalAction`
at the instantiation site rather than a link error or, worse, a silently wrong
measure.

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

- `core/sys/hd.hpp` — the `RETICOLO_HD` macro (`__host__ __device__` under nvcc,
  expands to nothing otherwise), so one annotated function compiles for both.
- `core/rng/philox.hpp` — the shared counter-based Philox, a pure `RETICOLO_HD`
  bijection so it compiles identically on host and device.

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
3. `cuda/actions/nn/phi4.hpp` — the `device_functors<action::Phi4<T>>` trait adapts
   a *host* action struct into device launchers (force / s_full / sample_momenta).
4. `cuda::DeviceAction<HostAction, Field>` + `cuda::Hmc<DAct, Integ>` — the
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
(`phi4_hmc`, `phi6_hmc`, `sine_gordon_hmc`, `xy_hmc`, `bose_gas_hmc`, `u1_hmc`,
`su2_hmc`, `su3_hmc`), one Metropolis sim per action (`*_metropolis`, same
list), the LLR sims (`phi4_llr`, `phi4_llr_metropolis`, `phi6_llr`,
`sine_gordon_llr`, `xy_llr`, `bose_gas_llr`, `u1_llr`, `u1_llr_smoothed`,
`su2_llr`, `su3_llr`), `f32` variants of the HMC and Metropolis sims, and the
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

- `tests/unit/` — types in isolation: `Site`, lattice geometry, `Lattice`,
  `FastRng`, `Parser`, observers, analysis.
- `tests/physics/` — force-vs-FD consistency for every action,
  HMC reversibility & integrator-order across the gauge groups.
- `tests/io/` — Writer round-trips, metadata stamping, phase collision
  rejection, `/vars` stamping.
- `tests/apps/` — every reference app smoke-tested end-to-end: run the
  binary on tiny inputs, open the HDF5, assert the schema.

CI matrix: `macos-appleclang`, `macos-llvm`, `linux-gcc`, `linux-clang`,
plus `clang-format` (pinned 22.1.5) and `clang-tidy` (pinned 22, via the amalgamation
pattern in `src/lint/`).
