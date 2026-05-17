# Reticolo v3 — fresh-start plan

**Scope:** lightweight, serial-only, scalar-field-only library for generic
lattice quantum field theory MCMC. No gauge theory. No parallelism (no
OpenMP, no MPI, no threading in the hot loop). Modern C++23. Performance
at the core. The library hands the user a small, composable set of types
and gets out of the way.

This plan is informed by two independent audits — of the current branch
(`refactor/disciplined-v2`, 401 commits of accumulated friction) and of
`main` (2-commit "v0.2.0 full rewrite") — but is not a port of either.
Patterns that both audits converge on are kept. Everything else is
designed from scratch against the constraints above.

---

## 1. What both audits agreed on

**Carry forward (the design wins):**

1. Action as a plain struct constrained by a small **refinement lattice
   of concepts** (`LocalAction` baseline + `HasForce` / `HasProposal` /
   `HasSEff` / `WolffEmbeddable` opt-ins). Zero virtuals. Action methods
   inline into the templated updater. This is the single decision both
   trees got right.
2. **Hand-written `apps/*.cpp`** as the public ergonomic. ~30–50 LOC each.
   The for-loop is visible. No closure drivers, no string-dispatched
   action selection, no YAML schema between user and types.
3. **One HDF5 sink + typed `Series<T>` handles.** Schema-by-convention.
   No backend abstraction, no per-action `*Hdf5.hpp` specialisation.
4. **Shape-pooled `Indexing`** (`shared_ptr<const NeighbourTable>` keyed
   by shape). Bench-validated 2/3 memory reduction in HMC at L=32 with
   no hot-loop regression. The inline-bitmask variant tried on the
   current branch lost 22–47% on `force` throughput — keep the table.
5. **Value-semantic `Lattice<T>`** (composition, not `: std::vector`).
6. The **physics-correctness test surface** from the current branch:
   force-vs-∂S/∂φ, HMC reversibility, integrator order p, free-field
   analytic limits, detailed balance, Wolff sanity, indexing-pool
   ref-count. This is the most valuable artefact in the repo.

**Never again (the design failures):**

1. **N×K hand-maintained registration tables.** The current branch has
   four parallel registries (`runtime.hpp` 16-branch switch,
   `k_builtin_actions`, `Hdf5TypeOf<>` per impl, YAML adapter table) that
   the compiler does not cross-check. Adding an action means editing
   4–6 unrelated files. The compiler must be the source of truth.
2. **Two contradictory app surfaces shipped half-built.** Current branch
   has both hand-written apps (no HDF5 output) and a YAML-driven
   `reticolo_run` dispatcher (no exploratory ergonomics). Pick one.
3. **Pure header-only when libhdf5 is involved.** `main`'s 675-line
   `hdf5_writer.hpp` lands in every TU; cold full build is ~3–4 min for
   17 apps. HDF5 is C, drags global init state, deserves one TU.
4. **Preprocessor inside a concept body.** `main`'s `HmcAction` flipping
   `HasForce` vs `HasForceInParallel` under `#ifdef RETICOLO_HAVE_OPENMP`
   is an ODR landmine. Concepts must not be macro-conditional.
5. **Dead subsystems left in the tree.** Current branch has unmigrated
   LLR code and empty `examples/01_throughput/` skeletons that look
   functional from a directory listing. Either build it or delete it.
6. **Schema-defining compile flags that are not stamped in output.**
   `RETICOLO_HDF5_LEGACY_COMPLEX_COMPOUND` changes the on-disk layout
   of complex datasets but leaves no trace in the file. Silent data
   divergence class. Every schema-relevant flag must land in `/run@`.

---

## 2. Constraints (non-negotiable)

- **Serial only.** No OpenMP, no `#pragma omp`, no thread pool, no
  `<thread>` in the hot path. `<mutex>` only in the IO layer (for
  defensive RAII flush ordering), nowhere else.
- **Scalar fields only.** No `GaugeField`, no Lie group machinery, no
  `Wilson*` actions, no heatbath, no overrelaxation, no parallel
  transport, no Eigen dependency.
- **C++23**, single tool-enforced style (`.clang-format`, `.clang-tidy`,
  `.clangd`, `.editorconfig` at repo root; CI runs both in `--Werror`
  mode).
- **Concepts over CRTP over virtual.** No virtual member function
  anywhere in the simulation hot path. The only acceptable virtual is
  internal to the IO layer if it simplifies error reporting.
- **Apps own the for-loop.** Library does not ship range-for helpers
  over phases, no `run(closure)`, no driver classes. The for-loop is
  the user's surface.
- **No central registries.** Adding an action = writing one header and
  one app. The compiler discovers everything by template instantiation
  at the app's call sites.
- **No two ways to do the same thing.** One `Series<T>`, one writer,
  one CLI helper, one logger.

---

## 3. Library shape

### 3.1 Targets

```
reticolo::core         INTERFACE   header-only (pure templates)
reticolo::io           STATIC      one .cpp, owns libhdf5
reticolo::reticolo     INTERFACE   aggregates the two above
```

Header-only buys easy embedding (`FetchContent_MakeAvailable(reticolo)`
+ `target_link_libraries(my_sim PRIVATE reticolo::reticolo)`). The
static `reticolo::io` keeps libhdf5 out of the user's TUs and removes
the global libhdf5-init coupling.

### 3.2 Directory layout

```
reticolo/
├── CMakeLists.txt
├── CMakePresets.json
├── README.md
├── .clang-format / .clang-tidy / .clangd / .editorconfig
├── include/reticolo/
│   ├── reticolo.hpp                # umbrella, no <hdf5.h>
│   ├── core/
│   │   ├── lattice.hpp             # Lattice<T>, const-correct
│   │   ├── indexing.hpp            # NeighbourTable + shape pool
│   │   ├── site.hpp                # Site, Parity, Shift
│   │   ├── bc.hpp                  # per-direction BC mask
│   │   ├── rng.hpp                 # Rng concept + FastRng (xoshiro256++)
│   │   └── log.hpp                 # tiny logger, sections, ostream-only
│   ├── action/
│   │   ├── concepts.hpp            # LocalAction + refinements
│   │   └── builtins/
│   │       ├── phi4.hpp
│   │       ├── phi6.hpp
│   │       ├── xy.hpp
│   │       ├── on_sigma.hpp
│   │       ├── sine_gordon.hpp
│   │       └── bose_gas.hpp
│   ├── algorithm/
│   │   ├── metropolis.hpp
│   │   ├── hmc.hpp                 # template<HasForce A, Integrator I>
│   │   ├── integrators.hpp         # Leapfrog, Omelyan PQP / PQPQP
│   │   └── wolff.hpp               # WolffEmbeddable only
│   ├── obs/
│   │   ├── concepts.hpp            # Observer concept
│   │   └── catalog.hpp             # mean, sq, magnetization, two_point, ...
│   └── io/
│       └── writer.hpp              # thin facade; no <hdf5.h>
├── src/io/
│   ├── writer.cpp                  # single TU touching libhdf5
│   └── hdf5_types.cpp
├── apps/
│   ├── phi4_hmc.cpp
│   ├── phi4_metropolis.cpp
│   ├── xy_wolff.cpp
│   ├── on_sigma_wolff.cpp
│   ├── sine_gordon_hmc.cpp
│   └── bose_gas_hmc.cpp
├── examples/
│   ├── phi4_tuning/                # README + driver app + py analysis
│   └── on_sigma_critical/
├── tests/
│   ├── physics/                    # the crown jewels — re-port from current branch
│   ├── unit/
│   └── io/
├── cmake/                          # ReticoloDeps, ReticoloWarnings, presets toolchains
└── docs/
    ├── architecture.md
    ├── writing_an_app.md
    └── action_concepts.md
```

### 3.3 Public umbrella header

`include/reticolo/reticolo.hpp` includes core, action/concepts, every
builtin action, every algorithm, observer catalog, and the **thin**
`io/writer.hpp` (no `<hdf5.h>`). One include per app. No transitive
`<hdf5.h>`, no transitive `<Eigen/Dense>` (we have no Eigen), no
transitive `<format>` outside the IO TU.

---

## 4. Concept lattice (the heart of the design)

```cpp
namespace reticolo::action {

// Baseline: any action knows its local change.
template <class A, class F>
concept LocalAction = requires(A const& a, Lattice<F> const& L, Site x, F nv) {
    { a.s_local (L, x)     } -> std::convertible_to<double>;
    { a.ds_local(L, x, nv) } -> std::convertible_to<double>;
};

// Refinement: full action (used by HMC dH and by acceptance diagnostics).
template <class A, class F>
concept HasSEff = LocalAction<A, F>
    && requires(A const& a, Lattice<F> const& L) {
    { a.s_full(L) } -> std::convertible_to<double>;
};

// Refinement: action can compute its molecular-dynamics force.
template <class A, class F>
concept HasForce = LocalAction<A, F>
    && requires(A const& a, Lattice<F> const& L, Lattice<F>& force) {
    { a.compute_force(L, force) };
};

// Refinement: action provides its own Metropolis proposal kernel.
template <class A, class F, class R>
concept HasProposal = LocalAction<A, F>
    && requires(A const& a, Lattice<F> const& L, Site x, R& rng) {
    { a.propose(L, x, rng) } -> std::convertible_to<F>;
};

// Refinement: action admits a Wolff cluster embedding.
template <class A, class F>
concept WolffEmbeddable = LocalAction<A, F>
    && requires(A const& a) {
    typename A::axis_t;
    { a.wolff_random_axis(std::declval<auto&>()) } -> std::same_as<typename A::axis_t>;
    { a.wolff_reflect (std::declval<F>(),  std::declval<typename A::axis_t>()) } -> std::same_as<F>;
    { a.wolff_link_p  (std::declval<F>(),  std::declval<F>(),
                       std::declval<typename A::axis_t>(), std::declval<double>()) } -> std::convertible_to<double>;
};

}  // namespace reticolo::action
```

Notes:

- All concepts take `Lattice<F> const&`. This fixes the const-correctness
  liability that forced `LocalAction` on the current branch to take a
  mutable reference. Const at the lowest layer unlocks observers,
  diagnostics, cross-replica reads without launder casts.
- No `#ifdef` inside any concept body. There is one kernel per
  capability; we are serial. If parallelism returns later, it lands in
  a separate concept (`HasForceInParallel`) and a separate `HMC`
  specialisation, not toggled by a macro.
- Adding an action is one struct definition. Wolff-supporting actions
  satisfy `WolffEmbeddable` automatically by exposing the four members.
  No registration table touches.

---

## 5. Hot-loop architecture

### 5.1 Lattice and Indexing

- `Lattice<T>` is **directly constructible** — `Lattice<double> phi({L,L,L,L});`
  — not built through a `make<>` dispatcher. The main-branch `make<>` template
  added a layer with no payoff; the only thing it bought was `describe(Section&, T)`
  logging, which moves to a free `log::dump(phi)` overload that users call
  explicitly when they want it.
- `Lattice<T>` holds a contiguous `std::vector<T>` (or a 64-byte-aligned
  custom buffer if benches show it matters) plus a
  `shared_ptr<Indexing const>` pulled from a shape-keyed pool.
- Iteration: `phi.sites()`, `phi.even_sites()`, `phi.odd_sites()` are
  cheap `views::iota` adapters over the contiguous index space. Wolff,
  parity-sweep Metropolis, observers, and tests all use them.
- `Indexing` owns `vector<size_t> next, prev` of size `nsites * ndims`,
  plus the bulk/skin partition for non-fully-periodic shapes. All
  members are `const`-qualified.
- Pool: `static unordered_map<ShapeKey, weak_ptr<Indexing const>>`
  guarded by a `std::mutex` only during `acquire/release` (never on the
  hot path). HMC's three intermediate lattices (`mom`, `force`,
  `old_field`) share the same `Indexing` as the field — three lattices,
  one neighbour table.
- Inner loop uses one memory read per neighbour: `next[site*ndims + mu]`.
  Branchless. The 4477d75 inline-bitmask experiment is not re-attempted
  — current branch perf data settles that question.

### 5.2 Bulk / skin split

The bulk/skin pattern from the current branch is correct but is moved
**out of action bodies** and **into the lattice iteration helper**:

```cpp
lattice::for_each_bulk(L, [&](Site x){ /* BC-blind kernel */ });
lattice::for_each_skin(L, [&](Site x){ /* BC-aware kernel  */ });
```

Action authors write one kernel per call site that uses
`L.next(x, mu)` (which, for the bulk, is exactly `next[…]` with no
branch; for the skin, applies BC). For all-periodic shapes, the helper
collapses to a single contiguous sweep with no branch, byte-identical
to the no-BC fast path.

### 5.3 HMC

```cpp
template <action::HasForce<F> A, class Integrator = integ::Leapfrog>
class Hmc {
    A const&            action_;
    Lattice<F>&         field_;
    Rng&                rng_;
    Lattice<F>          mom_;        // value-typed, set up once
    Lattice<F>          force_;
    Lattice<F>          old_field_;
    Integrator          integ_;      // type-parameter; no runtime switch
public:
    Hmc(A const&, Lattice<F>&, Rng&, HmcSpec);
    HmcStep trajectory();            // one full HMC update, returns dH, accepted
};
```

Wins over current branch:

- No `switch (integrator_)` in the trajectory loop. `Integrator::step`
  is fully inlined; the optimiser sees through `kick / drift`.
- No `unique_ptr<Lattice>` triple — value-typed members, default
  constructed in `setup()` to match the field's shape.
- `static_assert(HasForce<A,F>)` at the class-template level; misuse is
  a compile error, not a runtime throw.
- No allocations per trajectory.

### 5.4 Wolff

`template <action::WolffEmbeddable<F> A> class Wolff` constrained at
declaration; non-periodic BCs fail to satisfy a private
`AllPeriodicAction` concept and produce a compile error, not the
runtime throw the current branch emits.

### 5.5 Metropolis

```cpp
template <action::LocalAction<F> A, class R>
class Metropolis { ... };
```

Two specialisations: one for `HasProposal<A,F,R>` (action-provided
proposal kernel — e.g. XY angle increment), one fallback for `+δ` Gaussian
proposal on `LocalAction`. Both selected at instantiation by `if constexpr`.

---

## 6. IO

### 6.1 `io::Writer`

```cpp
namespace reticolo::io {

class Writer {
public:
    explicit Writer(std::filesystem::path const&, RunMetadata const& = {});
    ~Writer();

    template <class T>
    Series<T> series(std::string_view path, std::size_t chunk = 4096);

    // Single-value attribute write, e.g. for /vars@kappa.
    template <class T>
    void attr(std::string_view path, T value);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;     // PIMPL — keeps <hdf5.h> in writer.cpp
};

}  // namespace reticolo::io
```

- `Writer` opens with truncate-or-fail, **stamps on construction**:
  - `/run@cmdline` (full argv joined)
  - `/run@version`, `/run@commit`, `/run@build_type`
  - `/run@compile_flags`
  - `/run@hostname`, `/run@started_utc`
  - `/run@hdf5_complex_schema`  ← fixes the silent-schema-mode bug
  - `/run@hdf5_library_version`
- `Series<T>` is the only way to write a time series. Owns a
  `std::vector<T>` buffer of `chunk` rows, flushes on full / dtor.
  Returned **by value** (not by reference into a map) — fixes the
  rehash-invalidation hazard.
- One `std::mutex` inside `Impl` guards every HDF5 call; serial
  consumers pay one uncontended acquire per `append`, which is below
  the noise floor.
- No path-component walker bug class: the writer creates intermediate
  groups eagerly via a small `make_path` helper that does not query
  existence via libhdf5's error stack.

### 6.2 Schema

```
/run@<metadata...>
/vars@<name>                       # every parameter the app resolved
/<phase>/stats/{step, s, dH, accept, ...}   # algorithm-intrinsic
/<phase>/obs/{step, <name>, ...}            # observer outputs
/<phase>/field/<step>                       # optional, opt-in
```

`<phase>` is a free-form string the app chooses (`therm`, `prod`, etc).
The writer refuses to open a phase whose root group already exists —
catches typo'd phase reuse early, copies the current-branch behaviour
that worked.

### 6.3 No backend abstraction, no registry

There is one `Writer` class, one HDF5 backend, no `StorageBackend`
concept, no `Hdf5TypeOf<>` specialisation per action. `hdf5_types.cpp`
maps C++ scalars + `std::complex<T>` to native HDF5 types via a small
trait table, and that is the entire breadth of the dispatch.

---

## 7. CLI / configuration

**No YAML schema, no `reticolo_run`, no `--set`, no TOML.** Every app
parses its own CLI flags. To avoid per-app boilerplate divergence we
ship one tiny header:

```cpp
#include <reticolo/cli.hpp>

int main(int argc, char** argv) {
    cli::Parser p{"phi4_hmc", "HMC for phi^4"};
    auto L      = p.req<int>   ("L",      "linear size");
    auto kappa  = p.req<double>("kappa",  "hopping");
    auto lambda = p.req<double>("lambda", "quartic coupling");
    auto seed   = p.opt<uint64_t>("seed", 42);
    auto out    = p.opt<std::string>("out", "phi4.h5");
    p.parse(argc, argv);
    ...
}
```

`cli::Parser` is ~150 lines wrapping `cxxopts` (existing dep on both
branches). It records every resolved value into a small global the
`Writer` reads when stamping `/vars@*`, so reproducibility is automatic
without the app touching the writer for each parameter.

If a future app needs nothing fancy, it can bypass `cli::Parser` and
call `Writer::attr` directly per parameter. The convention is opt-in,
not enforced.

---

## 8. Apps (the public ergonomic)

Target: every app is ≤ 60 LOC including HDF5 output. Concrete shape:

```cpp
#include <reticolo/reticolo.hpp>
using namespace reticolo;

int main(int argc, char** argv) {
    cli::Parser p{"phi4_hmc"};
    auto L       = p.req<int>("L");
    auto kappa   = p.req<double>("kappa");
    auto lambda  = p.req<double>("lambda");
    auto tau     = p.opt<double>("tau", 1.0);
    auto n_md    = p.opt<int>   ("n_md", 20);
    auto n_therm = p.opt<int>   ("n_therm", 1000);
    auto n_prod  = p.opt<int>   ("n_prod",  10000);
    auto seed    = p.opt<uint64_t>("seed", 42);
    auto outpath = p.opt<std::string>("out", "phi4.h5");
    p.parse(argc, argv);

    Lattice<double> phi({L, L, L, L});
    FastRng         rng{seed};
    act::Phi4<double> S{.kappa = kappa, .lambda = lambda};

    io::Writer out{outpath};
    auto s_therm = out.series<double>("therm/stats/s");
    auto dH      = out.series<double>("prod/stats/dH");
    auto s_prod  = out.series<double>("prod/obs/s");
    auto mag     = out.series<double>("prod/obs/mag");

    alg::Hmc hmc{S, phi, rng, {.tau = tau, .n_md = n_md}};

    for (int i = 0; i < n_therm; ++i) {
        hmc.trajectory();
        s_therm.append(S.s_full(phi));
    }
    for (int i = 0; i < n_prod; ++i) {
        auto step = hmc.trajectory();
        dH    .append(step.dH);
        s_prod.append(S.s_full(phi));
        mag   .append(obs::magnetization(phi));
    }
}
```

Six apps ship in `apps/`:

| App                   | Action       | Updater         |
| --------------------- | ------------ | --------------- |
| `phi4_hmc`            | `act::Phi4`  | `alg::Hmc`      |
| `phi4_metropolis`     | `act::Phi4`  | `alg::Metropolis` |
| `xy_wolff`            | `act::XY`    | `alg::Wolff`    |
| `on_sigma_wolff`      | `act::OnSigma<N>` | `alg::Wolff` |
| `sine_gordon_hmc`     | `act::SineGordon` | `alg::Hmc` |
| `bose_gas_hmc`        | `act::BoseGas` | `alg::Hmc` (complex `s_eff`) |

That is the entire app surface. Anything else a user wants, they write
themselves by copying one of these — that is the design, not an
oversight.

---

## 9. Tests

Re-port the physics suite from the current branch verbatim in intent
(rewrite for the new API):

- `tests/physics/test_<action>_force_consistency.cpp` — for every
  `HasForce` action, finite-difference `∂S/∂φ_x` vs `compute_force`.
- `tests/physics/test_hmc_reversibility.cpp` — reverse-momentum + reverse-
  integrate returns the field to start within `1e-10`.
- `tests/physics/test_hmc_integrator_order.cpp` — `|dH|` scaling vs `dt`
  matches integrator's `p` (Leapfrog p=2, Omelyan p=2 with smaller
  prefactor, etc.).
- `tests/physics/test_<action>_free_limit.cpp` — for actions with a free
  Gaussian limit (Phi4 at λ=0, Phi6 at λ₆=0, SineGordon at low β),
  measured ⟨φ²⟩ matches the lattice propagator to bench tolerance.
- `tests/physics/test_<action>_high_temperature.cpp` — XY / OnSigma
  high-T expansion checks.
- `tests/physics/test_detailed_balance.cpp` — small lattice +
  test-action enumeration.
- `tests/physics/test_wolff_sanity.cpp` — XY / Ising-like Wolff cluster
  size distribution.

Plus the structural tests:

- `tests/unit/test_indexing_pool.cpp` — two `Lattice` of the same shape
  share one `Indexing`; different shapes do not.
- `tests/unit/test_lattice.cpp` — value semantics, iteration order,
  bulk/skin partition.
- `tests/unit/test_rng.cpp` — FastRng determinism + range stats.
- `tests/io/test_writer_roundtrip.cpp` — `Series<T>::append` then
  reopen-and-read.
- `tests/io/test_writer_metadata.cpp` — `/run@*` stamping, including
  `hdf5_complex_schema`.
- `tests/io/test_writer_phase_collision.cpp` — re-using a phase prefix
  throws at start, not mid-run.

CI matrix at minimum: `macos-appleclang` (no OpenMP — fine, we're
serial), `macos-llvm`, `linux-gcc`, `linux-clang`. All four use the
**same** `RELEASE+ASAN+UBSAN` test config — current branch's CI gap
(`macos-llvm` and `linux-clang` not covered) is closed at v3.

---

## 10. Performance budget (single thread)

Targets (from current-branch bench data on M1 Pro Release):

| Kernel              | Target (Msites/s) | Notes                            |
| ------------------- | ----------------- | -------------------------------- |
| `Phi4::force`       | ≥ 300             | current branch achieves 158–338  |
| `Phi4::s_full`      | ≥ 400             | current branch achieves 405      |
| `Lattice` deep copy | ≥ 50 GB/s         | ≈ memory bandwidth               |
| HMC trajectory at L=32, n_md=20 | ≤ 1.2× the bare 20×(force + leapfrog kick/drift) | measures dispatch overhead |

Rules that keep us on budget:

- No virtual dispatch in any updater hot path. Static asserts in CI.
- HMC integrator is a type parameter; no runtime `switch` in
  `trajectory()`.
- Action methods are non-virtual and templated on `Lattice<F>` by
  reference — the optimiser inlines `s_local` / `ds_local` /
  `compute_force` into the updater.
- Zero heap allocation per trajectory after `setup()`.
- `Indexing` is shared across the three HMC lattices of equal shape.
- `<hdf5.h>` enters exactly one TU; user TUs compile in ~1 s.

---

## 11. Build, tooling, CI

- CMake 3.25, presets for the four supported platforms above.
- C++23, `-O3 -march=native` by default (`RETICOLO_TUNE_NATIVE=ON`).
- Single dependency floor: HDF5 1.14, cxxopts (vendored or fetched).
  **No Eigen.** **No yaml-cpp.** **No OpenMP.**
- `.clang-format`, `.clang-tidy` (`readability-identifier-naming` set
  once), `.clangd` (with the `-fopenmp=libomp` shim removed — we have
  no OpenMP), `.editorconfig`. CI runs `clang-format --dry-run --Werror`
  and `clang-tidy` in `--warnings-as-errors` mode.
- CI matrix runs **all four** presets and the **full** ctest set,
  including the physics suite, under `-fsanitize=address,undefined` in
  one matrix leg.

---

## 12. Repo hygiene from day one

- No empty `examples/` skeletons. If a directory exists, it contains
  a `README.md`, a `CMakeLists.txt`, and a buildable `src/main.cpp`.
- No `archive/` subtree. Old code lives on the dead `refactor/disciplined-v2`
  and `main` branches, not in the rewrite's tree.
- No `proposals/` doc shipped without a referenced issue or milestone.
- No CLAUDE.md scattered through subdirectories. One repo-root
  `CLAUDE.md` that points at `docs/architecture.md`.
- Every schema-defining `RETICOLO_*` compile flag has a matching
  `/run@<flag_name>` stamp in `Writer` construction; this is a CI
  assertion, not a convention.

---

## 13. Implementation milestones

Each milestone is one branch + one PR + one passing CI run. No
multi-milestone branches.

| # | Milestone               | Deliverable                                                                                |
| - | ----------------------- | ------------------------------------------------------------------------------------------ |
| 0 | Scaffold                | `CMakeLists.txt`, presets, `.clang-format`/`.clang-tidy`/`.clangd`, empty CI green.        |
| 1 | Core lattice            | `Site`, `Parity`, `Lattice<T>`, `Indexing` + pool, `bc.hpp`, unit tests.                    |
| 2 | RNG + log               | `FastRng`, `Rng` concept, `log::Logger` + `Section`, unit tests.                            |
| 3 | Concepts + Phi4         | `action/concepts.hpp`, `act::Phi4`, force-consistency + free-limit tests pass.              |
| 4 | Updaters                | `alg::Metropolis`, `alg::Hmc<A, Integrator>`, `integrators.hpp`, reversibility + order tests. |
| 5 | IO                      | `io::Writer`, `Series<T>`, `/run@` stamping, schema-mode assertion, round-trip tests.       |
| 6 | Observer catalog        | `obs::mean`, `obs::sq`, `obs::magnetization`, `obs::two_point`, `cli::Parser`.              |
| 7 | First app               | `apps/phi4_hmc.cpp` end-to-end with HDF5 output ≤ 60 LOC.                                   |
| 8 | Action zoo              | Phi6, XY, OnSigma, SineGordon, BoseGas, each with its physics test.                         |
| 9 | Wolff                   | `alg::Wolff`, XY + OnSigma apps, Wolff-sanity test.                                         |
| 10 | Apps + examples        | Five remaining apps, `examples/phi4_tuning`, `examples/on_sigma_critical`.                  |
| 11 | Docs                   | `docs/architecture.md`, `docs/writing_an_app.md`, `docs/action_concepts.md`, README.        |

Stop conditions at each milestone: CI green on all four presets;
clang-tidy clean; ASAN+UBSAN clean; performance bench within budget.

---

## 14. Out of scope (explicit)

To prevent scope creep mid-rewrite, the following are **deferred** and
must not influence v3 design decisions:

- Parallelism (OpenMP, MPI, GPU).
- Gauge theories (U(1), SU(2), SU(3), Wilson, heatbath, overrelaxation).
- LLR / density-of-states algorithms.
- Checkpoint/restart from binary state. (Re-runnable from `argv` +
  seed; that is the v3 reproducibility story.)
- Multi-replica / parallel-tempering drivers.
- YAML / TOML / JSON configuration of runs.
- Plugin-style runtime action selection.

Any of these can be added later **on top of** v3 without breaking the
v3 surface — because v3 has no central registry or dispatcher for them
to collide with. That is the architectural payoff of the constraints.

---

## 15. Why this respects every constraint

| Constraint                          | How v3 satisfies it                                                                |
| ----------------------------------- | ---------------------------------------------------------------------------------- |
| Lightweight                          | Two CMake targets; three deps (HDF5, cxxopts, std); single-include umbrella.       |
| Easy to use                          | Six ≤ 60-LOC reference apps; one concept lattice the user reads end-to-end.        |
| Generic LQFT                         | Add an action by writing one struct; concepts pick up its capabilities.            |
| Serial only                          | No `<thread>`, no OpenMP anywhere. Concepts are macro-free.                        |
| No gauge theories                    | Entire `gauge_*` subsystem deleted; Eigen dep deleted; 17 → 6 apps.                |
| Clean modern C++                     | C++23, concepts > CRTP > virtual, one tool-enforced style identity.                |
| Performance at the core              | No virtuals in hot path; HMC integrator templated; shape-pooled Indexing; PIMPL'd IO. |

---

## 16. Decision log (so we don't relitigate)

1. **Header-only for core, static for IO.** Not pure header-only.
   Driver: 675-line `hdf5_writer.hpp` per-TU compile cost on `main`.
2. **No YAML, no `reticolo_run`.** Driver: current branch's two-app-
   surface incoherence. The cmdline + git SHA in `/run@` is the run
   record; apps are the configuration.
3. **HMC integrator is a type parameter.** Driver: per-trajectory
   `switch` plus virtual `kick/drift` on current branch.
4. **`LocalAction` takes `Lattice<F> const&`.** Driver: const-correctness
   liability on the current branch blocked observers from holding const refs.
5. **No central registry, ever.** Driver: four parallel uncross-checked
   registries on the current branch. The compiler is the source of truth.
6. **Schema-relevant compile flags stamp in `/run@`.** Driver: the silent
   `RETICOLO_HDF5_LEGACY_COMPLEX_COMPOUND` data-divergence bug.
7. **Bulk/skin lives in the lattice helper, not in action bodies.**
   Driver: keeps action code BC-agnostic; helps the all-periodic
   common case collapse to a single-loop fast path.
8. **No empty directories. No dead modules.** Driver: LLR and empty
   examples on current branch.
9. **No `make<>()` dispatcher; types are directly constructible.** Departure
   from main's locked-in v0.2.0 invariant. Driver: `make<>` added a template
   layer with no payoff beyond invoking `describe()`. Direct construction is
   shorter and clearer; logging moves to an explicit `log::dump(phi)` call.
10. **Flat top-level namespace, with `act = action` / `alg = algorithm` /
    `obs = observables` aliases inside `namespace reticolo`.** One
    `using namespace reticolo;` per app. Matches main's invariant; users
    write `Lattice<double>`, `FastRng`, `act::Phi4`, `alg::Hmc` directly.

---

*End of plan. Next action when the user gives the green light: branch
`rewrite/v3` off an empty tree, do M0 in one PR.*
