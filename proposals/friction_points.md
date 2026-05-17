# Friction points & latent issues — refactor/disciplined-v2

A read-through audit of the current branch. Ordered by severity, not by file.
The goal is to flag things that *will* bite later, not to prescribe fixes.

## A. Structural drift

### A1. LLR module is unmigrated dead code

`src/reticolo/modules/llr/*` is not part of the v2 tree; it is a copy from
`archive/pre-rewrite` that nothing builds and nothing can build.

- Includes paths that don't exist on this branch:
  `reticolo/modules/factory/ModuleBase.hpp`, `reticolo/tools/io_utils.hpp`,
  `reticolo/tools/logger.hpp`, `reticolo/tools/timer.hpp`,
  `reticolo/types/core.hpp`, `reticolo/types/core_math.hpp`.
  Current paths are `reticolo/core/tools/...` and `reticolo/core/types/...`.
- No `add_subdirectory(modules/llr)` or `add_executable` referencing it
  anywhere in CMake.
- `LLRController` inherits a removed `ModuleBase` and overrides
  `setup()` / `execute()` as `final = 0` — won't compile against any
  current header.

It still shows up in `examples/04_llr_density_of_states/` (empty skeleton)
and was listed in older memory as a v2 invariant. Risk: someone (you or me)
recommends LLR features as if they exist. Either port it or move it under
`archive/` so it stops being part of the apparent surface.

### A2. Example skeletons that aren't examples

`examples/{01_throughput, 03_hmc_tuning, 04_llr_density_of_states}/` each
contain only `build/` and `results/` directories — no source, no scripts,
no README. They are placeholders that look like working examples from a
directory listing. Either flesh them out or delete them; right now they
invite citation in plans and documentation that turns out to be aspirational.
`examples/02_autocorrelation/` is untracked, same hazard.

### A3. CI only covers two presets, not the OpenMP path

`.github/workflows/ci.yml` builds `linux-gcc` and `macos-appleclang`.
Neither exercises OpenMP — `macos-appleclang` deliberately disables it, and
`linux-gcc` doesn't enable it via preset config either. The `macos-llvm` and
`linux-clang` presets aren't covered at all. OpenMP regressions and the
`-fopenmp=libomp` shim issue (the one [[project_clangd_lsp]] documents) will
not be caught by CI.

---

## B. Two app styles, one production story

### B1. Hand-written apps cannot produce HDF5 output

Every hand-written app under `apps/` (`phi4_hmc`, `xy_metropolis`,
`onsigma_metropolis`, `bosegas_hmc`, `weak_field_gr_hmc`, `testaction`)
just prints to `std::cout`. None of them open an `Hdf5Writer`, write run
metadata, save configurations, or buffer measurements. So in practice:

- For exploration / spot-checks → hand-written apps work.
- For any real run that needs reproducible HDF5 output, autocorrelation
  analysis, or `analyze.py` consumption → you must use `reticolo_run` +
  YAML + python.

This is the *opposite* of the design intent in
[[feedback_lean_over_framework]] / [[project_no_toml_config]], which framed
the hand-written app as the primary surface and `reticolo_run` as the
fallback. As of today, `reticolo_run` is the only path that produces
analysable data; the hand-written apps are demos. `examples/phi4_tuning/`
confirms this — it drives `reticolo_run`, not the per-action apps.

The friction will compound: every new physics-paper run needs HDF5, so the
user is being funneled toward the generic YAML driver they explicitly
designed away from. Either lift HDF5 + metadata into a small reusable
helper that hand-written apps can call in ~5 lines, or accept that
`reticolo_run` is the real interface and demote the hand-written apps to
documentation.

### B2. `Series<T>` is built but unused

`Hdf5Writer::cached_series<T>(path)` and `Series<T>::append(v)` implement
exactly the typed, auto-buffered time-series API the old memory described.
Grep shows the only callers are inside `hdf5_writer.hpp` itself
(`series_cache_` infrastructure) — `MonteCarloHandler::measure_utility`
ignores it and re-implements the same buffering manually with
`obs_buffer_`, `mc_stats_buffer_`, `max_buffer_size_`,
`max_buffer_write_delay_`, explicit `flush` on size/time.

Two equivalent buffer-and-flush mechanisms living side by side. Pick one
or the other before someone fixes a bug in only half of them.

---

## C. Hand-maintained N×K registration tables

Adding a new action currently requires touching **four** places, all
hand-maintained, none cross-referenced by the compiler:

1. `src/reticolo/action/<NewAction>.hpp` — the action itself.
2. `src/reticolo/action/yaml/<NewAction>Yaml.hpp` — `apply_yaml` overload.
3. `src/reticolo/action/registration/ActionDescriptor.hpp` —
   `k_builtin_actions` table.
4. `src/reticolo/runtime/runtime.hpp` — `else if (action_name == "...")`
   branch (one per impl_type — currently 16 branches for 8 actions).
5. `src/reticolo/io/action_observables.hpp` — `Hdf5TypeOf<Obs>`
   specialisation (one per impl_type → 2 each).
6. Optionally a YAML adapter for any new `MonteCarloData<action_type>`
   compound if `action_type` is new.

Three issues:

- The `runtime.hpp` if-else chain (16 branches) and `k_builtin_actions`
  list (8 entries) are *not synced by code*. `runtime.hpp` does
  `throw std::runtime_error("Unknown action; available: " + ...)` and
  iterates `k_builtin_actions` for the error message — but a typo in the
  switch wouldn't be detected against the registry. The registry is only
  used cosmetically.
- Forgetting step 5 (HDF5 type spec) is a link-time error, but only when
  `reticolo_run` is invoked with that action — tests don't exercise
  `MonteCarloHandler` for every action. You can ship a half-registered
  action that passes `ctest` and crashes the first user.
- The whole point of `ActionDescriptor` was "explicit dispatch, no
  registry"; what we have is a registry plus an explicit dispatch that
  ignores it. Pick one. Either drive the switch from the descriptor table
  (template-instantiate the cross product programmatically) or delete the
  descriptor table.

### C1. YAML config keys are stringly-typed with no schema

`apply_yaml(Phi4&, params)` does `if (params["kappa"]) a.p.kappa = ...`.
A misspelled key (`kapppa`) is silently ignored and the action runs with
its default. YAML keys mismatched between `runtime.hpp` and
`apply_yaml` (`field_init`, `therm_steps`, `measure_step`, etc.) are also
silent. There is no schema, no `--strict`, no unknown-key warning.

For a research codebase where a wrong κ means a wrong physics result, this
is a footgun. A minimum-viable mitigation is logging the *applied*
parameter values after `apply_yaml` so you can diff what the file claimed
vs what the binary used; a real one is a one-line "unknown key" warning.

### C2. CLI ↔ YAML naming is per-app freelance

`phi4_hmc --md-steps` → builds `hmc_params["steps"]`.
`xy_metropolis --prop-width` → builds `params["prop_width"]`.
`bosegas_hmc --md-steps` → builds `hmc_params["steps"]`.

There is no shared convention mapping CLI dash-flags to YAML
snake/camel keys, and no shared helper. Adding a sixth hand-written app
will reinvent this again. A two-function helper
(`add_hmc_options(opts)` / `hmc_params_from_args(args)`) would cut every
app by ~10 lines and unify the surface — and would be exactly the kind of
"lean utility" that [[feedback_lean_over_framework]] approves of.

---

## D. HDF5 schema lock-in & reproducibility gap

### D1. Complex-schema mode is invisible in the file

`RETICOLO_HDF5_LEGACY_COMPLEX_COMPOUND` is a compile-time flag that
changes the on-disk representation of complex-valued data (`{r,i}` compound
vs HDF5 2.x native complex). The flag is baked into the binary; the
written HDF5 file does not record which schema was used.

Consequence: a `RelativisticBoseGas` run produced by a binary built with
legacy-compound can be opened by a binary built with native-complex and
silently misinterpret `(r, i)` as `(re, im)` (same wire bytes, different
named fields), or fail with a cryptic typecheck.

Fix is one line: stamp `/run@hdf5_complex_schema = "legacy_compound" |
"native_complex"` next to the existing `/run@hdf5_version`. Without it,
the dual-mode flexibility is a slow-motion data-integrity bug waiting for
the first cross-build read.

### D2. `RETICOLO_COMPILE_FLAGS` is stamped, but not the action source

`/run@cmdline`, `/run@version`, `/run@commit`, `/run@compile_flags` get
written — good. The actual action *parameters* (κ, λ, β, …) are not
auto-stamped; only what `MonteCarloHandler` chooses to log via its own
`logger_` text channel ends up in `.log` files, not in the HDF5. So given
a `measurements.h5`, you can prove which binary and commit produced it but
not which κ. The `apply_yaml` outcome should be dumped to a `/run/action`
group as attributes; right now it depends on hand-written cooperation
that doesn't exist.

---

## E. Concept boundaries and silent capability gaps

### E1. `LocalAction` non-const `field` leaks an `Indexing` access pattern

```cpp
{ action.compute_s(field) } -> ...;           // field: Lattice<F>&  (mutable)
{ action.s_local(field, site) } -> ...;       // mutable
{ action.ds_local(field, site, new_phi) };    // mutable
```

`compute_s` semantically does not mutate the field; the non-const ref is
there because `field.is_interior(site)` and `field.idx.skin_ids()` are
mutating-looking on the current `Lattice` API. This means **no algorithm
can hold a `const Lattice<F>&` and call `compute_s`** — observers,
diagnostics, checkpoint readers all have to launder constness or take a
copy. This will hit when the LLR replica-exchange code (which compares
neighbouring replicas without modifying them) is ported.

Cheapest fix: mark `is_interior`, `interior_ranges`, `skin_ids` as
`const` (they appear to be already in `Indexing`; the issue is the
`Lattice` accessors). Verify and tighten the concept signatures.

### E2. `WolffEmbeddable` doesn't encode the periodic-only restriction

`Wolff::setup` throws `std::runtime_error("Wolff: only Periodic boundary
conditions are supported in v1.")` if any direction is non-periodic.
Concept admits the action; runtime rejects the lattice. If a sweep
includes an antiperiodic-BC config by accident, you find out only after
allocating + thermalising. Either move the check into a static assertion
parametrised on the lattice's BC, or surface it before any heavy work.

### E3. `HasForce` and the `HmcSlot<...>` trick

```cpp
template <class A, class G, bool = action::HasForce<A>> struct HmcSlot {};
template <class A, class G> struct HmcSlot<A, G, true> {
    std::unique_ptr<HMC<A, G>> ptr;
};
```

Nice — avoids instantiating `HMC<A>` for actions without forces. But the
`if constexpr (action::HasForce<Action>)` guard at the *call site*
duplicates the concept check; if the slot is true but the check is false
(or vice versa) the throw message lies. Not currently possible but easy
to break in a refactor — extract a single `supports_hmc_v<A>` and use it
in both places.

---

## F. Performance & micro-architecture

### F1. HMC integrator switch in the hot loop

`update_field` does `switch (integrator_)` *every trajectory*. Trivial
cost per call but every per-step force evaluation is fronted by the
`compute_force / kick / drift` virtual layer too. Templating `HMC` on the
integrator (or storing a function-pointer set at `setup`) would let the
optimiser inline the integrator body. With ranlux gone (commit `6f68980`)
the integrator share is exactly where remaining wins are. `proposals/
perf_overview.md` flagged this already.

### F2. `HMC` allocates three `Lattice` via `unique_ptr`

`mom_`, `forces_`, `old_field_` are heap-allocated `unique_ptr<Lattice>`
because Lattice needs a shape at construction. Each per-site access goes
through a pointer hop. If `Lattice` gained a default + `resize`
constructor (or HMC stored value-typed Lattices initialised at `setup`)
the indirection would vanish.

### F3. `MonteCarloHandler` is a 500-line template instantiated per action

Every `(Action, Rng)` pair instantiates the whole class — workspace
management, HDF5 writer, logger init, buffered measurement, the lot. With
8 actions × 2 impls × 2 RNGs = 32 instantiations in `reticolo_run.cpp`,
each pulling `<hdf5.h>`, `<filesystem>`, `<format>`, … through the inline
template body. Compile times of `reticolo_run.cpp` are about to dominate
the build.

The handler logic is largely orthogonal to `Action`; only the buffer
element types and `apply_yaml(*action_, ...)` actually depend on it.
Splitting the handler into a non-template "workspace + writer + logger"
core + a templated thin shell that owns `field_` / `action_` / buffers
would cut instantiation cost dramatically.

### F4. `<hdf5.h>` is in a header on the public include path

`Hdf5Writer.hpp` includes `<hdf5.h>` and is included transitively via
`runtime.hpp` → `MonteCarloHandler.hpp`. Anything that touches `runtime.hpp`
pulls libhdf5's full C header set. Hand-written apps that don't need HDF5
still pay this if they ever include `runtime.hpp` (they don't today —
keep it that way).

### F5. `unif_` / `norm_` typed on `impl_type`

For `Action::impl_type = float`, the HMC momentum sampling uses
`std::normal_distribution<float>`, which is unusual in MC literature.
Quality is fine but it's an unstated trade — flag if the user does single-
precision HMC and gets surprising acceptance.

---

## G. Concurrency & ownership

### G1. `cached_series<T>` returns a reference into an `unordered_map`

```cpp
auto it = series_cache_.emplace(std::move(path), std::move(holder)).first;
return static_cast<SeriesHolder<T>*>(it->second.get())->handle;
```

The reference is valid as long as the map isn't rehashed. With the
mutex held throughout, single-threaded users are safe. But if a second
thread later inserts (after the lock is released by the caller), the
returned reference can dangle — `unordered_map` invalidates references
to elements on rehash. The dtor of `Series<T>` then writes through a
freed pointer.

Today everything is serialised by `mu_`, so this is latent. If a future
change adds "long-lived `Series<T>&` handed to a measurement loop", it
breaks. Either return `Series<T>*` after a stable allocation (the holder
is already heap-allocated, so just return `&holder->handle` outside the
map → the map only owns the holder, never moves the handle).

### G2. `Series` flush in dtor with `try/catch`

The dtor swallows exceptions silently. On a power-fail or disk-full
during the last flush you get no error and a truncated file with no
indication. Logging the swallowed exception to `stderr` (not throwing —
correct, dtors shouldn't throw) is one line.

### G3. `Indexing` shared across lattices is great, but rebuild semantics?

`Lattice` copy-assign comment says "indexing preserved when compatible".
What's "compatible"? Same shape + same BCs presumably. If a user does
`field = other_field` where shapes differ, what happens — silent
re-indexing, throw, or undefined? Worth a unit test (`test_indexing_pool`
covers the pool, not the assignment edge case).

---

## H. Smaller things

- **H1.** `runtime.hpp::reticolo_init` returns `std::nullopt` for both
  `--help` and missing-config — caller can't distinguish.
  Today nobody cares; tomorrow a CI smoke test that runs `reticolo_run --help`
  and expects `exit 0` will conflict with one that runs no args and expects
  `exit 1`.
- **H2.** `MonteCarloHandler` calls `logger_ << "[...] Initialization started"` —
  multiple typos in user-visible log strings (`Implmentations`, `loggind`,
  `memeory`, `bee`). Cosmetic but visible in every run log.
- **H3.** `OnSigma` is dispatched only as `O3Sigma` / `O3Sigma_F` /
  `O4Sigma` — `O4Sigma_F` is missing from the switch. Asymmetric coverage.
- **H4.** `BoundaryCondition` is stored as `std::uint8_t` keyed `BC` enum
  in the neighbour-pool key; `SMALL_LATTICE` switches `size_type` to
  `unsigned short` but the pool still keys on `size_t`. A `SMALL_LATTICE`
  build will compute a pool key that doesn't match between Indexing
  constructions on the same data — verify there's no silent rebuild.
- **H5.** `apps/testaction.cpp` shares a name prefix with
  `tests/test_testaction_detailed_balance.cpp`. Pick a less collisional
  name for the app (`testaction_demo`?) to avoid future grep confusion.
- **H6.** `apps/reticolo_run.cpp` is a thin shell, but everything it
  needs lives in `runtime.hpp`. If a future plan adds a second generic
  binary (e.g. `reticolo_replica`), the dispatch will get copy-pasted
  unless extracted into a `.cpp` translation unit.
- **H7.** `proposals/perf_overview.md` and `proposals/refactor_plan.md`
  predate several large commits (`6389653`, `2c987e9`, `66c44a3`,
  `e2715b5`). The numbers and TODO lists in them have probably drifted;
  worth a one-pass refresh or a "last verified at $commit" stamp.

---

## Priority ranking (my opinion, not yours)

1. **A1** (LLR dead code) and **B1/B2** (hand-written apps can't produce
   HDF5; `Series<T>` unused) — these define what the surface actually
   *is*, and they contradict the stated design.
2. **D1** (HDF5 complex schema not stamped) — one-line fix, prevents a
   data-corruption class of bug.
3. **C** (registration tables out of sync, no schema for YAML keys) —
   will determine how painful the next 5 actions are to add.
4. **F3** (handler over-templated) — bites when you add an action and
   wonder why `reticolo_run.cpp` takes 30 seconds to compile.
5. Everything else is real but lower-stakes; address opportunistically.
