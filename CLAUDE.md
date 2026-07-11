# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`reticolo` is a C++20 header-mostly library for Hybrid Monte Carlo simulations of lattice quantum field theories (scalar fields, U(1)/SU(2)/SU(3) gauge), plus a small set of hand-written reference apps under `apps/`. HMC is the only update algorithm (behind the `updater::Updater` contract). Orchestration — running many concurrent simulations — is a subsystem under `orch/`: a physics-free spine plus concrete orchestrators (parameter span; LLR = Logarithmic Linear Relaxation). Single-replica, parameter-span ensembles, and LLR replica ensembles. OpenMP threads the CPU at three composable layers (see §CPU parallelism): the lattice-traverse stage inside a single trajectory, the per-trajectory HMC region, and the LLR replica ensemble. Without `-fopenmp` every layer degrades to serial.

`README.md` and `docs/architecture.md` are the canonical references — read them before any non-trivial change. `docs/writing_an_app.md` walks through building a new app and adding a test for it.

## Build / test / run

CMake presets (configured in `CMakePresets.json`):

```sh
cmake --preset macos-appleclang    # macos-appleclang | macos-llvm | linux-gcc | linux-clang | debug | linux-nvcc
cmake --build --preset macos-appleclang
ctest --preset macos-appleclang
```

- Generator is Ninja. Build dirs land in `build/<preset>/`.
- `macos-appleclang` forces `RETICOLO_ENABLE_OPENMP=OFF` (Apple Clang has no usable OpenMP). All other presets build with OpenMP and link `libomp`.
- `debug` preset adds asan + ubsan and disables `-march=native`.
- `RETICOLO_WARNINGS_AS_ERRORS=ON` is the default in all presets — fix warnings, don't suppress them.

Run a single test:

```sh
ctest --preset macos-appleclang -R test_phi4_force_consistency --output-on-failure
# or invoke the binary directly to use Catch2 tag/name filters:
./build/macos-appleclang/tests/test_phi4_force_consistency "[force]"
```

Each `tests/{unit,physics,io}/*.cpp` is registered via `reticolo_add_test(<name> <source>)` in `tests/CMakeLists.txt`; tests are linked against a shared `tests/test_main.cpp` that calls `log::off()` so test output stays clean.

`tests/apps/` contains end-to-end smoke tests that exec the actual app binaries and inspect the HDF5 output — they depend on the app being built (`RETICOLO_BUILD_APPS=ON`, the default).

## Top-level layout

```
include/reticolo/      header-only core, actions, updaters (HMC), orchestration (spine + span + LLR), observers, CLI shim, app-setup helpers, optional CUDA backend
src/io/                the ONLY TU that #includes <hdf5.h> (PIMPL'd writer)
src/cli/               cli::Parser implementation (wraps cxxopts)
src/lint/              amalgamation TU so clang-tidy can scan the umbrella header in one pass
apps/                  builtin reference simulations + benchmarks (built in-tree); apps/cuda/ = .cu ports
examples/              numbered standalone consumer studies: own source + CMake (find-or-fetch reticolo) + bash sweep + python analyze
tests/                 unit / physics / io / apps smoke tests (Catch2)
tools/                 local CI-gate runner (check.sh) + CPU scaling benches (bench/) + GPU cloud runners (modal/kaggle/runpod) + validate/profile
templates/             copy-and-fill scaffolds (action / updater / orchestrator) — un-built `// FILL IN` stubs, see templates/README.md
docs/                  architecture + app/test how-to
```

Inside `include/reticolo/`: `core/` (indexing, lattices, RNG, logger, `parallel.hpp` = the `exec::` threading primitives), `action/` (the action families — `nn/` scalar nearest-neighbour (`NNAction`, covers site + bond), `complex/` sign-problem (real base + `ImagPart` mixin), `gauge/`; plus `windowed_action.hpp` (a decorator action — base + Gaussian window on a `Constraint` observable, used by LLR but not LLR-specific), the shared sweep engine, and concepts), `updater/` (the update-algorithm subsystem: the `updater::Updater` concept at top + `updater/hmc/` = `updater::Hmc` + integrators; folder mirrors `action/` — one flat `updater::` namespace), `orch/` (the orchestration subsystem: a physics-free spine — `Worker`/`Checkpointable` concepts, `ThreadPlan`/`plan_threads`, `parallel_workers`, ensemble checkpoint — plus concrete orchestrators in `orch/span/` (parameter span) and `orch/llr/` (replica-exchange LLR, namespace `orch::llr`)), `obs/`, `app/` (shared pre-loop scaffolding: `app::common_flags` + `app::open_writer`), `cli/`, `math/` (gauge group math), `io/`, `cuda/` (optional device backend).

The umbrella include is `<reticolo/reticolo.hpp>` — apps include exactly that.

## Architectural invariants (don't break these)

These are load-bearing design choices, not preferences. Most of them are described at length in `docs/architecture.md`.

- **Three CMake targets, not one.** `reticolo::core` (INTERFACE, header-only), `reticolo::io` (STATIC, the only thing that links HDF5), `reticolo::cli` (INTERFACE). `reticolo::reticolo` is the umbrella. `hid_t` and `<hdf5.h>` must never leak out of `src/io/writer.cpp`.
- **No virtual dispatch in the hot loop.** Actions are plain structs satisfying the field-agnostic HMC concepts in `<reticolo/action/concepts.hpp>` — `HmcAction` (`s_full` + `compute_force`), with opt-in refinements `HasFusedKick` and `HasImagPart`. One concept set covers site (`Lattice<F>`) and gauge (`MatrixLinkLattice<G,F>`) actions alike. The sole updater `updater::Hmc<A,R,Integ>` is a class template constrained by `HmcAction`; `orch::llr::Replica` is the orchestration layer on top. No base class, no `register_action`, no string dispatch.
- **HMC integrator is a type parameter**, not a runtime switch. `updater::integ::Leapfrog`, `Omelyan2` (the default), `Omelyan4` are structs with a static `run(...)`. A new integrator = new struct + instantiate `Hmc<A, R, NewInteg>`.
- **CPU thread counts are per-`Hmc` state, not a global.** `HmcSpec.n_threads` / `slabs_per_thread` (0 = ambient, resolved ONCE at construction and frozen; `set_spec` refuses threading changes) are bound each trajectory via `exec::team_scope` / `exec::slab_scope` so an `Hmc` can be pinned to a fixed team regardless of the OpenMP global — the basis for mixing replica-level and site-level parallelism. Results stay bit-identical for fixed `n_threads` + `slabs_per_thread` (deterministic fixed-partition reductions).
- **`Hmc` owns its randomness: one RNG stream per canonical slab.** The ctor takes a freshly-seeded family generator (`FastRng`/`PhiloxRng`/`RanlxdRng`/`Mt19937Rng`) BY VALUE and builds an internal `StreamSet<R>` (`core/rng/stream_set.hpp`) with one site stream per `exec::partition` item + a driver stream for serial draws (accept, exchange, hot starts). Slab i draws strictly from site stream i via `exec::field_visit_indexed` — bound by partition index, never by thread — and every fill re-checks the partition (throws on drift). Never pass the same rng lvalue to two `Hmc`s: by-value copies give both IDENTICAL chains; derive distinct seeds instead. Checkpoint/resume goes through `hmc.rng()` (`io::save_config`/`load_config` validate family kind + stream count on load).
- **One way to do each thing.** One writer, one CLI helper (`cli::Parser`), one logger (`reticolo::log`). No central registries. The compiler is the source of truth.
- **The updater has a contract.** `updater::Updater` (`updater/concepts.hpp`) is the sampler surface apps/orchestration depend on (`step` → `{dH, accepted}`, `last_s_full`, `rng`) — the updater-level analogue of `HmcAction`. `updater::Hmc` is the sole implementation; both orchestration workers `static_assert` their sampler against it. A new updater models the concept, reusing the `updater::integ::*` type-params — no base class.
- **Orchestration is a physics-free spine + clients.** `orch/` holds the concurrency / checkpoint / thread-plan spine (`orch::Worker`, `parallel_workers`, `save_ensemble`, `plan_threads`); concrete orchestrators live in subfolders (`orch/span`, `orch/llr`) as clients that own the schedule and any coupling. Dependency is one-way `orch/<name> → orch → core`; the spine never mentions windows / tilt / exchange. `parallel_workers` is concurrent-only — the thread-unsafe drain stays a serial loop in the client (orchestrators own their loop, like apps do). A new orchestrator = a `Worker` type + a driver over `parallel_workers`; do not fold workflow specifics into the spine.
- **Lattices share an `Indexing` via a process-wide weak-ptr pool.** `Indexing::acquire(shape)` returns `shared_ptr<Indexing const>`. Sibling lattices (HMC momentum/force/rollback) are constructed from another lattice's `indexing()` so three buffers share one geometry (shape + packed strides). There is no stored neighbour table anymore: `next(s, mu)` / `prev(s, mu)` are computed from the strides, and the hot kernels roll their own row-nested strided offsets rather than looking neighbours up per site. Dimension is capped at d ≤ 4; parity / even-odd tables are gone.
- **Periodic BCs only.** `l.next(x, mu)` / `l.prev(x, mu)` wrap periodically. Don't add per-site BC guards to action bodies; that was deliberately removed.
- **Apps own the for loop.** Each app under `apps/` is its own `main()` — roughly 75–130 lines including CLI setup and I/O (HMC apps ~75–120, LLR apps ~100–130), no closure driver, no range-for helper, no string-dispatch `reticolo_run`. The trajectory `for` stays plainly visible. To add an updater/action variant, copy an existing app and edit.
- **Writer is self-describing.** `io::Writer` stamps `/run@*` (cmdline, version, commit, compile flags, hostname, started_utc, hdf5 schema) on construction. Passing the `cli::Parser*` also stamps every registered flag to `/vars@<name>`. There is no separate config file format — argv + git SHA is the run record.
- **`Series<T>` is the only way to write a time series.** Move-only, returned by value, buffered with chunked flush. `extern template`'d in `writer.cpp` for the supported scalar set.

## Code style

- C++20. Concepts over CRTP over virtual. Modern C++; tool-enforced (`clang-format`, `clang-tidy` via `src/lint/amalgamation.cpp`, `clangd` via `.clangd`). CI pins `clang-format` to 22.1.5 (the latest on PyPI) and `clang-tidy` to 22 (`WarningsAsErrors` on — a different clang major reflows/diagnoses differently, so match the pin). Reproduce the gate locally with `tools/check.sh` (format + tidy + build + ctest) or the `/check` skill; don't run ad-hoc `clang-format` / `clang-tidy` against a mismatched compile DB.
- `clang-tidy` checks enabled: `bugprone-*`, `cppcoreguidelines-*`, `modernize-*`, `performance-*`, `readability-*` (minus `use-trailing-return-type`, `identifier-length`, `magic-numbers`). Run via the amalgamation TU so the umbrella header gets analysed once. Never auto-rename to satisfy a naming check — the `.clang-tidy` config encodes the real convention and blanket renames break the build.
- Short namespaces in apps: `reticolo::alg`, `reticolo::obs`, `reticolo::io`, `reticolo::cli`. The umbrella adds `namespace act = action;` so apps say `act::Phi4`.
- No comments unless logic is non-obvious. No speculative abstractions.
- CI matrix is appleclang + macos-llvm + linux-gcc + linux-clang. macOS-only local checks can miss GCC-only `-Wuseless-cast` and Linux package gaps — don't claim a change is CI-clean from appleclang alone.

## Logger

`reticolo::log` is a thread-safe (mutex around `std::cout`/`std::cerr`), OpenMP-aware logger. Severity = sigil (`·` debug, `┃` info, `⚠` warn, `✖` error), each line carries elapsed `HHH:MM:SS.mmm` + a 4-char tag. `log::start(workspace, out_name[, replicas])` is the single init: creates the workspace folder, opens `<workspace>/<stem>.log` (stem = out_name minus extension) and mirrors every entry into it, then prints the banner. `replicas=true` (LLR apps) adds the run-id column and per-replica `<stem>.<rNNN>.log` files. RAII `log::scope("rNNN")` binds a thread-local replica/run tag — `orch::llr::Replica` binds its own id inside its public methods; apps/drivers bind a scope only when their own code logs inside a parallel region (the warm-up loops). Transitively-called code picks up the tag automatically.

Lattice / RNG / Writer / action / updater / `orch::llr::Replica` constructors auto-announce themselves — most apps call `log::info` rarely. Tests and benchmarks call `log::off()` to short-circuit formatting.

## CLI

`cli::Parser` wraps cxxopts. `req<T>(name, desc)` and `opt<T>(name, default, desc)` return `T const&` to stable `unique_ptr`-backed storage — apps plant references at registration time and read them after `parse(argc, argv)`. `io::Writer{path, argc, argv, &parser}` then stamps every registered var into `/vars@<name>`.

## CPU parallelism

OpenMP threads the CPU at three composable layers, all mediated by the `exec::` primitives in `core/parallel.hpp`:

- **Lattice-traverse (site-level).** `parallel_map` / `parallel_reduce` (and their `_ranges` variants) are the only holders of `#pragma omp for`. The action sweep engine (`action/sweep/stencil.hpp`) drives `compute_force` / `s_full` through them. `in_traverse_region(want, body)` decides whether to open a region: it opens ONE `omp parallel` when `want` and not already nested; otherwise it worksplits (nested in our own region) or runs serial (nested in a foreign region, or the pass is too small). Chunking targets ~64 KiB working sets per work item.
- **Per-trajectory (`updater::Hmc`).** Each trajectory binds `team_scope{n_threads}` + `slab_scope{slabs_per_thread}` so the drift/kick/force passes thread under a fixed team (see the invariant above). Kernels are pure workers over a flat work-item index — they never see OpenMP. Momentum sampling threads through the Hmc-owned per-slab RNG streams (`StreamSet` + `field_visit_indexed`; see the RNG invariant above).
- **LLR replicas (ensemble-level).** `#pragma omp parallel for schedule(dynamic, 1)` over the replica vector.

Composability rule: inside the LLR replica team the traverse stage stays serial (the `in_traverse_region` flag is false in a foreign region) so site-level threads don't fight the replica team. Prior benchmarking: field traversal scales ~22–32×, but the axpy-style ops (drift/kick/copy/momentum) are bandwidth-bound and cap near ~6×; a full trajectory lands around ~10×. Per-op fork/join beat a persistent MD region on libgomp (a persistent-region + manual-SPMD prototype was built, verified, then reverted).

## Orchestration (`orch/`)

`orch/` is the orchestration subsystem: a physics-free spine (`reticolo::orch`) plus concrete orchestrators as clients. The spine is `Worker`/`Checkpointable` concepts (`orch/concepts.hpp`), `ThreadPlan` + `plan_threads` (`orch/thread_plan.hpp`), the one concurrent primitive `parallel_workers(workers, plan, body)` (`orch/ensemble.hpp` — concurrent-only; the serial drain stays in the caller), and generic `save_ensemble`/`load_ensemble` (`orch/checkpoint.hpp`, with per-worker `save_extra`/`load_extra` hooks). Dependency is one-way `orch/<name> → orch → core`; the spine never mentions windows, tilt, or exchange. A new orchestrator = a `Worker` type + a driver that loops over `parallel_workers`.

**Parameter span** (`orch::span`): `orch::span::Chain<Action,…>` is a generic HMC worker; the app builds the worker grid and hands it to `orch::span::Orchestrator` by move — `setup(out)` opens the series, `run(Schedule)` sweeps the grid concurrently (`apps/param_span_hmc`).

### LLR (`orch::llr`)

`orch::llr::Replica` wraps a base action in `action::WindowedAction` (a decorator action in `action/`, `S_win = S_base + a·Q + (Q−E_n)²/2δ²`) and is an `orch::Worker`. **What defines the window `Q` is a `Constraint` policy**: `SelfConstraint` (Q = the action itself; the default for real actions, fused single-pass), `ImagConstraint` (Q = `s_imag`; default for sign-problem actions), or `ObservableConstraint<Obs>` for ANY observable — "simulate this action, window on that quantity." The observable `Obs` is *itself an action* (an `NNAction` leaf, etc.): you define only the per-site formula kernels and its `s_full`(=Q)/`compute_force`(=−dQ/dφ) run through the parallel sweep engine automatically — no hand-rolled loop. LLR is one consumer; the default constraint reproduces the previous auto-selected modes bit-for-bit. Newton-Raphson warm-up + Robbins-Monro adapt the per-replica `a`. Periodic even/odd replica exchange. `orch::llr::Orchestrator` (`orch/llr/orchestrator.hpp`) is the client of the spine: a two-phase object where `setup(out)` plans threads, builds the replica ladder (a uniform E_n sweep synthesised from `orch::llr::Spec` — the old hand-rolled per-replica `make_unique` loop is gone), resumes, and opens the series, and `run()` / `run_smoothed(SmoothConfig)` drive `parallel_workers` and add exchange as a serial coupling between waves; each replica's HMC stays single-threaded inside the worker team. The one genuinely app-specific step — initial field content (cold-to-identity for gauge SU(N), hot-start for U(1)/complex, nothing for real scalars) — runs between `setup()` and `run()` via `replicas()`, guarded by `resuming()`.

Watch-outs (from prior incidents in this tree):
- Newton/RM step is `C·<dE>/δ²` where `C=12` for the hard window and `C=1` for the Gaussian window. Mixing them diverges geometrically.
- In the paper convention (`s_full = -a·S - window`), `a` flips sign at the natural peak, so `a_init = -1` and DoS reconstruction uses `(1+a)`. Energy convention keeps `a_init = 0` and integrates `a` directly.

## CUDA backend (optional, `RETICOLO_ENABLE_CUDA`; branch `feat/cuda-extension`)

A 4th target `reticolo::cuda` (INTERFACE / header-only — the device stack lives entirely in `include/reticolo/cuda/*.{hpp,cuh}`; nvcc compiles only the consumers that include it, `apps/cuda/` + `tests/cuda/`). The `<cuda_runtime.h>` quarantine that `reticolo::io` enforces for HDF5 is here automatic: `cuda/` headers are only ever pulled into a `.cu` TU, so a CPU build never sees them. Off by default. Canonical refs: `docs/cuda_architecture.md`, `docs/writing_a_cuda_app.md`, `docs/architecture.md` §CUDA backend.

**Ported:** HMC only (the workhorse), all three integrators (reused as type params), via a generic `cuda::DeviceAction<HostAction,Field>` + `cuda::Hmc`. Actions: Phi4/Phi6/SineGordon/Xy (scalar, f64+f32), Wilson\<U1/SU2/SU3\> (gauge), BoseGas (complex, f64+f32). LLR is also ported (stream-concurrent replicas — `cuda/llr/{driver,replica}.hpp`, `apps/cuda/*_llr_cuda.cu`).

**Invariants (don't break):** dependency is one-way `cuda → core` (core never includes `cuda/`); the only core GPU-awareness is `core/hd.hpp` (`RETICOLO_HD`) and the `__CUDACC__`-guarded Philox in `core/rng.hpp`. One shared per-site **formula** (`action/<family>/formula/<name>_formula.hpp` or `action/gauge/formula/<name>_formula.hpp`, `RETICOLO_HD`) — CPU and device must not diverge. **Gather-only, no `atomicAdd`** (reversibility); deterministic fixed-config reductions. `.cuh` = files with device kernels (nvcc-only); `.hpp` under `cuda/` = host-callable API. Abelian U(1) uses `gauge_u1.cuh`; matrix groups use the generic `gauge_sun.cuh` + a `group_device<G>` trait — keep families separate. A GPU app = `<reticolo/reticolo.hpp>` + `<reticolo/cuda/cuda.hpp>`.

**No local NVIDIA GPU — GPU build/test runs on a cloud GPU host (Modal or Kaggle), both driving the shared `linux-nvcc` CMake preset (the same preset workflow every system uses).** Note the `linux-nvcc` preset has `RETICOLO_ENABLE_OPENMP=OFF` and `RETICOLO_TUNE_NATIVE=OFF` (`-arch=native` breaks device code and `-march=native` SIGILLs the HDF5 binaries on A100 containers) — so CPU-threading benches must run under `linux-gcc`, not `linux-nvcc`.

```sh
cmake --preset linux-nvcc && cmake --build --preset linux-nvcc && ctest --preset linux-nvcc

uv run tools/modal/app.py build    # Modal runner (preferred): per-second GPU billing, pick the GPU (--gpu T4|A100|H100|...), warm build cache on a Volume; run a binary with `run --app NAME`
tools/kaggle/push.sh               # Kaggle runner: ships the working tree as a dataset, runs run.py, downloads to tools/kaggle/output/
tools/runpod/run.sh                # RunPod runner (SSH-driven): the ONLY host where nsys GPU profiling works; see tools/runpod/README.md
```

Modal is a single self-contained uv script `tools/modal/app.py` (PEP 723 inline deps, pinned to the `reticolo-cuda` Modal environment). Three commands: `build` (the preset gate), `run --app <target>` (build one binary, run it in a temp dir, export its output) — also how you run `bench_hmc_cuda` / `profile_hmc_cuda` / `nightly_cuda` — and `pull <run_id>` (fetch artifacts to `tools/modal/output/`). Kaggle: `push.sh` tars the **local working tree** → dataset (`reticolo-src`), ships `run.py` (toggle `RUN_GATES` / `RUN_PROFILE`), reflecting the current checkout with **no git commit/push**. For double-precision Modal runs use `--gpu A100` (T4 has no real FP64, ~40× slower). **`ncu` is blocked on Modal + Kaggle** (`ERR_NVGPUCTRPERM` — locked perf counters); `nsys` works there but is no longer wired into a sweep. RunPod (`tools/runpod/`, SSH into a bare pod: `setup.sh` → `build.sh` → `run.sh`/`profile.sh`) is the fallback where `nsys` GPU-kernel profiling actually produces a usable `.nsys-rep`.

## Examples

`examples/` are **standalone consumer projects**, not in-tree apps. Each `NN_short_name/` carries its own driver source(s) + a `CMakeLists.txt` that links `reticolo::reticolo` via an inline **find-or-fetch** block (reuse the target if built in-tree, else `add_subdirectory` a sibling checkout at `../..`, else `FetchContent` a pinned git tag) — so an example dir copied out of the repo still builds. The bash `run.sh` calls `build_example` (in `examples/_common/preset.sh`) to configure+build the example's own CMake project, then sweeps its `$example_bin/<binary>` and post-processes the HDF5 in `analyze.py`. Subdirs are numbered in landing order — clear orphan files before adding one.

`apps/` is the separate **builtin** set: one canonical reference sim per action + the `bench_*` suite, built only in-tree (opt-in `RETICOLO_BUILD_APPS=ON`, set in every preset). Examples that mirror an action copy its driver as a starting point; the canonical app stays in `apps/`. Study-only drivers (e.g. `tune_phi4_*`) live in their example, not `apps/`.

Build flags default **OFF** (`RETICOLO_BUILD_APPS`/`_EXAMPLES`/`_TESTS`) so a bare or fetched configure builds core/io/cli only; presets turn them on for the dev/CI workflow. OpenMP auto-degrades to serial when the toolchain lacks it (no longer `REQUIRED`).

## Common workflows

- **Add a new action**: three steps — pick a family base, write the formula, wire it up. (1) Pick by interaction shape: **`NNAction<Derived,T>`** (`action/nn/`) for any scalar nearest-neighbour action — the leaf gives the finalizers `action_kernel` (`(self, fwd_agg)→S`) + `force_kernel` (`(i, self, agg)→F`), and OPTIONALLY a per-bond `action_combine`/`force_combine` (default identity → "self + neighbour-sum" as in Phi4/Phi6/SineGordon; a real combine → endpoint-difference bond model as in XY) + `action_scale` (default 1). **`ComplexAction<Derived,T>`** (`action/complex/`) for the real (phase-quenched) part of a sign-problem action (anisotropic split-last), PLUS the **`ImagPart<Derived,T>`** mixin for the imaginary extension (`s_imag`/`compute_force_imag` + the fused combined kick) — a complex leaf is `struct BoseGas : ComplexAction<…>, ImagPart<…>`. **`Wilson<G>`** for gauge — usually you just write a group model `G` under `math/group/`. (2) Put **all the physics** in a `RETICOLO_HD` formula file (`action/<family>/formula/<name>_formula.hpp`) — the single source of truth shared with the CUDA functors. (3) Write a small aggregate leaf (`action/<family>/<name>.hpp`) deriving the base(s), with the couplings, `describe`, and coupling-hoisting kernels that bind the formula. The base supplies `s_full`/`compute_force`/`compute_force_and_kick`/caches; deriving keeps the leaf a designated-init aggregate, so `{.kappa=…}` still works. Add the header to the matching family aggregator (`action/nn.hpp` / `action/complex.hpp` / `action/gauge.hpp`). The `action/` tree is three families — `nn/` (scalar NN: leaves + `nn_action.hpp` + `formula/`), `complex/` (real base + `imag_part.hpp` mixin + leaf + `formula/`), `gauge/` — with the shared dimension-generic traversal engine once in `action/sweep/` (namespace `action::sweep`), concepts in `action/concepts.hpp`, gauge groups in `math/group/` (namespace `math::group`), their Wilson kernels in `action/gauge/formula/`. See `docs/writing_an_app.md` §Adding an action. Concept failures at the `updater::Hmc` site point at the missing member — read the error.
- **Add a new app**: copy an existing one in `apps/` and edit. Register it in `apps/CMakeLists.txt`. The universal pre-loop scaffolding (the `L,size` / `seed` / `workspace` / `out` flag block + workspace/writer open) is single-sourced in `include/reticolo/app/setup.hpp` (`app::common_flags` + `app::open_writer`) so it can't drift — reuse it; everything else (couplings, `ndim`, trajectory counts, and the trajectory `for` loop itself) stays in the app. If it's worth smoke-testing, add a `tests/apps/test_<name>_smoke.cpp` using `tests/apps/smoke_helpers.hpp`.
- **Add a new orchestrator**: put it in its own `orch/<name>/` subfolder as a client of the spine — never fold its specifics into `orch/`. Define a `Worker` type (a self-contained simulation unit satisfying `orch::Worker`; add `field()`/`rng()`/`save_extra`/`load_extra` if it should checkpoint) and a driver that loops over `orch::parallel_workers(workers, plan, body)` with a serial drain, reusing `orch::plan_threads` + `orch::save_ensemble`. Both shipped orchestrators are two-phase `Orchestrator` objects (`setup(out)` = threading/build/resume/IO, `run(...)` = the waves + serial drain): `orch/span/` (parameter span, app-built workers) is the minimal template; `orch/llr/` synthesises its replica ladder in `setup()` and shows exchange as a serial coupling between waves. The spine stays physics-free.
- **Add a new test**: pick `tests/unit/`, `tests/physics/`, or `tests/io/`. Register it in `tests/CMakeLists.txt` with `reticolo_add_test(<name> <relpath>)`. Canonical patterns (force-vs-FD, reversibility) are in `docs/writing_an_app.md`.
- **Multi-file refactors**: pause and confirm the plan before bulk-building. Don't run `cmake --build` across a large refactor without an explicit "build it".
