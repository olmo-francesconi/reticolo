# CUDA extension plan

A staged plan to add an optional CUDA backend to reticolo without breaking
the architectural invariants in [`architecture.md`](architecture.md). Read
[`cuda_architecture.md`](cuda_architecture.md) first — it justifies every
target chosen here (device-resident trajectory, SoA gauge layout already in
place, the site/replica parallelism axes).

> This document was reviewed against the codebase and by GPU-computing and
> lattice-QFT lenses. Where the first draft was optimistic, the corrected
> position is stated inline and tagged **(review)**.

## Design goals (and the invariants they must respect)

1. **Optional, isolated, off by default.** CUDA is a fourth CMake target
   that the core never depends on — analogous to how `reticolo::io`
   quarantines `<hdf5.h>`. A build without a CUDA toolchain is unaffected;
   `RETICOLO_ENABLE_CUDA=OFF` is the default everywhere except a new preset.
   **(review)** The analogy is asymmetric and must be handled: io works
   because *nothing in core includes `<hdf5.h>`*, but the reverse is not
   true for core's own host-only deps — `reticolo_core` (INTERFACE) links
   `sleef` and several action headers `#include <sleef.h>` + SIMD intrinsics
   transitively (`CMakeLists.txt:116-119`, `math/vec_libm.hpp:6-13`).
   Keeping `<cuda_runtime.h>` out of core is trivial; **keeping core's Sleef
   baggage out of the `.cu` reuse path is the actual hard part** (see M1).
2. **Same physics, same schema.** A CUDA app writes the identical HDF5
   layout (`/run@*`, `/<phase>/stats`, `/<phase>/obs`) via the existing host
   `io::Writer`. Only the trajectory compute moves to the device.
3. **Concepts over virtual, still.** No virtual dispatch reaches a kernel.
   Device updaters are class templates constrained by analogous action
   concepts; dispatch from app to kernel is resolved at compile time per
   (action, group, precision). The compiler stays the source of truth.
4. **Apps still own the for loop.** A CUDA app is its own `main()` — copy a
   host app, swap field/updater types for their `cuda::` mirrors, keep the
   visible trajectory `for`. No generic GPU driver.
5. **Device-resident trajectory.** `U/P/F/old_field` live on the device for
   the whole run. Host↔device traffic happens only at `meas_every`, and
   carries reduced scalars, not whole fields, wherever possible.
6. **Determinism is a correctness invariant, not a perf detail. (review)**
   HMC is exact only if the force `F(U)` is a deterministic function of `U`
   and the integrator is exactly reversible. Therefore: **force/action
   kernels are gather-only (each output owns its sum) — no `atomicAdd`
   scatter**, and **all reductions use a fixed, deterministic launch config**
   (`cub::DeviceReduce`, not atomics). f64 accumulation does *not* rescue a
   non-deterministic reduction order — f64 addition is non-associative, so a
   varying tree shape makes the reverse trajectory fail to retrace the
   forward one and silently violates detailed balance. The existing SoA +
   explicit `next_/prev_` layout already supports atomic-free gather.

## Target & build integration

A new target alongside the existing three:

- `reticolo::cuda` — **STATIC**, compiled by `nvcc` (lean toward nvcc over
  clang-cuda: cub/thrust, `__nv_*` libdevice intrinsics, and CUDA Graphs
  ergonomics are better supported; the clang-tidy/format integration gain of
  clang-cuda is secondary). The only target that includes
  `<cuda_runtime.h>` / cuRAND / cub. The umbrella `reticolo::reticolo` links
  it only when `RETICOLO_ENABLE_CUDA=ON`. Structurally this mirrors the io
  target wiring (`src/io/CMakeLists.txt`, `CMakeLists.txt:185-190`) cleanly.

```
RETICOLO_ENABLE_CUDA      (default OFF; new preset `cuda-linux` sets it ON)
CMAKE_CUDA_ARCHITECTURES  (e.g. 80;90 — Ampere/Hopper)
```

`.cu` translation units live under `src/cuda/`. A device-side header tree
under `include/reticolo/cuda/` carries host-callable wrapper types
(`cuda::Field`, `cuda::Hmc`, …) that are plain C++; kernels are defined in
the `.cu` TUs and reached through explicitly instantiated `launch_*`
functions, so apps including `<reticolo/reticolo.hpp>` never feed CUDA
syntax to the host compiler.

## What can be shared with the CPU code — and what must fork

**(review — this section replaces the first draft's optimistic "annotate the
math `__host__ __device__`" claim.)** Ground truth from the headers:

- **Genuinely shareable (pure arithmetic, no Sleef, no `std::vector`):**
  - `Phi4`'s per-site force/action *formula* (`phi4.hpp:73-146`) — pointer
    arithmetic only. (The surrounding `visit_nn` host iteration is *not*
    reused; you re-express the loop as a kernel.)
  - The SU(2)/SU(3) **batched matmul / `plaq_re_tr` / staple / TA-projection
    formulas** (`su2.hpp:189-298`, `su3.hpp:176-291`, `su{2,3}_ops.hpp`
    mul/adj/TA) — stack arrays only. These are not yet `__host__ __device__`
    or `constexpr`; annotating them is real work but the math ports.
- **Blocked / must fork to libdevice:**
  - **Everything transcendental.** The `double` paths of `SineGordon`,
    `CompactU1`/`Wilson<U1>`, and the gauge **drift (`expi_lmul_slab`) and
    momentum sampling (`sample_algebra_slab`)** call `math::*_batch` → Sleef
    (`sine_gordon.hpp:62-133`, `u1.hpp:122-256`, `su{2,3}_ops.hpp`). These
    fork to libdevice `sincos`/`acos`/`exp`.
  - **The Sleef chokepoint.** `math/vec_libm.hpp:6-13` includes `<sleef.h>`
    + intrinsics *unconditionally* and defines an `inline` namespace-scope
    Sleef-calling initializer. It is pulled transitively by every gauge
    action header. **M1's first job is to guard it (`#ifndef __CUDACC__`) or
    split the Sleef-free arithmetic into its own header**, or no gauge header
    compiles under nvcc at all.
  - **`std::complex` (BoseGas, complex momenta in `hmc.hpp`).** Not usable in
    device code → needs `cuda::std::complex` and a `norm`/sampling fork.
    BoseGas is therefore a non-trivial fork, *not* a routine M3 instance.
- **Action structs cannot be reused as-is. (review)** `Phi4`/`SineGordon`
  carry `mutable std::vector<T> scratch` and `mutable double last_s_full_`
  (`phi4.hpp:149-153`), making them non-trivially-copyable — they cannot be
  passed by value into a kernel or live in device memory. The device side
  uses a **stripped POD action** holding only the couplings (e.g.
  `{kappa, lambda}`); the cache/scratch members stay host-side.

## How dispatch stays compile-time

- Kernels are templated on `<Action, Group, Scalar, ndim>` and **explicitly
  instantiated** in `src/cuda/*.cu` for the supported combinations — the same
  pattern as `Series<T>`'s `extern template` set in `writer.cpp`.
- Host-side `cuda::Hmc<A,R,Integ>` is a **parallel reimplementation** of
  `alg::Hmc`, not an instantiation of it on device buffers: it keeps ΔH/ΔS
  scalars host-side (exactly as `alg::Hmc` already does, `hmc.hpp:134-168`)
  and routes `s_full`/`compute_force`/drift to `launch_*` wrappers. It need
  not model the optional `HasSFullCache` concepts — rollback is the host
  buffer copy + host `s0`, so there is no hidden device-cache coupling.
- Adding a new action = POD action struct + the shared/forked per-site math +
  one explicit-instantiation line. No string dispatch, no registry.

## RNG: counter-based Philox, fully specified

The CPU `FastRng` (xoshiro256++) is one sequential stream per replica;
reproducing its bit sequence across thousands of threads is pointless. Use
**counter-based Philox** (cuRAND `philox4_32_10`). This breaks bit-identity
with the CPU path *by design* — validate by physics, not stream identity.

**(review) The counter layout is a correctness invariant — specify it, don't
hand-wave "keyed by site."** Keying by `site_index` alone collides the
multiple independent normals a single site needs (SU(3): 8 Gell-Mann coords
× ndim directions per site) → correlated momenta → biased HMC. Define:

- High words (key): `(seed, draw_kind, trajectory)`.
- Low words (128-bit counter): a collision-free packing of
  `(site, direction, generator_component, pair_index)`.

This guarantees every independent draw has a unique counter with **no
per-thread carried state**. Two consequences that are also invariants:

- **Box–Muller pair-at-a-time.** The CPU caches the spare normal
  (`rng.hpp`); on device, generate normals in pairs per thread and never
  carry a spare (carried state reintroduces divergence and breaks replay).
- **Momentum variance must match the kinetic term exactly.** Scalar:
  `P ~ N(0,1)` for `K=½P²`. SU(N): the Gell-Mann convention
  `Tr λ_aλ_b = 2δ_ab`, `h_a ~ N(0, 1/√2)` (see `su3_ops.hpp` sampling
  comment) reproduced on device. A wrong variance biases the canonical
  ensemble while still "looking thermalized" — caught only by the
  `⟨exp(−ΔH)⟩=1` gate below, so add a momentum-moment test (mean 0, exact
  variance, traceless/anti-hermitian structure for su(N)).

## Kernel inventory

Each maps to a CPU hot loop characterised in `cuda_architecture.md`.
**All force/action kernels are gather-only (no atomics).** Reductions are
deterministic cub with f64 accumulation. Fuse where it removes a launch.

| Kernel | Mirrors | Launch dim | Notes |
|--------|---------|-----------|-------|
| `force_scalar<A,T,d>` | `A::compute_force` | nsites | arithmetic periodic wrap; gather neighbours |
| `force_and_kick<A,T,d>` | `A::compute_force_and_kick` | nsites | fused; avoids F buffer + a streaming pass |
| `drift_axpy<T>` / `kick_axpy<T>` | `integ_ops` | nelem | pure bandwidth |
| `s_full_kinetic_reduce<A,T,d>` | `s_full` + `kinetic_` | nsites → 2 scalars | **fused** reduction, deterministic, f64 accum |
| `sample_momenta<T,G>` | `Hmc::sample_momenta_` | nelem | Philox; exact variance convention |
| `plaq_force<G,T,d>` | Wilson `G::compute_force` | **TBD by profiling (M5)** | gather staples + TA[U·V]; SU(N) register-bound — see below |
| `plaq_action<G,T,d>` | Wilson `s_full` | nsites·planes → scalar | gather `ReTr U_p` reduce |
| `expi_lmul<G,T>` | `G::expi_lmul_slab` | nlinks | **single fused kernel** (see below) |
| `obs_reduce<Obs,T>` | `obs::*` | nsites → scalar | deterministic; on-device at `meas_every` |

**(review) `expi_lmul` — do NOT port the CPU multi-pass.** The CPU
`expi_lmul_slab` is a ~5-pass, 14-slab design that exists only to feed Sleef
SIMD lanes (`su3_ops.hpp:731,808-828`). Porting it pass-by-pass means 5+
launches per drift and streaming 14·n doubles per pass — turning a
compute-bound exp into a bandwidth thrash. Fuse the whole Cayley–Hamilton
exp (`c0/c1`, `det Q`, `acos`, `sincos`, coefficients, `U ← exp(iQ)·U`) into
**one kernel, one link per thread, intermediates in registers** via
libdevice. The coefficient math stays f64 regardless of storage precision.

**(review) `plaq_force` SU(3) is register-bound — don't commit to
thread-per-link.** A thread assembling a full SU(3) link force holds U (18
regs) + staple accumulator V (18) + matmul temporaries → ~80–120 live f64
registers, spilling or capping occupancy well before the FLOP ceiling (the
real cost is ~2400 FLOP/link, higher than the architecture doc's order
estimate). At M5 measure two candidates: (a) thread-per-link with
`__launch_bounds__` accepting low occupancy (compute-bound tolerates it),
vs (b) warp-/quarter-warp-per-link with lanes cooperating on the 9 complex
outputs (matmul → warp-shuffle reduction). Stage shared neighbour links into
shared memory to reuse staple sub-products across the μ-loop.

Out of scope for the first passes (keep on host): Wolff cluster DFS
(sequential), replica exchange (cheap, serial), Metropolis accept loop
(parallelise per colour only if profiling demands it).

## Latency: CUDA Graphs are a first-class deliverable, not an afterthought

**(review)** With Omelyan4 (4 stages) × N_md × {force, drift, kick}, a
trajectory issues dozens of tiny launches. On the small lattices the tests
target — and across the whole bandwidth-bound scalar regime — 5–20 µs launch
latency per kernel dominates wall time; you measure the scheduler, not the
GPU. Therefore:

- **Capture the full MD trajectory (`Integrator::run`, including the fused
  reductions) into one CUDA Graph and replay it per trajectory.** Topology is
  static; only data changes. This is the single highest-leverage perf item
  for everything except large SU(3), and it lands in **M2**, not "open
  questions." A fully fused single-kernel MD (cooperative-groups grid sync)
  is a later alternative; graphs get ~90% of the benefit with far less risk.
- **Pinned (page-locked) host memory** for the ΔH copy-back and measurement
  transfers — otherwise every `cudaMemcpy` bounces through a pageable buffer
  on the per-trajectory critical path.
- **Overlap measurement** (`meas_every` reductions) with the next
  trajectory's MD on a second stream so measurement doesn't serialize the
  chain.
- **Error discipline:** wrap every CUDA/cuRAND/cub call in `CUDA_CHECK` and
  call `cudaGetLastError()` after each launch (at least in debug). A silent
  kernel error corrupts the Markov chain invisibly.

## Precision policy

- **All reductions and the Cayley–Hamilton coefficient math run in f64,
  always**, independent of storage precision. ΔH precision is non-negotiable
  for accept/reject; the CPU already casts per-component loads to double for
  this reason.
- **Storage precision is a knob only for the bandwidth-bound scalar
  actions.** f32 there is safe: H0/H1 are evaluated on near-identical configs
  so their f32 errors correlate and cancel in ΔH (the binding f32 limit is
  the *reversibility-test tolerance*, not acceptance).
- **(review) SU(N) links default to f64 even in "f32 mode."** A near-unitary
  exp truncated to f32 every drift step accumulates `U†U ≠ 1` / `det U ≠ 1`;
  the CPU avoids in-loop reunitarization only because f64 drift is ~1e-16.
  Drifted links lose group meaning → biased ensemble and broken
  reversibility. If f32 links are ever wanted, port `project_slab` and
  reunitarize **only on accepted configurations, outside the trajectory**
  (mid-trajectory projection is not symplectic and re-breaks reversibility),
  cadence driven by measuring `‖U†U−1‖` on device.
- f16/TF32 stay off the symplectic path entirely — disqualified by
  *reversibility* (a non-deterministic tensor-core reduction) before
  accuracy even enters.

## Milestones

Ordered to de-risk: prove the device-resident loop on the cheapest
bandwidth-bound action, then climb to compute-bound gauge, then scale out.

- **M0 — Build skeleton.** `reticolo::cuda` target, `RETICOLO_ENABLE_CUDA`,
  `cuda-linux` preset, nvcc decision locked, a trivial `.cu` that round-trips
  a device buffer. Establish `CUDA_CHECK` discipline. CI: new preset builds;
  existing presets untouched.
- **M1 — Sleef-isolation + nvcc-compile proof. (rescoped, review)** Guard or
  split `math/vec_libm.hpp` so gauge headers parse under nvcc; prove the
  *Phi4 per-site formula* and the *SU(3) batched matmul/`plaq_re_tr`/staple/
  TA* compile device-side (annotate `__host__ __device__`/`constexpr`). This
  gates how much math is shared vs forked — do it before committing M2+.
- **M2 — Phi4 device HMC (f64) + CUDA Graphs.** Full device-resident
  trajectory for the simplest action: POD action, gather force/drift/kick,
  fused deterministic s_full+kinetic reduction, Philox momenta (specified
  counter layout), host-side accept/reject (one ΔH scalar over pinned
  memory), MD trajectory captured as a CUDA graph. New app
  `apps/phi4_hmc_cuda.cpp`. **Proves the whole pipeline + the latency story.**
- **M3 — Scalar coverage + f32 + O(N) SoA.** Phi6, SineGordon (libdevice
  transcendentals), XY, and the O(N) sigma field — which needs an explicit
  AoS→SoA transpose, an actual task not an afterthought. BoseGas (complex)
  flagged as its own fork (`cuda::std::complex`). Add f32 scalar
  instantiations.
- **M4 — U(1) gauge.** **(review) Two paths exist:** `CompactU1` over
  `LinkLattice<T>` (direction-major) and `Wilson<U1>` over
  `MatrixLinkLattice<U1>` with `n_real_components=1`. Pick `CompactU1` first;
  validates the link device type + sin-scatter-as-gather force before SU(N).
- **M5 — SU(2)/SU(3) gauge.** The compute-bound payoff: gather `plaq_force`,
  `plaq_action`, fused `expi_lmul`. Profile thread-per-link vs warp-per-link
  + shared-mem staple staging here; set Nsight gates (achieved occupancy,
  memory throughput vs HBM peak, warp-stall reasons).
- **M6 — On-device observables.** Move `obs::*` reductions on-device so
  measurement streams only scalars to host.
- **M7 — LLR replica batching. (review)** Run the N replicas as a **batched
  grid axis** (`grid = nsites·ndim·nreplica`), *not* streams — LLR lattices
  are small by design, so one replica can't saturate the GPU and N streams
  fight the scheduler. The MD math is identical per replica (no warp
  divergence); only host-side accept/reject, exchange, and the `a`-update
  differ and stay on host. Per-replica reductions must be correctly
  **segmented** (no cross-replica contamination); assert a single-replica run
  equals one slice of the batched run.

## Validation & testing — tiered

**(review) The first draft's "thermalize then compare ⟨S⟩ within statistics"
is both too weak and a rule violation** (it is a full MC-to-equilibrium
simulation, which CLAUDE.md / `feedback_tests_no_simulations` forbids in
`tests/`, and an O(dt²) bias hides inside statistical error bars). Split the
tiers explicitly:

- **`tests/` — fast, deterministic, math-only (the CI gates):**
  - **Force-vs-finite-difference on device** (central difference,
    precision-scaled tolerance) — the single most valuable kernel test.
  - **Reversibility** — `N_md` steps forward, negate P, run back; assert
    `‖U_rev−U₀‖`, `‖P_rev+P₀‖` at roundoff (f64) / bounded tolerance (f32).
    This is the direct catch for non-deterministic force/reductions.
  - **Integrator-order scaling** — `⟨ΔH⟩ ∝ dt^{2k}`; fit the log-log slope,
    assert the exponent (2 / 2 / 4 for Leapfrog/Omelyan2/Omelyan4). Catches
    wrong Omelyan coefficients ported to device.
  - **Momentum-moment check** — mean 0, exact per-convention variance,
    traceless/anti-hermitian structure for su(N).
  - **Device-vs-host `S(U)` agreement** — same config, assert match
    (critical for LLR, where `S` scales the force every MD step).
  - HDF5 schema smoke (reuse `tests/apps/` pattern).
- **App-level / nightly GPU harness (NOT `tests/`):**
  - Physics-equivalence CPU-vs-GPU observables within statistics (smoke, not
    gate).
  - `⟨exp(−ΔH)⟩ = 1` (Kennedy–Pendleton) — powerful joint check of momentum
    normalization + integrator + accept consistency.
  - Plaquette at known β (strong/weak-coupling limits); Gaussian/free-field
    acceptance vs analytic `erfc(dt)` law.
  - Autocorrelation comparison as a gross smoke check only (different RNG
    legitimately gives different autocorrelation — never CI-gating).

Gate all CUDA tests behind `RETICOLO_ENABLE_CUDA`; add one GPU CI lane if a
runner is available.

## LLR-specific risks (M7)

Keeping exchange, the `a`-update (Newton-Raphson / Robbins-Monro), and DoS
reconstruction on host correctly insulates the CLAUDE.md watch-outs (window
coefficient `C=12` vs `C=1`, paper-vs-energy convention, `a_init` sign,
`(1+a)` reconstruction). **But the windowed force carries a factor
`(a + (S−E_n)/δ²)` multiplying the base force, so the scalar `S(U)` feeds
into the force every MD step.** Under the determinism invariant this makes a
non-deterministic `S` corrupt the *force itself*, not just ΔH — strictly
worse than for plain actions. Requirements: the `S(U)` scaling the force and
the `s_full` in ΔH come from the *same* deterministic kernel; assert
device-vs-host `S` agreement (value, normalization, sign) before wiring the
`a`-update, or the documented "geometric divergence" resurfaces.

## Open questions (resolve before/within M0–M1)

- **Toolchain:** leaning nvcc (above); confirm at M0 against the existing
  clang-tidy/clang-format discipline.
- **Multi-GPU / MPI:** out of scope, but the device-resident + replica-batch
  (M7) design is the right substrate; the replica axis maps cleanly to
  multi-GPU later.
- **Memory budget:** four resident buffers (`U/P/F/old`) at f64 — confirm the
  largest target lattice fits; make the fused force+kick path (drops `F`) the
  default on device to save one buffer + a streaming pass.
