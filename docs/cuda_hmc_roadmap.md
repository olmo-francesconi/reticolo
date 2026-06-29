# CUDA HMC roadmap — full, generic port

The concrete engineering roadmap for porting the **entire HMC trajectory** to
CUDA. Companion to [`cuda_extension_plan.md`](cuda_extension_plan.md) (the
backend-wide plan) and [`cuda_architecture.md`](cuda_architecture.md) (the
hotspot/memory analysis). This document is narrower and deeper: it commits to
a *generic* device HMC — one that is agnostic to the action, the integrator,
the field type, and the precision — and lays out the phases, types, kernel
skeletons, and exit criteria to build it.

## North star: keep the genericity the CPU already has

The CPU HMC is already fully generic and the port must not regress that:

- `alg::Hmc<A, R, Integ, Field, Scalar>` is templated on action, RNG,
  integrator, field, and scalar (`algorithm/hmc.hpp:89`). It contains **no
  action-specific and no integrator-specific code** — it calls
  `action_.s_full`, `action_.compute_force`, `Integrator::run`, and the
  field-generic atoms.
- `Integrator::run(action, field, mom, force, tau, n_md)` (Leapfrog /
  Omelyan2 / Omelyan4, `algorithm/integrators.hpp`) contains **no
  field-specific code** — it calls only `detail::kick_` and `detail::drift_`,
  which dispatch on the field type via overload.
- The field types (`Lattice`, `LinkLattice`, `MatrixLinkLattice`) plug into
  those atoms through `flat_size` / `drift_field` / `kick_add` overloads and
  the `compute_force_and_kick` fused path.

**The design principle for the port: preserve these three seams.** The device
backend supplies (1) device field types, (2) device overloads of the
integrator atoms, and (3) a device action protocol — and the *existing*
integrator and driver logic is reused, not reimplemented per action or per
integrator. Anything that forces a `switch(action)` or `switch(integrator)`
in device code is a design failure.

```
        ┌─────────────────────────────────────────────────────────┐
        │  GENERIC (written once, never per-action/integrator)      │
        │                                                           │
        │  cuda::Hmc<A,R,Integ,Field>   ── mirrors alg::Hmc         │
        │     step(): sample → snapshot → H0 → Integ::run → H1 → MH │
        │                         │                                 │
        │  Integ::run  ───────────┤ reused UNCHANGED via atom       │
        │  (Leapfrog/Omelyan2/4)  │ overloads on cuda::Field        │
        │                         ▼                                 │
        │  device atoms: drift_field / kick_add / reduce / sample   │
        │  generic kernel skeletons (stencil / plaquette / reduce)  │
        └───────────────┬───────────────────────────┬──────────────┘
                        │ plugs in via               │ plugs in via
                        ▼ ACTION PROTOCOL            ▼ FIELD PROTOCOL
        ┌───────────────────────────┐   ┌──────────────────────────┐
        │ per-action device functors │   │ device field types       │
        │  Force, Action, Sampler    │   │  Scalar / Link / Matrix  │
        │  (pure math, M1-shared)    │   │  (buffer + access policy) │
        └───────────────────────────┘   └──────────────────────────┘
```

## The five generic pillars

### 1. `cuda::DeviceField<T, Layout>` — one resident buffer type

A single class template covers all three field shapes; the differences are a
**Layout policy**, not separate classes:

```cpp
// include/reticolo/cuda/device_field.hpp  (plain C++, host-callable)
template <class T, class Layout>
class DeviceField {
    T*                       d_;       // device buffer, run-lifetime
    std::shared_ptr<DeviceTopology const> topo_;  // shared like Indexing
    // Layout∈{ScalarLayout, LinkLayout, MatrixLayout<G>} supplies:
    //   - flat element count (nsites / nsites·d / nsites·d·2N²)
    //   - element → (site, dir, comp) decoding for kernels
public:
    T* data();   std::size_t size() const;
    DeviceTopology const& topo() const;
    // sibling ctor from another field's topo_ — mirrors the Lattice pattern
    explicit DeviceField(std::shared_ptr<DeviceTopology const>);
};
```

- **Sibling construction** mirrors the host `Lattice{phi.indexing()}` trick:
  `mom/force/old` share one `DeviceTopology`. One allocation per buffer for
  the whole run; nothing is freed/reallocated inside the trajectory.
- `flat_size(DeviceField)` overload feeds the *unchanged* integrator atoms.
- RAII owns the `cudaMalloc`/`cudaFree`; copy is deleted, move transfers.

### 2. `DeviceTopology` — periodic indexing, shipped once

The host `Indexing` neighbour tables become device-side. Two options, decided
in M1 by microbench; the *interface* is fixed either way:

```cpp
struct DeviceTopology {
    int  ndim;
    int  shape[RETICOLO_MAX_DIM];
    long stride[RETICOLO_MAX_DIM];
    long nsites;
    // Option A: arithmetic wrap, computed per thread (no extra loads)
    // Option B: int* d_next/d_prev gathered (mirror Indexing::next_/prev_)
    __device__ long next(long s, int mu) const;   // closed-form periodic
    __device__ long prev(long s, int mu) const;
};
```

Shape/stride live in `__constant__` memory (or are passed by value). Periodic
BCs are unbranched, so `next/prev` are closed-form — Option A is the default
unless gather wins on a target arch. Pooled by shape like `Indexing`.

### 3. Device atom overloads — the integrator reuse seam

The integrators call `detail::drift_(field, mom, c)` and
`detail::kick_(action, field, mom, force, k)`. Provide overloads for
`DeviceField` so **`Leapfrog::run` / `Omelyan2::run` / `Omelyan4::run`
compile and run unchanged on device fields**:

```cpp
// include/reticolo/cuda/atoms.hpp
template <class T, class L>
void drift_field(DeviceField<T,L>& f, DeviceField<T,L> const& p, double c) {
    launch_drift_axpy(f.data(), p.data(), c, f.size(), stream_);   // enqueue
}
template <class A, class T, class L>
void kick_(A const& act, DeviceField<T,L>& f, DeviceField<T,L>& m,
           DeviceField<T,L>& force, double k) {
    if constexpr (HasDeviceFusedKick<A>) act.force_and_kick(f, m, k); // 1 launch
    else { act.compute_force(f, force); launch_kick_axpy(...); }
}
// MatrixLayout overload: drift_field → launch_expi_lmul (group exp), per dir
```

This is the crux of "generic, not per-integrator": adding `Omelyan4` cost
**zero** CUDA work because its `run()` body already only touches atoms.
New integrators are likewise free.

> The naive cost of this reuse is one kernel launch per atom call (the launch
> storm). **CUDA Graph capture reconciles reuse with latency** (pillar 5):
> the first trajectory's atom launches are *captured*, not executed eagerly;
> subsequent trajectories replay the graph. The generic code is written once
> and the launch overhead is paid once.

### 4. The device action protocol — small, pure, M1-shareable

An action plugs into the generic kernels by exposing **per-element device
functors**, not loops. This is the only thing an action author writes, and it
is the per-site math that M1 proves is `__host__ __device__`-shareable with
the CPU (so there is one source of truth for the physics):

```cpp
// A device action satisfies cuda::HmcAction if it provides:
template <class A>
concept HmcActionDevice = requires {
    typename A::access_pattern;          // NnStencil | FwdReduce | Plaquette
    // per-element force:   F(local_value, gathered_neighbours) -> value
    // per-element action:  S(local_value, fwd_neighbours)      -> double
    // per-element sampler: draw momentum with the correct variance/structure
};
```

- `access_pattern` selects which **generic kernel skeleton** (pillar belowin
  this list, item 5's kernels) drives the functor — NN stencil for scalars,
  forward-reduce for `s_full`, plaquette for gauge. The action never writes a
  loop, a gather, or a wrap.
- The functor is a POD (couplings only) — satisfies the "stripped POD action"
  requirement; the host keeps the `last_s_full_`/scratch members.
- Adding an action = a functor file + one explicit-instantiation line per
  (precision) — mirrors `Series<T>`'s `extern template` set.

### 5. Generic kernel skeletons + RNG — written once, parameterized by functor

A *small fixed set* of kernel skeletons covers every action; each is a
template over the action functor, so no action gets its own kernel:

| Skeleton | Drives | Used by |
|----------|--------|---------|
| `stencil_nn<F,T,d>` | gather 2d neighbours, call `F` per site | all scalar/sigma force + fused kick |
| `reduce_fwd<F,T,d>` | gather d fwd neighbours, reduce `F` → scalar (f64) | all scalar `s_full` |
| `plaquette_force<G,T,d>` | gather plane links, gather-staple + TA, per link | all `Wilson<G>` force |
| `plaquette_reduce<G,T,d>` | gather plane links, reduce `ReTr U_p` (f64) | all `Wilson<G>` action |
| `axpy<T>` | `a += c·b` | drift/kick atoms (field-agnostic) |
| `expi_lmul<G,T>` | fused group exp left-multiply, per link | matrix drift atom |
| `device_reduce<F,T>` | generic block→grid f64 reduction, deterministic | s_full, kinetic, observables |

- **All gather, no scatter** (reversibility invariant): every output element
  owns its sum. No `atomicAdd` anywhere in force/action/reduce.
- **`device_reduce` is one generic deterministic reduction** (`cub`, fixed
  launch config, f64 accumulation) reused for `s_full`, kinetic, and every
  observable. `s_full` + kinetic are fused into one pass over the site grid.
- **Generic Philox momentum fill:** one `sample_momenta<Sampler>` skeleton; a
  per-field-type `Sampler` functor encodes the variance/structure (scalar
  `N(0,1)`; su(N) Gell-Mann `N(0,1/√2)` + anti-hermitian layout). Counter
  layout `(seed,draw_kind,traj | site,dir,comp,pair)` is fixed once, used by
  every action — no per-action RNG code, no carried Box–Muller spare.

### Putting it together: `cuda::Hmc` mirrors `alg::Hmc` once

```cpp
// include/reticolo/cuda/hmc.hpp  — generic; no action/integrator switches
template <class A, class R, class Integ = integ::Leapfrog,
          class Field = /* deduced from A::access_pattern + scalar */>
class Hmc {
    HmcResult step() {
        sample_momenta_();                       // generic Philox skeleton
        copy_device(old_, field_);               // async D2D
        double h0 = reduce_H_(field_, mom_);     // fused s_full+kinetic (f64)
        graph_.launch_or_capture([&]{ Integ::run(action_,field_,mom_,force_,tau_,n_md_); });
        double h1 = reduce_H_(field_, mom_);
        bool acc = mh_accept_(h1 - h0);          // host MH, pinned ΔH copy
        if (!acc) copy_device(field_, old_);
        return {.dH = h1-h0, .accepted = acc};
    }
};
```

The body is line-for-line the host `step()` with device atoms. `Integ::run`
is the unmodified host integrator. This is the whole point: **one generic
device HMC, reused across all actions and all integrators.**

## File layout

```
include/reticolo/cuda/
    device_field.hpp      DeviceField<T,Layout> + layout policies
    device_topology.hpp   DeviceTopology + pool
    atoms.hpp             drift_field/kick_/flat_size overloads (integrator seam)
    hmc.hpp               cuda::Hmc (generic)
    rng.hpp               Philox counter scheme + sample_momenta skeleton
    reduce.hpp            device_reduce (deterministic, f64)
    action_concepts.hpp   HmcActionDevice + access_pattern tags
    actions/*.hpp         per-action POD functors (host/device shared math)
src/cuda/
    kernels_stencil.cu    stencil_nn / reduce_fwd instantiations
    kernels_plaquette.cu  plaquette_force / plaquette_reduce / expi_lmul
    kernels_axpy.cu       axpy, copy, sample_momenta
    launch.cu             explicit launch_* instantiations per (action,group,T)
```

## Phased roadmap (HMC-focused)

Each phase has a hard exit criterion (a passing deterministic test, per the
tiering in the extension plan — `tests/` is math-only; sims are nightly).

### Phase 0 — toolchain & skeleton  *(exit: empty `.cu` builds in CI)*
- Add `reticolo::cuda` STATIC target, `RETICOLO_ENABLE_CUDA`, `cuda-linux`
  preset, nvcc locked, `CUDA_CHECK` macro + per-launch `cudaGetLastError()`.
- Pinned-host-memory helper; a single owned `cudaStream_t` per `cuda::Hmc`.

### Phase 1 — generic substrate + Sleef isolation  *(exit: device round-trips a field; `stencil_nn`+`reduce_fwd` skeletons pass a force-vs-FD test on a trivial functor; gauge headers parse under nvcc)*
- Guard/split `math/vec_libm.hpp` (`#ifndef __CUDACC__` or a Sleef-free
  arithmetic header) — **gates everything gauge**.
- `DeviceField` + `DeviceTopology` + pool; `axpy`/copy kernels;
  `device_reduce` (deterministic, f64).
- `stencil_nn` / `reduce_fwd` skeletons with a dummy functor.
- Prove the Phi4 per-site formula + SU(3) matmul compile `__host__ __device__`.

### Phase 2 — generic HMC end-to-end on Phi4 (f64) + graphs  *(exit: device reversibility test at roundoff; integrator-order test gives slopes 2/2/4 for Leapfrog/Omelyan2/Omelyan4 — proving integrator-genericity; force-vs-FD passes)*
- `cuda::Hmc`, atom overloads, Philox `sample_momenta`, fused
  `reduce_H_`, host MH over pinned ΔH.
- CUDA Graph capture/replay of `Integ::run`.
- New app `apps/phi4_hmc_cuda.cpp` (copies `phi4_hmc.cpp`, swaps types).
- **All three integrators must pass here with zero integrator-specific CUDA
  code** — this is the genericity gate.

### Phase 3 — scalar/sigma action coverage  *(exit: force-vs-FD + reversibility green for each new action; one app each)*
- Phi6, SineGordon (libdevice transcendentals), XY via the *same* skeletons —
  each is a functor + instantiation line, no new kernels.
- O(N) sigma: add the AoS→SoA `MatrixLayout`-style policy (explicit task).
- f32 instantiations for the bandwidth-bound scalar actions.
- BoseGas (complex): `cuda::std::complex` fork — its own sub-task.

### Phase 4 — U(1) gauge  *(exit: force-vs-FD + reversibility on `CompactU1`; plaquette at known β in nightly harness)*
- `LinkLayout` device field; `plaquette_force`/`plaquette_reduce` skeletons
  with the U(1) sin-scatter expressed as **gather**.
- Targets `CompactU1`/`LinkLattice` first; `Wilson<U1>` (MatrixLayout, nc=1)
  validates the matrix path for free before SU(N).

### Phase 5 — SU(2)/SU(3) gauge  *(exit: force-vs-FD + reversibility f64; momentum-moment test for Gell-Mann variance/anti-hermiticity; Nsight occupancy/throughput gates met)*
- `MatrixLayout<G>` device field; gather-staple `plaquette_force`; fused
  single-kernel `expi_lmul` (registers, libdevice — *not* the 14-slab CPU
  multi-pass).
- Profile thread-per-link vs warp-per-link + shared-mem staple staging.
- SU(N) links stay f64 (group-drift policy); f32-links deferred behind
  accepted-config reunitarization if ever needed.

### Phase 6 — on-device observables  *(exit: a CUDA app's HDF5 matches the host schema; only scalars cross PCIe)*
- `obs::*` reductions via `device_reduce`; measurement overlaps the next
  trajectory on a second stream.

### Phase 7 — hardening & docs  *(exit: full deterministic suite green under `RETICOLO_ENABLE_CUDA`; nightly physics harness green)*
- `⟨exp(−ΔH)⟩=1`, Gaussian-limit acceptance vs `erfc(dt)`, CPU-vs-GPU
  observable equivalence (nightly, not CI-gating).
- `docs/writing_a_cuda_app.md`; update `architecture.md` with the backend.

> **LLR replica batching is explicitly out of scope for this HMC roadmap** —
> it builds *on top of* a working generic device HMC and is tracked as M7 in
> `cuda_extension_plan.md`. The generic `cuda::Hmc` is designed so a batched
> replica axis is an outer grid dimension over the same kernels, not a
> rewrite.

## Cross-cutting invariants (apply to every phase)

1. **No `switch(action)` / `switch(integrator)` in device code.** Dispatch is
   compile-time via functor + explicit instantiation. A violation means the
   generic seam was broken.
2. **Gather only — no `atomicAdd`** in force/action/reduce (reversibility).
3. **Deterministic reductions** (fixed cub config, f64 accumulation).
4. **f64 for all reductions and group-exp coefficient math**, independent of
   storage precision; SU(N) links f64.
5. **One source of truth for the physics:** per-site math is the same
   `__host__ __device__` functor the CPU uses; divergence between CPU and GPU
   formulas is forbidden.
6. **CUDA headers never leak into `reticolo::core`**; Sleef never leaks into a
   `.cu`.

## Key decisions to lock early

| Decision | Default | Decide by |
|----------|---------|-----------|
| nvcc vs clang-cuda | nvcc | Phase 0 |
| topology: arithmetic wrap vs gathered arrays | arithmetic | Phase 1 microbench |
| graph capture vs fused mega-kernel | graph capture | Phase 2 |
| SU(3) force thread mapping | TBD | Phase 5 profiling |
| f32 link support | no (f64 only) | revisit post-Phase 5 |
</content>
