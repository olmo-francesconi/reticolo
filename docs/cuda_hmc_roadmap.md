# CUDA HMC roadmap — full, generic port

The concrete engineering roadmap for porting the **HMC trajectory** to CUDA.
Companion to [`cuda_extension_plan.md`](cuda_extension_plan.md) (backend-wide
plan) and [`cuda_architecture.md`](cuda_architecture.md) (hotspot/memory
analysis). This document commits to a device HMC that is **generic across
integrators and precisions, and as generic across actions/fields as the
physics actually allows** — and is honest about where that genericity stops.

> Reviewed against the codebase and by GPU-computing and methods lenses.
> The first draft over-promised "zero per-action CUDA code via 3 access
> patterns + a pure functor." Code review found that several actions don't
> fit; the corrected, narrower thesis is below, with the leaks called out
> **(review)** where they bit.

## North star: the integrator/driver axis is genuinely free; the action axis is not

The CPU HMC is already generic and the port must not regress it:

- `alg::Hmc<A,R,Integ,Field,Scalar>` (`algorithm/hmc.hpp:89`) contains no
  action- and no integrator-specific code — it calls `action_.s_full`,
  `action_.compute_force`, `Integrator::run`, and field-generic atoms.
- `Integrator::run` (`algorithm/integrators.hpp:66-177`) contains no
  field-specific code — **verified**: it touches only `detail::kick_` /
  `detail::drift_`, which dispatch by field-type overload, and the kick
  coefficient is correctly `real_scalar_t<F>`. The matrix-link group-exp
  drift is *not* a special case in `run` — it is a `drift_field` overload
  (`integ_ops.hpp:50-59`). So `run` bodies are reused verbatim.

**The one genericity that fully holds — and is the biggest single win — is
that the integrator axis drops out of the device kernel matrix entirely.**
Kernels are force/action/reduce/axpy; none depend on the integrator (that is
host-level `Integ::run`). Adding `Omelyan4`, or any future integrator, costs
**zero** CUDA code. Preserve this seam at all costs.

**(review) What does *not* hold:** "every action is a pure per-site functor
over one of 3 access patterns." Concretely (file-verified):

| Action | Reality | Consequence |
|--------|---------|-------------|
| **OnSigma** | has **no `compute_force`** — Metropolis-only via `propose()`; HMC on the sphere needs RATTLE-style constrained integration that does not exist on CPU (`on_sigma.hpp:26-28`) | **Out of this roadmap.** Not an HMC action; the plain `field += c·mom` drift is wrong on the manifold. Tracked as a separate constrained-integrator workstream. |
| **XY** | per-bond term is nonlinear (`Σ sin/cos(θ_x−θ_y)`), so it **bypasses `visit_nn`** and reads each neighbour individually (`xy.hpp:61-104`) | needs a *per-neighbour* stencil, not the neighbour-**sum** stencil phi4/phi6/sine_gordon use |
| **BoseGas** | anisotropic split-last hopping (`reduce_fwd_split_last`, `visit_nn_split_last`), plus `s_imag`/`compute_force_imag` over a **time-slab**, plus `std::complex` (`bose_gas.hpp:100-279`) | needs split-last + time-slab access patterns *and* a `cuda::std::complex` element — its own workstream, not "a complex fork" |
| **U(1) gauge** | `compute_force` is a **scatter** into 4 links/plaquette — for *both* `CompactU1` (`compact_u1.hpp:148-176`) and `Wilson<U1>` (`gauge_group/u1.hpp:74-76`) | the device must **re-derive** U(1) force as a per-link **gather** (sum the 2(d−1) plaquettes through each link, recompute `sin`) — the gather-only invariant forbids the CPU scatter. So `Wilson<U1>` does **not** "validate the matrix path for free": U(1) matrix path is scatter, SU(N) is gather. |
| **SU(2)/SU(3)** | link-centric gather already (`su3.hpp` `compute_force_batched_`) | fits the gather skeleton; the *formula* shares, the CPU *slab code* does not (see expi note) |

So the access-pattern set is an **open taxonomy**, not three closed tags, and
the honest scope of "one source of truth for the physics" is **the per-site
formula, not the loop structure** (gather vs scatter, slab vs per-site differ
between CPU and device for U(1) and the group exp).

```
        ┌─────────────────────────────────────────────────────────┐
        │  GENERIC — written once, never per-action/integrator      │
        │  cuda::Hmc<A,R,Integ,Field>     (mirrors alg::Hmc)         │
        │  Integ::run  ── reused UNCHANGED via atom overloads ───────┤
        │  device atoms: drift_field / kick_ / flat_size            │
        │  generic skeletons, each (functor × thread-map policy):    │
        │    stencil / plaquette / reduce / axpy / expi / sample    │
        └───────────────┬───────────────────────────┬──────────────┘
                        ▼ ACTION PROTOCOL            ▼ FIELD PROTOCOL
          per-action POD functors over an          device field types
          OPEN access-pattern taxonomy             Scalar / Link / Matrix
          (sum-stencil, nbr-stencil, fwd-reduce,   (buffer + layout policy)
           split-last, time-slab, plaq-gather)
```

## The generic pillars

### 1. `cuda::DeviceField<T, Layout>` — one resident buffer type

One class template, three layout policies (`ScalarLayout`, `LinkLayout`,
`MatrixLayout<G>`); the policy supplies the flat element count and the
element→(site,dir,comp) decode. Sibling construction mirrors the host
`Lattice{phi.indexing()}` trick so `mom/force/old` share one topology; one
`cudaMalloc` per buffer for the whole run. `flat_size(DeviceField)` overload
feeds the unchanged integrator atoms. RAII owns alloc/free; move transfers
ownership (pointer value preserved — see graph invariants).

### 2. `DeviceTopology` — passed by value, not `__constant__`

**(review)** Hold shape/stride in the struct and **pass it by value as a
kernel argument** (it lands in the constant param bank and broadcasts
identically) rather than a process-global `__constant__` symbol — the
topology pool deliberately allows multiple shapes to coexist, and a global
symbol aliases across them and across the MD/measurement streams.

```cpp
struct DeviceTopology {
    int  ndim; int shape[RETICOLO_MAX_DIM]; long stride[RETICOLO_MAX_DIM]; long nsites;
    __device__ long next(long s, int mu) const;   // closed-form periodic
    __device__ long prev(long s, int mu) const;
};
```

**(review)** The M1 microbench is **integer-div ALU throughput vs L2 table
bandwidth**, *not* "broadcast vs divergence" — periodic wrap is a predicated
select (no warp divergence); the real cost of arithmetic indexing is the
per-site coordinate decode (ndim integer mods, ~20+ cyc each). Defaults:
**bit-mask wrap (`& (L−1)`) for power-of-two extents** (division vanishes,
arithmetic wins outright); precomputed `next_/prev_` tables for general
extents at d≥3.

### 3. Device atom overloads — the integrator-reuse seam

Overload `drift_field`/`kick_` for `DeviceField` so `Leapfrog/Omelyan2/
Omelyan4::run` compile and run unchanged. The device `kick_` must replicate
the host `if constexpr (requires … compute_force_and_kick)` branch
(`integrators.hpp:41-48`) exactly — the fused path is **per-action/group
gated** (`HasFusedKick`; `Wilson<G>` gates it behind `G::compute_force_and_kick`),
not uniform. Atom calls **enqueue** kernels on the Hmc's stream; latency is
reclaimed by graph capture (pillar 5), not by giving up the reuse.

### 4. The device action protocol — streaming functor + access-pattern policy

An action plugs in via a POD functor (couplings only — no `std::vector`
scratch, no `mutable` cache; those stay host-side) plus an `access_pattern`
tag selecting the skeleton.

**(review) The functor streams neighbours into its own accumulator — it does
NOT receive a materialized neighbour array:**

```cpp
struct Phi4Force {            // __host__ __device__, shared with CPU formula
    T kappa, lambda;
    __host__ __device__ void init(T phi);
    __host__ __device__ void accumulate(int mu, T fwd, T bwd);   // or (mu, nbr)
    __host__ __device__ T    finalize() const;                   // → force/site value
};
```

This is the difference between the abstraction being free and a 2× regression
on SU(3): a materialized `neighbours[]` becomes a runtime-indexed local-memory
array (spills the 18-double links × 2d staples); a streaming accumulator keeps
SU(3)'s staple sum in registers and degenerates to a scalar add for phi4.

**(review) Host-includability is load-bearing.** Apps own `main()` and
include `<reticolo/reticolo.hpp>` → `cuda::Hmc` → the POD functor header is
also compiled by the *host* compiler. Its `__host__ __device__` annotations
must be macro-gated to no-ops when `!defined(__CUDACC__)` (a tiny
`reticolo/cuda/macros.hpp`), with no transitive `<cuda_runtime.h>` on the
host path. This, alongside M1's Sleef isolation, is what keeps "apps hold
concrete types and own the loop" true.

### 5. Generic skeletons + graph capture

A small fixed set of skeletons, each parameterized by **(action functor ×
thread-mapping policy)** so one skeleton serves both light and heavy actions:

| Skeleton | Drives | Notes |
|----------|--------|-------|
| `stencil<F,Map,T,d>` | scalar force / fused kick | sum- and per-neighbour variants (XY) |
| `reduce_fwd<F,T,d>` | scalar `s_full` | f64 accumulate |
| `plaquette<G,Map,T,d>` | `Wilson<G>` force/action (gather) | U(1) = re-derived gather |
| `expi_lmul<G,T>` | matrix drift | fused single kernel (below) |
| `axpy<T>` | drift/kick atoms | field-agnostic |
| `device_reduce<F…,T>` | s_full / kinetic / observables | hand-rolled, fixed-config, `doubleN`, segmented-ready |
| `sample<Sampler,T>` | momentum fill | per-field Sampler functor |

**(review) Thread-mapping is a policy axis** (`ThreadPerLink` /
`WarpPerLink`): U(1) wants high occupancy / multiple links per thread; SU(3)
is register-bound and wants `__launch_bounds__` + possibly warp-per-link with
shuffle reduction. M5 selects per group; it stays "one parameterized
skeleton," not two hand-written kernels.

**(review) `expi_lmul` is a single fused kernel** (c0/c1, detQ, acos, sincos,
coefficients, `U←exp(iQ)·U` in registers via libdevice) — **not** a port of
the CPU 5-pass 14-slab `expi_lmul_slab`. The Cayley–Hamilton *formula* is
shared; the CPU slab *code* is not (it is structured for Sleef SIMD lanes).
State this as an explicit, accepted exception to "one source of truth."

**(review) `device_reduce` is hand-rolled, not cub-bent.** A fixed
launch-config block→grid reduction with a `doubleN` accumulator (a) fuses N
functors trivially, (b) needs no `TransformInputIterator` gymnastics, (c)
gives the **segmented** per-replica reduction M7 needs with no cross-replica
contamination, and (d) is run-to-run deterministic by construction
(reversibility requires identical forward/reverse order). Use cub only for
plain single-functor reductions. **The "fused s_full+kinetic in one pass" is
scalar-only** — for gauge, kinetic reduces over links (`nsites·d·2N²`) while
the action reduces over plaquettes (different grid); `reduce_H_` is two
reductions there, not one.

### Putting it together: `cuda::Hmc` mirrors `alg::Hmc` once

```cpp
template <class A, class R, class Integ = alg::integ::Leapfrog, class Field = …>
class Hmc {                      // generic; no action/integrator switches
    HmcResult step() {
        sample_momenta_();                    // sample<Sampler>; traj counter via DEVICE BUFFER
        copy_device(old_, field_);
        graph_.run([&]{                       // captures H0 … MD … H1
            h0_ = reduce_H_(field_, mom_);     // scalar-fused / gauge-split
            Integ::run(action_, field_, mom_, force_, tau_, n_md_);
            h1_ = reduce_H_(field_, mom_);
        });
        bool acc = mh_accept_(h1_ - h0_);     // host MH; ΔH over pinned mem
        if (!acc) copy_device(field_, old_);
        return {.dH = h1_-h0_, .accepted = acc};
    }
};
```

`sample_momenta_` (host `hmc.hpp:203-231` has six branches) collapses to one
`sample<Sampler>` skeleton + a per-field Sampler functor — the SU(N) sampler
scatters 8 Gell-Mann coords into the 18-real anti-hermitian layout with
variance `N(0,1/√2)`; the float-narrowing branch is dropped (its purpose was
matching the CPU RNG stream, which Philox doesn't anyway).

## Graph capture — the latency mechanism, and its one sharp trap

**Why.** Every kernel launch costs ~5–20 µs of CPU-side driver overhead
before the GPU does anything. With the reused integrator, one Omelyan4
trajectory at `n_md=20` enqueues ~180 atom launches plus the reductions; on a
small/bandwidth-bound lattice where each kernel *runs* for only a few µs, that
overhead is most of the wall-clock — you would be benchmarking the driver,
not the GPU. This launch storm is the price of the genericity (reusing
`Integrator::run` means it emits the same fine-grained atom calls it does on
CPU, now as launches).

**What a graph is.** A pre-recorded DAG of GPU operations built once and
replayed with a single `cudaGraphLaunch` — the per-op launch overhead is paid
at build time, not per trajectory. The trajectory is the ideal fit: the
*sequence* of operations is identical every trajectory; only the *numbers in
the buffers* change. Static topology, mutable data.

**How — stream capture.** You don't hand-build the DAG. You wrap the reused
integrator in `cudaStreamBeginCapture(stream) … cudaStreamEndCapture`: the
enqueued kernels are *recorded*, not executed, while the host C++ around them
runs normally. Capture the region **`H0 … Integ::run … H1`** and replay it per
trajectory, copying back only the ΔH scalar. Host control flow captures
correctly: the `for(step<n_md)` runs at capture time and emits an **unrolled**
node sequence; the `step+1<n_md ? dt : half_dt` ternary bakes a host scalar
into each node; `if constexpr` is compile-time and never reaches capture. This
is exactly why the reuse works — unmodified host integrator code, turned into
a replayable graph.

**(review) [CRITICAL invariant] Nothing that varies per trajectory may be a
baked kernel literal inside the captured region.** The Philox key includes
the trajectory counter; if that counter (or `dt`, if ever varied) is passed
as a kernel argument, capture freezes it and **every replay regenerates the
same momenta → a non-ergodic, silently-broken chain that still looks
thermalized.** Per-trajectory-varying scalars must be read by kernels from a
small device buffer that the host bumps between replays. (This is also why
`sample_momenta_` is best left *outside* the graph, or fed via buffer if
brought in.)

Supporting requirements (all owned by the `graph_` wrapper, which is the real
deliverable — not "reuse unchanged"):
- **cub/reduction temp storage pre-allocated once** outside capture, fixed
  size — otherwise capture fails or allocates mid-graph.
- **`set_spec` invalidates the cached `graphExec`.** `n_md` change → node
  count change → re-instantiate (ms-scale, document as expensive); `tau`
  change → cheap iff coefficients are buffer-fed (the critical-invariant fix
  covers this). The `tune_phi4_*` study drivers do change these.
- **Lifetime:** `graphExec` bakes raw `d_` pointers + topology-by-value;
  member-declaration order in `cuda::Hmc` must destruct `graphExec` **before**
  the fields/topology, and no `DeviceField` may be reallocated while a graph
  is live (run-lifetime buffers + move-preserves-pointer make this safe, but
  it is an unstated invariant — state it).

## Graphs vs. fusion: two orthogonal optimizations (and what "specialised integrators" should mean)

Graphs and kernel fusion fix **different** costs and are not alternatives:

- **Graphs remove launch overhead** but not memory traffic — between the
  force kernel and the next drift, `U/P/F` still round-trip through HBM.
- **Fusion removes inter-kernel HBM traffic** by keeping state in
  registers/shared across sub-steps — but does nothing graphs don't for
  launch count.

For bandwidth-bound scalar actions, the HBM round-trip can dominate *even
after* graphs. That — not launch overhead — is the only thing a specialised
kernel buys. So the design treats fusion as an **opt-in optimization layer
behind the same `launch_*` interface**, default off, justified by profiling
(Phase 2 reports both the launch-overhead fraction *and* the inter-kernel
HBM-traffic fraction; only a large latter justifies building a fused kernel).

The spectrum, with verdicts:

- **Per-integrator host orchestration in `.cu`** (hand-written
  `leapfrog_run`/`omelyan4_run` still launching separate kernels): gains
  nothing over graphs, loses the free-integrator property. **Never.**
- **Fused force+kick** (`compute_force_and_kick`): already planned; one
  full-lattice pass, no grid sync needed. The safe first rung.
- **Fused per-step**: blocked by physics — drift must update *all* sites
  before the next force reads neighbours, a grid-wide barrier a normal kernel
  can't cross. So ≥2 kernels/step (force-kick, drift) is the floor without
  cooperative launch.
- **Cooperative mega-kernel** (whole trajectory in one kernel, `grid.sync()`
  as the barrier, `(φ,p)` in registers across steps): the only thing that
  truly kills inter-step HBM traffic. Three teeth: (1) occupancy ceiling —
  the lattice must fit one co-resident wave, else fall back to multi-kernel;
  (2) neighbours still go through global/shared, so the win is partial; (3)
  register pressure makes it **counterproductive for SU(3)** (force alone is
  ~80–120 regs). Realistic scope: small/medium **scalar** lattices only.

**(decision) The integrator stays generic even inside a fused kernel — pass
the step schedule as DATA, not code.** Every integrator is a sequence of
`(KICK, c·dt)`/`(DRIFT, c·dt)` ops (Leapfrog `KDK`, Omelyan4 the 9-op
BABAB…); the host already computes these coefficients. A fused/mega-kernel
takes a small `step_schedule[]` array and loops over it (`grid.sync()` between
ops). Then `Omelyan4` is still zero CUDA code — a different schedule, not a
different kernel — and the action functor + access pattern remain the single
source of truth. **Hard rule: no `LeapfrogKernel`/`Omelyan4Kernel` types ever;
any fused kernel is templated on the action functor and consumes the
integrator schedule as data.** This is what keeps "specialised for speed" and
"project-wide generic" from being in conflict.

## Instantiation matrix

**(review)** The atom-reuse design removes the integrator from the matrix:
kernels are `action × group × scalar × ndim`, not `× integrator`. Keep it
small: **make `ndim` a runtime `DeviceTopology` field by default** (compile-time
ndim buys nothing for bandwidth-bound kernels and is a silent 3× multiplier);
use compile-time ndim *only* where M5 shows the SU(3) force benefits from an
unrolled direction loop for register allocation. Result ≈ 40 rows per kernel
family — same order as the `Series<T>` `extern template` set. Dispatch crosses
the nvcc TU boundary exactly like `Series<T>`: app sees an `extern launch_*`
decl; the `.cu` includes the functor header, selects the skeleton via
`if constexpr (same_as<typename A::access_pattern, …>)`, and is explicitly
instantiated.

## File layout

```
include/reticolo/cuda/
    macros.hpp            __host__/__device__ gating for host inclusion
    device_field.hpp      DeviceField<T,Layout> + layout policies
    device_topology.hpp   DeviceTopology (by-value) + pool
    atoms.hpp             drift_field/kick_/flat_size overloads (integrator seam)
    graph.hpp             graphExec wrapper: capture/replay, spec-invalidation, cub temp
    hmc.hpp               cuda::Hmc (generic)
    rng.hpp               Philox counter scheme + sample skeleton (buffer-fed traj)
    reduce.hpp            hand-rolled deterministic doubleN / segmented reduction
    action_concepts.hpp   HmcActionDevice + open access-pattern tags
    actions/*.hpp         per-action POD functors (host/device-shared FORMULA)
src/cuda/
    kernels_stencil.cu    stencil / reduce_fwd instantiations
    kernels_plaquette.cu  plaquette (gather) / expi_lmul
    kernels_axpy.cu       axpy, copy, sample
    launch.cu             explicit launch_* instantiations per (action,group,T)
```

## Phased roadmap (HMC-focused)

Exit criteria are deterministic and CI-legal unless tagged **[nightly]**
(short seeded chains or sims — kept out of `tests/` per CLAUDE.md). Device
tests are **literal ports** of the named CPU files in `tests/physics/`.

### Phase 0 — toolchain & skeleton  *(exit: empty `.cu` builds in CI)*
`reticolo::cuda` target, `RETICOLO_ENABLE_CUDA`, `cuda-linux` preset, nvcc
locked, `CUDA_CHECK` + per-launch `cudaGetLastError()`, pinned-host helper,
the `graph_` wrapper skeleton, `macros.hpp` host-include gating.

### Phase 1 — substrate + Sleef isolation + gauge-math proof  *(exit: device round-trips a field; `stencil`+`reduce_fwd` pass force-vs-FD on a dummy functor; gauge headers parse under nvcc; **device `expi_lmul` matches host `su3::expi_lmul_slab` on one config**)*
- **(review)** Guard/split `math/vec_libm.hpp`. Correct scope: this gates
  **every nvcc TU that transitively includes `vec_libm`** — `rng.hpp`,
  `metropolis.hpp`, `sine_gordon.hpp`, `compact_u1.hpp`, `*_ops.hpp`, the
  gauge groups, *and the umbrella* — not "everything gauge."
- `DeviceField`/`DeviceTopology`/pool; `axpy`/copy; hand-rolled `device_reduce`.
- Prove Phi4 per-site formula + SU(3) matmul compile `__host__ __device__`.
- **(review)** Land the device-vs-host `expi_lmul` math test **now** (pure
  math, no HMC) — CLAUDE.md flags the Cayley–Hamilton path as historically
  buggy and Phase 5 otherwise has no earlier proxy (Wilson<U1> nc=1 is a
  trivial scalar phase, not su(N) exp).

### Phase 2 — generic HMC end-to-end on Phi4 (f64) + graphs  *(the flagship)*
Exit criteria:
- Reversibility at roundoff (port `test_hmc_reversibility.cpp`).
- Force-vs-FD (port `test_phi4_force_consistency.cpp`).
- **Integrator-order slopes 2/2/4** — **(review)** this is **f64-only** and a
  *seeded short-chain ensemble average* (the CPU `test_hmc_integrator_order.cpp`
  averages |ΔH| over 64 traj on 4⁴); cheap with graph replay, runs in CI, but
  classify it honestly (not single-shot). This is the integrator-genericity
  proof.
- **(review) "Zero integrator-specific CUDA code" as a lint gate**, not a
  reviewer claim: grep `src/cuda/*.cu` for `Leapfrog|Omelyan|integ::` → empty;
  `static_assert` no `cuda::integ::` namespace exists; `cuda::Hmc` instantiates
  `alg::integ::*`.
- **(review) Graph-replay-vs-eager equivalence**: one trajectory eager vs via
  replay from the same `(q0,p0)` → roundoff-identical (catches stale-pointer /
  baked-counter capture bugs).
- **(review) RNG reproducibility**: same Philox counter ⇒ identical bytes
  (host-computed vs device), and replay does **not** repeat momenta (guards
  the CRITICAL capture invariant).
- **(review) Performance baseline**: record traj/s for the reference Phi4
  lattice to a tracked file — the regression anchor for later phases. Report
  **both** the launch-overhead fraction (graphs target) and the inter-kernel
  HBM-traffic fraction (fusion target); the latter alone decides whether the
  opt-in mega-kernel is worth building.
- **(review) [nightly]** `⟨exp(−ΔH)⟩=1` + free-field acceptance vs `erfc(dt)`
  — force-vs-FD + reversibility can all pass while a wrong momentum variance
  or broken MH biases the ensemble; this is the only joint catch.

**Status: DONE** (16/16 CUDA gates green on Tesla P100, sm_60, CUDA 12.8, GCC-13).
Includes 2d (full-MD graph capture/replay-vs-eager) and **2e — host-free
trajectory streaming**: the *entire* trajectory (sample → H0 → MD → H1 →
device-side Metropolis accept → conditional rollback → device counter bump) is
captured in one CUDA graph and replayed with a single `cudaGraphLaunch`. Nothing
returns to the host between trajectories; `cuda::Hmc::run(k)` replays `k`
trajectories with zero host syncs, the host touches the chain only at
measurement (`sync()`). The MH accept draws its own Philox uniform on a separate
key stream (`seed ^ salt`); the trajectory counter is device-resident and bumped
inside the captured body (capture-trap-safe). Determinism gate: two `run(32)`
chains from the same seed agree bit-for-bit.

**Performance baseline (regression anchor)** — Phi4 HMC, f64, τ=1.0, n_md=10,
Leapfrog, host-free `run(k)` steady state, Tesla P100-PCIE-16GB:

| dims | V | ms/traj | traj/s |
|------|------|---------|--------|
| 4⁴   | 256       | 0.144 | 6964 |
| 8⁴   | 4096      | 0.152 | 6590 |
| 12⁴  | 20736     | 0.208 | 4813 |
| 16⁴  | 65536     | 0.518 | 1932 |
| 24⁴  | 331776    | 2.062 | 485  |
| 32⁴  | 1048576   | 6.260 | 160  |

Read: small volumes are **launch-bound** — the ~0.144 ms/traj floor is the graph
replay itself (n_md×(force + 2 kicks + 2 drifts) + 4 reductions + accept/resolve,
all sub-saturation), and host-free streaming dropped it from the ~0.17 ms/traj
device-scalar-reduction baseline (~15%, the removed per-step host sync). Large
volumes are **compute/HBM-bound** — 6.26 ms at 32⁴ is the force+reduction work,
where graphs/host-free barely move the needle (as expected). The launch-overhead
fraction (graphs target) is what 2d/2e attacked and is now amortized; the
inter-kernel HBM-traffic fraction (the fusion / opt-in mega-kernel decision) is
still unmeasured — defer to an Nsight pass before committing to fusion. Block
size is a uniform untuned 256 everywhere (no `__launch_bounds__`, no per-GPU
tuning) — a known ±10–20% knob, also deferred to profiling.

### Phase 3 — scalar coverage  *(exit: force-vs-FD + reversibility per action; [nightly] ⟨e^−ΔH⟩)*
- **Phi6, SineGordon, XY — DONE (3a–3c).** Each is a `device_functors<Action<T>>`
  specialization + a functor pair calling a shared `RETICOLO_HD` per-site formula
  in `action::detail::<name>_formula.hpp`; the CPU actions were rewired to call
  the same formulas (one source of truth). Phi6 is pure-poly so it matches the
  device to roundoff (1e-10); SineGordon and XY are transcendental and match to a
  bounded tolerance (1e-9 — device `sin`/`cos` vs Sleef/libm, ~1 ULP/site). The
  shared formulas take the transcendental as an argument (SineGordon) so the CPU
  f64 path keeps its Sleef-batched `sin`/`cos` while the algebra stays single-
  source. Gates: `DeviceAction<{Phi6,SineGordon,Xy}>` vs CPU `s_full` + force in
  `scalar_probe.cu`. **Correction to the original plan: XY needed NO second
  stencil** — `stencil_kernel`/`reduce_fwd_site_kernel` already call
  `accumulate(mu, nbr)` per-bond (passing each neighbour value, not a pre-summed
  one), so XY's `sin(θ−nbr)` accumulation drops straight in. The "sum-stencil vs
  per-neighbour" distinction the plan worried about never materialized.
- **3d — f32 DONE.** The device HMC stack instantiates in single precision: the
  MD drift/kick run in field precision (`axpy_f32`), the field/momenta are
  `DeviceField<float>`, but the volume reductions still accumulate the sum (and
  sum-of-squares) in **double** — a float volume sum loses ~log2(V) bits and
  corrupts ΔH, exactly as on the CPU. The reduce kernels are templated on input
  type (`block_sum/sumsq_kernel<In>`) with f32 + f64 entry points; `cuda::Hmc`
  and `DeviceAction` were already generic over `Field` and needed no change. Two
  added entry points beyond the kernels: `axpy_f32` and an f32-momenta
  `reduce_sumsq_into` overload; `fill_normals` became a template. Gates (bounded
  tolerance, NOT roundoff — and no order-scaling test, the f32 caveat):
  `DeviceAction<Phi4<float>>` vs CPU `s_full`+force (1e-5 / 1e-4), and f32
  Leapfrog reversibility (1e-3). f32 for the transcendental/other scalar actions
  is a free follow-on (same functors, `<float>`); only Phi4 is gated so far.
- **(remaining) BoseGas is its own sub-workstream**: `cuda::std::complex`,
  split-last + time-slab patterns, `s_imag`/`compute_force_imag` — not a drop-in
  functor.

### Phase 4 — U(1) gauge — DONE  *(exit: force-vs-FD + reversibility on `CompactU1`; [nightly] plaquette at known β)*
`LinkLayout` field (`ndim·nsites`, direction-major = host `LinkLattice` order, so
a flat raw-pointer round-trip is exact); `gauge_u1.cuh` plaquette skeleton with
the **U(1) force re-derived as a per-link gather** — `plaq_force_gather_kernel`
(one thread per link, `F = −β·Σ_{ν≠μ}[sin Q_ν(x) − sin Q_ν(x−ν)]`, reads
neighbours, writes only its own link) and `plaq_energy_site_kernel` (per-site
forward-plane `β·Σ(1−cos)` → `reduce_sum`). The CPU scatter is **not** ported.
- **Unified `DeviceAction`.** Rather than a separate gauge wrapper, the
  access pattern moved into the `device_functors<HostAction>` trait as static
  launchers (`compute_force` / `s_full` / `s_full_into`); `DeviceAction` is now a
  thin generic shell that delegates and never branches, constrained by a
  `DeviceActionTraits` concept (clean error on a mis-shaped trait). Scalar traits
  wrap the site-stencil skeletons (`site_launchers.hpp`); the gauge trait wraps
  the plaquette skeletons. `cuda::Hmc` is unchanged — it ran on
  `DeviceField<double, LinkLayout>` with no modification (momentum sampling,
  reductions, device accept all generic over the buffer).
- **One source of truth:** the plaquette *angle* is the shared `RETICOLO_HD`
  helper `action::detail::u1_plaq` (CPU `CompactU1` rewired to it); the cos/sin
  and the loop structure (gather vs the CPU's Sleef-batched scatter) deliberately
  differ, as the gather-only device invariant requires.
- **Gates (P100):** `DeviceAction<CompactU1>` vs CPU `s_full` (1e-10) + force
  (1e-7); gather force vs central FD of `s_full` (independent of the CPU);
  reversibility (roundoff); host-free `cuda::Hmc` smoke over the link field.
  Plaquette-at-β equilibration is a simulation → nightly only, not a gate.
- Validates the gather plaquette path before SU(N) — but **does not** validate
  the matrix path (`Wilson<U1>` is also scatter).

**Cross-action throughput baseline (P100, host-free `cuda::Hmc`, Leapfrog
τ=1.0 n_md=10, ms/traj):**

| action | L=8 (V=4096) | L=16 (V=65536) | L=32 (V=1.05M) |
|---|---|---|---|
| Phi4 f64       | 0.151 | 0.518 | 6.257 |
| Phi4 f32       | 0.150 | 0.499 | 5.615 |
| Phi6 f64       | 0.153 | 0.519 | 6.251 |
| SineGordon f64 | 0.161 | 0.533 | 6.379 |
| Xy f64         | 0.200 | 0.648 | 7.157 |
| CompactU1 f64  | 0.299 | 2.119 | 29.984 (DOF = 4·V) |
| Wilson<SU2> f64 | 0.559 | 7.006 | 121.929 (DOF = 32·V) |
| Wilson<SU3> f64 | 1.349 | 16.133 | 270.884 (DOF = 72·V) |

Reading (the input for any fusion/tuning decision): **the kernels are
HBM-bandwidth-bound, not compute-bound** — Phi6 ≈ Phi4 at every volume (a free
degree-6 term proves the ALU is idle waiting on memory). Small V (L=8) is
launch-bound at the ~0.15 ms graph-replay floor (per-action cost hidden). f32
helps only at large V (L=32 Phi4: 5.61 vs 6.26 ms, ~11% — the bandwidth regime);
at small V it is a wash. Cost ordering at L≥16: Phi4≈Phi6 < SineGordon (sin/cos)
< Xy (2 transcendentals/bond) < CompactU1 (4× DOF + plaquette gather). Block size
is still the untuned 256 everywhere — an Nsight occupancy pass is the obvious
lever now that we know it is memory-bound.

### Phase 5 — SU(2)/SU(3) gauge  *(exit: force-vs-FD + reversibility f64 at 1e-9 (port `test_su3_hmc_reversibility.cpp`); momentum-moment test for Gell-Mann variance + anti-hermiticity; [nightly] ⟨e^−ΔH⟩ + plaquette at β; **perf target met**)*
- **SU(2) — DONE** (31/31 CUDA gates green on Tesla P100). `MatrixLayout<SU2>`
  (nc=8, `[ndim][nc][nsites]` = host `MatrixLinkLattice<SU2>` order, flat copy
  exact). Generic SU(N) kernels (`gauge_sun.cuh`) templated on a device-traits
  type `GD = group_device<G>::type` (`SU2Device` in `gauge/su2_device.cuh`,
  RETICOLO_HD register-local 2×2 ops validated vs `math::su2`): per-site
  plaquette energy, per-link staple-gather force `scale·TA[U·V]`, group-exp drift
  `U←exp(dt·P)·U` (ADL `drift_field` atom over `MatrixLayout<G>`), Gell-Mann
  momentum sampler (h_a~N(0,½)). `device_functors<Wilson<SU2>>` plugs into the
  unchanged `DeviceAction`/`Hmc`; momentum sampling is now a 4th trait launcher
  (`sample_momenta`) so the scalar/U(1) iid-normal draw and the gauge algebra
  draw share one `Hmc` path. Gates in `src/cuda/probes/su2_probe.cu`: device-ops-vs-CPU,
  s_full+force vs CPU `Wilson<SU2>` (1e-9), MD energy conservation (2nd-order
  ratio), reversibility (1e-9), momentum moments, host-free HMC smoke. The matrix
  `drift_field` overload re-proves the integrator seam for matrix fields (Phase 2
  only proved it for scalar atoms).
- **SU(3) — DONE** (37/37 CUDA gates green on Tesla P100). Same generic SU(N)
  kernels with `GD = SU3Device` (`gauge/su3_device.cuh`, nc=18, n_gen=8): the
  register-local 3×3 ops and the Morningstar-Peardon Cayley-Hamilton group exp
  are copied verbatim from `math::su3` (RETICOLO_HD), validated bit-faithful by
  `su3_probe.cu` (device-ops-vs-CPU, s_full+force vs CPU `Wilson<SU3>` 1e-9,
  energy conservation, reversibility, momentum moments=n_gen, host-free HMC).
  Zero new kernels and no `DeviceAction`/`Hmc` change — SU(2)→SU(3) is purely a
  new traits struct + `group_device<SU3>` specialization.
- **GPU-vs-CPU baseline (Wilson<SU{2,3}>, Leapfrog), all measured.** Normalized
  to `link·force-evals/s` (links × force-evals/s, n_md/integrator-invariant; M1
  single core, n_md=20 MD-only — single-replica HMC is serial by design — vs P100
  n_md=10 full host-free HMC). The M1 falls out of cache at L≥16 so the speedup
  *rises* with volume and then plateaus:

  | action | 4D L=8 | 4D L=16 | 4D L=32 |
  |---|---|---|---|
  | SU(2): M1 ms → P100 ms | 33.6 → 0.56 | 780.6 → 6.97 | 13821 → 121.9 |
  | SU(2): speedup (link·fe/s) | ~31× | **~59×** | **~59×** |
  | SU(3): M1 ms → P100 ms | 112.1 → 1.35 | 2522 → 16.13 | 42599 → 270.9 |
  | SU(3): speedup (link·fe/s) | ~44× | **~82×** | **~82×** |

  M1 link·fe/s: SU(2) 1.02e7 (L=8, in cache) → 6.4e6 (L=32); SU(3) 3.07e6 →
  2.07e6. P100 link·fe/s: SU(2) ~3.8–4.1e8, SU(3) ~1.7–1.8e8. SU(3)'s denser 3×3
  + Cayley-Hamilton work gives the larger GPU win. Still bandwidth-bound: on the
  P100 at L=16, SU(3)/SU(2) = 16.13/6.97 = 2.31× ≈ the DOF ratio 18/8 = 2.25× —
  even the heavy matrix exp is hidden behind HBM traffic. Block size is the
  untuned 256 everywhere; an Nsight occupancy pass is the obvious lever.
- **DEFERRED** — Profile `ThreadPerLink` vs `WarpPerLink` + shared-mem staple
  staging. Not started; the bench shows the win is in occupancy/block-size, not
  the thread-mapping (kernels are bandwidth-bound, one thread per link).
- **DONE (concrete target)** — measured GPU-vs-CPU above (~59× SU(2) / ~82× SU(3)
  vs one M1 core, n_md-normalized). The remaining lever is the untuned 256 block
  (Nsight occupancy pass), tracked as the deferred tuning item — not a blocker.
- SU(N) links stay f64 (group-drift policy). ✓
- **DONE** — this phase re-proved the integrator seam for matrix fields (the
  `MatrixLayout<G>` `drift_field` group-exp overload, selected over the generic
  axpy drift by partial ordering); Phase 2 only proved it for scalar atoms.

### Phase 6 — observables, measurement overlap, checkpoint/restart — DONE
*(exit: CUDA app HDF5 matches host schema; only scalars cross PCIe;
**checkpoint round-trip**: device field + RNG counter → HDF5 → resume equals an
uninterrupted run)*

**Status: DONE** (39/39 CUDA gates green on Tesla P100). Done **natively**, not
through a façade: the CUDA apps are plain `.cu` twins of the host apps
(`apps/cuda/phi4_cuda_hmc.cu`, `apps/cuda/su2_cuda_hmc.cu`) that include the full
`<reticolo/reticolo.hpp>` + `<reticolo/cuda/cuda.hpp>` umbrellas, use
`cuda::Hmc` + `DeviceField` for the
device work and `io::Writer` for output, and keep the trajectory for-loop
plainly in `main()`. `io::Writer` PIMPLs HDF5, so nvcc never sees `<hdf5.h>`; it
links the prebuilt `reticolo::io` archive. The enabling fix was guarding the
`-Wall`/`-Werror` interface flags behind `$<COMPILE_LANGUAGE:CXX>` in
`ReticoloWarnings.cmake` (+ OpenMP OFF so no `-fopenmp` leaks onto nvcc).

- **Observables** — reduced on-device (`cuda::reduce_sum_f64`/`reduce_sumsq_f64`);
  only the scalar action + magnetisation moments cross PCIe, measured per
  host-free block of `meas_every` trajectories.
- **Checkpoint/restart** — `io::save_config_counter` / `load_config_counter`
  write `/field` + `/rng@{seed,counter}` + `/traj@i`. Because the device RNG is
  counter-based Philox, restoring (seed, trajectory counter) + field continues
  the chain **bit-for-bit** (Test #71: resumed run == uninterrupted run, exactly).
- **New gates**: Test #71 (checkpoint round-trip, bit-exact) and Test #89
  (phi4_cuda_hmc HDF5 schema, mirrors `test_phi4_hmc_smoke`).
- **(review)** First build to compile the full umbrella + io + apps under nvcc;
  it surfaced four latent issues now fixed: `std::views::zip` (C++23 lib,
  unavailable to nvcc `-std=c++20`) in `obs::analysis`; a `string_view`→`char*`
  bug in `Writer::field` for matrix links (only instantiated via SU(N)
  checkpoint); an umbrella-only `act::` alias used without the umbrella; and the
  `reticolo::log` vs CUDA `::log` ambiguity from a namespace-scope
  using-directive.
- **Deferred**: second-stream measurement overlapping the next trajectory (the
  current path syncs between blocks). Cheap to add when measurement cost matters.

### Phase 7 — hardening & docs  *(exit: full deterministic suite green under `RETICOLO_ENABLE_CUDA`; nightly physics harness green)*
`docs/writing_a_cuda_app.md`; update `architecture.md` with the backend;
consolidate the nightly harness.

> **LLR replica batching is out of scope here** — it sits on top of a working
> generic device HMC (M7 in `cuda_extension_plan.md`). The generic `cuda::Hmc`
> + hand-rolled **segmented** `device_reduce` are designed so a batched replica
> axis is an outer grid dimension over the same kernels, not a rewrite.

## Cross-cutting invariants (every phase)

1. **No `switch(action)` / `switch(integrator)` in device code** — compile-time
   functor + explicit instantiation; enforced by the Phase-2 lint gate.
2. **Gather only — no `atomicAdd`** in force/action/reduce (reversibility).
   U(1) force is re-derived as a gather; the CPU scatter is not ported.
3. **Deterministic reductions** — hand-rolled fixed launch config, f64
   accumulation; cub only for plain single-functor reductions.
4. **f64 for all reductions and group-exp coefficient math**, independent of
   storage precision; SU(N) links f64.
5. **One source of truth = the per-site FORMULA**, shared as a `__host__
   __device__` functor. **(review)** The *loop structure* may legitimately
   differ CPU↔device (U(1) scatter→gather; expi slab→fused); these are the
   only sanctioned divergences and each has a device-vs-host math test.
6. **(review) Nothing per-trajectory-varying may be a baked kernel literal in
   a captured region** — trajectory counter, `dt`, coefficients are
   device-buffer-fed. The chain's ergodicity depends on it.
7. **CUDA headers never leak into `reticolo::core`; Sleef never leaks into a
   `.cu`; `__host__ __device__` is macro-gated for host inclusion.**
8. **(review) No `DeviceField` realloc while a graph is live; `graphExec`
   destructs before its fields/topology.**

## Key decisions to lock early

| Decision | Default | Decide by |
|----------|---------|-----------|
| nvcc vs clang-cuda | nvcc | Phase 0 |
| topology by-value vs `__constant__` | by-value kernel arg | Phase 1 |
| index wrap: bit-mask / arithmetic / gathered table | bit-mask (pow2), table (general d≥3) | Phase 1 microbench |
| `ndim` template vs runtime | runtime (compile-time only for SU(3) force) | Phase 1 |
| graph capture vs explicit graph-builder | capture + buffer-fed scalars | Phase 2 |
| kernel fusion beyond force+kick (mega-kernel) | off; opt-in if HBM-traffic fraction dominates after graphs | Phase 2 profiling → revisit |
| fused-kernel integrator handling | schedule-as-data (never per-integrator kernels) | design rule |
| reduction: hand-rolled vs cub | hand-rolled (fused/segmented), cub (single-op) | Phase 1 |
| SU(3) thread mapping | TBD (`ThreadPerLink`/`WarpPerLink` policy) | Phase 5 profiling |
| f32 link support | no (f64 only) | revisit post-Phase 5 |
