# CUDA architecture analysis

A map of where reticolo spends compute and memory bandwidth, and how the
data structures and trajectory loop are shaped — written as the groundwork
for a CUDA backend. Read [`architecture.md`](architecture.md) first; this
document assumes its vocabulary (`Lattice`, `Indexing`, `Hmc`, the action
concepts, LLR replicas).

## The one structural fact that drives everything

A Monte Carlo run is a **Markov chain**: trajectory `n+1` depends on the
accepted state of trajectory `n`. The outer `for` loop is therefore serial
in MC time and cannot be parallelised. Every bit of exploitable parallelism
lives in one of three places:

- **Site / link-parallel** — inside a single force / action / integrator
  sweep, all sites (or links) are independent except for a final reduction.
  This is the GPU bread-and-butter.
- **Replica-parallel** — LLR runs N independent replicas, each its own HMC.
  Today this is the only OpenMP axis; on a GPU it is a batch/grid dimension.
- **Colour-parallel** — Metropolis even/odd sublattices are independent
  given the other colour frozen.

Serial and not portable: the trajectory chain itself, Wolff cluster DFS,
replica exchange.

## Hotspot ranking

Per-trajectory cost for single-replica HMC:

```
sample momenta        O(V)          bandwidth-bound, RNG
─ per MD step (×N_md, ×{1,2,4} integrator stages) ─
   compute_force       O(V)·f        ◄── THE HOTSPOT (≈90% for gauge)
   drift  (U += dt·P)  O(V)          pure axpy, bandwidth-bound
   kick   (P += dt·F)  O(V)          pure axpy (fused into force for some actions)
─────────────────────────────
ΔH = kinetic + s_full  O(V)          reduction
accept / reject        O(V)          flat-buffer copy on reject
```

Descending arithmetic weight (numbers are order-of-magnitude per site or
per link, double precision):

| Rank | Kernel | Action | ~FLOPs | Arith. intensity | Character |
|------|--------|--------|--------|------------------|-----------|
| 1 | `compute_force` (staples + TA[U·V]) | Wilson SU(3) | ~2400 | ~0.15 F/B | compute-bound, register-heavy |
| 2 | `expi_lmul_slab` (drift, Cayley–Hamilton exp) | SU(3) | ~200 multi-pass | ~0.15 F/B | transcendental (sincos/acos) |
| 3 | `compute_force` | Wilson SU(2) | ~200 | ~0.25 F/B | matrix-mul |
| 4 | `s_full` plaquette `ReTr U_p` | SU(3) | ~72 | ~0.13 F/B | bandwidth-bound |
| 5 | `compute_force` (sin scatter) | CompactU1 / Wilson U(1) | ~10 | ~0.6 F/B | transcendental, bandwidth |
| 6 | `compute_force` NN stencil | Phi4/Phi6/SineGordon/XY/BoseGas | ~10–20 (+1 sin) | low | memory-bound |
| 7 | drift / kick (`axpy`) | all | 2 | ~0.1 F/B | pure bandwidth |
| 8 | observables (mean / m2 / m4 / two_point) | all | O(V) reductions | low | bandwidth-bound |

**Conclusion:** SU(2)/SU(3) gauge is the only compute-bound corner (and the
SU(3) force is *register*-bound in practice — one thread holding the link,
staple accumulator, and matmul temporaries needs ~80–120 live f64 registers,
so occupancy, not the FLOP ceiling, gates it). Everything scalar and every
integrator atom is **memory-bandwidth bound**, so a GPU port wins on HBM
bandwidth first, FLOPs second.

The FLOP figures are order-of-magnitude per site/link; the SU(3) force in
particular is ~2400 FLOP/link (3 planes × fwd+bwd × ~2 complex 3×3 matmuls +
U·V + TA), revised up from an earlier under-count.

## Memory architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Indexing  (immutable, pooled by shape via process-wide weak_ptr map)       │
│   next_[nsites·ndims]  prev_[nsites·ndims]   (Site indices, layout s·d+μ)   │
│   parity_[nsites]      even_[] / odd_[]                                     │
│   ── ONE per shape; shared by all sibling lattices via shared_ptr ──        │
└──────────────────────────────────────────────────────────────────────────┘
        ▲ shared_ptr                  ▲                        ▲
┌───────┴────────┐         ┌──────────┴─────────┐   ┌──────────┴─────────────┐
│ Lattice<T>     │         │ LinkLattice<T>     │   │ MatrixLinkLattice<G,T> │
│ scalar field   │         │ U(1) links         │   │ SU(N) links            │
│ data: vector<T>│         │ data: vector<T>    │   │ data: vector<T>        │
│  one T / site  │         │ [ndim][nsites]     │   │ [ndim][2N²][nsites]    │
│  AoS-of-scalar │         │ direction-major    │   │  SoA per-component slab │
│  T∈{double,    │         │  (θ per link)      │   │  flat=((μ·2N²+k)·ns)+s │
│   float,       │         └────────────────────┘   │  stride-1 over sites ◄──┼ coalescing-ready
│   array<,N>,   │                                   └────────────────────────┘
│   complex}     │
└────────────────┘

HMC sibling buffers (share ONE Indexing, separate data arrays):
   field U ── momentum P ── force F ── old_field (rollback snapshot)
   (mom/force/old built from field.indexing(); no neighbour-table rebuild)
```

Layout facts that matter for CUDA:

- **Scalar fields are AoS** (`vector<T>`, one element/site, x-fastest
  row-major). Stencil reads are stride-1 in the inner axis → coalescing
  friendly as-is. `std::array<double,N>` (O(N)) is interleaved AoS and
  will want a transpose to SoA on device.
- **SU(N) links are already SoA** (`[μ][2N²][nsites]`, each real component
  a contiguous nsites slab). This is the GPU-ideal layout — straight
  `cudaMemcpy` + coalesced access, no repacking.
- **Neighbour table is explicit** (`next_/prev_` index arrays), not
  computed. On device, either ship the arrays (gather) or recompute the
  periodic wrap arithmetically per thread (no indirection, cheaper) — BCs
  are periodic-only and unbranched, so the index math is closed-form.
  Caveat: arithmetic wrap saves the index-array bandwidth but does **not**
  fix coalescing — neighbour reads along the fastest axis are contiguous,
  but slow-axis neighbours are strided by the sub-volume regardless. Lean on
  L2 reuse (adjacent sites share neighbours) and, for SU(N), shared-memory
  staging of staple sub-products.
- **U(1) has two distinct field layouts.** `CompactU1` runs on
  `LinkLattice<T>` (direction-major, one angle per link); `Wilson<U1>` runs
  on `MatrixLinkLattice<U1,T>` with `n_real_components = 1`. A device backend
  must pick one path per app.
- **Force kernels must be gather, not scatter.** For HMC reversibility the
  force must be a deterministic function of `U`; `atomicAdd` scatter makes it
  order-dependent and silently breaks detailed balance. The explicit
  `next_/prev_` layout already lets each output site/link sum its own
  contributions with zero atomics.
- The CPU stencils split each sweep into **bulk (stride-1, fixed offsets)
  \+ peeled wrap slab**. On GPU this is one kernel with modulo wrap, or a
  bulk kernel + a boundary kernel.

## Computation architecture

```
                         APP main()  (owns the for loop)
        ┌───────────────────────┴───────────────────────┐
   SINGLE-REPLICA                                     LLR ENSEMBLE
   for traj in 1..N:                          llr::Driver
     updater.step()                            #pragma omp parallel for (replicas) ◄ REPLICA-PARALLEL
     obs → Series → HDF5                         each Replica: phi, rng, HMC,
        │                                          WindowedAction(base, a, E_n, δ)
   ┌────┴─────┬───────────┬──────────┐            thermalize → sample → update a
   │ Hmc      │ Metropolis │ Wolff    │           ── serial even/odd EXCHANGE ──
   └────┬─────┴─────┬──────┴────┬─────             P=min(1,exp((a_i−a_j)(E_i−E_j)))
        │           │           └─ DFS cluster: SEQUENTIAL, path-dependent
        │           └─ even/odd sweep: RNG-serialized accept; checkerboard
        │              batches exp() but accept stays serial. SITE-PARALLEL/colour.
        ▼
   ┌──────────────────── HMC trajectory (the GPU target) ─────────────────────┐
   │ sample_momenta_   RNG normal_fill / group algebra sample      [O(V)]       │
   │ snapshot old_field                                            [O(V) copy]  │
   │ h0 = kinetic_() + action.s_full()                            [O(V) redux]  │
   │ Integrator::run  (Leapfrog | Omelyan2 | Omelyan4 — TYPE PARAM)             │
   │   repeat N_md × {1,2,4} stages:                                            │
   │      kick_:  compute_force(U,F) then P += k·F   ◄── SITE/LINK-PARALLEL     │
   │              (or fused compute_force_and_kick — no F buffer)               │
   │      drift_: U += c·P  (scalar)  |  U ← exp(i·c·P)·U  (SU(N), per μ)       │
   │ h1 = kinetic_() + action.s_full()                            [O(V) redux]  │
   │ accept iff ΔH≤0 or rand<exp(−ΔH); reject → restore old_field  [O(V)]       │
   └────────────────────────────────────────────────────────────────────────────┘

   RNG: FastRng (xoshiro256++, 32-byte state) — value type, per-replica/thread,
        copies diverge. normal_fill batches Box–Muller via Sleef sincos.
   Vector libm: Sleef (sin/cos/sincos/acos/exp batch) — the CPU SIMD path a
        CUDA backend replaces with libdevice / cuRAND.
```

## Parallelism axes, summarised

| Axis | Kernels | GPU mapping |
|------|---------|-------------|
| Site / link-parallel | `compute_force`, `s_full`, drift, kick, momentum sample, observables | one thread per site (scalar) or per link (`nsites·ndim`); reduction via cub/thrust |
| Colour-parallel | Metropolis even/odd | per-thread RNG, one kernel per colour |
| Replica-parallel | LLR ensemble | grid/stream batch dimension; saturates a GPU one lattice can't |
| Serial (not portable) | trajectory chain, Wolff DFS, replica exchange | stays host-side or single-thread |

## Implications for the backend

The clean target is a **device-resident HMC trajectory**: keep `U`, `P`,
`F`, `old_field` in device memory for the whole run and never copy per step.
The host touches the field only at measurement cadence (`meas_every`),
ideally by reducing observables on-device and copying back only scalars.

The architecture already supplies the two things a GPU needs — **flat
contiguous buffers** and **SoA gauge layout** — so `compute_force`, `drift`,
`kick`, and `s_full` map 1:1 to kernels over `nsites` / `nsites·ndim`. The
friction point is the header-only template design: device kernels can't be
arbitrary per-app template instantiations the way the CPU path is, so the
backend needs explicit kernel instantiations per (action, group, precision).
See [`writing_a_cuda_app.md`](writing_a_cuda_app.md) for the implementation pattern.
</invoke>
