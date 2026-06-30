# CUDA backend optimization plan

Actionable plan for a focused optimization session, derived from the Phase 9
profiling pass (`docs/cuda_hmc_roadmap.md` § Phase 9). **Read that section and
the two profiled SQLite analyses first.** Nothing here is implemented yet.

## What the profiling found (the starting point)

Both representative actions are **~68% force-kernel** at production volume and
scale linearly in V, but for *opposite* reasons:

| | phi4 L=64 (scalar) | su3 L=32 (gauge) |
|---|---|---|
| force registers/thread | 42 | **144** (drift `expi`: 156) |
| force occupancy (P100) | ~75% (6 blocks/SM) | **~12%** (1 block/SM) |
| bottleneck | **bandwidth** (4D-stencil locality) | **latency** (starved occupancy) |
| 2nd cost | s_full eval 14.7%, axpy 14.5% | drift+kick 26% |

Per-trajectory there is also a full-field **device→device rollback copy**
(`copy_device_(old←field)`): 134 MB (phi4) / 604 MB (su3) every trajectory.

P100 facts: 65,536 32-bit regs/SM, 2,048 threads (64 warps)/SM, ~732 GB/s HBM2,
~16 MB? no — 4 MB L2. `ncu` is **blocked on Kaggle** (`ERR_NVGPUCTRPERM`), so
occupancy/DRAM% are *inferred* from `registersPerThread` + grid/block in the
nsys SQLite, and bandwidth from the streaming-model in `profile_hmc.cu`.

## How to measure each change (the loop)

1. Edit the kernel(s).
2. Run the suite + reversibility on Kaggle (correctness gate — see invariants).
3. Re-profile: locally toggle `RETICOLO_PROFILE=1` in `tools/kaggle/run.py`
   (staged-only, revert after), `bash tools/kaggle/push.sh`. Pulls
   `tools/kaggle/output/profile/`.
4. Analyse: `python tools/profile/analyze.py tools/kaggle/output/profile`, and
   query the SQLite for `registersPerThread` / grid / per-kernel time:
   ```python
   # blocks/SM ≈ 65536 // (regs*blockThreads); occupancy ≈ blocks*blockThreads/2048
   ```
5. Compare `ms_per_traj` (throughput) and the force-kernel `regs`/occupancy.

**Correctness is non-negotiable and comes first every iteration:** keep the 47
deterministic gates green and HMC reversibility bit-exact. A faster kernel that
changes results is a regression. The invariants that constrain *how* we optimize:
gather-only (no `atomicAdd` — reversibility), deterministic fixed-config
reductions, one shared `RETICOLO_HD` per-site formula (CPU == GPU), no
integrator-specific kernels.

---

## Lever 1 — su3 occupancy — REFUTED (measured 2026-06-30)

**Outcome: the occupancy hypothesis is wrong; launch_bounds is counterproductive.**
A single-run `--lb-sweep` (profiler instantiates the force/drift kernels at 7
configs, regs+spill from `cudaFuncGetAttributes`) at su3 L=16/32:

| config (MaxT,MinB) | force regs | spill (B) | occ | force µs (L=32) |
|---|---|---|---|---|
| (256,0) baseline | 252 | 112 | 12.5% | **15 804** |
| (256,2) | 128 | 432 | 25% | 40 986 (2.6×) |
| (256,3) | 80 | 1128 | 37.5% | 94 576 (6.0×) |
| (256,4) | 64 | 1360 | 50% | 122 287 (7.7×) |
| (128,4/6/8) | = 256 equiv | | | = 256 equiv |

The SU(3) staple genuinely needs ~250 regs; capping to raise occupancy spills the
3×3 complex matrices to local memory and the spill traffic dwarfs the gain — the
kernel is **spill/register-bound, not occupancy-latency-bound** (the "Risk" case
below). Production stays at `kSuMinBlocks=0`; `(256,0)` == the no-launch_bounds
baseline (271 ms/traj L=32, unchanged). The harness (`gauge_sun.cuh` templated
kernels + `profile_hmc --lb-sweep`) is retained for the structural route. **A real
su3 win must be structural** (stage the staple through shared memory / accumulate
the projection incrementally / warp-per-link), not occupancy tuning.

<details><summary>original Lever 1 hypothesis (kept for context)</summary>

**80% of su3 runtime (force + drift) runs at ~12% occupancy**, pinned to 1
block/SM by 144–156 registers/thread. With ~8 warps/SM the P100 cannot hide DRAM
latency → the kernels stall. Raising occupancy is the single biggest lever in the
backend.

**Hypothesis:** capping registers (accepting some spill) to fit ≥2–4 blocks/SM
nets a large win because the kernel is latency-bound, not register-throughput-bound.

**Experiments (empirical — measure each, in `gauge_sun.cuh`):**
- `__launch_bounds__(256, 2)` on `su_plaq_force_kernel` + `su_expi_lmul_kernel`
  → caps ≤128 regs/thread → 2 blocks/SM (~25% occ).
- `__launch_bounds__(256, 4)` → ≤64 regs → 4 blocks/SM (~50% occ), more spill.
- Block size 128 (`kBlock=128`) ± launch_bounds → finer occupancy granularity.
- Structural (if launch_bounds plateaus from spill): stop holding the full 3×3
  complex staple live — accumulate `ReTr`/the algebra projection incrementally,
  or stage the staple through shared memory; consider a **warp-per-link** mapping
  (the 18 reals across a warp's lanes) for SU(3) only.

**Measure:** force/drift `regs`, derived blocks/SM, `ms_per_traj` at su3 L=16 and
L=32. **Target:** occupancy 12% → ≥25–50%; expect a meaningful force-kernel
speedup if the latency-bound theory holds. Watch for spill (`localMemoryPerThread`
in the SQLite > 0) eating the gain — find the sweet spot.

**Risk:** over-capping forces local-memory spills (slow). The matrix math may
genuinely need the registers; if so the win is smaller and the structural route
(shared-mem staple / warp-per-link) is the real fix. Keep SU(2)/SU(3) generic —
any change lives once in `gauge_sun.cuh` / the `GD` traits.

</details>

(↑ This Risk is exactly what happened — see the REFUTED outcome at the top.)

---

## Lever 2 — phi4 stencil memory reuse

phi4 is well-occupied (~75%) and **bandwidth-bound**: the 4D stencil re-reads
each site ~9× (self + 8 neighbours), and 3 of 4 directions are large strides
(L, L², L³) that blow the 4 MB L2 at L≥48 → poor coalescing/reuse, far above the
minimum field traffic.

**Experiments (in `stencil.cuh` / `reduce_fwd.cuh`):**
- *Cheap first:* mark the field pointer `const __restrict__` and read neighbours
  via `__ldg` (read-only data cache) — may recover some reuse with zero
  restructuring. Measure before investing in tiling.
- *Real win:* shared-memory **halo tiling** — a block loads its sites + a
  one-site halo into shared memory once, then every thread reads neighbours from
  shared memory instead of DRAM. 4D halo indexing is fiddly; start with a 2D/3D
  tile over the fastest dims (x,y[,z]) and stream the slowest (t).
- Confirm the `+x` neighbour is already coalesced (x-fastest) — the gains are on
  ±y/±z/±t.

**Measure:** phi4 L=48/64 force `ms`, streaming-model effective GB/s (a tiling
win shows as higher effective GB/s = less redundant traffic). **Target:** cut
force time toward the field-streaming floor (≈2×field/peak).

**Risk:** shared-mem tiling adds complexity to a currently-simple generic
`stencil_kernel` used by *all* scalar actions (phi4/phi6/sine_gordon/xy/bose_gas).
Keep it generic or gate behind the functor; don't fork per action. If `__ldg`
alone gives most of the win, prefer it.

---

## Lever 3 — the per-trajectory rollback copy (investigate, low priority)

`copy_device_(old←field)` is a full-field D2D copy every trajectory (~0.5 ms phi4
/ ~2.2 ms su3, ~0.5–1% of runtime). **It is entangled with the CUDA-graph design**
and is *not* a simple win:

- Naive "double-buffer + pointer swap" breaks graph capture — the captured graph
  bakes device pointers, so the read/write buffer roles can't alternate between
  replays.
- Out-of-place MD (write the proposal to a scratch, leave `field` intact) just
  moves the copy from always-save to copy-on-accept; at ~90% acceptance that's
  only ~10% savings.
- A genuine fix needs a **ping-pong of two captured graphs** (A→B and B→A,
  replayed alternately, resolve picks the winner in place) — worth it only if the
  host-free block size grows or this fraction rises. **Defer** unless profiling
  later shows it dominant.

---

## Secondary / opportunistic

- **f32 phi4 throughput:** phi4 is bandwidth-bound, so `Phi4<float>` should be
  ~2× faster (half the bytes). Already supported; worth quantifying in the
  profiler as a "free" lever for production f32 runs. (su3 links stay f64.)
- **phi4 s_full (14.7%):** two evaluations per trajectory for ΔH; correctness-
  critical, leave unless it becomes dominant after Lever 2.

## Suggested order for the session

1. **Lever 1** (su3 launch_bounds/block-size sweep) — biggest, most measurable.
2. **Lever 2** `__ldg`/`__restrict__` quick try; tiling only if it pays.
3. Quantify **f32 phi4** in the profiler (cheap, informative).
4. Lever 3 only if 1–2 reveal the copy as a larger fraction.

Each step: implement → 47 gates + reversibility green on Kaggle → re-profile →
record the before/after (regs, occupancy, ms/traj) in `cuda_hmc_roadmap.md`.
