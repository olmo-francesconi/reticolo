# GPU LLR — Milestone B plan (stream-concurrent replicas)

Status: **plan only, no code written.** Target branch `feat/cuda-extension`
(CUDA is off by default; current checkout is `main` — switch first).

## Goal & model

Port the LLR orchestration layer to the CUDA backend. Model **B**
(stream-concurrent): every replica owns its own `cuda::Hmc` on its own
non-blocking stream; the driver launches *all* replicas' trajectory before any
sync, so the GPU scheduler overlaps them. This is the honest analog of the CPU
`#pragma omp parallel for` over replicas — the parallel axis is the replica
vector; each replica's trajectory is the existing host-free graph.

Scope: **real actions only** (mode A: `(1+a)·S + (S−Eₙ)²/2δ²`). First app is
Phi4 (scalar, real force — simplest path, no matrix/link layout), mirroring
`apps/phi4_llr.cpp`. Complex/mode-B (BoseGas) is out — it needs device
imag-force / `s_imag` functors that don't exist yet.

Regime note: this is small-V / large-N (many DoS windows). B overlaps well while
N is under the GPU's concurrent-kernel ceiling, then plateaus into serial
batches. That plateau is the signal that model C (batched mega-field) is worth
building — a later milestone, not this one. Do not tune B past it.

## Two locked design decisions

- **Exchange = swap window params**, not configs. Keeps each `DeviceField` (and
  the graph pointer baked into it) in place; swaps only `{a, Eₙ}` between the two
  slots. `δ` is uniform so it never moves. Windows random-walk across slots
  (parallel-tempering style). Output is written per-slot and grouped by `Eₙ` in
  python.
- **Add a device-resident `s_full` accessor to `cuda::Hmc`** (two-phase) instead
  of the malloc-per-call `meas.s_full(field)`.

## Shared formula (honors the CPU/device "one formula" invariant)

New `include/reticolo/llr/detail/window_formula.hpp` — two `RETICOLO_HD` inlines,
the single source of truth for both CPU `llr::WindowedAction` and the device
kernels:

```cpp
namespace reticolo::llr::detail {
// mode A windowed action value
template <class R> RETICOLO_HD R windowed_value(R s, R a, R E_n, R delta) noexcept {
    R const d = s - E_n;
    return (R{1} + a) * s + (d * d) / (R{2} * delta * delta);
}
// mode A force multiplier  d/dS of windowed_value
template <class R> RETICOLO_HD R force_scale(R s, R a, R E_n, R delta) noexcept {
    return (R{1} + a) + (s - E_n) / (delta * delta);
}
}
```

Modify `include/reticolo/llr/windowed_action.hpp` (mode-A branches of `s_full`
and `compute_force`) to call these, so CPU and device cannot drift.

## New device windowed action

`include/reticolo/cuda/llr/windowed_action.cuh` — `cuda::WindowedAction<HostAction, Field>`.
Mirrors CPU `llr::WindowedAction<Base>`: owns a base `DeviceAction`, adds the
window state. Satisfies exactly what `cuda::Hmc` calls on its action
(`sample_momenta`, `s_full_into`, `compute_force`) plus `constraint_s_full_into`
for the accessor.

```cpp
template <class HostAction, class Field>
class WindowedAction {
public:
    using value_type = typename Field::value_type;
    WindowedAction(HostAction host, DeviceTopology topo, double a, double E_n, double delta);

    // --- interface cuda::Hmc consumes ---
    void sample_momenta(value_type* mom, long n, std::uint64_t seed,
                        std::uint64_t const* traj, cudaStream_t s) const;   // -> base_

    // windowed S into out[0]:  base S -> s_tmp_, then window_combine_kernel
    void s_full_into(double* out, Field const& f, double* partials, cudaStream_t s) const;

    // F_win = force_scale(S_base) * F_base :
    //   base_.compute_force -> force; base_.s_full_into -> s_tmp_; scale_force_kernel
    void compute_force(Field const& f, Field& force) const;

    // mode A constraint == base S (device scalar, no combine)
    void constraint_s_full_into(double* out, Field const& f, double* partials, cudaStream_t s) const;

    // host-side window controls (re-upload to params_; graph-safe, buffer ptr stable)
    void set_a(double a);
    void set_window(double a, double E_n);      // for exchange param-swap
    [[nodiscard]] double a() const;
    [[nodiscard]] double E_n() const;

private:
    DeviceAction<HostAction, Field> base_;
    mutable DeviceBuffer<double> params_;    // {a, E_n, delta} device-resident (read by kernels)
    mutable DeviceBuffer<double> s_tmp_;     // base-S scalar between reduce and combine/scale
    mutable DeviceBuffer<double> partials_;  // reduction scratch for compute_force's S reduction
    double a_h_, E_n_h_, delta_h_;           // host mirrors
};
```

Two small kernels (templated for weak linkage, like `mh_accept_kernel`):

- `window_combine_kernel<<<1,1>>>(out, s_tmp, params)` → `out[0] = detail::windowed_value(s_tmp[0], a, E_n, delta)`.
- `scale_force_kernel<T><<<grid,256>>>(force, n, s_tmp, params)` → `force[i] *= detail::force_scale(...)`; operates on the field's raw storage elements (real for U(1); element-wise real scaling for matrix groups later).

Graph-capture correctness: `params_` is device-resident so `set_a`/`set_window`
(an H2D of ≤3 doubles) never forces a re-capture — same pattern as `traj_buf_`.
`s_tmp_` is reused across the pot0/pot1 `s_full_into` calls and the force
reduction, but all run ordered on the one capture stream, so reads precede the
next write. Force S-reduction adds one reduction per force eval (perf cost of
device windowing — acceptable at small V).

## `cuda::Hmc` change — the device `s_full` accessor

Modify `include/reticolo/cuda/hmc.cuh`. Add one member and three methods (member
functions of a class template instantiate only when called, so plain
`DeviceAction` Hmc users are unaffected):

```cpp
// enqueue base-constraint reduction of the CURRENT (post-resolve) field into
// cons_buf_, ordered on md_stream_ after the trajectory. No sync.
void enqueue_constraint() {
    action_.constraint_s_full_into(cons_buf_.data(), field_, partials_.data(), md_stream_);
}
// D2H the enqueued scalar; syncs this stream (the trajectory already launched).
[[nodiscard]] double read_constraint() {
    double s = 0; cons_buf_.copy_to_host(&s, md_stream_); sync(); return s;
}
// convenience = enqueue + read (used by energy(), warmup); syncs.
[[nodiscard]] double constraint_value() { enqueue_constraint(); return read_constraint(); }
```

New member `DeviceBuffer<double> cons_buf_{1};`. Reuses the existing `partials_`
(free by the time the trajectory's own reductions are done — same stream). Reads
the resolved field, so it is correct after reject-rollback (why we recompute
rather than expose `eng_[3]`).

## Device replica wrapper

`include/reticolo/cuda/llr/replica.hpp` — `cuda::llr::Replica<HostAction, Integ, Field>`.
Non-movable (holds `Hmc`), so drivers use `vector<unique_ptr<Replica>>` exactly
like the CPU side.

```cpp
template <class HostAction, class Integ, class Field>
class Replica {
public:
    Replica(HostAction host, typename Field::SizeVec shape,
            double E_n, double delta, double tau, int n_md, std::uint64_t seed, double a_init = 0.0);

    // two-phase for stream overlap:
    void launch_trajectory() { hmc_.run(1); hmc_.enqueue_constraint(); }   // async
    [[nodiscard]] double read_dE() { return hmc_.read_constraint() - E_n_; } // gather (syncs own stream)

    void thermalize(int n) { hmc_.run(n); hmc_.sync(); }        // host-free, no measure
    [[nodiscard]] double energy() { return hmc_.constraint_value(); }        // base S

    [[nodiscard]] double a() const;      void set_a(double v);
    [[nodiscard]] double E_n() const;
    void swap_window(Replica& other);    // exchange: swap {a, E_n} both sides + re-upload
    void warm_into_window(int max_batches, int batch, double threshold_sigmas);

private:
    Field field_;
    Hmc<WindowedAction<HostAction, Field>, Integ, Field> hmc_;
    double E_n_;
    double delta_;
};
```

`sample(n)` is *not* a replica method here (unlike CPU): the averaging loop lives
in the driver so it can interleave launch-all / gather-all across replicas. The
replica exposes only the per-trajectory `launch_trajectory` / `read_dE`.

## Device LLR driver

`include/reticolo/cuda/llr/driver.hpp` — `cuda::llr::run(...)`. Host-callable, no
kernels. Reuses `llr::nr_update` / `llr::rm_update` from
`include/reticolo/llr/update_a.hpp` verbatim (pure host math). Structure differs
from the CPU driver: the trajectory loop is **outer**, the replica loop inner, to
get overlap.

```cpp
template <class Replica>
void run(std::vector<std::unique_ptr<Replica>>& reps, ExchRng& exch_rng,
         DriverSpec const& spec, io::Writer& out);
```

Per NR/RM iteration:

```
for r in reps: r->thermalize(n_therm);            // host-free warm (each own stream)
for r in reps: r->sync-if-needed                  // (thermalize already syncs; or defer)
// measurement sweep — overlapped:
zero accum[r]
for t in 0..n_meas:
    for r in reps: r->launch_trajectory();        // ALL async first -> streams overlap
    for r in reps: accum[r] += r->read_dE();      // gather (overlap already happened)
for r in reps: dE[r] = accum[r]/n_meas
for r in reps: a[r] = (NR|RM)_update(r->a(), dE[r], delta, k); r->set_a(a[r])
// serial HDF5 drain (as CPU): append a[r], dE[r], AND E_n[r] (now time-varying)
// param-swap exchange (RM only), serial even/odd:
for i in even|odd pairs:
    e_i=reps[i]->energy(); e_j=reps[i+1]->energy();
    if log_p>=0 || u<exp(log_p): reps[i]->swap_window(*reps[i+1]); ++accepted
```

Schema change vs CPU driver: `/replica_NNN/E_n` becomes a **series** (was the
fixed `/cfg/E_n`). Python analysis groups per-slot series by `E_n` to recover
per-window DoS.

## App

`apps/cuda/phi4_llr_cuda.cu` — mirrors `apps/phi4_llr.cpp` structure (CLI flags,
grid `n_rep = max(2, lround((E_max−E_min)/δ)+1)`, build
`vector<unique_ptr<Replica>>`, warm into window, `cuda::llr::run`). Uses
`DField = cuda::DeviceField<double>` (scalar layout) and
`WindowedAction<action::Phi4<double>, DField>`. Register in
`apps/CMakeLists.txt` under `if(RETICOLO_ENABLE_CUDA)`.

## Tests / validation

- Smoke test `tests/cuda/test_phi4_llr_cuda_smoke.cpp` — exec the app binary on a
  tiny lattice/few windows, assert HDF5 has `/replica_000/a`, `/dE`, `/E_n`
  series of the right length and finite values. (No convergence sim in ctest,
  per project rule.)
- Physics validation is a **manual Modal run**, not ctest: run a few windows on
  GPU and CPU (`phi4_llr`) at matched params, check the reconstructed `a(Eₙ)`
  agrees within statistics. Document in the PR, do not wire into a sweep.

## File-level surface

New:
- `include/reticolo/llr/detail/window_formula.hpp` (RETICOLO_HD shared formula)
- `include/reticolo/cuda/llr/windowed_action.cuh`
- `include/reticolo/cuda/llr/replica.hpp`
- `include/reticolo/cuda/llr/driver.hpp`
- `apps/cuda/phi4_llr_cuda.cu`
- `tests/cuda/test_phi4_llr_cuda_smoke.cpp`

Modified:
- `include/reticolo/llr/windowed_action.hpp` (use shared HD formula, mode A)
- `include/reticolo/cuda/hmc.cuh` (constraint accessor + `action()` + `cons_buf_`)
- `apps/CMakeLists.txt`, `tests/CMakeLists.txt`

NOT modified: the `cuda.hpp` umbrella stays as-is. `driver.hpp` needs
`io::Writer`, and the lean CUDA tools (`bench_cuda_hmc` / `profile_cuda_hmc`)
link only `reticolo::cuda` without io — so the app includes the three cuda/llr
headers directly rather than pulling them through the umbrella.

Reused unchanged: `include/reticolo/llr/update_a.hpp`, `llr/log.hpp`.

## Build / verify (no local GPU)

```sh
git switch feat/cuda-extension
uv run tools/modal/app.py build                 # linux-nvcc preset gate
uv run tools/modal/app.py run --app phi4_llr_cuda  # run one binary, export HDF5
```

## Validation results (2026-07-01)

Milestone B verified end to end:
- CPU: 181/181 tests, `-Werror` clean (windowed-action refactor behavior-preserving).
- GPU (Modal, T4): 48/48 CUDA-preset tests incl. the phi4_llr_cuda smoke.
- **Physics gate PASS:** phi4 L=8 ndim=3, 9 windows. CPU `phi4_llr` vs GPU
  `phi4_llr_cuda` reconstructed `a(Eₙ)` agree to **max 1.16σ** (abs Δa ~0.001–0.007).
  Reusable check: `tools/validate/compare_llr.py A.h5 B.h5` (groups the GPU's
  time-varying per-slot `E_n` series by window before averaging).

**Perf finding — model B is sync-bound at small V.** Same run: CPU (1 core,
OpenMP off) 10.2s total vs GPU (T4) 40.8s — **GPU ~4× slower** at 8³=512 sites.
Cause: the driver's measurement loop syncs every replica every trajectory
(`read_dE` → `cudaStreamSynchronize`), ~4050 host syncs/sweep each waiting on a
microsecond trajectory; stream overlap can't hide a per-trajectory host barrier.
**Fix applied — device-side `⟨dE⟩` accumulation** (`Hmc::begin_measure` /
`measure_trajectory` / `end_measure`, driver measure loop): a whole measurement
block runs host-free, `(S−Eₙ)` accumulates into a device buffer per trajectory,
one readback per sweep (~4050 syncs/sweep → 9). Result at 8³: GPU 40.8s → **28.9s
(1.4×)**, RM/sweep 0.90s → 0.63s; physics unchanged (1.16σ). Removing the sync
exposed the next floor: **per-trajectory host launch overhead** (~4 async launches
per `measure_trajectory` — graph + base-S reduction + accumulate — dominate when
512-site kernels are microseconds). Still ~2.8× slower than 1 CPU core at this
pessimal small V.

Further launch reduction, if a target size needs it: (a) capture N trajectories +
accumulates into one measurement graph (~1 host launch/block, stays model B), or
(b) model C batched replicas. The real lever is **V** — at 8³ the GPU is
latency-bound; larger lattices grow per-traj compute while launch overhead stays
fixed, so the crossover is a volume question (see crossover study).

## Force+action fusion (2026-07-01)

nsys (RunPod V100, `--cuda-graph-trace=node`) flagged the force-scale reduction:
`reduce_fwd_site_kernel` runs ~121k instances in lockstep with the force
`stencil_kernel` (~115k) — **two full field gathers per MD force eval** (~41/traj),
the reduction ~21% of GPU time. The windowed `compute_force` did `stencil` (force)
then a separate `reduce_fwd` (base-S) over the same field.

**Fix — a dual-output stencil** (`cuda/stencil_fwd_fused.cuh`): one thread-per-site
gather emits both `force[i]` (all 2d neighbours) and `site_energy[i]` (forward
neighbours), then the existing `reduce_sum_into` (a tree reduction, not a gather —
already part of the s_full path). One gather instead of two; the `reduce_fwd`
gather is gone. The functor keeps a **separate forward accumulator** alongside the
full 2d sum (a free register on device; the CPU `s_full_and_force` can't afford it
and halves the hopping weight instead) so `energy()` calls the unchanged
`phi4_action_site` on the same forward sum → **S_base bit-identical to
`s_full_into`**, no new formula, no reproducibility divergence.

Opt-in per action: `device_functors<A>::s_full_and_force` → `DeviceAction`
exposes it (detected by concept `HasFusedForce`) → the device `WindowedAction`
picks it via `if constexpr (requires{...})`, exact mirror of the CPU
`WindowedAction`'s `s_full_and_force` gate. Wired for all **real-scalar site
actions** — Phi4, Phi6, SineGordon, Xy (Xy carries two bond accumulators since its
force/energy use different per-bond transcendentals). **BoseGas** (complex) is left
for later: it needs the time-dir cosh(μ) weight (per-`mu` accumulation) and a
complex-capable force-scale kernel. Gauge actions keep the two-pass fallback.

**Validated (RunPod V100, sm_70):** full `linux-nvcc` build + 48/48 CUDA tests
(incl. the `phi4_llr_cuda` smoke), and the physics gate — CPU `phi4_llr` vs fused
GPU `phi4_llr_cuda`, defaults (L=8 ndim=3, 9 windows) — `a(Eₙ)` agrees to **max
1.08σ** (was 1.16σ pre-fusion), confirming the fused S_base is physics-equivalent.

**Profile confirms the win (nsys, `--cuda-graph-trace=node`):** in the fused run
`reduce_fwd_site_kernel` drops from lockstep-with-force (~21% GPU time, instance
count ≈ the force stencil's) to **3,932 vs 72,570 instances (~5%, 1.3% GPU
time)** — only the 2 Hamiltonian s_full evals per trajectory. `stencil_fwd_fused`
now carries force + base-S in one gather; the redundant per-force-eval gather is
gone. What's left dominant is the S-reduction *tree* (`block_sum`+`final_reduce`,
~35%) — inherent to the windowed force's global reduction, a separate lever, not
what this fusion targeted.

**Measured wall-clock A/B (Modal A100 arch-80, `stress` matrix, baseline 5b60a3e
vs fused — only the fusion differs):** the win grows with volume, since at large
4D V the strided neighbour *gathers* dominate the force path and the fusion drops
one of the two per force eval:

| L (4D) | sites | Δ wall @N=128 |
|---|---|---|
| 8  | 4,096  | −16% |
| 12 | 20,736 | −25% |
| 16 | 65,536 | **−29.5%** (127.2s → 89.6s) |

Small V / low replica count is launch-bound (−5% at L=8 N=8), converging to the
asymptote as replicas fill the GPU. So the 18% single-kernel estimate is the
floor; production 4D volumes see 25–30%.

## Open sign-off points

1. Output schema: per-slot `E_n` **series** + python grouping (recommended,
   consequence of param-swap) vs. keeping windows slot-fixed via config D2D-swap.
2. ~~Force-scale reduction cost: one extra base-S reduction per force eval.~~
   **Resolved** — folded into the force stencil (fusion above), Phi4.
3. First app = Phi4 (scalar). Add CompactU1 / Wilson variants after the scalar
   path is validated.
