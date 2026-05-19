# Kernel-throughput bench tools — design proposal

Three new bench apps that measure the computational hot paths in
isolation — *no* HMC accept-reject, *no* Metropolis driver, *no* I/O.
The goal is a single-table answer to three questions:

  1. "how fast is the **force** kernel of action X on a 4D L=8 lattice?"
     (`bench_actions`)
  2. "how fast is **momentum sampling** of field type Y with RNG Z?"
     (`bench_rng`)
  3. "how fast is a full **MD trajectory** at fixed (τ, n_md) for each
     integrator on action X?" (`bench_integrators`) — the answer is
     *not* `(1·force) + n_md·(force + drift + kick)`-style napkin math
     because matrix-group drift (Cayley-Hamilton exp for SU(3),
     closed-form Pauli exp for SU(2)) and kick (`mom += dt·F`) cost
     comparable to one force eval each — only the bench tells you the
     true ratio.

The existing `apps/bench_*` apps measure full HMC trajectories (force +
integrator + accept-reject + momentum sampling). Useful as a top-line
number, but they conflate four costs into one. The three new tools
isolate each piece so a perf regression can be attributed to the
specific kernel it lives in.

## Non-goals

- No HMC, no integrator, no Metropolis. Pure kernel timing.
- No HDF5 output. Single printf table to stdout. (Optional `--csv`
  switch if you want to re-plot — covered under "Open questions" below.)
- No autocorrelation, accept-rate, dH conservation, or any HMC diagnostics.
- No multi-thread scaling sweeps (kernels are serial by design at the
  current scope; OpenMP only enters at the LLR-replica level).

## What gets benchmarked — actions

### Scalar (`apps/bench_actions.cpp`)

| action          | field                                | kernels                                 |
|-----------------|--------------------------------------|-----------------------------------------|
| `Phi4`          | `Lattice<double>`                    | `s_full`, `compute_force`, `compute_force_and_kick` |
| `Phi6`          | `Lattice<double>`                    | same                                    |
| `SineGordon`    | `Lattice<double>`                    | same                                    |
| `Xy`            | `Lattice<double>`                    | `s_full`, `compute_force` (no fused)    |
| `OnSigma<3>`    | `Lattice<std::array<double,3>>`      | `s_local` only — no force; report per-site |
| `BoseGas`       | `Lattice<std::complex<double>>`      | `s_full`, `compute_force`, `s_imag`, `compute_force_imag` |
| `CompactU1`     | `LinkLattice<double>`                | `s_full`, `compute_force`, `compute_force_and_kick` |
| `Wilson<U1>`    | `MatrixLinkLattice<U1, double>`      | `s_full`, `compute_force` (no fused)    |
| `Wilson<SU2>`   | `MatrixLinkLattice<SU2, double>`     | `s_full`, `compute_force`               |
| `Wilson<SU3>`   | `MatrixLinkLattice<SU3, double>`     | `s_full`, `compute_force`               |

Each action gets a row per (kernel, ndim, L). Scalar actions don't have
`HasLinkForce` so the gauge/scalar dispatch is `if constexpr` on
`HasForce` vs `HasLinkForce` — same pattern as `bench_gauge_vs_scalar.cpp`.

### Volumes

Conservative default grid (keeps total runtime under ~2 min on a recent
M-series box):

| ndim | L values    |
|------|-------------|
| 2    | 32, 64      |
| 3    | 12, 16, 24  |
| 4    | 4, 6, 8     |

`OnSigma<3>` and `BoseGas` only at 3D + 4D (their hot path is already
exercised by the scalar volumes).

### Hot-start

Each field is initialised to a "hot" random configuration so the inner
loops actually see non-trivial data (compilers can short-circuit zero
loads in tight kernels). Per type:

| field                                | hot init                                       |
|--------------------------------------|-----------------------------------------------|
| `Lattice<double>`                    | each cell `rng.normal()`                       |
| `Lattice<std::array<double,N>>`      | each cell `(rng.normal(), …)` then normalise   |
| `Lattice<std::complex<double>>`      | each cell `(rng.normal(), rng.normal())`       |
| `LinkLattice<double>`                | each link `rng.normal()` (an angle in [-π,π])  |
| `MatrixLinkLattice<G, double>`       | identity → `G::expi_lmul_slab(U, sampled P, 0.5)` per direction (gives random SU(N)) |

Init time is excluded from the kernel measurement.

## What gets benchmarked — RNG (`apps/bench_rng.cpp`)

For each (RNG, fill mode) tuple, measure throughput of filling a buffer
the size of one trajectory's worth of momenta for each field type.

### Fill modes

| mode                       | applies to                                   |
|----------------------------|---------------------------------------------|
| `normal_fill(buf, n)`      | batched Box-Muller path used by scalar HMC for `Lattice<double>` / `LinkLattice<double>` |
| `cast normal_fill` (×2)    | `Lattice<std::complex<double>>` (HMC reinterprets `mom.data()` as `double[2*n]`) |
| `sample_algebra_slab`      | each gauge group model's algebra sampler; `MatrixLinkLattice<G, double>` |
| per-element `rng.normal()` | catch-all fallback for `OnSigma` (array-valued field) |

### RNGs to compare

All three are first-class library types satisfying the existing `Rng`
concept (`uniform_u64`, `uniform`, `uniform_int`, `normal`,
`normal_fill`, `reseed`, `state_type` = `std::uint64_t`):

| RNG          | source                                          | flavour |
|--------------|-------------------------------------------------|---------|
| `FastRng`    | `include/reticolo/core/rng.hpp` (existing)      | xoshiro256++ + SplitMix64 seed, native 64-bit |
| `RanluxRng`  | **new** `include/reticolo/core/ranlux_rng.hpp`  | thin wrapper around `std::ranlux48` (subtract-with-carry + Lüscher discard); native 48-bit, two calls to fill uint64 |
| `Mt19937Rng` | **new** `include/reticolo/core/mt19937_rng.hpp` | thin wrapper around `std::mt19937_64`; native 64-bit |

Each new RNG is a small (~60 LOC) class with the same public surface as
FastRng so all existing call sites (Hmc, Replica, gauge_group::*::
sample_algebra_slab) accept it as a drop-in template parameter. To keep
the comparison apples-to-apples, every RNG uses the **same Box-Muller
polar form** that FastRng uses (cached pair + batched `normal_fill`) —
the only thing that varies is the underlying uniform-bit source. This
isolates the cost of the engine itself from the Gaussian transform.

`static_assert(Rng<RanluxRng>)` and `static_assert(Rng<Mt19937Rng>)` at
the bottom of each header lock the concept conformance at compile time.

A new test `tests/unit/test_rng_concept.cpp` parametrises the existing
RNG correctness checks (period of uint64 sequence, normal mean/variance,
Lemire `uniform_int` exact distribution, `reseed` reproducibility) over
all three types via a typed-test list.

### Volumes (same grid as actions)

Per field type, run at 3D L=24 and 4D L=8 (the two biggest cases that
actually hit memory bandwidth). Other volumes optional — RNG throughput
is pretty volume-independent until the buffer outgrows L2.

## What gets benchmarked — integrators (`apps/bench_integrators.cpp`)

For each (action, integrator, ndim, L) tuple, time **one full MD
trajectory** at a fixed `(τ, n_md)`. No Metropolis, no momentum
resampling between trajectories — back-to-back calls to
`Integrator::run(action, field, mom, force, τ, n_md)` with adaptive
repetition (same protocol as `bench_actions`).

### Integrators

| name       | force evals/traj   | drift updates/traj | per-traj cost ratio (cheap drift) |
|------------|--------------------|--------------------|-----------------------------------|
| `Leapfrog` | `n_md + 1`         | `n_md`             | 1×                                |
| `Omelyan2` | `2·n_md + 1`       | `2·n_md`           | ~2×                               |
| `Omelyan4` | `4·n_md + 1`       | `4·n_md`           | ~4×                               |

For scalar actions where drift = elementwise add and kick = elementwise
add, the per-traj ratio tracks the force-eval count cleanly. For
matrix-link actions the drift is `U ← exp(dt·P)·U` per direction —
non-trivial cost (SU(3) Cayley-Hamilton is ~250 flops/link). The bench
will show whether drift shifts the L:O2:O4 ratio meaningfully on the
matrix-link actions.

### Parameters

Fixed point: `τ = 1.0`, `n_md = 20`. Same actions and volumes as
`bench_actions`. The bench does **not** sweep `n_md` or tune for fixed
acceptance — that's the realm of `tune_phi4` and the `03_phi4_algorithm_
tuning` example, which already exist. This bench is just "how much
wall time does one trajectory take, broken down by integrator".

### Output

```
INTEGRATORS — full MD-trajectory throughput at τ=1.0, n_md=20

%-14s %-14s %-12s %-10s %-12s %-14s %-14s %-12s
"ndim x L"  "action"  "integrator"  "dofs"  "wall/traj"  "force evals/s"  "dof upd/s"  "MFLOPS"
--------------- -------------- ------------ ---------- ------------ -------------- -------------- ------------
3D L=24       Phi4           Leapfrog     13824      3.0e-03      7.0e+03         9.2e+07 M       1180
3D L=24       Phi4           Omelyan2     13824      5.8e-03      7.1e+03         9.5e+07 M       1190
3D L=24       Phi4           Omelyan4     13824      1.1e-02      7.5e+03         9.7e+07 M       1220
…
4D L=8        Wilson<SU3>    Leapfrog     24576      0.32         63              1.5e+06 M       …
4D L=8        Wilson<SU3>    Omelyan2     24576      …            …               …               …
4D L=8        Wilson<SU3>    Omelyan4     24576      …            …               …               …
```

The interesting column is `force evals/s` — for scalar actions it
should be roughly integrator-independent (each force eval is the same
work). For matrix-link actions where drift cost is comparable to force
cost, it'll appear lower for L (because drift is amortised over fewer
forces) and higher for O4 (drift amortised over more forces). The
delta is the quantitative answer to "how much does the matrix-group
drift cost relative to the force".

## Output format

Hand-rolled printf table to stdout, no HDF5, no CSV by default. Two
tables per bench (header + rows). Mockup for `bench_actions`:

```
ACTIONS — kernel throughput on hot random fields
Integrator: NONE   (kernels called in isolation, no MD loop)
Flop accounting: k_sin ≈ 15.0 ops per sin/cos call

%-14s %-14s %-10s %-22s %-12s %-14s %-12s
"ndim x L"  "action"  "dofs"  "kernel"  "wall [s]"  "dof upd/s"  "MFLOPS"
--------------- -------------- ---------- ---------------------- ------------ -------------- ------------
3D L=24       Phi4           13824      s_full                 1.42e-04      9.74e+07 M    1240.3
3D L=24       Phi4           13824      compute_force          1.55e-04      8.92e+07 M    1180.4
3D L=24       Phi4           13824      compute_force_and_kick 1.51e-04      9.16e+07 M    1212.7
3D L=24       Phi6           13824      s_full                 ...
…
4D L=8        Wilson<SU3>    24576      s_full                 ...
4D L=8        Wilson<SU3>    compute_force                 ...
```

`dofs` is sites for scalar actions, links for link/matrix-link.
`dof upd/s` = `dofs / wall_seconds` (per kernel call).
`MFLOPS` = `dofs · flops_per_dof / wall_seconds / 1e6`.

For `bench_rng`:

```
RNG — momentum-sampling throughput

%-30s %-12s %-12s %-14s %-14s
"field"  "rng"  "doubles"  "wall [s]"  "doubles/s"
------------------------------ ------------ ------------ -------------- --------------
Lattice<double>  3D L=24       FastRng      13824        7.0e-06        1.98e+09
Lattice<double>  3D L=24       Ranlux48     13824        ...
LinkLattice<double>  4D L=8    FastRng      24576        …
MatrixLinkLattice<SU3>  4D L=8 FastRng (alg) 24576       …
…
```

## Implementation approach

### Timing

Standard `std::chrono::steady_clock`. Adaptive repetition count: each
case starts with `n_calls = 1`, doubles until total wall time exceeds
`k_target_seconds` (default 0.3 s), reports the per-call median over
the last batch. This keeps measurements stable across kernels with
~6 orders of magnitude in cost (a Phi4 force at L=24 takes microseconds;
a Wilson<SU3> force at L=8 takes tens of milliseconds).

### Cache effects

Run a `k_warmup = 3` repetitions before the timed pass to fill caches.
No flushing between calls (we want steady-state throughput, not
cold-cache).

### Force buffer reuse

`compute_force` writes into a force lattice. Allocate it once outside
the timed loop, reuse across reps. Same for `compute_force_and_kick`'s
momentum buffer.

### Hot-init cost

Excluded from timing. Each case has the structure:

```cpp
{
    Lattice<F> phi{shape};
    Lattice<F> force{phi.indexing()};
    FastRng rng{seed};
    hot_init(phi, rng);
    Action const action{…};
    for (int i = 0; i < k_warmup; ++i) action.compute_force(phi, force);
    auto const t0 = clock::now();
    for (int i = 0; i < n_calls; ++i) action.compute_force(phi, force);
    auto const t1 = clock::now();
    print_row(…);
}
```

## Flop accounting

Per-kernel flop constants live as `constexpr double` in a small
`apps/_bench/flops.hpp` (or inlined in each bench TU; see "File layout"
below). Reuses the formulas already encoded in
`bench_gauge_vs_scalar.cpp` plus new ones for the SU(N) actions.

Reference values (per *force* eval, per dof, ndim = `d`):

| action          | flops/dof          | rationale                                       |
|-----------------|--------------------|------------------------------------------------|
| `Phi4`          | `2d + 7`           | nn-sum + polynomial                             |
| `Phi6`          | `2d + 11`          | extra `g6·φ^6` (4 mul + 2 add)                  |
| `SineGordon`    | `2d + 3 + k_sin`   | nn-sum + sin + cubic                            |
| `Xy`            | `2d·(1 + k_sin)`   | per-bond sin (no factorisation)                 |
| `OnSigma<N>`    | `2d·N + N²`        | nn-sum is vector-valued; force scales w/ N²     |
| `BoseGas`       | `~3d + cosh + 12`  | anisotropic last-dim hop + chemical potential   |
| `CompactU1`     | `0.5·(d−1)·(3 + k_sin + 9)` | per-link via plane-centric scatter   |
| `Wilson<U1>`    | same as `CompactU1` but no batched Sleef on the generic path; slower by ~2× |
| `Wilson<SU2>`   | `(4(d−1) + 1)·56 + 10` ≈ `4d² + …`  | link-centric: 4(d−1) staple cmms + 1 final cmm + TA |
| `Wilson<SU3>`   | `(4(d−1) + 1)·198 + 18` ≈ `15d² + …` | same shape, 3×3 cmm is 198 flops |

`s_full` cost ≈ `compute_force` cost minus the staple-scatter overhead;
within ~10–20%. The flop constants are intentionally rough — they exist
so the MFLOPS column is comparable across kernels with very different
arithmetic densities, not as a definitive count.

For matrix groups specifically, `flops_per_link` accounts for all 2N²
real operations per matrix entry, then folds in the number of plaquette
planes touched per link `(d−1)`. The constant `k_sin = 15.0` follows the
existing convention in `bench_gauge_vs_scalar.cpp`.

## File layout

```
include/reticolo/core/
    rng.hpp                 EXISTING — Rng concept + FastRng (unchanged)
    ranlux_rng.hpp          NEW — RanluxRng wrapping std::ranlux48
    mt19937_rng.hpp         NEW — Mt19937Rng wrapping std::mt19937_64
include/reticolo/
    reticolo.hpp            EXTENDED — umbrella includes the two new RNGs
tests/unit/
    test_rng_concept.hpp    EXISTING — extended with parametrised cases for the
                              new RNGs (period, normal stats, reproducibility)
apps/
    bench_actions.cpp       NEW — kernel throughput per action
    bench_rng.cpp           NEW — RNG throughput per field type, all three RNGs
    bench_integrators.cpp   NEW — full-trajectory wall time per (action, integrator)
    _bench/                 NEW header subdir (only included by the bench TUs)
        timing.hpp          chrono helpers, adaptive-rep loop, table formatter
        hot_init.hpp        per-field hot-start functions
        flops.hpp           per-kernel flop-per-dof constants
```

Wired into `apps/CMakeLists.txt` as standalone executables. The two new
RNGs DO get tests (correctness of the library types matters); the bench
apps themselves do not.

Existing `bench_*` apps stay as-is — they measure full-HMC throughput
which is still a useful upper-bound number, just not the same question.

## Open questions

1. **CSV output flag?** Single printf is enough for eyeballing. If we
   want to plot deltas across commits (e.g., perf regressions), a
   `--csv` flag would help. Defer until needed.
2. **Single precision (`float`)?** The actions are double-only in the
   current tree (Sleef paths gated on `T == double`). Skip float for
   now; revisit if/when a perf push wants it.
3. **`OnSigma<N>` for multiple N?** Default to `N=3` (the Heisenberg
   model used in example 02). If the user cares about `N=4`, easy to
   add another row.
4. ~~Two RNGs only, or include more?~~ **Settled — three library RNGs:
   FastRng, RanluxRng, Mt19937Rng. All first-class in `include/reticolo/
   core/`, all satisfying the existing `Rng` concept, all drop-in to
   the existing call sites.**
5. **Bench-app perf in CI?** These would be slow enough to skip in CI;
   they're for local perf tracking only. No CTest registration.
6. **Larger volumes?** The default grid maxes out at 4D L=8 (≈ 4k sites,
   ≈ 16k links). If you regularly run at 4D L=10 or 12 we should add
   those — but they're long enough that adaptive repetition might still
   take many seconds per row at the high end. Flag for approval.

## Estimated effort

- `ranlux_rng.hpp` + `mt19937_rng.hpp`: ~120 LOC.
- `test_rng_concept.hpp` extension: ~80 LOC (parametrise existing checks).
- `_bench/` shared headers: ~150 LOC.
- `bench_actions.cpp`: ~250 LOC (10 actions × small per-action block).
- `bench_rng.cpp`: ~150 LOC (6 field types × 3 RNGs).
- `bench_integrators.cpp`: ~200 LOC (10 actions × 3 integrators × small block;
  reuses hot-init from bench_actions).
- CMakeLists update.

Total ~950 LOC. Two new public types in `core/`, no dep additions, no
change to existing public APIs (RanluxRng/Mt19937Rng instantiate at any
existing call site that already takes a templated `Rng`).
