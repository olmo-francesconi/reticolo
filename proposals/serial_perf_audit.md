# Serial performance audit — 2026-05-18

Scope: single-thread wall-time only. Parallelism is explicitly out of scope.
Build: `macos-appleclang` preset, Release, `-O3 -DNDEBUG -march=native -fno-math-errno`,
Apple clang via `/usr/bin/c++`. Lattice volumes are physics-realistic (3D L=8..24,
4D L=4..10). All numbers are from `apps/bench_*` plus a `sample`-based attribution
of `bench_phase_quenched`, `u1_hmc -L 16`, and `sine_gordon_hmc -L 16`.

The library is already in good shape — the scalar real-field hot paths
(Phi4, Phi6, SineGordon-MD) are within a small factor of memory-bandwidth
peak, and the bulk-vs-slab `visit_nn` / `visit_plane` pattern is doing what
it advertises. The wins below are concentrated in three places: one action
that was missed by the refactor (BoseGas), one allocation on the LLR force
path, and the libm sin/cos calls that dominate every transcendental action.

---

## Baseline (M/s = millions of real-DOF updates per second, HMC traj, Omelyan2)

| action      | 3D L=16 | 4D L=8  | comment                                |
| ----------- | ------- | ------- | -------------------------------------- |
| Phi4        | 663     | 249     | reference; pure NN + polynomial        |
| Phi6        | 627     | (~248)  | matches Phi4 within noise              |
| SineGordon  | 154     | 46      | ~4× slower than Phi4 → sin/cos bound   |
| BoseGas PQ  | 154     | 136     | 2.16× slower than Phi4 per real DOF    |
| CompactU1   | 36      | 24      | sin/cos × n_links/n_sites (= 4 in 4D)  |

(MC sweep throughput is ~25 M sites/s across the board → RNG-bound, separate from HMC.)

---

## Findings, ranked by serial-wall-time impact

### 1. `BoseGas` hot kernels miss the `visit_nn` fast path  — **expected 2–3× on bose_gas_*** apps**

`include/reticolo/action/builtins/bose_gas.hpp`, lines for `s_full`, `compute_force`,
`compute_force_imag`, `s_imag`.

All four use the pattern the codebase explicitly warns against:

```cpp
for (Site x : l.sites()) {
    complex_t const phi = l[x];
    ...
    staple += c * (l[l.next(x, nu)] + l[l.prev(x, nu)]);  // gather through Indexing
}
```

`include/reticolo/action/helpers.hpp` line ~9 says verbatim:

> Don't use these helpers from inside those hot paths; the function-call
> boundary inhibits the compiler's hoist.

But that's exactly what `BoseGas` does. Every other scalar action
(`Phi4`, `Phi6`, `SineGordon`) uses `detail::visit_nn` / `detail::reduce_fwd`
with stride-1 bulk + wrap slabs, which Apple clang turns into clean NEON.
`BoseGas` instead pays a 2d-element gather through the neighbour table at
every site, in a body the compiler can't prove is safe to hoist.

**Evidence:** `sample` on `bench_phase_quenched` (4D L=8) attributes ~50% of
trajectory time to `BoseGas::compute_force` and the bulk of the rest to
`Omelyan2::run` (which is the kick loop calling that same force). The
phase-quenched / phi4 throughput ratio is 2.16× at 3D L=16 and 1.87× at 4D
L=8, with phi4 = 1.0× the DOF cost — they should be comparable per real DOF
since both are NN + on-site polynomial, just with `T = std::complex<double>`
vs `T = double` (which the compiler handles natively).

**Fix:** port `s_full`, `compute_force`, `compute_force_imag` to
`detail::visit_nn` / `detail::reduce_fwd` parametrised on `std::complex<T>`.
The neighbour-sum body already exists; only the outer loop needs swapping.
The asymmetric mu-dependent factor (cosh(mu) on the time direction) is a
loop-invariant constant per (mu, nu) plane, not a problem for the helper —
or alternatively split the time direction into its own pass. Plaq-bench
showed the same pattern already works for real complex-valued bodies (see
`bose_gas_llr`'s Metropolis-cascade path which uses `ds_imag_local`).

The hot kernels for `bose_gas_hmc.cpp`, `bose_gas_llr.cpp`,
`bench_phase_quenched.cpp` would all benefit.

---

### 2. `llr::WindowedAction<...>` allocates a `Lattice<T>` per force call in complex mode  — **expected 5–15% on bose_gas_llr**

`include/reticolo/llr/windowed_action.hpp` — `compute_force` and
`compute_force_and_kick`, in the `if constexpr (k_complex)` branches:

```cpp
Lattice<T> imag_force{force.indexing()};   // malloc + zero-fill
base.compute_force_imag(l, imag_force);
... // pointwise combine
```

This runs on every MD kick step. With Omelyan2 and n_md=20 that's 41 allocations
+ frees per trajectory, each of size `nsites * sizeof(complex<double>)`.
For 4D L=8 that's 4096 * 16 B = 64 KiB per allocation — large enough to thrash
the allocator's small-bin path, small enough that `Lattice<T>` constructs it via
`std::vector` (so default-construct + zero).

Same issue in the real-LLR `compute_force` path — there it uses an in-place
`fp[i] *= scale` rescale with no extra buffer, so it's already fine. Only the
complex path allocates.

**Fix:** lift `imag_force` to a member of `WindowedAction` (lazy mutable scratch)
or, cleaner, plumb it through as a constructor-time scratch lattice on the LLR
Replica that owns the Hmc instance. The `Replica<...>` already holds the
`WindowedAction` by value (`include/reticolo/llr/replica.hpp` line ~50), so this
is a one-field addition.

Bonus: `BoseGas` could expose `compute_force_imag_and_kick(field, mom, k_dt)`
that fuses the loop, matching the existing fused-kick pattern of every other
action — then `WindowedAction::compute_force_and_kick` doesn't need an imag scratch
buffer at all in complex mode. Same shape as `Phi4::compute_force_and_kick`.

---

### 3. Scalar `std::sin` / `std::cos` are not vectorised  — **expected 2–4× on SineGordon, XY, CompactU1**

Every transcendental action's hot loop calls scalar libm:

- `action/builtins/sine_gordon.hpp` — `std::cos` in `s_full`, `std::sin` in `compute_force` + fused kick.
- `action/builtins/xy.hpp` — `std::cos` in `s_full`, two `std::sin` per site per direction in `compute_force`.
- `gauge/builtins/compact_u1.hpp` — `std::cos`/`std::sin` per plaquette.

Apple clang on M-series will NOT vectorise these into `vvcos` / `vvsin`
automatically — it needs either `-fveclib=libmvec` (Linux only), explicit
Accelerate calls (`vvsin`, `vvcos`, `vvsincos` from `<Accelerate/Accelerate.h>`),
or `#pragma omp simd` with a vectorised libm. The samples on `u1_hmc` and
`sine_gordon_hmc` show **100% of the visit_plane / reduce_fwd time inside the
body lambda**, with no other identifiable hotspot — i.e. the sin/cos call itself
is the entire cost.

SineGordon at 4D L=16 is 4× slower than Phi4 (154 → 35 M/s normalised). Each force
needs 1 sin call per site vs Phi4's ~5 polynomial fma — so 4× says the sin call
costs about as much as 20 fmas, which is roughly what scalar `sincos` runs on M-series.

**Fix options (in order of effort):**

1. **`std::sin` / `std::cos` via Accelerate `vvsin` / `vvcos`** in a buffered pass:
   pre-compute `sin(plaq)` / `cos(plaq)` into a scratch span sized to the inner
   row (`L0` elements), then reduce / scatter. Adds one buffer pass; vectorised
   sincos is ~4-6× faster per element than scalar on M2/M3. Net ~2-3× on these
   actions. Mechanical change inside the `_4d_` / `_3d_` bodies of
   `visit_plane` and `reduce_fwd` only when the body lambda needs sincos —
   easiest to express as a sibling helper `reduce_fwd_with_phase` / etc.

2. **Sleef** (`sleef_sind4` / `sleef_cosd4`) gives portable vectorised libm
   without Apple-specific headers, single header dep. Same speedup.

3. **`#pragma clang fp reassociate(on)`** locally around the s_full reduction
   loops — lets the compiler emit a tree-reduction and overlap independent
   `sin` calls. Not as good as vector libm but zero new code and zero deps.
   Worth measuring first as a no-code-change baseline.

This is the single largest serial speedup available to the library across
multiple actions, but it's also the highest-effort fix.

---

### 4. `Hmc::trajectory` snapshot / rollback / `hamiltonian_` loop through `Site` views  — **expected 1–3% per trajectory**

`include/reticolo/algorithm/hmc.hpp`:

```cpp
for (Site x : field_.sites()) {
    old_field_[x] = field_[x];
}
...
for (Site x : field_.sites()) {
    field_[x] = old_field_[x];
}
...
// hamiltonian_(): for (Site x : field_.sites()) kin += static_cast<double>(mom_[x]) * mom_[x];
```

The gauge twin (`include/reticolo/gauge/algorithm/hmc.hpp`) already does the
right thing with raw pointers:

```cpp
F* const old        = old_field_.data();
F const* const fp   = field_.data();
std::size_t const n = field_.nlinks();
for (std::size_t i = 0; i < n; ++i) { old[i] = fp[i]; }
```

The scalar HMC should match. Snapshot is just a `std::memcpy` over `nsites *
sizeof(F)` bytes; rollback the same. `hamiltonian_` is a plain reduction over
`mom_.data()`. Sample says this is ~1-3% of trajectory time (not huge), but
it's free to fix and brings the two HMC paths to one style.

Same pattern shows up in `algorithm/integrators.hpp::drift_` and the non-fused
branch of `kick_`:

```cpp
for (Site x : field.sites()) {
    field[x] += static_cast<F>(c_dt) * mom[x];
}
```

vs the gauge version's flat `for (std::size_t i = 0; i < n; ++i)`. The
compiler usually does autovectorise this through the iota+transform view,
but not always — and on a hot path it's not worth the gamble. Match the
gauge pattern.

---

### 5. CompactU1 `s_full` per call is ~1.5× the cost of `compute_force_and_kick` per call

Sample on `u1_hmc -L 16 --ndim 4 --n_md 20` (Omelyan2 → 41 force-eval passes,
2 s_full passes per trajectory) attributes:

- `visit_plane_4d_<...CompactU1::s_full>` — 47 samples
- `visit_plane_4d_<...CompactU1::compute_force_and_kick>` — 32 samples

If they cost the same per call, the ratio should be 2/41 = 0.05. The
measured ratio is 1.47. So `s_full` per call is ~30× more expensive than one
force eval.

Likely cause: `s_full` does a serially-dependent reduction
`accum += std::cos(plaq)`, which blocks `cos` parallelism along the chain.
`compute_force_and_kick` scatters into `mmu[s]` / `mnu[s_pmu]` with no
cross-iteration data dependency, so the OoO engine can keep multiple `sin`
calls in flight.

**Fix:** introduce a parallel reduction (per-row partial accumulators, then
fold), or apply `#pragma clang fp reassociate(on)` locally around the s_full
body in `gauge/builtins/compact_u1.hpp`. Either way the reduction structure
needs to expose independent partial sums.

Caveat: `s_full` only runs twice per trajectory, so even fully eliminating it
saves ~3% of trajectory time at n_md=20. Worth doing only because it's a few
lines and the same trick helps every reduction-style `s_full` in the library
(Phi4, Phi6, SineGordon — all use `reduce_fwd` with strict-order `total +=`).

---

### 6. Wolff `wolff_link_p`: always computes `std::exp` even when accept-prob is 0

`action/builtins/xy.hpp` line ~115:

```cpp
double const w  = -2.0 * static_cast<double>(beta) * sx * sy;
return 1.0 - std::exp(std::min(0.0, w));
```

When `w >= 0`, `min` returns 0 and the result is `1 - exp(0) = 0`. `std::exp`
still runs. Short-circuit:

```cpp
if (w >= 0.0) return 0.0;
return 1.0 - std::exp(w);
```

`on_sigma.hpp` has the same idiom. Wolff `update()` calls this once per visited
neighbour link (4-8 per cluster site), so for a beta where about half the
links are unfavoured this saves ~half of the exp calls. Wolff isn't on most
hot paths but `xy_wolff` and `on_sigma_wolff` are the headline cluster-update
apps and this is one line.

---

### 7. Build flags worth re-checking

`CMakeLists.txt` sets `-fno-math-errno` for Release. That's the right baseline.
Worth considering for the action hot files specifically:

- `-ffp-contract=fast` (already default at -O3 on clang, but worth confirming
  generates FMAs in `visit_nn` lambdas — `sample` symbol output suggests yes)
- `#pragma clang fp reassociate(on)` *inside* the `reduce_fwd_*d_` bodies in
  `action/hot_loop.hpp` and the `visit_plane_*d_` bodies in `gauge/hot_loop.hpp`
  for the cos-reduction. Strictly localised — does not touch any code that
  cares about determinism. Same RNG stream, same accept decisions; only the
  s_full numerical value reorders adds, which is already noise at the metropolis
  test scale.

Don't enable `-ffast-math` globally — it would silently break sign-imbalanced
sums in observables.

---

## Suggested order of attack

1. **BoseGas → visit_nn** (item 1). Biggest single win, no new deps, mechanical
   diff. Confidence: high.
2. **WindowedAction scratch** (item 2). Tiny diff, removes a malloc/traj on the
   complex-LLR path. Confidence: high.
3. **HMC + integrator raw-pointer cleanup** (item 4). Free win, brings scalar HMC
   to parity with gauge HMC's style. Confidence: high.
4. **`#pragma clang fp reassociate(on)` on the s_full reductions** (items 5 + 7).
   One-line experiment. Confidence: medium — measure before keeping.
5. **Vector libm for sin/cos** (item 3). Biggest absolute win but biggest diff.
   Defer until 1-4 are done so the new baseline is clean. Confidence: high,
   but pick between Accelerate vs Sleef before starting.
6. **Wolff exp short-circuit** (item 6). Drive-by cleanup when touching XY/ON.

Items 1, 2, 3, 4 together: expect ~10-20% on Phi4/SineGordon-class workloads,
~2× on BoseGas-class workloads, no change on U1 (which is item 5 territory).
Item 5 alone: expect ~2-3× on SineGordon, XY, CompactU1.
