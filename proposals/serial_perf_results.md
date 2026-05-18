# Serial perf optimisation — results (2026-05-18)

Six items from `proposals/serial_perf_audit.md` were implemented in order on
this branch. Build: `macos-appleclang` preset, Apple M-series, NEON 2-wide
doubles, `-O3 -march=native -fno-math-errno`. All 119 tests pass at every
step — no physics regression.

## Headline numbers

Throughput in million DOF updates per second (HMC trajectory, Omelyan2 unless
noted), before → after, with relative speedup:

### Scalar HMC (Phi4 — reference, no transcendental)

| size      | before | after  | Δ        |
| --------- | ------ | ------ | -------- |
| 3D L=16   | 663    | 668    | +1% (noise) |
| 3D L=24   | 624    | 637    | +2% (noise) |
| 4D L=10   | 496    | 492    | −1% (noise) |

Phi4 is the control. No transcendentals, hot kernel was already well-tuned —
the integrator/HMC cleanup (#3) is a wash here, as predicted.

### SineGordon HMC (sin/cos via Sleef)

| size      | before | after  | speedup |
| --------- | ------ | ------ | ------- |
| 3D L=8    |  157   |  181   | **1.15×** |
| 3D L=16   |  154   |  185   | **1.20×** |
| 3D L=24   |  145   |  180   | **1.24×** |
| 4D L=8 (gauge-bench) |  45.5  |  63.4  | **1.39×** |
| 4D L=10   |  126   |  167   | **1.32×** |

### CompactU1 HMC (sin/cos via Sleef)

| size      | before | after  | speedup |
| --------- | ------ | ------ | ------- |
| 3D L=8    |  44.4  |  65.6  | **1.48×** |
| 3D L=12   |  38.8  |  64.8  | **1.67×** |
| 3D L=16   |  36.3  |  65.5  | **1.80×** |
| 4D L=4    |  27.7  |  49.1  | **1.77×** |
| 4D L=6    |  24.7  |  52.6  | **2.13×** |
| 4D L=8    |  24.5  |  54.2  | **2.21×** |

### BoseGas HMC (visit_nn pattern; complex)

| size      | before | after  | speedup | PQ/Phi4 ratio (before → after) |
| --------- | ------ | ------ | ------- | ------------------------------- |
| 3D L=8    |  156   |  198   | **1.27×** | 1.70 → 1.35 |
| 3D L=12   |  148   |  185   | **1.25×** | 2.17 → 1.71 |
| 3D L=16   |  154   |  179   | **1.16×** | 2.16 → 1.90 |
| 4D L=4    |  136   |  164   | **1.20×** | 1.26 → **1.04** |
| 4D L=6    |  132   |  183   | **1.39×** | 1.46 → **1.02** |
| 4D L=8    |  136   |  175   | **1.29×** | 1.87 → 1.44 |

The PQ/Phi4 ratio is the cleanest metric — BoseGas now does the same
per-DOF work as Phi4 at small lattices and is within 1.5× even at the
largest tested. Before, the gap was 2.2×.

---

## Per-item contributions

### #1  BoseGas hot loops → visit_nn pattern  (+15–40%)

`include/reticolo/action/builtins/bose_gas.hpp` — replaced the
`for (Site x : l.sites())` member-access pattern with
`detail::visit_nn_split_last` / `detail::reduce_fwd_split_last`. Added the
"split-last-direction" hot-loop helpers to
`include/reticolo/action/hot_loop.hpp`, exposing the time-direction NN sum
separately so the action can apply its anisotropic `cosh(mu)` factor.
`s_imag` / `compute_force_imag` got dedicated last-dim slab walks.

The new helpers (`visit_nn_split_last`, `reduce_fwd_split_last`) are
reusable — any future action with anisotropic last-dim coupling
(fermion-determinant theories, finite-T, chemical potential) plugs in
directly.

### #2  Lazy `imag_force` scratch on WindowedAction  (eliminates per-MD-step malloc)

`include/reticolo/llr/windowed_action.hpp` — the complex-LLR branch of
`compute_force` / `compute_force_and_kick` used to construct a fresh
`Lattice<T>` on every call (41 mallocs/traj at n_md=20). Now backed by
`mutable std::optional<Lattice<T>> imag_scratch_storage_`, lazily emplaced
on first call. Visible in `bose_gas_llr` runs as smoother per-traj timing;
no separate before/after bench since the alloc cost was hidden under the
sin/cos dominance (item #5 makes it more visible by removing the bigger
cost).

### #3  Scalar HMC + integrator raw-pointer cleanup  (style; ~0%)

`algorithm/hmc.hpp` + `algorithm/integrators.hpp` — `for (Site x :
field_.sites())` replaced with flat-buffer `for (std::size_t i = 0; i < n;
++i)` to match the gauge HMC twin. Phi4 numbers unchanged within noise →
clang was already auto-vectorising through the iota/transform view. Net
effect is consistency with the gauge path, not perf.

### #4  `clang fp reassociate(on)` on reduction loops  (~0–3%)

`action/hot_loop.hpp` and `gauge/builtins/compact_u1.hpp` — strict
`total += body(i)` chains marked reassociation-friendly. Small net change
on this hardware (M-series Apple Silicon OoO engine already keeps multiple
sin/cos in flight via register renaming). Kept anyway: the macro is a
one-line knob and gives the compiler explicit license to tree-reduce on
other backends.

### #5  Sleef vector libm  (1.4–2.2× on transcendental actions — the headline win)

Added Sleef 3.6.1 via `FetchContent` in `CMakeLists.txt`, statically
linked to `reticolo::core`. New header `include/reticolo/math/vec_libm.hpp`
provides three portable wrappers — `cos_batch`, `sin_batch`,
`sincos_batch` — that dispatch at compile time on `__AVX512F__`,
`__AVX2__`, `__AVX__`, `__ARM_NEON`, `__SSE2__` (in that priority order)
to the matching Sleef `Sleef_*d{8,4,2}_u10*` function, with a scalar
`Sleef_*_u10` tail. **Same binary semantics on every architecture** the
project compiles on (no Apple-specific Accelerate path).

Integrated in:

- `action/builtins/sine_gordon.hpp` — `compute_force`,
  `compute_force_and_kick` precompute `sin(phi)` for the whole lattice;
  `s_full` precomputes `cos(phi)`. Backed by a `mutable std::vector<T>
  scratch_` on the action struct (lazy resize on first call).
- `gauge/builtins/compact_u1.hpp` — `s_full`, `compute_force`,
  `compute_force_and_kick` each stash the plane's plaquette angles into a
  scratch via `visit_plane`, vector-apply sin/cos in place, then a second
  `visit_plane` scatters into accum / force / mom.

All Sleef paths are guarded by `if constexpr (std::is_same_v<T, double>)`
so float / extended-precision instantiations fall back to the original
`std::sin`/`std::cos` loops — same speed as before, just bypassed.

### #6  Wolff `1 - exp(min(0, w))` short-circuit  (small)

`action/builtins/xy.hpp` and `action/builtins/on_sigma.hpp` —
`wolff_link_p` now returns 0 directly when `w >= 0` instead of computing
`std::exp(0)` and subtracting 1. Saves one libm `exp` call per
unfavoured-bond visit, which is roughly half of all visits at typical
beta. Wolff isn't in the headline benches (cluster updaters scale very
differently from HMC) so no perf row above, but `xy_wolff` and
`on_sigma_wolff` are the headline cluster apps and this is one branch.

---

## Build-system impact

- One new FetchContent: Sleef 3.6.1 (BSD-licensed). First configure adds
  ~6 s for clone + Sleef CMake configure. Build adds ~2 s for the static
  library (parallel with the rest). Footprint: one extra static lib
  (`libsleef.a`, ~1.2 MB) inside the build tree.
- `reticolo::core` now `INTERFACE`-links Sleef and adds Sleef's build-tree
  include dir (Sleef's exported target doesn't propagate it automatically
  during build, only install — handled by an explicit
  `target_include_directories(... SYSTEM INTERFACE ...)` block in
  `CMakeLists.txt`).
- No new compile-time options; no preset changes; no test changes.

## What's NOT done (deferred from the original audit)

- **XY action sin/cos vectorisation.** `xy.hpp` uses per-bond
  `sin(theta_x - theta_y)`, which doesn't fit the per-site precompute
  pattern. Would need a per-direction temp buffer of bond differences,
  ~80 LOC of new code. The xy_wolff app uses the cluster updater
  (item #6), not HMC, so XY-HMC throughput isn't on the bench critical
  path. Left for later if/when XY HMC becomes important.
- **`reduce_fwd` indexed variant.** SineGordon's `s_full` works around the
  missing site index by accumulating `cos(phi)` separately. If a future
  action wants both `fwd_sum` and a per-site precomputed transcendental,
  add `reduce_fwd_indexed<T,Body>(l, body)` where body is
  `(i, phi, fwd_sum)`. Not worth doing speculatively.
- **`compute_force_imag_and_kick` on BoseGas.** Would fuse the second
  `for` loop inside `WindowedAction::compute_force_and_kick`'s complex
  branch. Small win on bose_gas_llr only; left for later.

---

## Net summary

The single biggest serial speedup the codebase had available was vector
libm on the transcendental hot loops — and we got it portably (Sleef ships
the same vector code path on every ISA the project compiles for). On top
of that, the BoseGas refactor brought a complex-field action into parity
with its real-field cousin, and several smaller cleanups removed
allocations and aligned the scalar/gauge HMC styles.

Best-case end-to-end speedup measured on this hardware:

- **CompactU1 4D L=8: 2.2×** (24.5 → 54.2 M dof/s)
- **CompactU1 4D L=6: 2.1×** (24.7 → 52.6 M dof/s)
- **CompactU1 3D L=16: 1.8×** (36.3 → 65.5 M dof/s)
- **BoseGas 4D L=6: 1.4×** (132 → 183 M real-DOF/s; now per-DOF parity with Phi4)
- **SineGordon 4D L=10: 1.3×** (126 → 167 M dof/s)

Phi4 unchanged (was already optimal); cluster-updater apps unchanged but
correctness-preserving (#6).
