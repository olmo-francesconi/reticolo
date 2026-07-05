# Review fix plan (untracked — do not commit)

Findings from the full-codebase review (2026-07-03). Working through P1→P6 in order.
Status: [ ] todo · [~] in progress · [x] done

## Progress
- P1 [x], P2 [x], P3 [x], P4 [x], P5 [x] — all verified: full CPU suite 188/188 green.
- P6 [x] — Modal linux-nvcc gate PASSED: full CUDA suite compiled, 64/64 CUDA
  tests green (incl. all *_llr_cuda smoke tests → exercises the P3 driver.hpp
  exchange change and the P6 reduce.cuh constant). Caveat: the gate is a Release
  (NDEBUG) build, so the check.hpp capture-guard lives in the untaken #else
  branch and was NOT compiled — verify with a Debug CUDA configure if ever used.

---

## P1 — Fix mode-B CPU exchange observable  [x]
**File:** `include/reticolo/llr/replica.hpp:146-148`
`energy()` unconditionally returns `base.last_s_full()` (= S_R). In the complex
(k_complex) path the window parameter `a` couples to **S_I**, and `sample()`
already uses `last_s_imag()`. The swap acceptance in `exchange.hpp` therefore
reweights the wrong observable → breaks detailed balance of the swap.
CUDA path is already correct (`cuda/llr/replica.hpp:111` → `constraint_value()`).
**Fix:** branch `energy()` on `Windowed::k_complex`, mirroring `sample()`:
return `last_s_imag()` in mode B, `last_s_full()` in mode A.

## P2 — Add FD force test for Wilson SU(2)/SU(3)  [x]
Largest testing hole: matrix staple force only exercised by reversibility +
float/double cross-checks, never force-vs-finite-difference.
**Fix:** add `tests/physics/test_wilson_force_consistency.cpp` (mirror the
scalar/compact-U(1) FD pattern), register in `tests/CMakeLists.txt`. FD on the
link algebra: perturb along each generator, central difference of S_full vs the
projected force component.

## P3 — Exchange Gaussian-window quadratic term  [x]
**File:** `include/reticolo/llr/exchange.hpp:22`
Current `log_p = (a_i-a_j)(E_i-E_j)` is exact only for a HARD window; the code
uses a GAUSSIAN window, so the exact acceptance carries a quadratic cross-term.
General (per-replica δ) form, with q = constraint value (= energy() after P1):
```
log_p = (a_i-a_j)(q_i-q_j)
      + (q_i-q_j)/2 * [ (q_i+q_j-2*E_n_i)/δ_i²  -  (q_i+q_j-2*E_n_j)/δ_j² ]
```
**Fix:** implement the exact term in `try_exchange` (read E_n() and delta() from
both replicas). Also check + align the CUDA exchange path for consistency.

## P4 — CPU perf one-liners  [x]
1. `include/reticolo/algorithm/integ_ops.hpp:29,41` — drift/kick coefficient is
   a full `std::complex` for complex fields (complex×complex + NaN-recovery
   branch, blocks vectorization). Use `real_scalar_t<F>` (already imported in
   integrators.hpp:40). Applies to `drift_field` and `kick_add`.
2. `__restrict` on force/kick/drift OUTPUT buffers (`integ_ops.hpp`
   drift_field/kick_add; scalar force bodies). Pattern already in `su3_ops.hpp`.
   Verify with the cross-ISA asm probe before/after.

## P5 — IO durability + LLR log placement  [x]
1. `src/io/writer.cpp:456,469` — per-replica `/a` and `/dE` series use default
   4096-row chunk → nothing flushes until the `Series` dtor at run end; a crash
   loses the entire adaptation history. Use a smaller chunk (or periodic flush)
   for these low-cardinality series.
2. `include/reticolo/llr/driver.hpp:109` — unconditional `iter("RM", …)` log
   call sits INSIDE the OpenMP replica region and does a mutex-protected
   ofstream flush per replica per RM iter. `smoothed_driver.hpp` does it right
   (serial section). Move it out / gate silent.

## P6 — CUDA guards  [x]
1. `include/reticolo/cuda/check.hpp:33-37` — debug `RETICOLO_CUDA_CHECK_LAUNCH()`
   calls `cudaDeviceSynchronize()`, illegal during graph capture. Make it
   capture-aware (`cudaStreamIsCapturing`) or document NDEBUG requirement.
2. `include/reticolo/cuda/reduce.cuh:25,207` — two `kMaxGrid=1024` constants
   must stay equal (public one sizes caller `partials_` scratch). Derive one
   from the other.

---

## Minor findings

### Done (batch, verified: 188/188 CPU tests green)
- [x] `phi4` s_full_and_force redundant phi*phi — added `phi4_force_site_p2`
  (phi²-taking) to the shared formula; plain overload delegates. Bit-identical,
  so CUDA Phi4 functor unaffected.
- [x] `math/su2_ops.hpp` project_su2 epsilon guard — degenerate quaternion →
  identity (mirrors exp_su2's h≈0 guard).
- [x] Gauge device-math cross-link — CPU su2_ops/su3_ops now point at the
  cuda::SU{2,3}Device copies + the test_cuda_su{2,3} drift guard.
- [x] Stale docs — CLAUDE.md app line count corrected to ~75–130; architecture.md
  now scopes the "no virtual" invariant (cli::Parser VarSlotBase is off-hot-path).
- [x] `alg::Hmc` lifetime guard — deleted rvalue-action constructor (field/rng
  are non-const lvalue refs, already reject temporaries).

### Left (deliberately not done)
- Bose combined force+kick streams the complex field twice/MD step — real win,
  needs a new split-helper variant; scoped separately.
- CUDA measurement-graph capture: fold {traj + reduction + accumulate} into one
  captured graph — biggest small-V GPU-LLR lever; separate work.
- CUDA scalar stencil 4D halo tiling — throughput ceiling; genuinely hard,
  deferred as the authors chose.
- Optional fused-kernel detection inline `requires` → named concept/static_assert
  (prevents silent slow-path fallback) — moderate, not urgent.
- No ⟨e^{-ΔH}⟩=1 test — belongs in examples/, not unit tests (no MC in tests).
- Marginal/skip: rng log_batch; su3::expi_lmul_slab double slab read; CUDA
  mh_accept+bump_counter node merge; Indexing pool concurrency test.
