# Cleanup review — findings & plan (untracked, gitignored)

Second full-codebase review (2026-07-07), focused on best-practices, elegance, and
structure. Three parallel review passes (structure / naming / elegance) + direct recon.
Status legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[keep]` decided-against.

Tracked files: 341. Headline: the project is not badly bloated — the real weight is
(a) stray committed files and (b) a ~200-line dispatch duplication introduced by the
step-3 traversal refactor. The rest is consistency polish.

---

## TIER 0 — stray files & dead code (deletions, ~zero risk)  ← doing now

- [x] `PLAN_review_fixes.md` — marked "do not commit", committed anyway. DELETE.
- [x] `include/reticolo/action/.impeccable/hook.cache.json` (+ dir) — impeccable skill
      cache leaked into the public header tree. DELETE + gitignore `.impeccable/`.
- [x] `scripts/read_output.py` (+ empty `scripts/`) — ZERO refs anywhere; superseded by
      `examples/*/analyze.py`. DELETE.
- [x] `runs/` — empty, untracked, not gitignored (commit trap). GITIGNORE.
- [keep] `tools/runpod/` (9 files) — flagged as unreferenced/undocumented, BUT user
      confirms it is in active use. KEEP. (Follow-up: document it in CLAUDE.md alongside
      Modal/Kaggle so it stops reading as stray.)
- [x] `docs/cuda_llr_plan.md` — header says "plan only, no code written", but the code
      shipped (`cuda/llr/` 347 LOC + 5 `*_llr_cuda.cu` apps). Stale/misleading. DELETE.
- [x] `hmc.hpp:149-154` — comment describes the ABANDONED persistent-region MD design;
      contradicts `:210-214` and the actual `run_md_`. DELETE the block. (flagged in
      review #1, never fixed.)
- [x] `action/gauge/detail/plane.hpp:56-63,87` — `wrap_lo` / `stride_nu` computed then
      only `(void)`-cast. Dead scaffolding. DELETE (keep `len_nu`, it's live).
- [keep→T1] gauge `s_full` middle tier + `s_full_plane_re_tr_sum` ×3 — the middle
      `else if constexpr` in `wilson.hpp:87` is unreachable for shipped groups, BUT
      `s_full_plane_re_tr_sum` is NOT dead: `tests/unit/test_gauge_batched_kernels.cpp:54`
      calls it directly. Removing it means updating that test → judgment call, DEFER to
      Tier 1 (not a pure dead-code deletion).

## TIER 0b — stale docs (small fixes)  ← doing now

- [x] `CLAUDE.md:106` — "Not ported: LLR" is false (5 `*_llr_cuda.cu` apps + smoke tests
      + `cuda/llr/`). FIX the claim.
- [x] `docs/writing_an_app.md` — 6 refs to RETIRED `CompactU1`/`LinkLattice`
      (lines ~282/331/342/391/440/441). UPDATED to `Wilson<gauge_group::U1>` /
      `MatrixLinkLattice`; dropped the deleted `test_wilson_u1_vs_compact` bullet.
- [x] NEWLY FOUND: a prior `LinkLattice→MatrixLinkLattice` rename MANGLED type names
      in the cuda docs — `MatrixMatrixLinkLattice<U1>` (double "Matrix"),
      `MatrixLinkLattice<U1><T>` (invalid double-bracket) + a stale "two U(1) layouts"
      (CompactU1) paragraph. Fixed the PROSE in `cuda_architecture.md:109-112` and
      `writing_a_cuda_app.md:61`.
- [ ] FOLLOW-UP: `cuda_architecture.md:75` ASCII-box diagram still has the mangled
      `MatrixLinkLattice<U1><T>` / `MatrixMatrixLinkLattice<U1><G,T>` cells — fixing the
      text breaks box alignment, so it needs a proper redraw pass (deferred, fiddly).

---

## TIER 1 — the real bloat: dispatch duplication (~250 lines)  ← deferred (needs re-verify)

- [x] **Dispatch/tiling dedup — DONE (flavor A).** Added `item_bases_<D>` (the
      neighbour-base loop: 4 copies → 1) and `traverse_dispatch_<Acc>` +
      `run_items_`/`run_ranges_` (the ndims switch + 4D tile decode: 4 copies → 1,
      shared by map/reduce AND site/complex via a C++20 templated-lambda item). The
      per-D partition + 1D vectorised path are preserved exactly → bit-identical.
      complex/traversal.hpp 320→222; traverse.hpp net −9 (added the shared dispatch,
      removed 2 internal switch copies). Net −107 lines, 4× duplication eliminated.
      186/186 macos; Modal linux-gcc OpenMP re-verify in flight.
- [ ] **`SFullCache` mixin (~24 lines).** Site/Bond/Complex/Gauge bases each hand-roll
      `last_s_full()/restore_last_s_full()/last_s_full_=NaN` (Complex repeats it for
      `s_imag`). One mixin `struct SFullCache{...}` (+ `SImagCache`). Also removes the
      asymmetry: gauge caches in the base wrapper, the others inside `s_full`.
- [ ] **`bose_gas.hpp` `std::complex ↔ cplx<T>` round-trip ×7** → one `to_cplx` helper
      in `core/cplx.hpp` (or overload the formula on `std::complex`).
- [ ] **`WindowedAction` 4 serial merge loops** (`llr/windowed_action.hpp:105-107,
      122-124, 151-153, 168-170`) are exactly `kick_add` / a folded scale — never
      converted when the integ atoms were parallelized. Reuse `kick_add`.
- [ ] `site/detail/traversal.hpp` (32-line thin wrapper binding `IdentityCombine`) could
      be dropped if `SiteAction` calls `visit_stencil(...IdentityCombine{}...)` directly.
      (`Comb` itself earns its keep — bond passes a real kernel.)

## TIER 2 — naming / namespace / structure consistency  ← deferred

- [ ] `math/gauge_group/` declares `reticolo::gauge_group` (skips `math`), inconsistent
      with `reticolo::math::su2/su3` dir-mates → `reticolo::math::gauge_group`.
- [ ] Gauge `formula/` holds host kernels, not formulas: `wilson_su2.hpp`/`wilson_su3.hpp`
      are `wilson_kernels<G>` specializations (belong in `detail/`); U(1) uniquely has
      both `wilson_u1.hpp` + `wilson_u1_formula.hpp` while SU(2)/SU(3) have no `_formula`.
      Move the kernel specs to `detail/`, keep only HD `_formula.hpp` under `formula/`.
- [ ] `traverse.hpp` vs family `traversal.hpp` — rename engine → `stencil.hpp` (matches
      its `*_stencil` functions; disambiguates). (my naming.)
- [ ] Traversal verb mishmash: `visit_stencil` / `field_apply` / `parallel_map` for "map"
      across two `detail` namespaces (partly my `field_apply`). Settle on one verb pair.
- [ ] `llr/detail/window_formula.hpp` is an HD formula filed under `detail/` not
      `formula/` → move to `llr/formula/`.
- [ ] `core/` (17 headers) mild grab-bag → at least a `core/rng/` subdir (5 RNG headers),
      maybe `core/lattice/` (lattice/matrix_link_lattice/indexing/site).
- [ ] bench consolidation: `bench_scaling_site` + `bench_scaling_su3` superseded by
      `bench_scaling_all` (this session); consider folding `bench_scalars` into
      `bench_gauge_vs_scalar`. 12 → 9 benches. (Note: `bench_scaling_su3`'s CHECK/TRAJCHK
      bit-invariance probe is now covered by `test_traverse_parallel` + Modal OpenMP.)
- [ ] Lower: `u1_detail`→`detail`; `RETICOLO_IO_DECLARE`→`_WRITER_DECLARE`; uneven `obs`
      nesting (`obs::mag` vs bare `obs`); examples 05 missing README, 09 python-only
      (move to tools?), 10 is a non-standard consumer example.

## Confirmed already-clean (no action)
Trailing-`_` convention; macro→`constexpr` idiom (`RETICOLO_TRAVERSE_L2_BYTES`,
`RETICOLO_GAUGE_K_BATCH_MAX`); `alg` dir/namespace abbreviation; integrators
(`Leapfrog`/`Omelyan{2,4}` share `detail::kick_/drift_`); the RNG menu (FastRng workhorse
+ tested Mt19937/Ranlux/Philox alternatives).
