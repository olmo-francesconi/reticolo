# Reticolo — in-depth performance audit (2026-05-19)

Branch: `rewrite/v3`. Synthesis of five parallel deep-dives over core layout,
action kernels, integrators+algorithms, RNG+math, and the build system.

## M-tier outcomes (second pass)

| Item | Status | Notes |
|------|--------|-------|
| **M1+M2+M5** LLR `s_full` reuse + complex fused-kick | 🟡 partial | Shipped a clean refactor: `hmc.hpp::hamiltonian_()` split into `kinetic_()` + explicit `s_full` call; `last_s_full_` cache populated correctly across accept/reject; `Replica::last_windowed_action()` accessor exposes it. Full user-facing reuse (per-kick s_full elimination, base-action caching for Exchange) needs either incremental s_full updates in the action API or fusing s_full into compute_force_and_kick (gauge-group API change) — both architecturally larger than the M-tier suggests. Plumbing in place for a future session. |
| **M3** `MatrixLinkLattice` aligned allocator | 🔴 skipped | On AArch64 NEON the unaligned penalty for `ldp q` is negligible. Mild possible win on x86 AVX-512 only; `std::vector` already aligns to 16 B; bumping to 64 B costs ergonomics on every gauge call site for marginal gain. |
| **M4** BoseGas combined force + fused kick | 🟢 SHIPPED | Added `BoseGas::compute_force_and_kick` (satisfies `HasFusedKick`) and `BoseGas::compute_force_combined_and_kick(l, mom, scale_r, scale_i, k_dt)`. `WindowedAction::compute_force_and_kick` mode B picks the combined kernel up via `requires`-detection — eliminates the imag_scratch round-trip the audit flagged. Concept-driven dispatch means any future complex-action that exposes the same primitive gets the speedup automatically. Tests pass 167/167. Bose-gas-LLR wall-time delta couldn't be cleanly benched (pre-existing HMC stability issues in the app at most parameter sets — happens in pre-M4 binary too). |
| **M6+M8** Sleef Box-Muller in `normal_fill` + batched `sample_algebra_slab` | 🟢 SHIPPED | Classical Box-Muller (no polar rejection) via Sleef `sincos_batch`; both `normal()` and `normal_fill()` use the same Sleef scalar path so they're bit-identical (the existing unit test enforces this). SU(2)/SU(3) `sample_algebra_slab` pre-fill via `normal_fill` then scatter (with `* k_inv_sqrt2` folded into the scatter so there's no extra full sweep). **`bench_rng` FastRng momentum-sampling: +17–19% on `Lattice<double>`, `Lattice<complex>`, `LinkLattice<double>` for both 3D L=24 and 4D L=8.** HMC wall-time impact stays in the noise (momentum init is &lt;1 % of trajectory) but the kernel is now correctness-clean and auto-vectorisation-friendly. |
| **M7** SU(3) `plaq_re_tr` row-0-only trace | 🔴 skipped | `plaq_re_tr` lives on the `s_full` path which is ~1 % of HMC trajectory cost (called 2× per traj for accept/reject). Even a 3× speedup saves &lt;0.05 s out of 7.6 s. Also the audit's "row 0 only" claim doesn't check out — `Re Tr(AB·DC†) = Σ_{i,j} AB_{ij}·conj(DC_{ij})` needs every row. |
| **M10** build flags (`-mcpu=apple-m1`, IPO, Sleef ISA pin) | 🔴 not shipped | Tested replacing `-march=native` with `-mcpu=apple-m1` on AppleClang/arm64 — neutral on FMA-bound kernels (the features `-march=native` silently strips on that target are crypto / fp16 / shuffle, irrelevant to double-precision FMA). Reverted to keep the build host-portable: `-march=native` works on every target (AppleClang/arm64 just treats it as a no-op rather than failing), whereas hard-coding `-mcpu=apple-m1` would break on Apple M2+, x86, and Linux. Real audit-cited Sleef-ISA pin was already in the build config. |

## Status — 2026-05-19 implementation pass

| Item | Status | Notes |
|------|--------|-------|
| **H4** link lattice loop order | ✅ shipped | `link_lattice.hpp` `for_each_update` swapped to μ-outer / site-inner. All 167 tests pass. |
| **H3** Wilson<U(1)> Sleef path | ✅ shipped | `gauge_group/u1.hpp` gained batched `compute_force` (sin_batch), `s_full_plane_re_tr_sum` (cos_batch), `compute_force_and_kick`. `Wilson<U(1)>` now matches CompactU1 throughput (1.0–1.8% gap vs. previous ~2× slowdown). Bench: `bench_wilson_vs_compact_u1`. |
| **H2** Wilson<G>::compute_force_and_kick | ✅ shipped | Added for all three gauge groups (U(1), SU(2), SU(3)). Generalised the integrator's fused-kick concept (`integrators.hpp::kick_`) so it accepts any `Field` type (no longer hardcoded `LinkLattice<F>`). Integrator now skips the materialise-then-stream `force` pass for matrix gauge actions. |
| **H1** SU(N) slab `compute_force` | 🟢 SHIPPED (portable batching) | After two failed attempts that confirmed the bottleneck — (1) 3-sweep `v_scratch[18][ns]` was 1.75× slower from memory round-trip, (2) disassembly showed 100% scalar emission from the per-site kernel due to interleaved Re/Im storage blocking SLP — the fix that worked: **B=8-site batching with split Re/Im scratch**. Per μ direction, process sites in batches of 8, load matrix data into `..._re[N²][8]` / `..._im[N²][8]` AoSoA, run nested matmul/TA/scatter with innermost `for b in 0..8` over stride-1 packed buffers. Each `cr += ar·br - ai·bi` becomes pure stride-1 SIMD with no lane permutation — the compiler auto-vectorises trivially on any target (NEON 2-wide, AVX2 4-wide, AVX-512 8-wide). Math identical to the scalar reference; tail sites (ns % 8 != 0) fall back to the original scalar code. **Result: SU(3) 4D L=8 HMC 10.3s → 7.63s mean = 26% wall-time reduction.** Disassembly: **0 scalar `fmla d`, 96 `fmla.2d` + 96 `fmul.2d` + 112 `fadd.2d` + 100 `ldp q` + 36 `stp q`** in the SU(3) hot path. SU(2) flat (~7.8s) because H6 SU(2) drift batching already captured its dominant headroom, but the code is now consistently structured and auto-vectorises portably. 167/167 tests pass. |
| **H6** slab `exp_su{2,3}` via Sleef | 🟢 SU(2) shipped (14%), 🟡 SU(3) neutral | SU(2) `expi_lmul_slab` rewritten as 3-pass (β scratch → `sincos_batch` → assemble V·U). Measured **14% gain** on SU(2) HMC 4D L=8 (7.77s vs 9.0s baseline mean, 200 trajectories) and the full test suite went from 12.0s → 10.6s. SU(3) `expi_lmul_slab` ships the same shape — multi-pass over 13 scratch slabs (ratio → `acos_batch` → t3 → `sincos_batch` → u, w → `sincos_batch` → assembly) including the small-c1 Taylor fallback per site. Correct (reversibility + weak-coupling tests pass) but **net-zero on SU(3) HMC wall time** at this lattice — drift is a smaller fraction of SU(3) HMC than the audit estimated, and the multi-pass scratch round-trip eats the trig-batching gain. Kept in tree: gain will materialise once an H1 attempt makes `compute_force` cheap enough for drift to dominate. Also added `acos_batch` to `vec_libm.hpp`. |
| **H5** constexpr-D Indexing + `uint32_t` neighbours | 🟡 already partly done | Audit impact overstated. Scalar HMC's compile-time-D fast path already exists as hand-written `visit_nn_1d_/2d_/3d_/4d_` (`detail/helpers.hpp`) — scalar Phi4/SineGordon/XY bypass the slow gather entirely. The `uint32_t` neighbour-table proposal halves the table from 128 KB → 64 KB at 4D L=8 — but it already fits L1 at `size_t`, so no measurable cache win at typical sizes. Real impact only at L≥12 4D where the table starts spilling L1. Mechanical change touching ~10 files; defer to a dedicated session for the larger-lattice regime. |

Numbers (M2 Pro, macOS, appleclang Release):
- `bench_wilson_vs_compact_u1` — Wilson<U(1)> = 5.6–7.0 × 10⁷ dof updates/s, **0–2 % gap to CompactU1** across 3D L=12, 4D L=6, 4D L=8 (was a latent ~2× regression target).
- SU(2) HMC 4D L=8, 200 traj, n_md=10: **7.79 s** post-H1+H6 vs **9.0 s** pre-pass mean → **~13% wall-time reduction**.
- SU(3) HMC 4D L=8, 50 traj, n_md=10: **7.63 s** post-H1+H6 vs **10.3 s** pre-pass → **26% wall-time reduction**.
- Full test suite: 12.0 s → 10.6 s wall (167/167 pass throughout).
- M10 (build flags): tried `-mcpu=apple-m1` vs `-march=native`; neutral on FMA-bound kernels (the features `-march=native` strips on appleclang/arm64 are crypto/fp16/shuffle, irrelevant to double-precision FMA).

All gains come from compiler-friendly portable C++ — no intrinsics. The H1 fix in particular works because the split-Re/Im AoSoA layout + nested `for b` innermost loop gives the auto-vectorizer a textbook stride-1 vector op pattern.

**Cross-compiler validation (M2 Pro, same source):**

| | Xcode AppleClang | Homebrew LLVM 22 |
|---|---|---|
| SU(3) HMC 4D L=8 | 7.63 s | 7.77 s |
| SU(2) HMC 4D L=8 | 7.79 s | 7.67 s |
| `fmla.2d` (whole binary) | 2221 | 2237 |
| `fmul.2d` | 1756 | 1813 |
| `fadd.2d` | 2048 | 2083 |
| All 167 tests | ✅ | ✅ |

Both compilers fully auto-vectorize the batched kernel to NEON 2-wide, with ~2 % timing variance and near-identical instruction counts. Same source → same SIMD strategy on two independent LLVM versions.

**Cross-target SIMD scaling (Homebrew LLVM 22, `clang++ -target ... -O3 -std=c++23 -S`):**

A minimal driver containing only `SU3::compute_force_and_kick` was cross-compiled to seven targets. The kernel auto-vectorizes to the widest available SIMD on each, with no source changes:

| Target ISA | 8-wide (zmm 512b) | 4-wide (ymm 256b) | 2-wide (xmm/NEON 128b) | scalar `sd` |
|---|---|---|---|---|
| aarch64 NEON (Apple M1) | — | — | **840** | 1468 |
| x86 haswell (AVX2) | 0 | **612** | 0 | 1180 |
| x86 zen3 (AVX2) | 0 | **612** | 0 | 1180 |
| x86 skylake-avx512 (default → 256b) | 0 | **836** | 0 | 1180 |
| x86 skylake-avx512 `-mprefer-vector-width=512` | **750** | 46 | 8 | 392 |
| x86 sapphirerapids `-mprefer-vector-width=512` | **750** | 46 | 8 | 392 |
| x86 zen4 (AVX-512, default) | **750** | 46 | 8 | 392 |
| x86 zen5 (AVX-512, default) | **750** | 46 | 8 | 392 |

- aarch64 maxes at 128-bit NEON → 840 2-wide vector ops.
- AVX2 targets land 612–836 ops in 256-bit ymm.
- AVX-512 targets land 750 ops in 512-bit zmm; scalar ops drop from 1180 → 392 (-67 %).
- Skylake-AVX512 defaults to 256-bit due to the original Skylake-X frequency throttling; Zen 4/5 and Sapphire Rapids have no penalty and clang picks 512-bit by default for them.

This is direct evidence that the H1 portable batched layout (`split Re/Im AoSoA` + innermost `for b in 0..K_BATCH`) is genuinely architecture-independent: same source, same hot path, every modern ISA target gets its natural SIMD width.

---


Impact tags:
- **H** = >10 % wall-time potential on a representative HMC trajectory
- **M** = 2–10 %
- **L** = <2 %

---

## 1. Top wall-time wins (do these first)

### H1. Slab / tile `Wilson<SU2|SU3>::compute_force` and `s_full`
- Files: `include/reticolo/action/detail/gauge_group/su{2,3}.hpp` (su3:91–156, su2:111–167), `include/reticolo/action/wilson.hpp:46,73,85`.
- Current pattern: per (μ, s) the `load_link` lambda scatter-reads 8 (SU2) or 18 (SU3) stride-`ns` columns into stack scratch, then runs scalar `mul_3x3` / `mul_adj_3x3`. The same `U_ν(s)` block is re-gathered 2·(d−1) times — for 4D SU(3) that's 6 redundant per-link gathers, all stride-`ns` ≥ 32 KB.
- Fix: process B≈64 sites per slab so the 18 stride-1 component streams stay hot in L1; for each slab, gather each block once, accumulate staples in a contiguous AoS-of-18 scratch, scatter once. Same shape as the U(1) `visit_plane` path already in `gauge_helpers.hpp`.
- Expected: **2–4× on SU(3) HMC**, ~1.5× on SU(2). This is the single biggest remaining structural win.

### H2. `Wilson<G>::compute_force_and_kick` missing for every matrix gauge group
- Files: `include/reticolo/action/wilson.hpp`, `include/reticolo/action/detail/gauge_group/{su2,su3}.hpp`, `include/reticolo/algorithm/integrators.hpp:36-41`, `include/reticolo/algorithm/integ_ops.hpp`.
- `HasLinkFusedKick` fails for SU2/SU3, so every MD kick materialises a full force `LinkLattice` (18·ns doubles SU(3)) then re-reads it in `kick_add`. That's an extra read+write+read pass on every step.
- Fix: add `Wilson<G>::compute_force_and_kick(field, mom, scale)`; SU(2)/SU(3) inline the final TA scatter as `m_mu_blk[k*ns+s] += k_dt * neg_b_oN * ta[k]`.
- Expected: **10–20 % on SU(N) HMC**.

### H3. `Wilson<U(1)>::compute_force` uses scalar `std::sin`
- File: `include/reticolo/action/detail/gauge_group/u1.hpp:59`.
- `CompactU1` already runs `sin_batch` via Sleef (`compact_u1.hpp:158–205`); the new unified `Wilson<U1>` path regresses to per-plaquette scalar sin. As soon as `CompactU1` is retired (per the v3 plan), U(1) HMC drops ~2× at L=8⁴.
- Fix: U(1) override of `Wilson<G>::compute_force` that mirrors the CompactU1 plane-pass + `sin_batch`.
- Expected: **~2× on Wilson<U1>** before CompactU1 is removed; required to avoid regression.

### H4. `LinkLattice::for_each_update` strides by `nsites` on the inner index
- File: `include/reticolo/core/link_lattice.hpp:97–107`.
- Loops site-outer, μ-inner against direction-major storage (`flat = μ·nsites + i`), so each μ step jumps `nsites` doubles → a guaranteed L1 miss per link, every link.
- Fix: μ-outer, site-inner. If the current order is mandated by RNG bit-stability, gate the swap behind a second overload (`for_each_update_unstable_order`) for HMC / heatbath where draw order doesn't matter.
- Expected: **bandwidth-bound → stride-1**; multi-× on any update that walks all links.

### H5. No neighbour-loop fast path: `Indexing` is runtime-sized, neighbour table is dependent-load
- Files: `include/reticolo/core/indexing.hpp:42–69`, `include/reticolo/core/lattice.hpp:46–47`.
- `next_` / `prev_` are flat `vector<size_t>` of stride `ndims`; each NN step is `next_[s·d + μ]` — runtime mul by `d` and a `size_t` (8 B) dependent load. For 4D that's 8 dependent loads per Phi4 update, blocking SIMD.
- Fixes (compound):
  1. Template `Indexing` (or a thin wrapper) on `Ndim` so `ndims()` is constexpr and the NN loop unrolls; expose `template<size_t D> visit_nn(Site,F&&)`.
  2. Add per-direction stride-1 arrays `next_mu(μ)` / `prev_mu(μ)` of length `nsites` so fixed-μ sweeps are contiguous, prefetchable.
  3. Store the neighbour table as `uint32_t` (≤ 2³¹ sites is plenty) — halves footprint, doubles L1/L2 residency on the latency-critical chain.
  4. Mirror (2) for `even_next_` / `odd_next_` so EO sweeps are one dependent load per neighbour, not two.
- Expected: **~2× on small-D scalar Metropolis / HMC** (Phi4, SineGordon, XY).

### H6. Batch `exp_su2` / `exp_su3` transcendentals across a slab
- Files: `include/reticolo/math/su{2,3}_ops.hpp` (su3:171–307, su2:171–191), `include/reticolo/math/vec_libm.hpp`.
- Each MD step calls per-site scalar `sin/cos/sqrt/acos` via libm — function calls force register spills, defeat vectorisation. Sleef + `sincos_batch` is already in the tree.
- Fix: 3-pass slab kernel: (1) compute `u`, `w` arrays for the slab, (2) `sincos_batch` over `u`, `w`, (3) assemble `f0/f1/f2`. Branch on `h > 1e-12` becomes a mask / branchless polynomial.
- Expected: **1.5–2× on the MD integrator** for gauge actions.

---

## 2. Medium wins

### M1. `WindowedAction::compute_force` re-sweeps `s_full` after the force pass
- File: `include/reticolo/llr/windowed_action.hpp:136`.
- Real LLR path calls `base.compute_force` then `base.s_full(l)` — two full sweeps per MD step.
- Fix: have base actions return `(force, S)` from one pass, e.g. `compute_force_and_full(l, force, &s)`; or extend the fused-kick concept to carry an incremental `S` along the trajectory.

### M2. `Replica::sample()` + `Exchange::energy()` re-evaluate `s_full` after each traj
- Files: `include/reticolo/llr/replica.hpp:62–87`, `include/reticolo/llr/exchange.hpp:22–30`.
- HMC's `hamiltonian_` already computed `s_full(phi)` for the accept/reject step; LLR sample / replica exchange recompute it. With N replicas × exchange attempts that's 2N full sweeps per cycle.
- Fix: cache last-known S inside `Replica` (record `s_full` at end of trajectory) and reuse in both `sample()` and `energy()`.

### M3. `MatrixLinkLattice` storage has no alignment guarantee
- File: `include/reticolo/core/matrix_link_lattice.hpp:43–50`.
- `data_` is `std::vector<double>` (16 B alignment). Slab kernels want 32/64 B aligned starts for AVX2/AVX-512 aligned loads + `__builtin_assume_aligned`.
- Fix: aligned allocator (64 B), pad each (μ, k) slab to a cache-line multiple, add `assume_aligned` at slab-kernel head.

### M4. `BoseGas::compute_force_imag` overwrites, no fused kick
- File: `include/reticolo/action/bose_gas.hpp:172`.
- LLR complex path keeps a separate imag scratch lattice and merges it back.
- Fix: `compute_force_combined_and_kick(l, mom, a, b, k_dt)` that scatters `mom[i] += k_dt·(a·F_R + b·F_I)` in one `visit_nn`. (Already on the `serial_perf_results.md` deferred list — confirm it's still planned.)

### M5. `WindowedAction` complex fused-kick: second streaming pass
- File: `include/reticolo/llr/windowed_action.hpp:146–167`.
- Real branch fuses; complex branch does base-fused-kick then a second `mom += k·ip[i]` over the imag scratch. Bandwidth-bound on large lattices.
- Fix: fold into a `force_imag_kick(mom, scale)` sibling on base actions.

### M6. Box-Muller polar rejection wastes ~21 % of uniforms
- File: `include/reticolo/core/rng.hpp:87–104`.
- `normal()` rejects on `s≥1 || s==0` (acceptance π/4 ≈ 0.785). Vol·Nd Gaussians per HMC traj makes this non-trivial.
- Fix: classical Box-Muller via Sleef `sincos_batch` in the `normal_fill` batched path. Ziggurat is faster still but is more code; classical-BM is free because the Sleef infra exists.

### M7. SU(3) `plaq_re_tr` does two full `mul_3x3` for a trace
- File: `include/reticolo/math/su3_ops.hpp:54–118`.
- Re Tr(A·B·C†·D†) needs only row 0 of A·B (6 reals) dotted into row 0 of D·C — ~3× cheaper than two full 9-output matmuls.
- Fix: dedicated `re_tr_quad(A, B, C, D)` doing row-0 only.

### M8. Slab `sample_algebra` draws Gaussians one at a time
- Files: `include/reticolo/math/su2_ops.hpp:312–329`, `include/reticolo/math/su3_ops.hpp:485–521`.
- 8 (SU2) / 8 (SU3) `rng.normal()` calls per site, each with the rejection branch.
- Fix: pre-fill a flat `N·ns` buffer with `rng.normal_fill()` once per momentum init, then scatter into the storage layout.

### M9. No site-parallel iteration helper
- Files: `include/reticolo/core/lattice.hpp:54–57`, `include/reticolo/core/link_lattice.hpp:86–89`.
- Nothing in the core iterates sites in parallel even when the project chooses to keep OpenMP wired in. The `RETICOLO_ENABLE_OPENMP` flag is ON by default but no `#pragma omp parallel for` exists in any hot kernel.
- Fix: either explicitly disable OpenMP (consistent with the "serial-only" v3 plan) **or** add `for_each_site_parallel(Body)` and wire it into HMC force / momentum refresh / kinetic. Mutually exclusive choices; pick one and stop paying for the libomp link in every binary.

### M10. Build flags — three sub-items
- File: `CMakeLists.txt:122–130`.
  - **`-march=native` is a no-op on AppleClang/AArch64.** Branch the option: `-mcpu=apple-m1` on `APPLE && AppleClang && arm64`, else `-march=native -mtune=native`. Expected 5–15 % on M-series.
  - **Add safe sub-flags to fast-math** without breaking MCMC reproducibility: `-fno-signed-zeros -fno-trapping-math -freciprocal-math -fno-rounding-math`. Avoid `-ffast-math`, `-fassociative-math`, `-ffinite-math-only`.
  - **Enable IPO/LTO for Release** via `check_ipo_supported` + `CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON`. Header-only core already inlines, but app TUs ↔ writer.cpp ↔ Sleef can still benefit (2–5 %).
- File: `CMakeLists.txt:58–74` — pin Sleef ISA flavours (`SLEEF_ENFORCE_AVX2`, `SLEEF_ENFORCE_AVX512F` where supported) so Linux CI hosts with AVX-512 don't silently fall back to AVX2.

---

## 3. Smaller wins / hygiene

- **L** `integrators.hpp:67-71 / 104-110 / 152-162` — Leapfrog / Omelyan2 / Omelyan4 have a runtime `step+1 < n_md ? dt : half_dt` branch in the trajectory loop. Peel manually: `n_md-1` full-stride iterations then a trailing half-kick.
- **L** `hmc.hpp:82–109` — snapshot / restore use hand loops; substitute `std::memcpy` / `std::ranges::copy` to ensure autovectorisation; consider hoisting to a `Field::assign(other)` helper.
- **L** `integ_ops.hpp:32-34,44-46` — drift/kick AXPY loops lack `__restrict__` / `#pragma omp simd` on the raw pointers. TBAA usually saves it on clang; explicit `__restrict__` closes rare load-after-store stalls on GCC.
- **L** `wolff.hpp:50–53,67–74` — `mark_` choice (uint32 generation bump) is good; consider 32-bit site indices for `stack_` / `cluster_sites_` if cluster sizes routinely exceed 10⁷.
- **L** `bose_gas.hpp:63,77,93` — `std::cosh(mu)` recomputed per call; lazy-cache.
- **L** `on_sigma.hpp:43,55,66` — bypasses `visit_nn` because the value type is `std::array<T,N>`; acceptable until OnSigma gets an HMC path.
- **L** `sine_gordon.hpp:121` — `mutable std::vector<T> scratch` is fine in serial; flag for the day parallelism returns.
- **L** `math/{su2,su3}_ops.hpp` — per-site stack scratch (`double a_s[18]`, `double b_s[8]`) is unaligned; `alignas(64)` helps autovec.
- **L** `wilson.hpp:73` — try `__restrict__` on `u_mu_blk` / `f_mu_blk` in the hot loops after profiling.
- **L** `cmake/ReticoloWarnings.cmake` + `CMakePresets.json:55–64` — Debug preset overwrites `CMAKE_CXX_FLAGS`; switch to `_DEBUG_INIT`. Not perf-relevant.
- **L** PCH / unity build — pure iteration-time win, not runtime.

---

## 4. Things that look fine

- No virtual dispatch, no `std::function`, no type erasure in any hot path — concepts/templates only.
- `shared_ptr<Indexing>` aliasing is dodged via `indexing_ref()`.
- `FastRng` is explicitly typed in every app — no accidental Ranlux default.
- Sleef build config (`DFT/QUAD/GNUABI/SCALAR/INLINE/TESTS off`, `LIBM on`, static) is correct.
- `-fno-math-errno` for Release is correct, and **not** pulling in `-ffast-math` is the right call for MCMC reproducibility.
- `metropolis.hpp` accept/reject: branchy short-circuit on `ds≤0` is the right choice (saves the exp); one uniform per site; no redundant draws.
- `hmc.hpp:128–150` momentum sampling already uses `normal_fill` batched paths where available.
- `wolff.hpp` `mark_` as `vector<uint32_t>` with generation bump is the right call over `vector<bool>`.
- LLR `imag_scratch_ = std::optional<Field>` lazy-allocated once and reused — the previous bose_gas_llr regression is properly fixed.

---

## 5. Suggested order of attack

1. **H4** (link-lattice loop order) — one-file fix, multi-× on link sweeps.
2. **H5** (indexing fast path: constexpr `D`, per-μ tables, `uint32_t`) — touches many hot kernels but mechanical.
3. **H1 + H2** (SU(N) slab `compute_force` and fused-kick) — the biggest single gain for SU(2)/SU(3) HMC; do as one work item since they share scaffolding.
4. **H3** (Wilson<U(1)> Sleef path) — required before CompactU1 retires.
5. **H6** (slab `exp_su{2,3}` via Sleef) — naturally follows H1.
6. **M10** (`-mcpu=apple-m1`, safe fast-math, IPO, Sleef ISA pin) — cheap build-side wins, do before benchmarking the H-items.
7. **M1 + M2 + M5** (LLR `s_full` reuse + complex fused-kick) — close the LLR-specific re-sweep.
8. **M3** (aligned `MatrixLinkLattice` storage) — small and complementary to H1.
9. **M6 + M8** (Sleef Box-Muller, batched algebra Gaussians) — momentum init speedup.
10. **L** items as opportunistic cleanup during the above.
