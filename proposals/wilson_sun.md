# Wilson SU(N) gauge action — design proposal

Extend the current `CompactU1` Wilson action to a generic `Wilson<G>` family
covering U(1), SU(2), SU(3). End state: pure-gauge HMC/Metropolis/LLR for QCD
(no fermions, no improved actions) running on the same templated
`Hmc<A, R, Integrator, Field>` / `Replica<...>` / `WindowedAction<Base, Field>`
machinery already in tree. The structural unification in commits `8183be3 →
52b052c` makes this an additive change — no plumbing rework, only a new
field type and a new action.

## Non-goals

- **Fermions.** No clover, no staggered, no Wilson-Dirac. Pure gauge only.
- **Improved gauge actions.** No Iwasaki, no DBW2, no Symanzik tree-level
  improvement. Plain plaquette Wilson `S_W = β·Σ_p (1 − (1/N)·Re Tr U_p)`.
- **GPU / distributed.** Same single-node OpenMP-replica + serial-trajectory
  envelope as today. Vectorisation target: AVX2/AVX-512/NEON via autovec
  + Sleef batched transcendentals.
- **External gauge libraries.** No Grid / QDP++ / Chroma dep. The whole
  point of `reticolo` is to *be* small. We hand-roll 2×2 and 3×3.

## Math summary

Field: `U_μ(x) ∈ SU(N)` on every link. Storage: full N×N complex matrix
(not Cayley-Klein for SU(2), not algebra-coords for SU(3)) — projection
back to the group every k trajectories via Gram-Schmidt + det fix handles
numerical drift.

Action:
```
S_W = (β/N) Σ_x Σ_{μ<ν} Re Tr (N·𝟙 − U_{μν}(x))
    = β · n_plaq − (β/N) Σ Re Tr U_p
U_p(x;μν) = U_μ(x) · U_ν(x+μ̂) · U_μ(x+ν̂)† · U_ν(x)†
```

For N=1 this reduces line-for-line to the existing `CompactU1`
(`Re Tr U_p → cos θ_p`, β/N → β, the constant absorbs identically).

Momentum: anti-hermitian traceless N×N matrices `P_μ(x) ∈ su(N)`. Storage:
same shape as a link (N² complex), with the algebra constraint enforced
implicitly by the way `P` is generated and by `traceless_antiherm()`
projections at sample/force time. Kinetic energy:
```
K = (1/2) Σ_{x,μ} Tr(P_μ(x)† P_μ(x))    (positive, gauge-invariant)
```

Force on the algebra (standard result, e.g. Gattringer-Lang §7.1.3):
```
F_μ(x) = − dS/dU_μ projected on su(N)
       = −(β/(2N)) · [ U_μ(x)·V_μ(x) − (U_μ(x)·V_μ(x))† ]_traceless
V_μ(x) = Σ_{ν≠μ}  staple_μν(x)        (sum of 2(d−1) staples)
```
The staple `V` is the sum of the two "U-shaped" link products that close
each plaquette through link `(x,μ)`. This is one matrix per link, computed
once per HMC force evaluation.

HMC drift step (the new design point — see §3 below):
```
U_μ(x) ← exp(i · dt · P_μ(x)) · U_μ(x)
```
This *replaces* the elementwise `q += dt·p` used by scalar fields. Kick
step (`P += dt·F`) is unchanged: `P` and `F` both live in `su(N)` which is
a real vector space, so additive update is correct.

Matrix exponential:
- **SU(2).** Closed-form Pauli decomposition. `P = i·α·σ/2` with
  `α ∈ ℝ³`; `exp(i·dt·P) = cos(|α|·dt/2)·𝟙 + i·sin(|α|·dt/2)·(α̂·σ)`.
  One sincos per link. Trivially vectorises across sites with Sleef.
- **SU(3).** Cayley-Hamilton via Morningstar-Peardon (Phys. Rev. D 69,
  054501; ~2004). No eigendecomposition, no complex sqrt; just compute
  Tr(Q²), Tr(Q³), solve a small real characteristic polynomial, evaluate
  three real coefficients `f₀,f₁,f₂` and form `exp(iQ) = f₀·𝟙 + f₁·iQ +
  f₂·(iQ)²`. ~50 flops per link plus one `cos`, one `sin`, one `acos`.
  Vectorises cleanly across sites (all scalar ops on per-site
  coefficients).

Reunitarisation: every `n_reunit` trajectories (default 50), pass over
links projecting back to SU(N). SU(2): normalise to unit determinant
quaternion. SU(3): modified Gram-Schmidt on the three column vectors,
then divide row 3 by det. Cost: negligible at the chosen cadence.

## Storage layout (the key vectorisation decision)

Current `LinkLattice<T>`: direction-major, one `T` per link. Memory is
`ndim` contiguous blocks of `nsites` doubles. Hot loops over a `(μ,ν)`
plaquette plane stride through one `nsites` block per direction with
fixed `(s_pmu − s, s_pnu − s)` offsets — `visit_plane` peels the wrap
slabs so the bulk inner loop autovectorises.

For SU(N) each link is `2·N²` real doubles. Two viable layouts:

| Layout | Memory shape | Vectorises across | Cache per slab |
|---|---|---|---|
| AoS-per-link | `[ndim][nsites][2N²]` | matrix elements within one site | great per site, bad across sites |
| **SoA-per-component** | `[ndim][2N²][nsites]` | **lattice sites** | great for slab kernels |

We pick **SoA-per-component**. Reasons:
1. Direct extension of the existing direction-major scalar-link pattern.
   U(1) is the `2N² = 1` degenerate case — no migration needed.
2. Slab kernels over a `(μ,ν)` plane become `2N²` parallel stride-1
   streams per input link. A 3×3 complex-complex multiply across a slab
   of W sites is W parallel 3×3 mat-mats; the compiler sees 9 independent
   FMA chains each W lanes wide. Excellent autovec without intrinsics.
3. Matrix exp is per-site scalar work on `2N²` coefficients — also slab-
   friendly when the precomputed scalars (`f₀,f₁,f₂` for SU(3)) live in
   their own nsites scratch arrays.

Implementation: a new field type
```cpp
template <class Group, class T = double>
class MatrixLinkLattice {
    // shape:       SizeVec
    // raw buffer:  T[ndim * Group::n_real_components * nsites]
    // accessors:
    T const* mu_comp_data(std::size_t mu, std::size_t k) const;  // stride-1 in s
    T*       mu_comp_data(std::size_t mu, std::size_t k);
    // plus the existing LinkLattice surface: nlinks(), ndims(), nsites(),
    //  data(), begin(), end(), next(), prev(), for_each_update(...)
};
```

The integrator and HMC plumbing only see `data()` / `flat_size()` /
`for_each_update()` for the momentum buffer (which is *also* a
`MatrixLinkLattice<Group, T>`, since `P ∈ su(N)` has the same storage
shape as `U ∈ SU(N)`). All matrix-shape knowledge lives inside `Group`
and inside `Wilson<Group>`.

## Drift customisation (the only integrator change)

Today the integrators do `q[i] += dt · p[i]` inline. For SU(N) the drift
is `U_μ(x) ← exp(i·dt·P_μ(x)) · U_μ(x)` — group exp, not elementwise add.

Introduce two free-function customisation points used by every integrator:
```cpp
namespace reticolo::alg::integ {
  // Default: elementwise q += dt * p over flat_size() doubles.
  template <class Field, class Mom> void drift(Field&, Mom const&, double);
  // Default: elementwise p += dt * f.
  template <class Mom, class Force> void kick (Mom&, Force const&, double);
}
```
- Default templates cover `Lattice<T>` and `LinkLattice<T>` (U(1)) — no
  perf change, same loop the integrators do today, just hoisted into a
  free function. Bench step verifies bit-identical output.
- `MatrixLinkLattice<Group, T>` overloads `drift()` to call
  `Group::expi_lmul(U, P, dt)` per site, which dispatches per group.
  Both SU(2) and SU(3) write a single slab kernel.
- `kick()` default works for *all* fields including matrix links (force
  and momentum both live in `su(N)`, addition is the right op).

Leapfrog/Omelyan2/Omelyan4 change to call `integ::drift(...)` and
`integ::kick(...)` instead of inlining the loop. ~5 lines per integrator.
Existing `HasFusedKick` path stays as-is (only matters for scalar; gauge
will not get a fused kick at first — it can be added later if the perf
audit shows it matters).

## GaugeGroup concept

```cpp
template <class G>
concept GaugeGroup = requires {
    typename G::scalar_t;                  // double / float
    { G::N }                  -> std::convertible_to<int>;
    { G::n_real_components }  -> std::convertible_to<int>;   // 2*N*N
    { G::n_algebra_real }     -> std::convertible_to<int>;   // N*N - 1
} && requires (typename G::scalar_t* U, typename G::scalar_t const* V,
               typename G::scalar_t* P, typename G::scalar_t* out,
               typename G::scalar_t* scratch,
               std::size_t n, typename G::scalar_t dt) {
    // All slab kernels: stride-1 over n sites, take base pointers to
    // ndim-stripped component blocks. n_real_components arrays per link.

    G::mul_slab          (out, U, V, n);            // out = U * V
    G::mul_adj_slab      (out, U, V, n);            // out = U * V†
    G::adj_mul_slab      (out, U, V, n);            // out = U† * V
    G::re_tr_slab        (out, U, n);               // out[s] = Re Tr U[s]
    G::traceless_antiherm_slab(out, U, n);          // P_TA projection
    G::expi_lmul_slab    (U, P, dt, scratch, n);    // U ← exp(i·dt·P) · U
    G::project_slab      (U, n);                    // reunitarise in place
    G::sample_algebra_slab(P, rng, n);              // gaussian on su(N)
    G::kinetic_slab      (out, P, n);               // out[s] = (1/2)|P_s|²
};
```

Models:
- `gauge_group::U1`  — `N=1`, `n_real_components=1` (angle θ), the existing
  hot loop survives as-is; just wrapped to satisfy the concept. Lets us
  delete `compact_u1.hpp` in M4 and have `Wilson<U1>` cover it.
- `gauge_group::SU2` — `N=2`, `n_real_components=8`. Closed-form exp.
- `gauge_group::SU3` — `N=3`, `n_real_components=18`. Cayley-Hamilton exp.

Each model is one ~150-line header + one slab-kernel test
(`tests/unit/test_gauge_group_<g>.cpp`) hitting unitarity, hermitian
conjugate identity, exp(0)=𝟙, exp(iH)·exp(−iH)=𝟙, projection idempotent.

## The Wilson action

```cpp
template <gauge_group::GaugeGroup G, class T = double>
struct Wilson {
    using value_type = T;
    using group_type = G;
    using field_type = MatrixLinkLattice<G, T>;

    T beta = T{0};

    // LinkLocalAction (Metropolis) ---- s_local / ds_local over 2(d−1) plaqs
    // HasLinkSEff                  ---- s_full plane-by-plane via visit_plane
    // HasLinkForce                 ---- compute_force via staple sum
};
```

`s_full`: per plane, loop sites in a slab; for each site compute the
plaquette product `U_p` (four `mul_slab` calls into a per-component
scratch holding the running product) and reduce `Re Tr U_p` over the
slab via `re_tr_slab`. One scalar reduction at the end per plane.
Memory traffic per plane is exactly the same as U(1) scaled by
`2N²` — same slab pattern, same `visit_plane` driver.

`compute_force`: per link `(x, μ)` we need the staple `V_μ(x)` (sum of
`2(d−1)` staples). Two clean implementation options:

  (a) **Plaquette-centric scatter** (parallels current U(1)): walk each
      `(μ, ν)` plane, for each site compute the four matrix products
      that form the plaquette, scatter the staple contributions into
      the four links it touches. Memory pattern matches the current
      `compute_force` in `compact_u1.hpp`.
  (b) **Link-centric gather** (more typical in production codes): walk
      each link, accumulate its staple by gathering from its `2(d−1)`
      neighbours. More flops but bigger working set per inner iter;
      probably worse on bandwidth-bound 4D-L≥8.

We start with (a) and bench. The slab scratch grows from `nsites
doubles` (U(1)) to `~10·nsites doubles` (running matrix products),
allocated lazily like the existing `scratch` member.

## File layout

```
include/reticolo/action/
    wilson.hpp                  # Wilson<G> generic
    compact_u1.hpp              # kept until M4 perf parity, then
                                # deleted and Wilson<U1> takes over
    detail/
        concepts.hpp            # add WilsonAction concept refinement
        gauge_helpers.hpp       # visit_plane (unchanged)
        gauge_group/
            base.hpp            # GaugeGroup concept + shared helpers
            u1.hpp              # 1-component, identity-as-1.0
            su2.hpp             # 4-complex storage, closed-form exp
            su3.hpp             # 9-complex storage, Cayley-Hamilton
include/reticolo/core/
    matrix_link_lattice.hpp     # [ndim][nc][nsites] field, sibling to
                                # LinkLattice; satisfies Field traits
include/reticolo/algorithm/
    integrators.hpp             # call integ::drift / integ::kick
    integ_ops.hpp               # NEW: default drift/kick free fns +
                                # matrix-link overloads dispatching to G
include/reticolo/math/
    su2_ops.hpp                 # mul/adj/exp slabs, hand-written
    su3_ops.hpp                 # same for 3×3
apps/
    su2_hmc.cpp                 # mirrors u1_hmc.cpp
    su3_hmc.cpp                 # 4D L=4/6/8 reference run
    bench_wilson.cpp            # U(1)/SU(2)/SU(3) at matched τ, n_md
tests/physics/
    test_wilson_su2_force.cpp   # numerical Δ vs analytic force
    test_wilson_su3_force.cpp
    test_su2_hmc_reversibility.cpp
    test_su3_hmc_reversibility.cpp
```

## External libraries

**Recommendation: none new.** Sleef stays as the only vector-math dep.

Considered and rejected:
- **Eigen.** Re-adds the dep the v3 plan deliberately killed. Overkill
  for 3×3 — Eigen's GEMM dispatch overhead is ~larger than the 27-mul
  3×3 itself. Doesn't help us across the site axis, which is where the
  SIMD gain lives (SoA-per-component layout).
- **xsimd / `std::simd` (C++26).** Would let us write explicit-lane
  kernels, but the SoA layout means the compiler already sees clean
  stride-1 FMA chains. Adding explicit lanes locks the code to a
  vector width and gives up portability. Revisit only if a post-M3
  perf audit shows the compiler is leaving >30% on the table.
- **OpenBLAS / Apple Accelerate.** Same as Eigen — 3×3 is below the
  GEMM crossover; the call overhead exceeds the math.

Hand-rolled kernels for SU(2) and SU(3) are tractable (~100 LOC each
for the full mul/adj/exp/project family) and let us inline straight
into slab loops where the compiler can fold per-site work.

## Milestones

Each commit ships green tests + a perf bench row in `bench_wilson`
showing no regression on the U(1) path (the only path with existing
numbers). Same discipline as the structural-unification series — keep
diffs bounded, keep perf tracked.

| # | Commit | Scope | LOC |
|---|---|---|---|
| 1 | `feat(core): MatrixLinkLattice<G,T> field type`        | new header + Field-traits hookup; concept-satisfies HMC/Metropolis/Replica generics | ~250 |
| 2 | `feat(integ): drift/kick customisation points`        | `integ_ops.hpp` with default templates; integrators switched over; bit-identical scalar + U(1) bench | ~120 |
| 3 | `feat(gauge_group): U1 model under GaugeGroup`        | wraps existing 1-component ops to satisfy `GaugeGroup`; `Wilson<U1>` instantiates; runs `u1_hmc` and matches `CompactU1` bit-identically | ~150 |
| 4 | `feat(gauge_group): SU2 closed-form ops`              | `su2_ops.hpp` (mul/adj/exp/project slabs) + tests + `gauge_group::SU2` | ~300 |
| 5 | `feat(action): Wilson<SU2> + su2_hmc app`             | force/s_full/Metropolis paths; reversibility test; small-β plaquette check vs 1−β/(2N²)·... weak-coupling expansion | ~250 |
| 6 | `feat(gauge_group): SU3 Cayley-Hamilton exp`          | `su3_ops.hpp` + tests (exp(0)=𝟙, exp(iH)·exp(−iH)=𝟙, unitarity of project) | ~400 |
| 7 | `feat(action): Wilson<SU3> + su3_hmc app`             | reversibility, weak-coupling plaquette check (3D and 4D), strong-coupling 1−β²/18 check | ~250 |
| 8 | `chore: delete compact_u1.hpp, route U(1) via Wilson<U1>` | proves the abstraction; perf bench shows no regression vs the special-case | -250 |
| 9 | `feat(llr): Replica<Wilson<G>, ...> example`          | one worked example replicating an SU(2) or SU(3) bulk-transition fingerprint via LLR (analog of example 05) | ~150 |

Optional follow-on (separate proposal, not in this plan):
- Improved gauge action `Wilson_imp<G>` (Lüscher-Weisz or DBW2) by
  adding a rectangle staple — pure additive change to the staple sum.
- Stout smearing for measurement — also additive, lives next to the
  action.
- HasLinkFusedKick for gauge — fold the staple → exp → mul chain into
  a single slab pass. Estimate ~15% on the integrator inner loop.

## Test plan (per action commit)

1. **Reversibility.** Same template-test pattern as
   `test_u1_hmc_reversibility.cpp`: integrate forward, flip momentum,
   integrate back, check `‖U − U₀‖ < 1e-10` and `‖P + P₀‖ < 1e-10`.
   Runs all three integrators (Leapfrog/Omelyan2/Omelyan4).
2. **Force consistency.** `compute_force` matches a finite-difference
   gradient of `s_full` on a random small lattice (4⁴) to ~1e-7.
   Per-direction loop, per-component perturbation.
3. **Weak-coupling check.** At β → ∞ the mean plaquette
   `⟨P⟩ ≡ ⟨(1/N) Re Tr U_p⟩` matches the leading-order expansion:
   - SU(2): `⟨P⟩ ≈ 1 − 3/(4β) + O(β⁻²)`
   - SU(3): `⟨P⟩ ≈ 1 − 1/(3β) + O(β⁻²)`
   Run at β=20 (SU(2)) / β=15 (SU(3)) on 6⁴, ≥4σ tolerance.
4. **Strong-coupling check.** At β → 0:
   - SU(2): `⟨P⟩ ≈ β/4 + O(β³)`
   - SU(3): `⟨P⟩ ≈ β/18 + O(β³)`
   Run at β=0.5 on 4⁴.
5. **Reunitarisation drift.** Without reunitarisation, log `‖U†U − 𝟙‖_F`
   over 1000 trajectories and verify the chosen cadence keeps it below
   1e-10 (Gattringer-Lang quotes 1e-12 with cadence 50 in double).

## Risks and fall-backs

- **SU(3) matrix exp accuracy.** Cayley-Hamilton has a known ill-
  conditioned branch when the two non-trivial eigenvalues of `Q` get
  close (small `θ` in Morningstar-Peardon notation). Standard
  remediation is a Taylor expansion in that regime; documented in
  Morningstar-Peardon §C. Plan: implement the Taylor branch from day
  one, gated on `|θ| < 1e-4`.
- **AoS-via-SoA syntactic burden.** Writing `mul_slab` cleanly with 9
  components in scope is awkward — easy to swap an index by accident.
  Mitigation: hand-write one canonical 3×3 inline kernel in
  `su3_ops.hpp` derived directly from the symbolic matrix product,
  exhaustively unit-tested against a reference using a single matrix
  multiplied by `std::complex<double>` literal kernel. Once that
  reference exists, the slab kernels are just "wrap this in a
  `for (size_t s=0; s<n; ++s)` and load/store from component arrays".
- **Compiler autovec failure on the inner mat-mul.** If clang/gcc
  refuse to vectorise the 9-FMA slab loops on some platform, fall back
  to `std::experimental::simd` lanes — well-isolated to `su3_ops.hpp`,
  no change to `Wilson<G>` or the field type. Measure first.
- **LLR over gauge action energy.** The `WindowedAction<Base, Field>`
  unification already covers this — no work needed once `Wilson<SU3>`
  satisfies `LinkLocalAction`. The single tricky decision is whether
  the LLR energy variable is `S_W` itself or `(1/N)·Re Tr U_p` summed
  (i.e., normalised plaquette). We pick `S_W` for parity with the
  existing U(1) LLR example. Window scaling stays in `S` units.
- **Perf delta of Wilson<U1> vs CompactU1 (M8).** If `Wilson<U1>` is
  >5% slower than the hand-tuned `CompactU1`, we keep `compact_u1.hpp`
  alive and ship `Wilson<G>` only for SU(2)/SU(3). Same dual-track
  pattern as the gauge/scalar split we just collapsed; happy to live
  with it if generality costs measurable cycles on the most-used path.

## Estimated effort

~2.5 kLOC net additions, ~3 weeks at the v3 pace. Biggest unknown is
M6 (SU(3) exp + tests) — historically the trickiest single piece in
any pure-gauge code. Plan to budget 2 days for that one commit.
