# Gauge module — design plan

Status: proposal, awaiting "build it".
Branch target: a fresh `feature/gauge` off `rewrite/v3`.
Scope: compact U(1) only. Metropolis + HMC + LLR all working.
Stretch goal (example): reproduce the 4D compact U(1) double-peak ρ(S) at the
weakly-first-order transition (Langfeld-Lucini-Rago 2012, EPJC).

## 1. What we are building

A self-contained gauge sub-library under `include/reticolo/gauge/` plus the
matching apps + example. Sits **alongside** the scalar stack — no edits to the
existing `Lattice<T>`, action concept lattice, `alg::Hmc`, integrators, or
`alg::Metropolis`. Two parallel families.

Only point of contact with the scalar stack: the LLR module gets a small
refactor (one extra type alias on each action) so a single `WindowedAction`
template works on both scalar fields and link fields.

User decisions, already settled:

- **Storage**: new `LinkLattice<T>` type. Indexed by `(Site, mu)`.
- **Concepts**: parallel `LinkLocalAction` family.
- **First LLR target**: 4D compact U(1) at β ≈ 1.0 (the famous double peak).

## 2. `LinkLattice<T>`

Sits next to `Lattice<T>` in `include/reticolo/core/link_lattice.hpp`. Same
shape-pooled `Indexing`, same value semantics, same shape pool. Storage is a
flat `std::vector<T>` of size `ndim · nsites`. Site-major layout: a site's
`ndim` outgoing links are contiguous (locality matches per-site sweep order;
matches what `Lattice<std::array<T, ndim>>` would do).

```cpp
template <class T>
class LinkLattice {
public:
    using value_type = T;
    using SizeVec    = Indexing::SizeVec;

    explicit LinkLattice(SizeVec shape);
    LinkLattice(SizeVec shape, T fill);
    explicit LinkLattice(std::shared_ptr<Indexing const> idx);   // sibling
    LinkLattice(std::shared_ptr<Indexing const> idx, T fill);

    // Value semantics, default copy/move.

    [[nodiscard]] T&       operator()(Site x, std::size_t mu) noexcept;
    [[nodiscard]] T const& operator()(Site x, std::size_t mu) const noexcept;

    // Bulk access for hot kernels.
    [[nodiscard]] T*       data() noexcept;
    [[nodiscard]] T const* data() const noexcept;
    [[nodiscard]] std::size_t nsites() const noexcept;
    [[nodiscard]] std::size_t nlinks() const noexcept;   // = ndim * nsites
    [[nodiscard]] std::size_t ndims() const noexcept;

    // Neighbour walk — delegates to the shared Indexing.
    [[nodiscard]] Site next(Site, std::size_t mu) const noexcept;
    [[nodiscard]] Site prev(Site, std::size_t mu) const noexcept;

    [[nodiscard]] auto sites() const noexcept;                 // same as Lattice
    [[nodiscard]] std::shared_ptr<Indexing const> indexing() const noexcept;
};
```

A `LinkLattice<T>` can share an `Indexing` with a sibling site `Lattice<F>` of
the same shape (the HMC pattern: field, mom, force all share one neighbour
table). No new `Indexing` machinery — the shape pool already handles this.

## 3. Concept lattice — parallel family

`include/reticolo/gauge/concepts.hpp`. Mirrors `action/concepts.hpp` but on
`LinkLattice<F>`:

- `LinkLocalAction<A, F>`: `s_local(l, x, mu) -> double`, `ds_local(l, x, mu, new_v) -> double`. Enough to drive a link-Metropolis sweep.
- `HasLinkSEff<A, F>`: refinement; `s_full(l) -> double`. Required for HMC.
- `HasLinkForce<A, F>`: refinement; `compute_force(l, force)` writes `-dS/dθ_μ(x)` into a sibling `LinkLattice<F>`.
- `HasLinkFusedKick<A, F>`: refinement; `compute_force_and_kick(l, mom, k_dt)` for the no-force-buffer fast path (same trick as the scalar `HasFusedKick`).

No `WolffEmbeddable` analogue — Wolff is for spin systems, not gauge.

## 4. `act::CompactU1<T, ndim>`

`include/reticolo/gauge/builtins/compact_u1.hpp`. Standard Wilson plaquette
action:

```
S = β Σ_x Σ_{μ<ν} [1 − cos(θ_μν(x))]
θ_μν(x) = θ_μ(x) + θ_ν(x+μ̂) − θ_μ(x+ν̂) − θ_ν(x)
```

Force per link (`F_μ(x) = −∂S/∂θ_μ(x)`):

```
F_μ(x) = −β Σ_{ν≠μ} [ sin θ_μν(x) − sin θ_μν(x − ν̂) ]
```

Each link is touched by `2(ndim−1)` plaquettes (forward + backward in each of
the `ndim−1` other directions) — same staple computation found in every gauge
code.

The members of `CompactU1`:

- `T beta;`
- `using value_type = T;`
- `using field_type = LinkLattice<T>;`   ← lets the LLR template find the field type
- `s_local(l, x, mu)`, `ds_local(l, x, mu, new_v)` — for Metropolis. Sum the
  `2(ndim−1)` staple contributions touching this link.
- `s_full(l)` — full plaquette sum. One pass over all plaquettes
  (positive-μν convention so each plaquette is counted once).
- `compute_force(l, force)` — writes a `LinkLattice<T>` of forces.
- `compute_force_and_kick(l, mom, k_dt)` — fused.

Hot-loop tuning lives in a follow-up. First cut uses a clean reference
implementation; a parallel `gauge/hot_loop.hpp` (mirroring the scalar one) is
out of scope for v1.

## 5. `alg::LinkMetropolis`

`include/reticolo/gauge/algorithm/metropolis.hpp`. Site-and-direction sweep:
for each site x, for each μ in 0..ndim-1, propose `Δθ ~ U[-δ, +δ]`, compute
`ds_local(l, x, mu, θ + Δθ)`, accept with `min{1, exp(−ds)}`. Plain layout —
no even/odd colouring needed for U(1) since the staple at one link doesn't
touch any other link that shares the same staple (gauge-link interactions are
indirect through the plaquette, never through a shared link variable).

Parameters: `δ` (proposal width). The existing `alg::Metropolis` is **not**
touched.

## 6. Link HMC + integrators

`include/reticolo/gauge/algorithm/hmc.hpp` + `gauge/algorithm/integrators.hpp`.
Sits parallel to the scalar ones; same structure, different field type.

- Momentum: `LinkLattice<T>` of the same shape and Indexing as the field.
- Kinetic term: `H_kin = (1/2) Σ_x Σ_μ π_μ(x)²` — straight sum over the flat
  data buffer.
- Momentum sampling: fill the flat buffer with Gaussian normals (one per link;
  matches the scalar `normal_fill` fast path for `T = double`).
- Integrators (`Leapfrog`, `Omelyan2`, `Omelyan4`) reproduce the scalar
  versions verbatim with `Lattice<F>` swapped for `LinkLattice<F>`. The
  `kick_` and `drift_` helpers in `gauge/integrators.hpp::detail` are
  near-copies of the scalar ones; `if constexpr (HasLinkFusedKick<A, F>)`
  picks the no-force-buffer path when available.

Refactor note: a future cleanup could template the existing integrators on
`Field`/`Mom`/`Force` types rather than `Lattice<F>`, deduplicating with the
scalar versions. Out of scope for v1. ~150 LOC of integrator duplication is
the price.

## 7. LLR refactor — one template, two field types

This is the only place where the existing tree changes. Currently:

```cpp
template <class Base, class T = Base::value_type>
struct WindowedAction {
    [[nodiscard]] T s_full(Lattice<T> const& l) const noexcept;
    void compute_force(Lattice<T> const& l, Lattice<T>& force) const noexcept;
    // ...
};
```

Change to discover the field type from the action:

```cpp
template <class Base>
struct WindowedAction {
    using Field = typename Base::field_type;   // e.g. Lattice<double> or LinkLattice<double>
    using T     = typename Base::value_type;

    [[nodiscard]] T s_full(Field const& l) const noexcept;
    void compute_force(Field const& l, Field& force) const noexcept;
    // ...
};
```

Existing scalar actions (`Phi4`, `Phi6`, `SineGordon`, `Xy`, `OnSigma`) each
get a single new line:

```cpp
using field_type = Lattice<T>;            // or Lattice<std::array<T, N>> for OnSigma
```

`act::CompactU1<T, ndim>` declares `using field_type = LinkLattice<T>;` and is
picked up automatically.

`Replica<Base, Rng, Integrator>` does the same — picks `Base::field_type` for
its lattice + momentum + force types. The HMC instance inside the replica is
the *gauge* HMC when `Base::field_type` is a `LinkLattice<T>`. Concept-driven
selection: a small `if constexpr` (or two `Replica` partial specialisations)
picks `alg::Hmc` vs `gauge::Hmc` based on the field type. Cleanest: two
specialisations.

Existing `apps/phi4_llr.cpp` keeps compiling unchanged.

## 8. Worked example: 4D compact U(1) double-peak ρ(S)

`apps/u1_llr.cpp` mirrors `apps/phi4_llr.cpp`: the app owns the NR loop, the
RM loop, and the even/odd exchange sweep. Same HDF5 schema, same `analyze.py`
shape.

`examples/05_u1_llr/`:
* `README.md` — citing Langfeld, Lucini, Rago (2012), EPJC 72:2086, as the
  literature target.
* `run.sh` — runs plain `u1_hmc` (also new) for an HMC ρ(S) estimate, then
  `u1_llr` over the chosen window. Default lattice 4^4 or 6^4 at β ≈ 1.0
  (the transition region for 4D compact U(1)). Default β values bracket the
  transition (e.g. 0.95, 1.00, 1.02, 1.05).
* `analyze.py` — same shape as example 04 (HMC histogram vs LLR
  reconstruction overlay + a_n convergence panel).

Headline expectation: at β ≈ 1.0 ρ(S) shows the characteristic double-peak
signature; HMC importance sampler concentrates on one of the peaks and
misses the other; LLR resolves both and the saddle between them.

Two new apps under `apps/`:
* `apps/u1_hmc.cpp` — plain HMC at fixed β.
* `apps/u1_metropolis.cpp` — plain link Metropolis at fixed β. Useful for
  thermalising before HMC and for a sanity Wilson-loop measurement.

## 9. File additions (concrete)

### Core
1. `include/reticolo/core/link_lattice.hpp` — `LinkLattice<T>` (~120 LOC).

### Gauge action / concepts
2. `include/reticolo/gauge/concepts.hpp` (~60 LOC).
3. `include/reticolo/gauge/builtins/compact_u1.hpp` (~200 LOC).

### Gauge updaters
4. `include/reticolo/gauge/algorithm/metropolis.hpp` (~80 LOC).
5. `include/reticolo/gauge/algorithm/integrators.hpp` (~180 LOC; mostly
   adapted from `algorithm/integrators.hpp`).
6. `include/reticolo/gauge/algorithm/hmc.hpp` (~140 LOC; mostly adapted).

### LLR refactor
7. `include/reticolo/llr/windowed_action.hpp` — discover `field_type` from
   `Base`. The existing scalar actions each get one new `using field_type = …;`
   line. `Replica` picks `alg::Hmc` vs `gauge::Hmc` via two specialisations.

### Umbrella
8. `include/reticolo/reticolo.hpp` — add gauge includes.

### Apps
9. `apps/u1_metropolis.cpp` (~80 LOC).
10. `apps/u1_hmc.cpp` (~100 LOC).
11. `apps/u1_llr.cpp` (~150 LOC).

### Example
12. `examples/05_u1_llr/{README.md, run.sh, analyze.py}`.

## 10. Test plan

Per the project rule, tests come once the feature shape is stable. Concretely:

- **Force consistency**: numerical-derivative check, `compute_force` vs `s_full`,
  random link configs and small β. Same template as
  `tests/physics/test_phi4_force_consistency.cpp`.
- **HMC reversibility**: integrator + momentum flip reproduces the original
  configuration to ~1e-10. Same template as
  `tests/physics/test_hmc_reversibility.cpp`.
- **Integrator order**: `dH` scaling vs `dt` gives slope 2 / 2 / 4 for
  Leapfrog / Omelyan2 / Omelyan4.
- **2D U(1) analytic Z**: in 2D compact U(1) the partition function is
  `Z = ∏_p ∫ dθ exp(−β(1−cos θ))` (Migdal). At small β a plaquette-average
  measurement should match the analytic `I_1(β)/I_0(β)`.
- **App smoke tests**: `u1_metropolis`, `u1_hmc`, `u1_llr` each get a
  `tests/apps/*.cpp` that runs the binary on tiny input and asserts the HDF5
  schema is right.

## 11. Out of scope (v1)

- SU(N) gauge theory (would need `Eigen` / N×N matrix arithmetic).
- Hot-loop tuning (`gauge/hot_loop.hpp` parallel to the scalar one).
- Gauge-fixing (only gauge-invariant observables in v1).
- Fermions of any kind.
- Multilevel / heatbath updaters.

## 12. Build-it sequence

When the user says "build it":

1. `LinkLattice<T>` + tiny construction/access/iteration tests built into the
   physics test suite as a smoke (not full Catch2 yet — per the rule, tests
   come after shape stabilisation).
2. `gauge/concepts.hpp` + `act::CompactU1<T, ndim>`. Build it against a small
   throwaway driver that creates a 2D 4^2 lattice, evaluates `s_full`, and
   prints. Sanity vs known analytic mean plaquette in 2D.
3. `alg::LinkMetropolis`. Drop into the throwaway driver, thermalise, measure
   mean plaquette, compare to the 2D analytic value `I_1(β)/I_0(β)`.
4. Link integrators + link HMC. Reuse the throwaway driver; check
   reversibility and integrator order *informally* before writing the formal
   tests.
5. LLR refactor (`field_type` everywhere). Verify `apps/phi4_llr.cpp` still
   builds and the example 04 still runs bit-identical at fixed seed.
6. `apps/u1_metropolis.cpp`, `apps/u1_hmc.cpp`, `apps/u1_llr.cpp` + CMake.
7. `examples/05_u1_llr/` scripts + literature replication target.
8. Tests come here, as a separate commit before merging.

## 13. Open question to settle before step 1

- **Dimensionality knob**: `act::CompactU1<T, ndim>` where `ndim` is a compile-time
  template parameter (mirrors what `OnSigma` does with `N`)? Or a runtime
  argument like the scalar `Phi4` does with `--ndim`? Runtime is more flexible
  for apps; compile-time gives better-optimised hot loops. Default suggestion:
  runtime via `Indexing` (the same way scalar actions work today — `ndim` lives
  in the shape, not in the action type). Confirm.
