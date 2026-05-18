# Complex-action LLR — design plan

Status: proposal, awaiting "build it".
Branch target: continue on `rewrite/v3`.
Scope: extend the LLR module so each replica samples a real (phase-quenched)
action under a Gaussian-penalty window constraining a *separate observable* —
typically the imaginary part of a complex action.
Stretch goal (example): reproduce Francesconi-Holzmann-Lucini-Rago
(arXiv:1910.11026), the 4D relativistic Bose gas at finite chemical potential.

## 1. What we are building

A small refinement to the action concept lattice + a compile-time dispatch on
`WindowedAction`, plus a new `act::BoseGas<T>` builtin with complex-field
storage and an example reproducing the paper's `⟨e^{iφ}⟩_pq(μ)` curve across
the silver-blaze threshold.

Three structural changes, then a builtin, then an example. Each step is
independently testable; steps 1 and 2 don't change any existing behavior.

User decisions, already settled:

- **Complex storage**: `Lattice<std::complex<T>>`, not split real/imag.
- **One action, optional imag refinement**: a single `act::BoseGas` exposes
  both its real-sampling half and its imag-window half through a new
  `HasImagPart` concept. `WindowedAction<Base>` dispatches at compile time —
  no two-template-parameter API, no separate constraint class.
- **No phase-factor measurement during the run.** The LLR run reconstructs
  the DoS `ρ(S_I) = ∫Dφ δ(S_I − S_I[φ]) e^{−S_R[φ]}`. The Fourier transform
  to `Z(μ)` and `⟨e^{iφ}⟩_pq` lives in `analyze.py`.

## 2. Phase 1 — complex-field HMC plumbing

**Goal:** make `Lattice<std::complex<T>>` a first-class field type for HMC, so
that a complex-valued action with real-valued `s_full` integrates correctly.

**Files touched:**
- `include/reticolo/algorithm/hmc.hpp` — kinetic energy + momentum sampling
  branches on `std::is_same_v<F, std::complex<T>>`.
- `include/reticolo/core/rng.hpp` — no new code; `normal_fill(double*, n)`
  already exists and works on `reinterpret_cast<double*>(complex_buf)` since
  `std::complex<double>` is layout-compatible with `double[2]` (mandated by
  the standard, §29.5.4).

**Kinetic energy specialization.** `Hmc::hamiltonian_()` currently does:

```cpp
auto const pi = static_cast<double>(p[i]);
kin += pi * pi;
```

Add an `if constexpr` branch for complex `F`:

```cpp
if constexpr (is_complex_v<F>) {
    kin += std::norm(p[i]);   // |p|² = p_re² + p_im²
} else {
    auto const pi = static_cast<double>(p[i]);
    kin += pi * pi;
}
kin *= 0.5;   // unchanged
```

**Momentum sampling specialization.** `sample_momenta_()` currently does:

```cpp
if constexpr (std::is_same_v<F, double>) {
    rng_.normal_fill(mom_.data(), mom_.nsites());
}
```

Add a complex branch that fills `2 · nsites` doubles into the reinterpreted
buffer — both real and imaginary parts get independent `N(0, 1)` draws, which
gives the correct sampling of `K = ½ Σ |p|² = ½ Σ (p_re² + p_im²)`:

```cpp
if constexpr (std::is_same_v<F, std::complex<double>>) {
    rng_.normal_fill(reinterpret_cast<double*>(mom_.data()), 2 * mom_.nsites());
}
```

**Integrator drift/kick:** unchanged. `std::complex<T>` supports `+=`, `*` on
matching scalars, so the existing `mom[i] += k_dt * force[i]` and
`field[i] += c_dt * mom[i]` work bit-for-bit.

**Force convention:** `compute_force` returns `−∂S/∂φ*` (the Wirtinger
derivative). Hamilton's equation `dp/dt = −∂H/∂φ*` then integrates correctly
via `mom += dt · force`. Reduces to the existing real-field convention when
Im(φ) ≡ 0.

**Tests (Phase 1):**
1. `test_hmc_complex_smoke.cpp` — instantiates `Hmc<TrivialComplexAction, ...>`
   on `Lattice<std::complex<double>>` with a Gaussian action
   `S = ½ Σ |φ|²`, runs 200 trajectories, checks momentum norm distribution
   matches `χ²(2V)` to within statistical tolerance.
2. `test_hmc_complex_reversibility.cpp` — same as the existing
   `test_hmc_reversibility.cpp` but with complex field. Reverse-then-forward
   should recover initial state to `~1e-10`.
3. `test_complex_real_equivalence.cpp` — define a `RealAsComplex` action that
   pins Im(φ)=0 and behaves as Phi4 on Re(φ). Verify trajectory output
   matches a same-seed Phi4 HMC run on `Lattice<double>` bit-for-bit (or
   within FP rounding).

**Done when:** all three tests green; no existing test regresses.

## 3. Phase 2 — `HasImagPart` concept + `WindowedAction` dispatch

**Goal:** allow `WindowedAction<Base>` to use a different observable for the
window constraint than the one HMC samples on.

**Files touched:**
- `include/reticolo/action/concepts.hpp` — add `HasImagPart` concept.
- `include/reticolo/llr/windowed_action.hpp` — compile-time dispatch.
- `include/reticolo/gauge/llr/windowed_action.hpp` — gauge twin, same
  dispatch.
- `include/reticolo/llr/replica.hpp` + gauge twin — `sample()` accumulates
  the constraint value, not the base value, when imag dispatch is active.

**Concept.** One new optional refinement, no breaking changes to the existing
concept lattice:

```cpp
template <class A, class F>
concept HasImagPart = requires(A const& a,
                               typename A::field_type const& field,
                               typename A::field_type& force) {
    { a.s_imag(field) } -> std::convertible_to<F>;
    a.compute_force_imag(field, force);
};
```

**Dispatch in `WindowedAction<Base>`.** Single template, branches at compile
time:

```cpp
template <class Base, class T = typename Base::value_type>
struct WindowedAction {
    Base const& base;
    T a, E_n, delta;

    // The constrained observable: s_imag if Base has it, else s_full.
    [[nodiscard]] T constraint(typename Base::field_type const& l) const noexcept {
        if constexpr (action::HasImagPart<Base, T>) {
            return base.s_imag(l);
        } else {
            return base.s_full(l);
        }
    }

    // Energy-convention windowed action:
    //   real LLR:    s_full = (1 + a)·S       + (S    − E_n)²/(2δ²)
    //   complex LLR: s_full =       S_R + a·S_I + (S_I − E_n)²/(2δ²)
    T s_full(typename Base::field_type const& l) const noexcept {
        T const q  = constraint(l);
        T const dq = q - E_n;
        T const w  = (dq * dq) / (T{2} * delta * delta);
        if constexpr (action::HasImagPart<Base, T>) {
            return base.s_full(l) + a*q + w;
        } else {
            return (T{1} + a)*base.s_full(l) + w;
        }
    }

    // Force is the gradient of s_full. In the real-LLR branch this collapses
    // to the current scalar (1 + a + (S − E_n)/δ²) · F_base. In the
    // complex-LLR branch it's F_base + (a + (S_I − E_n)/δ²) · F_imag —
    // two force-buffer fills then a fused add.
    void compute_force(typename Base::field_type const& l,
                       typename Base::field_type& force) const noexcept {
        if constexpr (action::HasImagPart<Base, T>) {
            base.compute_force(l, force);
            // scratch buffer for the imag-part force, scaled in-place
            // and added on top. ~10 LOC.
        } else {
            base.compute_force(l, force);
            T const s     = base.s_full(l);
            T const scale = T{1} + a + ((s - E_n) / (delta * delta));
            // ... unchanged from current code
        }
    }
};
```

Gauge twin follows the same pattern (paper-convention signs).

**Replica change.** `sample()` currently returns `⟨base.s_full − E_n⟩` —
straightforward edit to use `windowed_.constraint(phi)` instead. NR/RM
unchanged.

**Tests (Phase 2):**
1. Existing scalar + gauge LLR smoke tests (`test_phi4_llr`, `test_u1_llr`)
   must remain green. They use actions that *don't* satisfy `HasImagPart`,
   so dispatch picks the original branch; output should be byte-identical
   to pre-Phase-2 runs (compare HDF5 datasets directly).
2. New `test_windowed_action_imag_force_consistency.cpp` — instantiate a
   toy action that has *both* `compute_force` and `compute_force_imag`,
   verify the combined windowed force matches the central finite
   difference of the combined windowed `s_full`.

**Done when:** all existing LLR tests untouched and bit-identical; new
imag-dispatch force-consistency test green.

## 4. Phase 3 — `act::BoseGas<T>`

**Goal:** the 4D self-interacting Bose gas at finite chemical potential,
written natively in complex-field storage.

**File added:** `include/reticolo/action/builtins/bose_gas.hpp` (~150 LOC,
mirrors the layout of `act::Phi4`).

**Action definition** (paper eq. 10, with their factor-of-2 normalization
fixed by absorbing into the coupling constants — chosen to make `μ=0` reduce
exactly to O(2)-symmetric Phi4):

```cpp
template <class T = double>
struct BoseGas {
    using value_type = std::complex<T>;
    using field_type = Lattice<std::complex<T>>;

    T mass   = T{1};
    T lambda = T{1};
    T mu     = T{0};   // chemical potential, enters time-direction hopping

    // ---- HasSEff + HasForce ----------------------------------------------
    // S_R = Σ_x [ (2d + m²) |φ_x|² + λ |φ_x|⁴
    //           − 2 Σ_{i=1}^{d-1} Re(φ*_x φ_{x+î})
    //           − 2 cosh(μ) Re(φ*_x φ_{x+d̂})        ]
    T s_full(field_type const& l) const noexcept;
    void compute_force(field_type const& l, field_type& force) const noexcept;

    // ---- HasImagPart -----------------------------------------------------
    // S_I = Σ_x Im(φ*_x φ_{x+d̂})         (μ-independent)
    T s_imag(field_type const& l) const noexcept;
    void compute_force_imag(field_type const& l, field_type& force) const noexcept;
};
```

`d̂` is the last (time) direction. `S_I` is bilinear, gradient is one shift in
each of the two time-neighbours — no transcendentals, no bulk-path trickery
needed for v1.

**Hot-loop helper extension.** `action/hot_loop.hpp` currently assumes
isotropic hopping. The Bose gas needs **anisotropic** time-direction hopping
(`cosh(μ)` coefficient on the last-direction NN sum). Add a variant
`visit_nn_anisotropic_(l, body, time_coupling)` that splits the body into a
spatial-NN sum + a separate scaled time-direction term. ~40 LOC, same
bulk-vs-slab pattern.

**Tests (Phase 3):**
1. `test_bose_gas_force_consistency` — real-part force matches central FD of
   `s_full` to `~1e-7`. Twist both Re and Im of φ_x and check both components.
2. `test_bose_gas_imag_force_consistency` — imag-part force matches FD of
   `s_imag`.
3. `test_bose_gas_mu0_reduces_to_o2_phi4` — at μ=0 the action is O(2)
   symmetric; run on a random φ config and verify `S_R(BoseGas) == S(O2-Phi4)`
   to `~1e-12`.
4. `test_bose_gas_real_phi_reduces_to_phi4` — if Im(φ) ≡ 0 then `S_R` should
   match plain Phi4 on Re(φ) (up to the factor-2 normalization we picked).
5. `test_bose_gas_sym_imag_part` — `S_I[φ_1 ≡ φ_2]` should be exactly zero
   (the antisymmetric ε_{ab} structure).

**Done when:** all five tests green.

## 5. Phase 4 — Example 06: `examples/06_bose_gas_llr/`

**Goal:** reproduce the paper's `⟨e^{iφ}⟩_pq(μ)` curve across the silver-blaze
transition.

**Files added:**
- `apps/bose_gas_llr.cpp` — one binary per μ value, phase-quenched HMC +
  LLR over `S_I`. ~80 LOC, mirrors `u1_llr.cpp`.
- `examples/06_bose_gas_llr/run.sh` — parallel μ scan via `xargs -P JOBS`.
  Three stages, mirroring example 05:
  1. Phase-quenched HMC to estimate the empirical `S_I` range per μ.
  2. LLR over `[S_I_min, S_I_max]` with auto-δ (target ~30 replicas).
  3. Optional second pass with tightened range at the most-suppressed μ.
- `examples/06_bose_gas_llr/analyze.py` — per-μ load `ρ_μ(s)`, build the
  piecewise-exponential interpolant on a fine grid, compute
  `⟨e^{iφ}⟩_pq(μ) = ∫ρ_μ(s) cos(sinh(μ)·s) ds / ∫ρ_μ(s) ds` (eq. 7 of the
  paper), plot `⟨e^{iφ}⟩` and `ΔF = −(1/V) log⟨e^{iφ}⟩_pq` vs μ; mark the
  paper's silver-blaze transition value with a vertical line.

**Defaults shipped:** L = 6 (paper uses 4-16; L=6 hits the dynamics cleanly
in ~5-10 min per μ on 3 cores), m = λ = 1, μ ∈ {0.5, 0.8, 1.0, 1.2, 1.5}
bracketing the transition. The ambitious `O(10^{−480})` precision is not
the target — we aim for `O(10^{−10})` to `O(10^{−30})`, enough to show the
algorithm reproduces the qualitative curve.

**Done when:** the `⟨e^{iφ}⟩_pq(μ)` plot visibly tracks the paper's Fig. 3
shape (smooth decrease in the low-density phase, kink near the critical μ,
silver-blaze behavior above).

## 6. Out of scope (defer to follow-up work)

- 16⁴ thermodynamic-limit fits.
- Direct comparison to dualization / complex Langevin results in the paper.
- Per-window phase-factor measurement during sampling (the
  `sample_with_obs(...)` extension I sketched earlier). The Fourier-transform
  reconstruction we're building doesn't need it; it would be a small but
  separate addition for follow-up papers.
- Gauge analog (4D U(1) + θ-term). The same `HasImagPart` concept applies; a
  `gauge::action::CompactU1WithTheta<T>` is a future builtin.

## 7. Estimated effort

- Phase 1: ~50 LOC + 3 tests. ~1 hour.
- Phase 2: ~80 LOC + 1 test. ~1-1.5 hours.
- Phase 3: ~200 LOC (action + anisotropic helper) + 5 tests. ~2-3 hours.
- Phase 4: ~200 LOC (app + analyze.py + run.sh). ~1-2 hours interactive,
  plus a few hours of compute for the μ scan.

Total: one focused day of work to a green example reproducing the paper's
core result at moderate volume.

## 8. Risks / open questions to settle before coding

- **Force convention with Wirtinger derivatives.** Sign and factor-of-2
  conventions need to be pinned down explicitly before writing
  `compute_force`. Proposal: `compute_force` returns `−∂S/∂φ*` and the kick
  is `mom += dt·F`, mirroring the real-action convention. Verified by the
  Phase-1 reversibility + force-consistency tests.
- **Factor-2 in the paper's eq. 11 vs eq. 10.** Eq. 11 normalizes by an
  extra factor of ½ on the on-site terms (and the implicit factor of 2 on
  the hopping disappears). For us the safest choice is to implement eq. 10
  as written, then absorb the factor in coupling-constant defaults so a
  user-facing `BoseGas{.mass=1, .lambda=1, .mu=...}` matches the paper's
  numbers directly.
- **Numerical stability of `cosh(μ)` at large μ.** Paper goes up to μ ~ 2.0
  where `cosh ~ 3.76` — still fine in `double`. No overflow concerns at our
  target volumes.
