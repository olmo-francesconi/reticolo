# 11 — Generalized-observable LLR: density of states of an arbitrary Q

LLR normally reconstructs ρ(S), the density of states of the **action**. But
the windowing/adapting/exchanging machinery doesn't care *what* observable it
constrains. This example reconstructs the DoS of the **field amplitude**
Q = Σₓ φ(x)² for a φ⁴ scalar field — a quantity that is not the action — using
exactly the same `orch::llr::Orchestrator` driver.

The whole point is that the base action you *sample*, the observable you
*constrain*, the window, and the sampler are independent building blocks. Going
from example 03 (LLR on S) to this one is a **single template argument**:

```cpp
using Constraint = action::ObservableConstraint<example::FieldAmplitude<double>>;
using Llr        = orch::llr::Orchestrator<act::Phi4<double>, FastRng,
                                           updater::integ::Omelyan2, double,
                                           Lattice<double>, Constraint>;
```

## Defining the observable

An observable is just an action leaf: you write per-site formula kernels and its
value/gradient run through the shared parallel sweep engine — **no hand-rolled
lattice loop**, so it threads automatically like every other action
(`observable.hpp`):

```cpp
template <class T = double>
struct FieldAmplitude : reticolo::action::NNAction<FieldAmplitude<T>, T> {
    using value_type = T;
    auto action_kernel() const noexcept {         // Q_site = φ²
        return [](T self, T /*agg*/) { return self * self; };
    }
    auto force_kernel() const noexcept {          // −dQ/dφ = −2φ
        return [](std::size_t, T self, T /*agg*/) { return T{-2} * self; };
    }
};
```

Swap this leaf for any other observable — magnetization Σφ, a two-point
function, a gauge topological charge — and the same LLR machine reconstructs
its DoS.

## What the run does

The window is `S_win = S_φ⁴ + a·Q + (Q − Eₙ)²/2δ²`, so at the LLR fixed point
`⟨Q − Eₙ⟩ = 0` the converged per-replica tilt is `aₙ = d ln ρ / dQ|_{Eₙ}` and
`ln ρ(Q) = ∫ a dQ`.

1. **Stage 1 — plain HMC** (`phi4_hmc`). Stores `⟨φ²⟩` per trajectory; its
   histogram is an importance-sampled ρ(⟨φ²⟩), accurate only in the typical
   band — and doubles as the range-finder for the LLR window.
2. **Stage 2 — generalized LLR** (`generalized_dos_llr`). Runs
   Newton-Raphson warm-up + restarted Robbins-Monro + even/odd replica
   exchange over a fine ladder of amplitude windows. `δ` (per-site, in ⟨φ²⟩
   units) is the single knob — it also sets the replica spacing, so the app
   derives `n_rep` to cover the padded range.
3. **Comparison** (`analyze.py`). Integrates `aₙ` to `ln ρ(⟨φ²⟩)`, aligns it to
   the direct histogram over their overlap, and overlays the two. They agree in
   the band the plain run explores; the LLR curve extends smoothly into the
   suppressed tails.

CLI window units are per-site ⟨φ²⟩ = Q/V (intuitive); the app scales by the
volume V = Lⁿᵈⁱᵐ internally, and stamps `/cfg@V` so `analyze.py` can label the
axis in ⟨φ²⟩.

## Building & running

```sh
cd examples/11_generalized_dos_llr
cmake -S . -B build && cmake --build build      # standalone
# or:
bash examples/11_generalized_dos_llr/run.sh
```

Override the preset with `RETICOLO_PRESET=macos-llvm ./run.sh` (recommended on
macOS for OpenMP-enabled LLR replicas).

Outputs in `results/`:
* `hmc.h5`      — plain HMC output (per-trajectory ⟨φ²⟩ etc.)
* `amp_llr.h5`  — LLR slopes per replica + exchange stats + `/cfg@V`

And `generalized_dos_llr.png` next to the script: tilt adaptation, converged
`aₙ`, and `ln ρ` / `ρ` vs the direct histogram.

## Knobs (env vars)

| variable          | default            | meaning                                          |
| ----------------- | ------------------ | ------------------------------------------------ |
| `RETICOLO_PRESET` | `macos-appleclang` | which build preset to invoke                     |
| `NDIM`, `L`       | `3`, `8`           | dimensions / linear extent                       |
| `KAPPA`, `LAMBDA` | `0.18`, `1.0`      | φ⁴ couplings                                     |
| `N_THERM`, `N_PROD` | `2000`, `20000`  | HMC thermalisation / production trajectories     |
| `AMP_MIN`, `AMP_MAX` | `0.38`, `0.88`  | LLR window range, per-site ⟨φ²⟩ (fixed wide, straddling the peak ≈ 0.60) |
| `DAMP`            | `0.017`            | window δ = replica spacing, per-site ⟨φ²⟩ (the single LLR knob) |
| `N_NR`, `N_RM`    | `6`, `18`          | NR warm-up / RM sweep counts                     |
| `N_THERM_NR`, `N_MEAS_NR` | `80`, `250` | HMC traj counts inside each NR iter            |
| `N_THERM_RM`, `N_MEAS_RM` | `50`, `250` | HMC traj counts inside each RM sweep           |
| `TAU`, `N_MD`     | `1.0`, `20`        | HMC trajectory length / MD steps                 |
| `SEED`            | `42`               | RNG seed                                         |

Defaults are sized for a quick laptop run. For sharper tails, raise `N_RM` and
lower `DAMP`.
