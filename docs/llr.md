# LLR — Logarithmic Linear Relaxation

LLR (Langfeld–Lucini–Rago) is a density-of-states sampling method.
Instead of sampling configurations from the Boltzmann measure `e^{−S(φ)}`
and computing observables as ensemble averages, LLR estimates the density
of states ρ(E) directly. With ρ(E) in hand, the partition function and
any observable that's a function of the energy are computable by
quadrature for *any* coupling — no rerunning. This is the trick that
breaks freezing in first-order transitions and lets you reconstruct
sign-problem-bound theories (BoseGas at finite μ) from a phase-quenched
ensemble.

This document explains what's happening, then walks through one of the
shipped LLR apps line by line. Theory references at the bottom.

## 1. The idea in one paragraph

Slice the energy axis into N windows centred at E_0, E_1, …, E_{N-1}
with spacing δ. For each window, run an HMC trajectory that samples not
from `e^{−S}` but from a **tilted** measure `e^{−S − a·E − (E−E_n)²/2δ²}`,
where `a` is a per-replica reweighting parameter and the Gaussian
penalty pins the replica to its window. Tune `a` so that, on average,
`⟨E − E_n⟩ ≈ 0` — i.e. the replica sits centered on its window. The
fitted `a(E_n)` is the local log-slope of ρ(E): `a(E_n) ≈ d log ρ / dE`
at E = E_n. Integrate `a` across windows and you have log ρ(E).

The fitting loop has two phases: a **Newton–Raphson warm-up** that
quickly pulls each replica into its window, then a **Robbins–Monro**
phase that polishes with shrinking step size. Adjacent replicas
periodically exchange their fields to improve mixing across the energy
range.

## 2. The tilted action

For an action `S_base` and an energy observable `E(φ)`, the LLR
measure on replica `n` is

```
S_LLR(φ; a, E_n, δ) = S_base(φ) + a·E(φ) + (E(φ) − E_n)² / (2δ²)
```

In the shipped library the energy observable is the base action itself
(`E = S_base`), so the formulas collapse cleanly:

```
S_LLR = (1 + a)·S_base + (S_base − E_n)² / (2δ²)
F_LLR(x) = (1 + a + (S_base − E_n)/δ²) · F_base(x)
```

— compute the base force once, scale it by a runtime scalar. This is
what `llr::WindowedAction` does. It wraps any base action and exposes
the same `s_full` / `compute_force` interface, so `alg::Hmc` consumes
it identically to a raw action.

## 3. The update rules

After each sample of `⟨dE⟩ := ⟨E − E_n⟩` on a replica, advance its `a`:

```
NR phase (k = 0 … n_NR−1):    a ← a + C · ⟨dE⟩ / δ²
RM phase (k = 0 … n_RM−1):    a ← a + C · ⟨dE⟩ / (δ² · (k+1))
```

`C` is a window-shape coefficient: `C = 1` for the Gaussian penalty
(what the library uses), `C = 12` for a hard window (not shipped).
Mixing the coefficients across phases diverges geometrically — the
library locks `C = 1` everywhere.

`llr::nr_update(a, dE, δ)` and `llr::rm_update(a, dE, δ, k)` are
provided as free functions; apps call them between `sample()` and
`set_a()`.

## 4. Replica exchange

Every RM iteration (after the per-replica updates), adjacent replicas
attempt to swap fields with Metropolis acceptance:

```
P_swap(i, j) = min{1, exp((a_i − a_j) · (E_i − E_j))}
```

The exchange itself swaps the `φ` field between two replicas; the `a`
and `E_n` stay attached to their replica. Even/odd alternation across
RM iterations covers every adjacent pair on average.

`llr::try_exchange(rep_i, rep_j, rng)` does one attempt.

## 5. The Replica class

```cpp
template <class Base, class Rng,
          class Integrator = alg::integ::Leapfrog,
          class T          = typename Base::value_type,
          class Field      = Lattice<T>>
class llr::Replica;
```

Each `Replica` owns:

- a unique `id` (free-form string, e.g. `"r000"`) used as the log scope tag,
- its own `Field` (so OpenMP parallel-for over replicas is data-clean),
- its own `Rng`,
- a `WindowedAction<Base, T, Field>` wrapping a copy of the base action
  with this replica's `a` / `E_n` / `δ`,
- its own `alg::Hmc` instance threading the windowed action.

Key methods:

| method                                                | what it does                                          |
| ----------------------------------------------------- | ----------------------------------------------------- |
| `thermalize(n, mode = normal)`                        | run n HMC trajectories silently; log a summary on completion (`acc=…`) unless mode = silent |
| `sample(n, mode = normal)` → `scalar_t`               | run n trajectories, return ⟨dE⟩, log summary          |
| `a()`, `E_n()`, `delta()`, `set_a(v)`, `set_E_n(v)`   | window-parameter accessors                            |
| `id()` → `string_view`                                | for `log::scope(rep.id())` in the parallel-for body   |
| `phi()`                                               | direct field access for cold-start initialisation     |

Inside `thermalize` / `sample`, the inner HMC trajectories run with
`log::Mode::silent` — only the call boundary logs.

## 6. The app skeleton

Every shipped LLR app (`phi4_llr`, `u1_llr`, `su2_llr`, `bose_gas_llr`)
follows the same structure. Annotated:

```cpp
int main(int argc, char** argv) {
    using namespace reticolo;
    using Action   = action::Phi4<double>;
    using ReplicaT = llr::Replica<Action, FastRng>;

    cli::Parser p{ /* ... */ };
    /* parse, get L, ndim, kappa, lambda, delta, E_min, E_max,
       n_nr, n_therm_nr, n_meas_nr, n_rm, n_therm_rm, n_meas_rm,
       seed, outpath */
    if (!p.parse(argc, argv)) return 0;

    log::start(outpath);                                  // banner + per-run files

    Lattice<double>::SizeVec shape(ndim, L);
    Action const base{.kappa = kappa, .lambda = lambda};
    log::act(base);                                        // one act line

    // Compute n_rep so adjacent window centres are δ apart over [E_min, E_max].
    int const n_rep = std::max(2, int(std::lround((E_max − E_min)/delta)) + 1);

    // Build replicas. Each Replica auto-announces itself with its own scope.
    std::vector<std::unique_ptr<ReplicaT>> reps;
    for (int n = 0; n < n_rep; ++n) {
        double const e_n = E_min + n*delta;
        reps.push_back(std::make_unique<ReplicaT>(
            std::format("r{:03}", n),                      // id
            shape, base,
            FastRng{seed + 1 + n},                         // per-replica RNG
            e_n,                                           // window centre
            delta,                                         // window width
            /*a_init=*/0.0,
            alg::HmcSpec{.tau = tau, .n_md = n_md}));
    }

    io::Writer out{outpath, argc, argv, &p};
    /* declare per-replica series for a, dE; one shared series for exchange acceptance */

    // === Newton-Raphson warm-up ===
    log::info("llr", "NR phase  {} iters × {} replicas", n_nr, n_rep);
    for (int k = 0; k < n_nr; ++k) {
#pragma omp parallel for schedule(dynamic, 1)
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            auto _  = log::scope(std::string{reps[n]->id()});   // r0NN binding
            auto& r = *reps[n];
            r.thermalize(n_therm_nr);                            // logs as r0NN
            de_buf[n] = r.sample(n_meas_nr);
            a_buf[n]  = llr::nr_update(r.a(), de_buf[n], delta);
            r.set_a(a_buf[n]);
        }
        /* serialize the buffers into HDF5 (HDF5 writes are not threadsafe) */
        log::info("llr", "NR iter  {:>3}/{}  done", k + 1, n_nr);
    }

    // === Robbins-Monro + exchange ===
    log::info("llr", "RM phase  {} iters × {} replicas", n_rm, n_rep);
    for (int s = 0; s < n_rm; ++s) {
#pragma omp parallel for schedule(dynamic, 1)
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            auto _  = log::scope(std::string{reps[n]->id()});
            auto& r = *reps[n];
            r.thermalize(n_therm_rm, log::Mode::silent);         // silent for RM
            de_buf[n] = r.sample(n_meas_rm, log::Mode::silent);
            a_buf[n]  = llr::rm_update(r.a(), de_buf[n], delta, s);
            r.set_a(a_buf[n]);
            log::info("repl", "RM {:>3}/{}  a={:+.3f}  ⟨dE⟩={:+.3e}  ⟨dE⟩/δ={:+.3f}",
                      s + 1, n_rm, a_buf[n], de_buf[n], de_buf[n] / delta);
        }
        /* append to series */

        // Alternating even/odd exchange.
        int accepted = 0, attempts = 0;
        for (std::size_t i = (s & 1); i + 1 < reps.size(); i += 2) {
            ++attempts;
            if (llr::try_exchange(*reps[i], *reps[i + 1], exch_rng)) ++accepted;
        }
        log::info("exch", "step  {:>3}  accepted  {}/{}", s + 1, accepted, attempts);
        log::info("llr", "RM iter  {:>3}/{}  done", s + 1, n_rm);
    }
}
```

A few invariants worth flagging:

- **One scope per parallel-for iteration.** The `log::scope(...)` line at
  the top of the body is what makes every nested log call (including
  inside `Replica::thermalize` → `Hmc::step`) carry the replica's
  run id. Skip it and you'll see the `⚠ unscoped` warning on every line.
- **HDF5 serialisation is non-parallel.** All per-iteration appends are
  staged into `de_buf` / `a_buf` arrays inside the parallel-for, then a
  *serial* loop drains them into `Series<double>`. HDF5's library is not
  threadsafe by default; this is the simple workaround.
- **Therm is loud in NR, silent in RM.** During NR each replica's
  thermalisation is informative (you're watching it pull into its
  window); during RM (typically 10×–100× more iterations) it would
  flood. The library doesn't choose for you — apps pass `Mode::silent`.

## 7. Tuning knobs

| knob              | what it controls                                      | typical range          |
| ----------------- | ----------------------------------------------------- | ---------------------- |
| `delta`           | window half-width; also the NR/RM step coefficient denominator | tuned to the volume — start near `√V` |
| `n_rep`           | derived from `(E_max − E_min)/delta`                  | 8–32 for production    |
| `n_nr`            | NR warm-up iterations                                 | 6–12                   |
| `n_therm_nr`      | trajectories per NR sweep (per replica)               | 200–1000               |
| `n_meas_nr`       | measurement trajectories per NR sweep                 | 1000–5000              |
| `n_rm`            | RM iterations                                         | 100–1000               |
| `n_therm_rm`      | trajectories per RM iter                              | 50–200                 |
| `n_meas_rm`       | measurement trajectories per RM iter                  | 200–1000               |

Convergence diagnostic: in the RM phase, `⟨dE⟩/δ` should drift toward 0
on every replica. The per-iter `repl  RM …  ⟨dE⟩/δ=…` log line is the
quickest signal — values shrinking past ~0.1 in absolute terms are a
healthy sign that the window is centred.

## 8. Complex LLR (`bose_gas_llr`)

The BoseGas action splits as `S = S_R + i·sinh(μ)·S_I` (relativistic
Bose gas at finite chemical potential). HMC samples the real
phase-quenched part `S_R`; LLR's energy observable becomes `S_I`, the
imaginary part. Reconstructing the full partition function is a
Fourier-transform of ρ(S_I) against `e^{i·sinh(μ)·S_I}`.

This is implemented by:

1. `BoseGas::s_imag(φ)` — returns the imaginary observable.
2. `WindowedAction` specialisation for `k_complex = true` — uses
   `base.last_s_imag()` instead of `base.last_s_full()` for the window
   penalty.
3. Same Replica / NR / RM / exchange machinery as the real case.

The app needs a hot-start + per-replica Metropolis cascade thermalisation
because HMC alone can't reliably pull each replica into its `S_I`
window at finite μ; see `apps/bose_gas_llr.cpp` for the pattern.

## 9. References

- K. Langfeld, B. Lucini, A. Rago — *The density of states in gauge
  theories* (2012), Phys. Rev. Lett. 109, 111601.
- K. Langfeld, B. Lucini, R. Pellegrini, A. Rago — *An efficient
  algorithm for numerical computations of continuous densities of
  states* (2016), Eur. Phys. J. C 76, 306.
- Library proposal & implementation notes: `proposals/llr.md`,
  `proposals/complex_llr.md`, `proposals/llr_cache_wiring.md`.
