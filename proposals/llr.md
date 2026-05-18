# LLR module — design plan

Status: proposal, awaiting "build it".
Branch target: a fresh `feature/llr` off `rewrite/v3`.
Scope: real-valued actions only. RE-LLR with HMC/Omelyan kernel.

## 1. What we are building

A self-contained `llr/` sub-library under `include/reticolo/llr/` that adds the
Langfeld–Lucini–Rago density-of-states algorithm to the existing HMC stack,
parameterised over an arbitrary energy observable, with replica exchange.

User choices fixed up front:

- **Window**: Gaussian penalty `(E − E_n)^2 / (2 δ^2)` added to the action.
- **Kernel**: existing HMC + leapfrog / Omelyan2 / Omelyan4 — reused unchanged.
- **a-update**: Newton–Raphson warm-up (k=0…n_NR−1) then restarted
  Robbins–Monro (counter resets to k=0).
- **Geometry**: N replicas in one process, alternating even/odd NN exchange.
- **Energy E**: templated, user-supplied functor satisfying `EnergyObservable`.
  **Default = the base action itself** (`E(φ) ≡ S_base(φ)`). A
  hopping-only variant is a natural follow-up: if the action splits as
  `S = κ·E_hop + V_local`, then `Z(κ) = ∫ dE_hop ρ(E_hop) e^{−κ E_hop}`,
  so one LLR run on E_hop reconstructs Z over a κ-range via Laplace
  transform — same structural trick as Langfeld–Lucini–Rago use on the
  plaquette in lattice gauge theory. Whether the φ⁴ LLR literature has
  done this specific split: needs verification, not assumed.
- **First test bed**: φ⁴ scalar (existing `act::Phi4`) with E = S.

## 2. Tilted-action layout

Define on each replica the effective action for HMC sampling:

```
S_LLR(φ; a, E_n, δ) = S_base(φ) + a · E(φ) + (E(φ) − E_n)^2 / (2 δ^2)
```

Force per site (what HMC needs):

```
F(x) = F_base(x) − ( a + (E(φ) − E_n)/δ^2 ) · ∂E/∂φ(x)
```

so `EnergyObservable` must expose both a scalar `value(φ)` and a way to
accumulate a coefficient-scaled gradient into a force lattice.

**Default case `E = S_base`.** When the energy observable is the base
action itself the formulas collapse — no separate gradient is needed:

```
S_LLR = (1 + a) · S_base + (S_base − E_n)^2 / (2 δ^2)
F_LLR(x) = ( 1 + a + (S_base − E_n)/δ^2 ) · F_base(x)
```

i.e. compute the base force once, scale it by a runtime scalar. This is
what the default adapter `ActionAsEnergy` provides — no new force code,
no per-action work to bring up v1.

Restricted-ensemble expectation used by the updates:

```
<dE> := < E(φ) − E_n >    (sampled with S_LLR at current a)
```

Updates:

- NR step:  `a ← a + 12 <dE> / δ²` (for k = 0 .. n_NR − 1)
- RM step:  `a ← a + 12 <dE> / ( δ² · (k + 1) )` with k reset to 0
  at the start of the RM phase.

Exchange acceptance between replicas i, j (their E values, their a values):

```
P_swap = min{ 1, exp( (a_i − a_j) · (E_i − E_j) ) }
```

swap φ between replicas; keep a_i, E_n,i in place.

## 3. File additions

All new files, no edits to existing modules **except** the umbrella header.

### Headers — `include/reticolo/llr/`

1. `concepts.hpp` (~40 LOC)
   - `EnergyObservable<O, T>`: requires
     `O::value(Lattice<T> const&) -> T` and
     `O::add_force(Lattice<T> const&, Lattice<T>& out, T coeff)`
     (accumulates `coeff · ∂E/∂φ(x)` into `out`).
   - Default adapter `template<class Base> struct ActionAsEnergy`
     wrapping a base action: `value = base.s_full(phi)`,
     `add_force(phi, out, c)` calls `base.compute_force_and_kick(phi, out, c)`
     when present (which adds `c · F_base = −c · ∂S/∂φ`, sign already
     matches `−∂E/∂φ` since E = S).
     With this, `WindowedAction<Base, ActionAsEnergy<Base>>` only ever
     needs `base.s_full` + `base.compute_force(_and_kick)`, no new
     gradient code per action.

2. `windowed_action.hpp` (~80 LOC)
   - `template<class Base, class Energy> struct WindowedAction`
     holding `(Base& base, Energy& energy, T a, T E_n, T delta)`.
   - Satisfies the existing `act` concepts:
     - `s_full(phi)` → `base.s_full + a*E + (E−E_n)^2/(2δ²)`
     - `compute_force(phi, F)` → call `base.compute_force(phi, F)`,
       compute `E`, then `energy.add_force(phi, F, −(a + (E−E_n)/δ²))`.
     - Fused `compute_force_and_kick` only if `Base::HasFusedKick`
       and `Energy` exposes a fused variant — otherwise skip (HMC falls back).
   - No `s_local` — Metropolis path is out of scope for first cut.

3. `update_a.hpp` (~20 LOC) — pure free functions
   - `nr_update(a, mean_dE, delta) -> a`
   - `rm_update(a, mean_dE, delta, k) -> a`

4. `replica.hpp` (~80 LOC)
   - `template<class Base, class Energy, class Rng> class Replica`
   - Owns: `Lattice<T> phi`, `Rng rng`, `Base base`, `Energy energy`,
     `WindowedAction<...> tilted`, `alg::Hmc<...> hmc`, `T a`, `T E_n`, `T delta`.
   - API:
     - `void thermalize(int n)` — n HMC trajectories, discard
     - `T sample(int n)` — n trajectories, return running mean of `E − E_n`
     - `T energy() const` — current `energy.value(phi)`
     - `void set_a(T a_new)` — re-stamp into tilted action
     - `void swap_phi_with(Replica& other)` — used by exchange
   - Construction takes `HmcSpec` (tau, n_md, integrator), seed, lattice shape.

5. `exchange.hpp` (~30 LOC)
   - `try_exchange(Replica& a, Replica& b, rng) -> bool` applies the
     acceptance above and swaps configs on accept.
   - `sweep_even_odd(span<Replica>, int parity, rng)` — runs the alternation.

### Energy observables

No new builtin observable in v1 — `ActionAsEnergy<Base>` (in
`concepts.hpp`) is the default and is enough for the φ⁴ test bed.
Hopping-only and other custom observables can be added later by users
without touching the `llr/` core.

### Umbrella include

6. `include/reticolo/reticolo.hpp` — add a small block:
   ```
   #include <reticolo/llr/concepts.hpp>
   #include <reticolo/llr/windowed_action.hpp>
   #include <reticolo/llr/update_a.hpp>
   #include <reticolo/llr/replica.hpp>
   #include <reticolo/llr/exchange.hpp>
   ```

### App and example

7. `apps/phi4_llr.cpp` (~150 LOC) — single binary, app owns the for-loop.

   Sketch of `main`:
   ```cpp
   cli::Parser p{"phi4_llr", "LLR with replica exchange for phi^4"};
   // lattice, action, RE-LLR knobs (E_min, E_max, n_rep, delta, n_nr, n_rm, ...)
   // ...
   std::vector<Replica<...>> reps;
   reps.reserve(n_rep);
   for (int n = 0; n < n_rep; ++n)
       reps.emplace_back(/* shape, base, energy, a0=0, E_n, delta, hmc_spec, seed+n */);

   // NR warm-up
   for (int k = 0; k < n_nr; ++k) {
       for (auto& r : reps) {
           r.thermalize(n_therm_nr);
           auto m = r.sample(n_meas_nr);
           r.set_a(llr::nr_update(r.a(), m, delta));
       }
   }

   // RM with replica exchange
   for (int s = 0; s < n_rm; ++s) {
       for (auto& r : reps) {
           r.thermalize(n_therm_rm);
           auto m = r.sample(n_meas_rm);
           r.set_a(llr::rm_update(r.a(), m, delta, s));
       }
       llr::sweep_even_odd(reps, s & 1, exchange_rng);
       // append per-replica diagnostics to HDF5
   }
   ```

8. `apps/CMakeLists.txt` — add `add_app(phi4_llr)` (or whatever the
   existing macro is — confirm during impl).

9. `examples/04_phi4_llr/`
    - `README.md` — demo, not a literature replication. The published
      non-gauge LLR papers use discrete spins (Potts) or complex-action
      continuous fields (Bose gas at finite μ); plain real-action φ⁴ LLR
      does not appear in the literature we found, so we treat this
      example as a sanity demonstration of the implementation.
    - `run.sh` — same lattice, same λ, several κ values: a few clearly
      symmetric, a couple near the broken-symmetry side, and one or two
      close to κ_c for the chosen (ndim, λ). The reportable result is
      how `ln ρ(S)` changes shape as κ approaches criticality (broad
      single-peak away → narrowing / double-peak structure near transition).
    - `analyze.py` — read `a_n` history per replica per κ; integrate
      piecewise to reconstruct `ln ρ(S)` up to a constant; bootstrap over
      the last block of RM updates; overlay the curves at the different κ.

## 4. Output schema (HDF5)

Per replica `n`:
```
/replica_n/run@*             reproducibility metadata
/replica_n/cfg/E_n           scalar (window center)
/replica_n/cfg/delta         scalar (window width)
/replica_n/a_history         T-series, one append per NR or RM step
/replica_n/dE_means          T-series, paired with a_history
/replica_n/exch_with_lo      0/1 series, attempts indexed by sweep
/replica_n/exch_with_hi      0/1 series
/replica_n/E_obs             T-series, the running E sampled during prod
```

Global:
```
/llr/cfg/E_min, E_max, dE, n_rep, n_nr, n_rm, ...
/llr/cfg/integrator, tau, n_md
```

`a_history.last()` per replica is the final LLR slope used for ρ
reconstruction in `analyze.py`.

## 5. Reuse vs new code

- HMC, integrators, RNG, Lattice, Writer, Parser, action concepts: reused
  unchanged. `WindowedAction` satisfies the existing action concepts so
  `alg::Hmc` works on it with zero changes.
- `act::Phi4` reused as the base action. Hopping energy is a thin reader
  over the same lattice — no duplication.
- Apps own the loop (NR phase, RM phase, exchange sweep) — matches the
  established style; no closure driver, no string-dispatch.

## 6. Out of scope (first cut)

- Metropolis kernel path / hard-window reflection.
- Multicanonical or Wang–Landau crossover.
- MPI / inter-process replica exchange.
- Resume from HDF5 between binary invocations.
- Complex `a_n` (sign problem) — flagged explicitly: real actions only.
- Per-action `s_local` for the tilted action (Metropolis would need it).

## 7. Build-it sequence

When the user says "build it", implementation order:

1. `EnergyObservable` concept + `ActionAsEnergy<Base>` default adapter.
2. `WindowedAction` against the existing `act` concepts. Quick smoke
   test by running plain HMC on `WindowedAction` with `a = 0`, `δ → ∞`
   (should match running HMC on the bare base action up to roundoff).
3. `Replica` wrapper, `nr_update`, `rm_update`.
4. `exchange` + `sweep_even_odd`.
5. Wire `apps/phi4_llr.cpp`, add to CMake.
6. `examples/04_phi4_llr/` scripts + README; pick the literature target
   together before tuning the run.

Tests and validation per the project rule come once the feature shape is
stable (after step 5 builds and runs).

## 8. Open questions to settle before step 1

- **Literature target**: which exact paper / figure should the example
  reproduce? Picks the lattice size, λ, κ range, and `(E_min, E_max, δ)`.
- **Default integrator** for the example: Omelyan2 seems right (cheap,
  larger usable dt). Confirm.
- **n_rep, n_nr, n_rm** defaults in `run.sh`: tied to the literature
  target — defer to that decision.
