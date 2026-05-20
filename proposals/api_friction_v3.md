# API friction — audit residual (post-docs pass)

This is what's left of the 2026-05 UX audit after the documentation
shortcomings were addressed. Everything below is a code-level concern.

The findings are classified by impact, with file references and a
suggested approach for each. Nothing here is in flight — this is a
candidate punch-list for a future polish pass.

## 🔴 Real problems

### F1. Construction patterns aren't uniform

Actions use designated init; algorithms use positional braced init; and
the positional 4th argument's **type** drifts:

```cpp
act::Phi4<double>{.kappa = 0.13, .lambda = 0.05}        // designated
alg::Hmc<...>{action, phi, rng, {.tau = 1.0, .n_md = 20}}   // positional + struct
alg::Metropolis<...>{action, phi, rng, sigma}                // positional + scalar
alg::Wolff<...>{action, phi, rng}                            // positional, 3 args
```

A user reading these in sequence has no way to predict the 4th-arg
shape. Writing `alg::Metropolis{my_action, phi, rng, my_hmc_spec}` will
silently fail to compile with a "no matching ctor" template error.

**Files:** `include/reticolo/algorithm/{hmc,metropolis,wolff}.hpp:9–55` each.

**Suggested approach:** give each algorithm a `Spec` struct (mirror of
`HmcSpec`), accept it as the 4th designated-init argument:

```cpp
alg::Hmc<...>{action, phi, rng, alg::HmcSpec{.tau = …, .n_md = …}}
alg::Metropolis<...>{action, phi, rng, alg::MetropolisSpec{.sigma = …}}
alg::Wolff<...>{action, phi, rng, alg::WolffSpec{}}                  // empty for now
```

`MetropolisSpec` is currently a free `double sigma`; making it a
1-field struct costs nothing and aligns the surface. Empty
`WolffSpec` reserves the slot for future config (e.g. cluster size
caps) without breaking callers.

### F2. `llr::Replica` constructor + template surface is heavy

```cpp
template <class Base,
          class Rng,
          class Integrator = alg::integ::Leapfrog,
          class T          = typename Base::value_type,
          class Field      = Lattice<T>>
class Replica {
public:
    Replica(std::string id,
            SizeVec shape,
            Base const& base,
            Rng rng_init,
            scalar_t e_n_init,
            scalar_t delta_init,
            scalar_t a_init,
            alg::HmcSpec const& spec);
};
```

5 template params, 8 positional ctor args. Compare to `alg::Hmc` (4
ctor args). Apps use this 4× per LLR binary; the positional sequence is
undocumented inline and easy to mis-order. `e_n` and `a_init` are both
scalars — swap them silently and the simulation diverges.

**Files:** `include/reticolo/llr/replica.hpp:34–55`. Call sites:
`apps/{phi4,u1,bose_gas,su2}_llr.cpp` (4 places, identical pattern).

**Suggested approach:** introduce `llr::ReplicaSpec`:

```cpp
struct ReplicaSpec {
    std::string id;
    SizeVec     shape;
    scalar_t    e_n;
    scalar_t    delta;
    scalar_t    a_init = 0;
};

template <...>
Replica(Base const& base, Rng rng, ReplicaSpec spec, alg::HmcSpec hmc_spec);
```

Call site becomes:

```cpp
std::make_unique<ReplicaT>(base, FastRng{seed + n + 1},
    llr::ReplicaSpec{.id = std::format("r{:03}", n),
                     .shape = shape, .e_n = e_n, .delta = delta},
    alg::HmcSpec{.tau = tau, .n_md = n_md})
```

Designated init names every field; ordering mistakes become impossible.

### F3. Algorithm auto-announce is inconsistent with `llr::Replica`

`llr::Replica` self-announces in its constructor; `alg::Hmc`,
`alg::Metropolis`, `alg::Wolff` do not. The split is invisible to a
newcomer reading the apps: every LLR app drops `log::algo(*replica)`
calls, but every HMC / MC / Wolff app has them.

The root cause (Replica's ctor is non-trivial and already does
side-effects, algorithms try to stay leaner) isn't visible to the user.

**Files:**
- `include/reticolo/llr/replica.hpp:55` (the auto-announce block in
  ctor body)
- `include/reticolo/algorithm/{hmc,metropolis,wolff}.hpp` (no auto-announce)
- Apps: every non-LLR app has `log::algo(hmc);` or `log::algo(mc);`
  right after construction.

**Suggested approach:** add the same 2-line announce block to
`Hmc::Hmc`, `Metropolis::Metropolis`, `Wolff::Wolff`:

```cpp
log::Entry e{log::Level::info, log_tag};
describe(e);
```

Apps then drop their `log::algo(...)` lines. Tests stay silent because
`tests/test_main.cpp` already calls `log::off()`.

## 🟡 Friction

### F4. `alg::Hmc` template signature names are unhelpful

```cpp
template <class A, class R, class Integrator = integ::Leapfrog,
          class Field = Lattice<typename A::value_type>,
          class F     = typename A::value_type>
class Hmc;
```

`Field` and `F` are both meaningful template params, but `F` is named
non-descriptively, and *both* show up in every template error message
even though the user almost never needs to override either.

**File:** `include/reticolo/algorithm/hmc.hpp:86–92`.

**Suggested approach:** rename `F` to `Scalar` (or similar). Consider
moving `Field` and `Scalar` after a defaulted alias so they don't appear
in error messages for typical instantiations. (Not always possible
without breaking deduction; worth trying.)

### F5. `log::scope(std::string{rep.id()})` wrapper is noisy at every call site

`Replica::id()` returns `string_view`; `log::scope()` takes
`std::string`. Apps wrap with `std::string{…}` 1–2× per LLR app:

```cpp
auto _ = log::scope(std::string{reps[n]->id()});
```

8 occurrences across the 4 LLR apps.

**File:** `include/reticolo/core/log.hpp:367`.

**Suggested approach:** overload `log::scope` to accept
`std::string_view`:

```cpp
[[nodiscard]] inline Scope scope(std::string_view run_id) {
    return Scope{std::string{run_id}};
}
[[nodiscard]] inline Scope scope(std::string run_id) {     // existing
    return Scope{std::move(run_id)};
}
```

Both still construct an owning `std::string` inside `Scope` (TLS push
moves it onto the stack); the overload just removes the wrapper at the
call site.

### F6. The LLR per-iter line is duplicated across 4 apps

```cpp
log::info("repl", "RM {:>3}/{}  a={:+.3f}  ⟨dE⟩={:+.3e}  ⟨dE⟩/δ={:+.3f}",
          s + 1, n_rm, a_buf[n], de_buf[n], de_buf[n] / delta);
```

Identical string across `phi4_llr`, `u1_llr`, `bose_gas_llr`, `su2_llr`.
A format change requires 4 edits.

The earlier design decision was "no domain helpers" — apps emit
`log::info` inline. But "no specialised helpers" was meant to push back
against `log::hmc_step`, `log::llr_phase`, etc. — one-off events that
each had their own format. The per-iter LLR line is a different
category: it's a structural row (same 6-field schema across all apps).

**Files:** `apps/{phi4,u1,bose_gas,su2}_llr.cpp`, RM loop body.

**Suggested approach:** add a single helper:

```cpp
namespace reticolo::log {
inline void llr_iter(std::string_view phase, std::size_t k, std::size_t n_iters,
                     double a, double dE, double delta) {
    double const r = (delta != 0.0) ? (dE / delta) : 0.0;
    info("repl", "{}  {:>3}/{}  a={:+.3f}  ⟨dE⟩={:+.3e}  ⟨dE⟩/δ={:+.3f}",
         phase, k, n_iters, a, dE, r);
}
}
```

Or: accept the duplication and document it as intentional. The choice
depends on whether you expect to add a 5th LLR app.

### F7. `obs::` namespace is non-parallel

```cpp
obs::magnetization(phi)             // |<phi>|, takes Lattice<double>
obs::mean(phi)                      // <phi>, plain mean
obs::mean_sq(phi)                   // (<phi>)^2, mean squared
obs::m2(phi)                        // <phi^2>, mean of square
obs::xy_magnetization_sq(theta)     // <|M|^2>/V^2 for XY (angle lattice)
obs::vector_magnetization_sq<N>(phi) // O(N) analogue
```

5 names, no consistent convention. Reading the list, a user can't
predict whether the next O(N) observer is `obs::on_*` or
`obs::vector_*` or `obs::*_n<N>`; whether `mean_sq` is the mean
squared (yes) or the mean of squares (no).

**Files:** `include/reticolo/obs/*.hpp`.

**Suggested approach:** rename for parallelism. One option:

| current name                          | proposed                          |
| ------------------------------------- | --------------------------------- |
| `mean`, `mean_sq`, `m2`               | `mean`, `mean²`, `mean_of_sq`     |
| `magnetization` (scalar)              | `magnetization_abs`               |
| `xy_magnetization_sq`                 | `xy::magnetization_sq` (subns)    |
| `vector_magnetization_sq<N>`          | `on::magnetization_sq<N>` (subns) |

i.e. group field-shape-specific observers into sub-namespaces (`xy::`,
`on::`), use math notation (`²` / `_of_sq`) to disambiguate the two
square observables.

Breaking change — every app needs an update. Worth doing once if
done.

### F8. NaN / divergence handling is silent

`writing_an_app.md` line 214 (now removed in the docs pass, but the
behaviour is unchanged): "If `s_full` returns NaN you'll see NaN; debug
the integrator step size." No tooling helps with this. A user observing
crashed runs has only `dH` history to debug from.

**Files:** every action's `s_full` returns NaN silently when the field
diverges; HMC's accept/reject test `dH <= 0.0 || rng < exp(-dH)`
accepts any NaN (`NaN < x` is `false`, but the first leg is `NaN <= 0`
which is also `false`, so it rejects — except a rejected NaN-bearing
field still gets the rollback path which assumes a finite `s0` cache).

**Suggested approach:** at minimum, log a warning when `dH` is NaN or
exceeds a configurable threshold:

```cpp
if (std::isnan(dH) || std::abs(dH) > k_hmc_warn_threshold) {
    log::warn("phys", "|ΔH|={:.3e} on trajectory {} — integrator may be unstable",
              dH, step_count_);
}
```

Costs one branch per trajectory; lets a user notice instability before
hours of HMC sweep silently diverge.

### F9. CLI errors come from raw cxxopts

A user running `./phi4_hmc` without `--size` gets cxxopts' default
"Option L,size is required but not provided" message. No reticolo
context — the user has to know cxxopts to recognise it.

**Files:** `include/reticolo/cli/parser.hpp` — `parse(argc, argv)`
catches and rethrows cxxopts exceptions without reformatting.

**Suggested approach:** catch the cxxopts exceptions and emit a
`log::error("cli", "missing required flag: {}", flag_name)` line
instead of (or alongside) the raw cxxopts error. Low priority — the
existing message is technically correct, just impersonal.

## 🟢 What works (for context)

These were validated as clean during the audit and should NOT be
touched:

- **Concept refinement lattice** (`docs/action_concepts.md`,
  `include/reticolo/action/detail/concepts.hpp`). Adding a new action
  is a documented ≤50 LOC exercise; every refinement is one method.
- **PIMPL HDF5 split.** `<hdf5.h>` doesn't leak into any header. The
  3-target CMake layout (`reticolo::core` / `reticolo::io` /
  `reticolo::cli`) is the right shape.
- **Apps own the for loop.** No closure callbacks, no event dispatch.
  This is load-bearing for the project's character.
- **Sibling Lattice ctor** (sharing `Indexing`). The one design that
  pays off in every HMC trajectory.
- **`Rng` concept + 3 interchangeable RNGs**. `FastRng` / `Mt19937Rng`
  / `RanluxRng` plug in at every template parameter.
- **Examples directory.** End-to-end studies with bash + python
  post-processing recovering known critical exponents.
- **Build / preset story.** `cmake --preset` covers four platforms,
  one debug build. Reproducible.

## Priority order

If acting on this list, suggested sequence:

1. **F3** (algorithm auto-announce) — biggest user-visible win, smallest
   change. Removes ~3 lines from every non-LLR app.
2. **F5** (`scope(string_view)`) — tiny library change, removes
   wrapping at 8 call sites.
3. **F1 + F2** (Spec structs for algorithms and Replica) — biggest
   *cleanup*, breaks every app. Do both together if you do them.
4. **F6** (one LLR helper) — local fix, ~10 LOC added, ~6 lines saved
   per LLR app. Or formally accept the duplication.
5. **F8** (NaN warning) — small library change, real diagnostic value.
6. **F4** (Hmc template names) — cosmetic; helps error messages.
7. **F9** (CLI error wrap) — low value.
8. **F7** (observables rename) — most disruptive (every app touched);
   only worth it if you're already doing a breaking pass.

Items 1–2 are quick wins that don't break callers. Items 3–8 are at
least partially breaking and worth batching into a single "API
cleanup" pass.
