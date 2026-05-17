# Writing an app — 10 minutes to a working LQFT measurement

This is the canonical walkthrough. We'll build `apps/phi4_hmc.cpp`
line-by-line: 4D φ⁴ scalar field driven by HMC, full reproducibility
metadata in the HDF5 output, susceptibility + magnetisation observers
recorded as time series.

The completed app is ~55 LOC. No framework, no driver, no config file —
the binary's `argv` plus the git SHA stamped to `/run@` is the run record.

## Skeleton

Apps live in [`apps/`](../apps/) and get registered through one CMake line:

```cmake
# apps/CMakeLists.txt
reticolo_add_app(my_app my_app.cpp)
```

`reticolo_add_app` is a tiny helper in the same file — it adds an executable
and links it against `reticolo::reticolo` (the umbrella target that pulls
in core + io + cli).

## Step 1 — the umbrella include

```cpp
#include <reticolo/reticolo.hpp>

#include <cstddef>

int main(int argc, char** argv) {
    using namespace reticolo;
    // ...
}
```

One include gets you `Lattice`, `FastRng`, `act::Phi4`, `alg::Hmc`,
`io::Writer`, `cli::Parser`, `obs::*` — everything. `using namespace
reticolo;` is the idiomatic way to bring in the short namespace aliases
(`act`, `alg`, `obs`) so the body stays terse.

## Step 2 — declare the CLI flags

```cpp
cli::Parser p{"phi4_hmc", "Hybrid Monte Carlo for the phi^4 scalar field"};
auto const& L          = p.req<int>("L,size",       "linear lattice extent");
auto const& kappa      = p.req<double>("kappa",     "hopping parameter");
auto const& lambda     = p.req<double>("lambda",    "quartic coupling");
auto const& ndim       = p.opt<int>("ndim",         4,    "spatial dimensions");
auto const& tau        = p.opt<double>("tau",       1.0,  "HMC trajectory length");
auto const& n_md       = p.opt<int>("n_md",         20,   "MD steps per trajectory");
auto const& n_therm    = p.opt<int>("n_therm",      200,  "thermalisation trajectories");
auto const& n_prod     = p.opt<int>("n_prod",       1000, "production trajectories");
auto const& seed       = p.opt<unsigned long long>("seed", 42ULL, "RNG seed");
auto const& outpath    = p.opt<std::string>("out",  std::string{"phi4.h5"}, "HDF5 output path");
p.parse(argc, argv);
```

`req<T>` registers a required flag and returns `T const&` to stable
parser-owned storage. The reference is meaningful only **after** `parse()`
returns. `opt<T>` is the same but with a default value.

The `"L,size"` form is cxxopts' "short + long" spec — `-L 16` and
`--size=16` both work. Single-character names must be paired with a long
form (cxxopts requirement).

Pass `--help` and the parser prints the usage block and exits.

## Step 3 — wire the action and the field

```cpp
Lattice<double>::SizeVec shape(static_cast<std::size_t>(ndim),
                               static_cast<std::size_t>(L));
Lattice<double>  phi{shape};
FastRng          rng{seed};
act::Phi4<double> phi4{.kappa = kappa, .lambda = lambda};
```

- `Lattice<T>::SizeVec` is `std::vector<std::size_t>`; we build an
  `ndim`-tuple of `L`s for a cubic lattice. (Use any shape you like —
  rectangular lattices work, too.)
- `act::Phi4` is a plain struct with two fields; designated initialisers
  make the construction self-documenting.
- `FastRng` is seeded once. Copies of an `Rng` diverge from each other from
  the moment they're made, which is useful if you ever want reproducible
  parallel chains.

## Step 4 — open the writer and stamp metadata

```cpp
io::Writer out{outpath, argc, argv, &p};
```

Truncate-or-fail. The constructor stamps every reproducibility attribute in
[`/run@*`](architecture.md#io) and, because we pass a `Parser*`, every
resolved CLI flag in `/vars@<name>`. You don't have to write any of this
yourself. The file is now self-describing.

## Step 5 — declare phases and series

```cpp
out.start_phase("therm");
out.start_phase("prod");

auto s_therm  = out.series<double>("/therm/stats/s");
auto d_h      = out.series<double>("/prod/stats/dH");
auto accepted = out.series<int>("/prod/stats/accepted");
auto s_prod   = out.series<double>("/prod/obs/s");
auto mag      = out.series<double>("/prod/obs/mag");
auto mag_sq   = out.series<double>("/prod/obs/mag_sq");
```

- `start_phase("foo")` creates `/foo` and rejects re-use (typo'd phase names
  raise instead of silently overwriting).
- `series<T>(path)` returns a `Series<T>` handle by value (move-only). The
  handle owns a 4096-row buffer and flushes when full or on dtor. Distinct
  handles for `dH`, `accepted`, the observables — every dataset lives at
  its own HDF5 path.

Pick a path naming convention and stick to it (`/<phase>/stats/<name>` and
`/<phase>/obs/<name>` work well — stats are integrator diagnostics, obs are
physics observables).

## Step 6 — the updater

```cpp
alg::Hmc<act::Phi4<double>, FastRng> hmc{phi4, phi, rng, {.tau = tau, .n_md = n_md}};
```

The class template parameters are the action type and the RNG type. The
fourth (defaulted) parameter is the integrator — `integ::Leapfrog` is the
only one shipping today; swap in `integ::Yoshida4` later by extending the
type, no other change.

The `Hmc` constructor pre-allocates the momentum, force, and rollback
lattices as **siblings** of `phi` (same `Indexing`). One neighbour table,
three lattices.

## Step 7 — own the for loop

```cpp
for (int i = 0; i < n_therm; ++i) {
    (void)hmc.trajectory();
    s_therm.append(phi4.s_full(phi));
}
for (int i = 0; i < n_prod; ++i) {
    auto const step = hmc.trajectory();
    d_h.append(step.dH);
    accepted.append(step.accepted ? 1 : 0);
    s_prod.append(phi4.s_full(phi));
    mag.append(obs::magnetization(phi));
    mag_sq.append(obs::mean_sq(phi));
}
```

This is the philosophy: **apps own the for loop**. There's no closure-based
driver, no `simulate(action, n_steps)` helper. The trajectory cadence, the
measurement cadence, the conditional logic about when to write what — it's
all visible in plain `main`.

`hmc.trajectory()` returns a `HmcStep { dH, accepted }`. Stash both for
later diagnostics — the per-trajectory `dH` history is the cheapest
single check that the integrator step is sized right.

## Step 8 — build and run

```sh
cmake --build --preset macos-appleclang --target phi4_hmc
build/macos-appleclang/apps/phi4_hmc --size 8 --kappa 0.13 --lambda 0.02 --out phi4.h5
```

Inspect the output:

```sh
h5ls -r phi4.h5
# /                        Group
# /prod                    Group
# /prod/obs                Group
# /prod/obs/mag            Dataset {1000}
# /prod/obs/mag_sq         Dataset {1000}
# /prod/obs/s              Dataset {1000}
# /prod/stats              Group
# /prod/stats/accepted     Dataset {1000}
# /prod/stats/dH           Dataset {1000}
# /run                     Group
# /therm                   Group
# /therm/stats/s           Dataset {200}
# /vars                    Group

h5dump -A phi4.h5 | grep '@'
# everything you passed on argv + the git SHA + compile flags
```

The HDF5 is fully self-describing — push it through Python/numpy/h5py for
the analysis. See [`examples/01_phi4_tuning/analyze.py`](../examples/01_phi4_tuning/analyze.py)
for the block-jackknife + Bhattacharjee-Seno collapse pattern.

## When to start a new app

When you'd otherwise be adding a `--mode` flag to an existing app. The cost
of writing a new app is one file in `apps/`, one line in
`apps/CMakeLists.txt`, and one smoke test in `tests/apps/`
(template: [`tests/apps/test_phi4_hmc_smoke.cpp`](../tests/apps/test_phi4_hmc_smoke.cpp)).
That's *cheaper* than the conditional logic + documentation a multi-mode
app accumulates.

The library ships six apps for exactly this reason — same physics in some
of them, different updaters or different observables, hand-written.

## What's *not* in the app

- **No** logging library beyond `log::Section` (RAII timer / banner).
- **No** error recovery beyond what the standard library does. If
  `s_full` returns NaN you'll see NaN; debug the integrator step size.
- **No** result post-processing: that's Python's job downstream. The app
  writes raw per-trajectory time series; reductions happen later.

That's the whole framework.
