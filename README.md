# reticolo

Lightweight **C++23** library for **serial** Monte Carlo simulations of scalar
quantum field theories on a lattice.

The design priorities, in order:

1. **Small public surface.** A user writes a hand-rolled `main()` that wires a
   `Lattice`, an action struct, an updater, and an HDF5 writer — ~30–60 lines,
   no framework, no string-dispatch driver.
2. **Performance at the core.** No virtual dispatch in the hot loop; action
   methods inline into the templated updater; indexing tables shared across
   lattices of the same shape; HMC integrator is a type parameter.
3. **One way to do each thing.** One writer, one CLI helper, one logger,
   no central registries. The compiler is the source of truth.

Scope (deliberate): serial only, scalar fields only. No OpenMP, no MPI, no
gauge theories. The architecture leaves room for any of these later without
renegotiating the public surface — see
[`proposals/rewrite_plan_v3.md`](proposals/rewrite_plan_v3.md).

## A complete app

```cpp
#include <reticolo/reticolo.hpp>

int main(int argc, char** argv) {
    using namespace reticolo;

    cli::Parser p{"phi4_hmc", "HMC for the phi^4 scalar field"};
    auto const& L      = p.req<int>("L,size", "linear extent");
    auto const& kappa  = p.req<double>("kappa", "hopping");
    auto const& lambda = p.req<double>("lambda", "quartic coupling");
    auto const& seed   = p.opt<unsigned long long>("seed", 42ULL, "RNG seed");
    auto const& out    = p.opt<std::string>("out", std::string{"phi4.h5"}, "HDF5 path");
    p.parse(argc, argv);

    auto const l = static_cast<std::size_t>(L);
    Lattice<double>            phi{{l, l, l, l}};
    FastRng                    rng{seed};
    act::Phi4<double>          action{.kappa = kappa, .lambda = lambda};

    io::Writer file{out, argc, argv, &p};
    file.start_phase("prod");
    auto s_series = file.series<double>("/prod/obs/s");
    auto m_series = file.series<double>("/prod/obs/mag");

    alg::Hmc<act::Phi4<double>, FastRng> hmc{action, phi, rng, {.tau = 1.0, .n_md = 20}};

    for (int i = 0; i < 1000; ++i) {
        (void)hmc.trajectory();
        s_series.append(action.s_full(phi));
        m_series.append(obs::magnetization(phi));
    }
}
```

The whole stack (`Lattice`, action, integrator-templated HMC, CLI parser,
HDF5 writer, observers) is in `<reticolo/reticolo.hpp>` — one include.

## Build

```sh
cmake --preset macos-appleclang
cmake --build --preset macos-appleclang
ctest --preset macos-appleclang
```

Presets: `macos-appleclang`, `macos-llvm`, `linux-gcc`, `linux-clang`,
`debug` (asan + ubsan).

External deps (vendored via FetchContent): HDF5 (system), cxxopts, Catch2.

## Reading order

| document                                                        | what it gives you                                                                 |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| [`docs/writing_an_app.md`](docs/writing_an_app.md)              | step-by-step build of the snippet above                                           |
| [`docs/action_concepts.md`](docs/action_concepts.md)            | the concept-refinement lattice (write a new action in <50 LOC)                    |
| [`docs/architecture.md`](docs/architecture.md)                  | three CMake targets, PIMPL HDF5, value-semantic `Lattice<T>`, design rationale    |
| [`proposals/rewrite_plan_v3.md`](proposals/rewrite_plan_v3.md)  | full v3 plan with the prior-branch audit that motivated it                        |

## Worked examples

Both run end-to-end (parameter sweep → HDF5 → block-jackknife analysis → FSS
plot) with two commands from a fresh clone:

- **[`examples/01_phi4_tuning/`](examples/01_phi4_tuning/)** — 2D φ⁴ in the
  Ising universality class; HMC + susceptibility FSS; recovers γ/ν within
  ~2% of the exact Onsager value (7/4).
- **[`examples/02_on_sigma_critical/`](examples/02_on_sigma_critical/)** —
  3D O(3) Heisenberg sigma model via the Wolff cluster updater; Binder
  cumulant FSS; recovers β_c within ~0.1% of the Campostrini et al. 2002
  literature value (≈ 0.693).

Both scripts run sweeps in parallel via `xargs -P` (default = nproc); apps
themselves stay serial by design.

## Apps

| binary                                | action          | updater                       |
| ------------------------------------- | --------------- | ----------------------------- |
| `apps/phi4_hmc.cpp`                   | `act::Phi4`     | `alg::Hmc` (leapfrog)         |
| `apps/phi6_hmc.cpp`                   | `act::Phi6`     | `alg::Hmc`                    |
| `apps/sine_gordon_hmc.cpp`            | `act::SineGordon` | `alg::Hmc`                  |
| `apps/xy_wolff.cpp`                   | `act::Xy`       | `alg::Wolff` + `alg::Metropolis` |
| `apps/on_sigma_metropolis.cpp`        | `act::OnSigma<N>` | `alg::Metropolis`           |
| `apps/on_sigma_wolff.cpp`             | `act::OnSigma<N>` | `alg::Wolff`                |

Each app is ≤60 LOC and is its own configuration — there's no `reticolo_run`
dispatcher to look up. Copy one and edit.

## Layout

```
include/reticolo/      header-only core, actions, algorithms, observers, CLI
src/io/                the one TU that owns <hdf5.h> (PIMPL'd)
src/lint/              amalgamation TU for constant-time clang-tidy
apps/                  hand-written reference simulations
examples/              end-to-end studies with bash + python post-processing
tests/                 unit + physics + e2e smoke tests (100+ Catch2 cases)
proposals/             v3 plan + audit of the prior branch
```

## License

TBD.
