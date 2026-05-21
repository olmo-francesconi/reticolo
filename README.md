# reticolo

Lightweight **C++23** library for Monte Carlo simulations of quantum field
theories on a lattice. Scalar and gauge fields, single-replica or LLR
(Logarithmic Linear Relaxation) replica ensembles.

The design priorities, in order:

1. **Small public surface.** A user writes a hand-rolled `main()` that wires a
   `Lattice`, an action struct, an updater, and an HDF5 writer — ~30–60 lines,
   no framework, no string-dispatch driver.
2. **Performance at the core.** No virtual dispatch in the hot loop; action
   methods inline into the templated updater; indexing tables shared across
   lattices of the same shape; HMC integrator is a type parameter.
3. **One way to do each thing.** One writer, one CLI helper, one logger,
   no central registries. The compiler is the source of truth.

Scope:

- **Field types**: real and complex scalars, O(N) sigma vectors, U(1) link
  angles, SU(N) matrix links.
- **Actions**: Phi4, Phi6, SineGordon, Xy, OnSigma<N>, BoseGas, CompactU1,
  Wilson<G> for U(1)/SU(2)/SU(3).
- **Updaters**: HMC (Leapfrog / Omelyan2 / Omelyan4), Metropolis,
  Wolff cluster, LLR replica ensembles with Newton-Raphson warm-up and
  Robbins-Monro adaptation.
- **Parallelism**: the trajectory inner loop is serial. OpenMP parallel-for
  is used at the *replica* layer (LLR) — N replicas thermalise / measure
  concurrently, each on one thread. Apps themselves are otherwise serial.
- **Out of scope**: MPI, GPU offload, multi-node distribution.

The architecture leaves room for these later without renegotiating the public
surface — see [`proposals/rewrite_plan_v3.md`](proposals/rewrite_plan_v3.md).

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
        (void)hmc.step();
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

External dependencies:

| dep            | how it ships                | required for                                   |
| -------------- | --------------------------- | ---------------------------------------------- |
| HDF5           | system (Homebrew / apt)     | `io::Writer` — apps that record output         |
| OpenMP         | system + compiler-bundled   | LLR replicas (`#pragma omp parallel for` over the replica vector); HMC stays single-threaded |
| Sleef          | vendored via FetchContent   | batched cos / sin / exp in the action kernels (XY, SineGordon, CompactU1, …) |
| cxxopts        | vendored via FetchContent   | `cli::Parser`                                  |
| Catch2         | vendored via FetchContent   | tests only                                     |

CMake's `find_package(OpenMP REQUIRED)` is unconditional; build will fail if
the compiler has no OpenMP runtime. macos-appleclang implicitly uses
`libomp` from Homebrew; the `macos-llvm` preset uses Homebrew LLVM's
bundled runtime.

## Reading order

| document                                                        | what it gives you                                                                 |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| [`docs/writing_an_app.md`](docs/writing_an_app.md)              | step-by-step build of the snippet above                                           |
| [`docs/action_concepts.md`](docs/action_concepts.md)            | the concept-refinement lattice (write a new action in <50 LOC)                    |
| [`docs/architecture.md`](docs/architecture.md)                  | three CMake targets, PIMPL HDF5, value-semantic `Lattice<T>`, design rationale    |
| [`docs/logging.md`](docs/logging.md)                            | the logger surface: banner, `┃` sigil, scope/replica tags, `log::off()` for tests |
| [`docs/llr.md`](docs/llr.md)                                    | Logarithmic Linear Relaxation: window penalty, NR warm-up, RM, replica exchange   |
| [`docs/writing_a_test.md`](docs/writing_a_test.md)              | the canonical patterns: force consistency, reversibility, Metropolis limits       |
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

**Scalar HMC / Metropolis / Wolff:**

| binary                          | action              | updater                          |
| ------------------------------- | ------------------- | -------------------------------- |
| `apps/phi4_hmc.cpp`             | `act::Phi4`         | `alg::Hmc`                       |
| `apps/phi6_hmc.cpp`             | `act::Phi6`         | `alg::Hmc`                       |
| `apps/sine_gordon_hmc.cpp`      | `act::SineGordon`   | `alg::Hmc`                       |
| `apps/bose_gas_hmc.cpp`         | `act::BoseGas`      | `alg::Hmc` (phase-quenched)      |
| `apps/xy_wolff.cpp`             | `act::Xy`           | `alg::Wolff` + `alg::Metropolis` |
| `apps/on_sigma_metropolis.cpp`  | `act::OnSigma<N>`   | `alg::Metropolis`                |
| `apps/on_sigma_wolff.cpp`       | `act::OnSigma<N>`   | `alg::Wolff`                     |

**Gauge HMC / Metropolis:**

| binary                          | action                    | updater          |
| ------------------------------- | ------------------------- | ---------------- |
| `apps/u1_hmc.cpp`               | `act::CompactU1`          | `alg::Hmc`       |
| `apps/u1_metropolis.cpp`        | `act::CompactU1`          | `alg::Metropolis`|
| `apps/su2_hmc.cpp`              | `act::Wilson<SU2>`        | `alg::Hmc`       |
| `apps/su3_hmc.cpp`              | `act::Wilson<SU3>`        | `alg::Hmc`       |

**LLR (Logarithmic Linear Relaxation) — replica-parallel:**

| binary                          | base action            | what it samples              |
| ------------------------------- | ---------------------- | ---------------------------- |
| `apps/phi4_llr.cpp`             | `act::Phi4`            | density of states ρ(S)       |
| `apps/u1_llr.cpp`               | `act::CompactU1`       | ρ(S) for U(1) gauge          |
| `apps/su2_llr.cpp`              | `act::Wilson<SU2>`     | ρ(S) for SU(2) gauge         |
| `apps/bose_gas_llr.cpp`         | `act::BoseGas`         | ρ(S_im) for sign-problem reconstruction |

**Tuning + benchmarks:**

| binary                                  | what                                                 |
| --------------------------------------- | ---------------------------------------------------- |
| `apps/tune_phi4.cpp`                    | algorithm tuning rig (HMC vs Metropolis, integrator choice) |
| `apps/bench_actions.cpp`                | per-action kernel throughput                         |
| `apps/bench_integrators.cpp`            | Leapfrog vs Omelyan2 vs Omelyan4 cost-per-accept    |
| `apps/bench_scalars.cpp`                | per-scalar-action full-MD bench                      |
| `apps/bench_rng.cpp`                    | FastRng vs Mt19937 vs Ranlux throughput              |
| `apps/bench_gauge_vs_scalar.cpp`        | gauge vs scalar HMC head-to-head                     |
| `apps/bench_wilson_vs_compact_u1.cpp`   | `Wilson<U1>` vs `CompactU1` at N=1                   |
| `apps/bench_phase_quenched.cpp`         | BoseGas phase-quenched HMC throughput                |

Each app is its own configuration — there's no `reticolo_run` dispatcher to
look up. Copy one and edit.

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
