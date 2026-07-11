<p align="center">
  <img src="docs/banner.png" alt="reticolo — Monte Carlo for lattice quantum field theory" width="100%">
</p>

<p align="center">
  <a href="https://github.com/olmo-francesconi/reticolo/actions/workflows/ci.yml"><img src="https://github.com/olmo-francesconi/reticolo/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/C%2B%2B-20-00599c" alt="C++20">
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux-555" alt="platforms">
  <img src="https://img.shields.io/badge/GPU-CUDA-76b900?logo=nvidia&logoColor=white" alt="CUDA">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT"></a>
</p>

# reticolo

**reticolo** is a C++20 library for Monte Carlo simulation of lattice quantum
field theories. It provides scalar and gauge field actions on periodic
hypercubic lattices, samples them with Hybrid Monte Carlo, and reconstructs the
density of states with the LLR algorithm. An optional CUDA backend runs the same
simulations on the GPU. Configurations and observables are written to
self-describing HDF5.

## Features

- **Field theories** — scalar φ⁴, φ⁶ and sine-Gordon; the O(2)/XY model; a
  charged scalar with a sign problem (Bose gas); and Wilson gauge theory for
  U(1), SU(2) and SU(3).
- **Updates** — Hybrid Monte Carlo with Leapfrog, second-order Omelyan and
  fourth-order Omelyan molecular-dynamics integrators.
- **Density of states** — LLR reconstruction (Newton–Raphson and Robbins–Monro
  adaptation with replica exchange) over parallel windowed HMC chains.
- **GPU** — optional CUDA backend covering HMC for every action in single and
  double precision, sharing one per-site formula with the CPU code.
- **Random number generators** — xoshiro256++, Philox4×32, Mersenne Twister and
  RANLUX behind a common interface.
- **Output** — self-describing HDF5 carrying the full run record (command line,
  git commit, compile flags) with checkpoint/resume from any trajectory.

## Requirements

- A C++20 compiler (GCC, Clang or Apple Clang)
- CMake ≥ 3.25 and Ninja
- HDF5
- OpenMP — optional, for CPU parallelism (site, trajectory and replica levels)
- CUDA — optional, for the GPU backend

## Building

```sh
cmake --preset macos-appleclang   # or macos-llvm, linux-gcc, linux-clang, debug, linux-nvcc
cmake --build --preset macos-appleclang
ctest --preset macos-appleclang
```

## Usage

A simulation composes a field, an RNG, an action, an updater and a writer, then
runs the trajectory loop:

```cpp
#include <reticolo/reticolo.hpp>

int main() {
    using namespace reticolo;

    Lattice<double>   phi{{8, 8, 8, 8}};
    FastRng           rng{42};
    act::Phi4<double> action{.kappa = 0.18, .lambda = 1.0};
    alg::Hmc          hmc{action, phi, rng, {.tau = 1.0, .n_md = 20}};

    io::Writer writer{"phi4.h5"};
    auto       s = writer.series<double>("/prod/obs/s");

    for (int i = 0; i < 1000; ++i) {
        hmc.step();
        s.append(action.s_full(phi));
    }
}
```

The `apps/` directory contains a reference HMC and LLR program for each action.
Output is read back with any HDF5 tool; run metadata lives in the `/run` and
`/vars` attributes and each observable is a 1-D dataset:

```python
import h5py, numpy as np

with h5py.File("phi4.h5", "r") as f:
    s = np.asarray(f["/prod/obs/s"])
    print(f["/run"].attrs["commit"], s.mean())
```

## Parallelism

On the CPU, OpenMP threads three composable layers, all off by default (serial
without `-fopenmp`):

- **Site level** — the lattice-traverse passes (force, action sweep) inside one
  trajectory.
- **Trajectory level** — each HMC trajectory runs its drift/kick/force passes on
  a fixed thread team.
- **Replica level** — LLR runs its windowed HMC chains in parallel over the
  replica ensemble.

For a single-replica HMC app the ambient team is used — just set the
environment:

```sh
OMP_NUM_THREADS=8 ./build/linux-gcc/apps/phi4_hmc --L 32 --ndim 4
```

To pin a trajectory to a fixed team regardless of the global (so replica- and
site-level threads don't fight), set it on the spec — resolved once at
construction and frozen. Chains stay bit-identical for fixed
`n_threads` + `slabs_per_thread`:

```cpp
alg::Hmc hmc{action, phi, FastRng{42},
             {.tau = 1.0, .n_md = 20, .n_threads = 4}};
```

LLR apps saturate replicas first, then spill leftover threads into per-replica
site teams automatically — `OMP_NUM_THREADS` is the only knob most runs need.

## GPU

With `RETICOLO_ENABLE_CUDA`, the same actions run on the GPU. A CUDA app is the
umbrella header plus the CUDA header; it uses `cuda::DeviceField` and `cuda::Hmc`
directly, and the trajectory loop mirrors the CPU version one-for-one:

```cpp
#include <reticolo/reticolo.hpp>
#include <reticolo/cuda/cuda.hpp>

int main() {
    using namespace reticolo;
    using Field  = cuda::DeviceField<double>;
    using Action = cuda::DeviceAction<act::Phi4<double>, Field>;

    Field             phi{{8, 8, 8, 8}};
    act::Phi4<double> phi4{.kappa = 0.18, .lambda = 1.0};
    Action            meas{phi4, phi.topology()};                  // measures S
    cuda::Hmc         hmc{Action{phi4, phi.topology()}, phi, 1.0, 20, 42};

    io::Writer writer{"phi4_cuda.h5"};
    auto       s = writer.series<double>("/prod/obs/s");

    for (int i = 0; i < 1000; ++i) {
        hmc.step();
        s.append(meas.s_full(phi));
    }
}
```

The action is wrapped in a `cuda::DeviceAction` (move-only, so the measurement
keeps its own instance while `hmc` consumes the other). CPU and GPU share one
per-site formula, so the two backends sample the same ensemble. Build and run
with the CUDA preset:

```sh
cmake --preset linux-nvcc && cmake --build --preset linux-nvcc
```

`apps/cuda/` has HMC and LLR reference apps for φ⁴, Bose gas and U(1)/SU(2)/SU(3).
With no local GPU, the `tools/modal/` runner builds and runs on a cloud GPU —
see [`docs/cuda_architecture.md`](docs/cuda_architecture.md).

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — library design and data model
- [`docs/writing_an_app.md`](docs/writing_an_app.md) — adding an action, app or test
- [`docs/cuda_architecture.md`](docs/cuda_architecture.md) — the GPU backend

## License

Released under the MIT License — see [LICENSE](LICENSE).
