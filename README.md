<div align="center">

```

 ██████╗ ███████╗████████╗██╗ ██████╗ ██████╗ ██╗      ██████╗
 ██╔══██╗██╔════╝╚══██╔══╝██║██╔════╝██╔═══██╗██║     ██╔═══██╗
 ██████╔╝█████╗     ██║   ██║██║     ██║   ██║██║     ██║   ██║
 ██╔══██╗██╔══╝     ██║   ██║██║     ██║   ██║██║     ██║   ██║
 ██║  ██║███████╗   ██║   ██║╚██████╗╚██████╔╝███████╗╚██████╔╝
 ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝ ╚═════╝

```

**Modern C++ library for Monte Carlo simulation of lattice quantum field theories.**

</div>

Scalar fields (Phi4, Phi6, SineGordon, XY, O(N) sigma, complex Bose gas) and gauge fields
(compact U(1), Wilson SU(2)/SU(3)) behind a concept-refined action interface. Ships HMC
(Leapfrog / Omelyan2 / Omelyan4), Metropolis, Wolff cluster, and LLR density-of-states
updaters; OpenMP at the LLR replica layer only. Output is self-describing HDF5 — every run
stamps cmdline, git commit, compile flags, and every CLI flag the parser resolved. Apps under
`apps/` are ~60–130 LOC each: the trajectory `for` loop stays plainly visible. See
[`docs/architecture.md`](docs/architecture.md) for the design rationale.

## Build

```sh
cmake --preset macos-appleclang   # or macos-llvm, linux-gcc, linux-clang, debug
cmake --build --preset macos-appleclang
ctest --preset macos-appleclang
```

## Sample app

Phi^4 HMC on a 4D periodic lattice — compose `field × RNG × action × updater × writer`,
then own the trajectory loop. The updater is instantiated against the action's concept at
compile time; no virtual dispatch, no string registry:

```cpp
#include <reticolo/reticolo.hpp>

int main() {
    using namespace reticolo;

    Lattice<double>   phi{{8, 8, 8, 8}};                       // 4D periodic scalar field
    FastRng           rng{42};
    act::Phi4<double> action{.kappa = 0.18, .lambda = 1.0};    // plain struct; no inheritance

    // Default integrator is Leapfrog; pass alg::integ::Omelyan2 / Omelyan4 as a template arg.
    alg::Hmc<act::Phi4<double>, FastRng> hmc{action, phi, rng, {.tau = 1.0, .n_md = 20}};

    io::Writer writer{"phi4.h5"};                              // HDF5 + run metadata
    auto       s = writer.series<double>("/prod/obs/s");

    for (int i = 0; i < 1000; ++i) {
        hmc.step();                                            // sample momenta + integrate + accept/reject
        s.append(action.s_full(phi));
    }
}
```
