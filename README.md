# reticolo

**Modern C++ library for Monte Carlo simulation of lattice quantum field theories.**

Scalar actions (Phi4, Phi6, SineGordon, XY, O(N) sigma, Relativistic Bose gas) and gauge actions
(compact U(1), Wilson SU(2)/SU(3)) behind a concept-refined interface. Ships HMC
(Leapfrog / Omelyan2 / Omelyan4), Metropolis, Wolff cluster, and LLR density-of-states
updaters. Output in self-describing HDF5.

See [`docs/architecture.md`](docs/architecture.md) for the design rationale.

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

    // Default integrator is Leapfrog; pass alg::integ::omelyan2 / omelyan4 as a trailing tag.
    alg::Hmc hmc{action, phi, rng, {.tau = 1.0, .n_md = 20}};

    io::Writer writer{"phi4.h5"};                              // HDF5 + run metadata
    auto       s = writer.series<double>("/prod/obs/s");

    for (int i = 0; i < 1000; ++i) {
        hmc.step();                                            // sample momenta + integrate + accept/reject
        s.append(action.s_full(phi));
    }
}
```

## Reading the output

Every app writes a self-describing HDF5 file: reproducibility metadata under
`/run@*`, the resolved command-line flags under `/vars@*`, and one dataset per
time series. Read it with `h5py` — attributes carry the run record, datasets are
plain 1-D arrays:

```python
import h5py, numpy as np

with h5py.File("phi4.h5", "r") as f:
    print(dict(f["/run"].attrs))          # commit, compile_flags, started_utc, ...
    print(dict(f["/vars"].attrs))          # parsed CLI args

    s   = np.asarray(f["/prod/obs/s"])     # action per production trajectory
    acc = np.asarray(f["/prod/stats/accepted"])
    print(f"<S> = {s.mean():.4g} ± {s.std(ddof=1) / len(s)**0.5:.2g}")
    print(f"acceptance = {acc.mean():.3f}")
```

[`scripts/read_output.py`](scripts/read_output.py) is a runnable version that
prints the full metadata and a mean ± stderr summary of every series:

```sh
./build/macos-appleclang/apps/phi4_hmc --out phi4.h5 --n_prod 1000
python scripts/read_output.py phi4.h5
```

```
dataset                      rows           mean      std err
--------------------------------------------------------------
/prod/obs/s                  1000        797.566         2.14
/prod/stats/accepted         1000          0.842       0.0115
...
HMC acceptance: 0.842 over 1000 trajectories
```

## Performance

Single-thread kernel throughput. Numbers below are from an **Apple M1 Pro** with
the `macos-appleclang` preset (Apple clang 21, `-O3 -march=native`, OpenMP off).
Each cell is the mean over many batch samples; `p05`/`p95` are the 5th/95th
percentiles of throughput across those samples. Indicative — reproduce with:

```sh
cmake --build --preset macos-appleclang --target bench_readme
./build/macos-appleclang/apps/bench_readme
```

All three RNGs satisfy the same `Rng` concept and plug into every call site.
`uniform()` is a single draw; `gaussian` is the batched `normal_fill` path HMC
uses to sample momenta. Throughput in M draws/s:

| RNG          | draw     |   mean |  p05 |   p95 |
|--------------|----------|-------:|-----:|------:|
| FastRng      | uniform  |  809.7 | 772.9 | 841.1 |
| FastRng      | gaussian |  178.9 | 172.1 | 184.2 |
| Mt19937_64   | uniform  |  266.2 | 259.0 | 272.5 |
| Mt19937_64   | gaussian |   93.3 |  91.1 |  94.8 |
| Ranlux48     | uniform  |    3.2 |   3.2 |   3.3 |
| Ranlux48     | gaussian |    2.4 |   2.3 |   2.4 |

(`Ranlux48` is two orders of magnitude slower by design — `std::ranlux48` keeps
11 of every 389 values for decorrelation. `FastRng`, the default, is xoshiro256++.)

`compute_force` on a hot random 4D configuration, per degree of freedom (sites
for scalar, links for gauge). Phi4 is the scalar baseline; the SU(2)/SU(3)
staple force is correspondingly heavier per link. Throughput in M dof-updates/s:

| Action            | lattice |   mean |    p05 |    p95 |
|-------------------|---------|-------:|-------:|-------:|
| Phi4              | 4⁴      |  483.5 |  471.7 |  493.5 |
| Phi4              | 6⁴      |  561.9 |  545.7 |  575.8 |
| Phi4              | 8⁴      |  611.7 |  593.0 |  627.0 |
| Phi4              | 12⁴     |  972.2 |  939.1 |  994.2 |
| Phi4              | 16⁴     | 1021.2 |  993.3 | 1037.2 |
| Phi4              | 20⁴     | 1041.7 | 1001.8 | 1069.3 |
| Phi4              | 24⁴     | 1042.2 | 1001.9 | 1075.5 |
| Wilson&lt;SU2&gt; | 4⁴      |   12.0 |   11.5 |   12.4 |
| Wilson&lt;SU2&gt; | 6⁴      |   11.9 |   11.5 |   12.3 |
| Wilson&lt;SU2&gt; | 8⁴      |    9.8 |    9.3 |   10.3 |
| Wilson&lt;SU2&gt; | 12⁴     |   10.3 |    9.7 |   10.9 |
| Wilson&lt;SU2&gt; | 16⁴     |    8.1 |    7.9 |    8.3 |
| Wilson&lt;SU2&gt; | 20⁴     |    9.1 |    8.9 |    9.3 |
| Wilson&lt;SU2&gt; | 24⁴     |    8.5 |    8.4 |    8.6 |
| Wilson&lt;SU3&gt; | 4⁴      |    4.7 |    4.6 |    4.8 |
| Wilson&lt;SU3&gt; | 6⁴      |    4.7 |    4.5 |    4.9 |
| Wilson&lt;SU3&gt; | 8⁴      |    3.9 |    3.6 |    4.1 |
| Wilson&lt;SU3&gt; | 12⁴     |    3.9 |    3.8 |    4.1 |
| Wilson&lt;SU3&gt; | 16⁴     |    3.1 |    3.0 |    3.1 |
| Wilson&lt;SU3&gt; | 20⁴     |    3.8 |    3.7 |    3.9 |
| Wilson&lt;SU3&gt; | 24⁴     |    3.3 |    3.3 |    3.4 |
