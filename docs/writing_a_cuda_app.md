# Writing a CUDA app

The GPU twin of [`writing_an_app.md`](writing_an_app.md). A CUDA app is a plain
`.cu` that looks like its CPU sibling — same CLI, same HDF5 schema, the
trajectory `for` plainly in `main()` — but runs the trajectories on the device.
The completed `apps/cuda/phi4_cuda_hmc.cu` is ~95 LOC.

Read the [CUDA backend section of architecture.md](architecture.md#cuda-backend)
first; this file is the practical walk-through.

## Where it lives

GPU apps live in [`apps/cuda/`](../apps/cuda/) (the rest of `apps/` is CPU-only),
and register through one CMake line in [`apps/CMakeLists.txt`](../apps/CMakeLists.txt),
inside the `if(RETICOLO_ENABLE_CUDA)` block:

```cmake
reticolo_add_cuda_app(my_cuda_app cuda/my_cuda_app.cu)
```

`reticolo_add_cuda_app` links `reticolo::reticolo` (the full umbrella, including
`reticolo::cuda` and `reticolo::io`), applies the `$<COMPILE_LANGUAGE:CXX>`-guarded
warnings (inert on the `.cu` compile), and sets `CUDA_STANDARD 20`.

## Step 1 — the two umbrellas

```cpp
#include <reticolo/reticolo.hpp>     // core + io + cli (host)
#include <reticolo/cuda/cuda.hpp>    // the device stack (nvcc-only)

#include <cuda_runtime.h>
```

`<reticolo/cuda/cuda.hpp>` re-exports the public device API: `DeviceField`,
`DeviceAction`, `cuda::Hmc`, the on-device reductions, and the device-functor
adapters for every supported action + gauge group. The device-vs-CPU validation
gates are separate `tests/cuda/test_cuda_<name>.cu` files, not part of this
umbrella. `io::Writer` PIMPLs HDF5, so nvcc never sees `<hdf5.h>` — the app just
links the prebuilt `reticolo::io` archive.

## Step 2 — pick the device types

```cpp
using namespace reticolo;
using DField = cuda::DeviceField<double>;                       // scalar field
using DAct   = cuda::DeviceAction<act::Phi4<double>, DField>;   // generic device action
```

For a gauge action the field carries a layout policy:

```cpp
using Group  = gauge_group::SU2;
using DField = cuda::DeviceField<double, cuda::MatrixLayout<Group>>;
using DAct   = cuda::DeviceAction<action::Wilson<Group, double>, DField>;
```

CLI flags are declared exactly as in a CPU app (`cli::Parser`, `app::common_flags`).

## Step 3 — stage the field, build the HMC

Cold-start or `--resume` on the host `Lattice`/`MatrixLinkLattice`, copy to the
device, then construct the generic device HMC with a reused integrator tag:

```cpp
DField field{shape};
field.copy_from_host(host.data());
RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

DAct meas{phi4, field.topology()};                      // measurement action (own scratch)
cuda::Hmc<DAct, alg::integ::Leapfrog, DField> hmc{
    DAct{phi4, field.topology()}, field, tau, n_md, seed};
```

`alg::integ::Leapfrog` is the *same* type the CPU `alg::Hmc` uses — swap it for
`alg::integ::Omelyan2` / `Omelyan4` to change integrator.

## Step 4 — own the for loop (host-free blocks)

Run trajectories in blocks of `meas_every` so the whole block replays as a CUDA
graph with no host round-trips; only the scalar observables cross PCIe, reduced
on-device:

```cpp
for (long long i = start_i; i < n_prod; i += meas_every) {
    int const k = static_cast<int>(std::min<long long>(meas_every, n_prod - i));
    hmc.run(k);
    hmc.sync();
    s_prod.append(meas.s_full(field));
    auto const n      = static_cast<long>(field.size());
    double const mean = cuda::reduce_sum_f64(field.data(), n) / v;
    mag.append(std::abs(mean));
    m_sq.append(cuda::reduce_sumsq_f64(field.data(), n) / v);
}
```

## Step 5 — checkpoint (optional)

The device RNG is counter-based Philox, so `(seed, trajectory counter) + field`
fully determines the continuation — a resumed run is **bit-for-bit** identical to
an uninterrupted one. Copy the field out and write all three:

```cpp
field.copy_to_host(host.data());
RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());
io::save_config_counter(cfg_path(outpath, done), host,
                        hmc.seed(), hmc.rng_counter(), done, argc, argv, &p);
```

Resume reads them back and calls `hmc.set_rng_counter(counter)` before running.

## Step 6 — build, test, run

There is no local NVIDIA GPU in the dev setup; the backend is built and tested on
a GPU host (Modal or the Kaggle runner) through the shared `linux-nvcc` CMake
preset — the same preset workflow every other system uses:

```sh
cmake --preset linux-nvcc
cmake --build --preset linux-nvcc
ctest --preset linux-nvcc
```

The preset bakes in `RETICOLO_ENABLE_CUDA=ON`, OpenMP off, examples off, and
`CMAKE_CUDA_ARCHITECTURES=native` (it does not pin the host compiler, so `CC`/`CXX`
still apply). Add a smoke test by mirroring
[`tests/apps/test_phi4_cuda_hmc_smoke.cpp`](../tests/apps/test_phi4_cuda_hmc_smoke.cpp)
(execs the binary, checks the HDF5 schema) — register it under the
`RETICOLO_BUILD_IO AND RETICOLO_BUILD_APPS AND RETICOLO_ENABLE_CUDA` block in
`tests/CMakeLists.txt`.

## Adding a new action to the GPU backend

You do **not** rewrite the physics. The per-site formula already lives in
`action/detail/site/<name>_formula.hpp` (or `action/detail/gauge/<name>_formula.hpp`),
annotated `RETICOLO_HD`. To run it on the GPU:

1. Add `include/reticolo/cuda/actions/<name>.hpp`: device functors whose
   `finalize()` calls the *same* `action::detail::<name>_*_site<T>(...)` formula,
   plus a `device_functors<action::<Name><T>>` trait wiring them to the site/
   plaquette launchers (copy `actions/phi4.hpp` for a scalar, `actions/wilson.hpp`
   for a gauge action).
2. Re-export it from `include/reticolo/cuda/cuda.hpp`.
3. Add a native `.cu` Catch2 test `tests/cuda/test_cuda_<name>.cu` (copy an
   existing one) and register it with `reticolo_add_cuda_test(...)` in
   `tests/CMakeLists.txt`. The canonical gates are device-ops-vs-CPU,
   `s_full`+force vs the CPU action to ~1e-9, MD energy conservation, and
   reversibility.

The compiler enforces the seam: a missing trait member is a `DeviceAction`
instantiation error pointing straight at it.

## Gotchas (each cost a Kaggle round-trip once)

- **No C++23 features in headers the umbrella pulls** — the library targets C++20,
  and nvcc device compilation also caps at `-std=c++20`. (`std::views::zip` → indexed loop.)
- **Scope `using namespace reticolo;` inside `main()`/a function**, never at
  namespace scope — `reticolo::log` collides with CUDA's `::log` math function in
  device headers ("reference to 'log' is ambiguous").
- **`act::` is an umbrella-only alias.** In a TU that does not include
  `<reticolo/reticolo.hpp>` (e.g. a `tests/cuda/*.cu` device test), spell it
  `action::`.
