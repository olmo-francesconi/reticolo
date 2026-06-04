# Examples

Each subdirectory is a **standalone consumer project** — it carries its own
`CMakeLists.txt` that links `reticolo::reticolo` and can be copied out of
this repo and built independently.

## Find-or-fetch

Every example's `CMakeLists.txt` resolves `reticolo::reticolo` in three stages:

1. Reuse the target if already configured (in-tree aggregate build via a preset).
2. `add_subdirectory("../../")` if the sibling checkout is present.
3. `FetchContent_Declare` a pinned git tag otherwise.

This means you can copy an example directory anywhere and it will build against
a local checkout or fetch the library automatically.

## Building an example

**Standalone (the main use-case):**

```sh
cd examples/01_phi4_tuning
cmake -S . -B build && cmake --build build
```

**In-tree (CI compile-check):**

```sh
cmake --preset macos-appleclang   # RETICOLO_BUILD_EXAMPLES=ON is set by all presets
cmake --build --preset macos-appleclang
# binaries land under build/<preset>/examples/NN/
```

**Via run.sh:** every example's sweep script calls `build_example` (defined in
`_common/preset.sh`), which configures+builds the example and sets `$example_bin`
to the binary directory. Just run:

```sh
bash examples/01_phi4_tuning/run.sh
```

Override the preset with `RETICOLO_PRESET=macos-llvm` or `--preset macos-llvm`.

## Index

| # | directory | description |
|---|-----------|-------------|
| 01 | `01_phi4_tuning` | φ⁴ finite-size scaling in the 2D Ising universality class |
| 02 | `02_on_sigma_critical` | 3D O(3) Heisenberg critical point via Wolff cluster |
| 03 | `03_phi4_algorithm_tuning` | Algorithm tuning for φ⁴: Metropolis vs HMC Leapfrog/Omelyan2/Omelyan4 |
| 04 | `04_phi4_llr` | φ⁴ density of states: HMC histogram vs LLR reconstruction |
| 05 | `05_u1_llr` | Compact U(1) gauge density of states with LLR + replica exchange |
| 06 | `06_bose_gas_llr` | Relativistic Bose gas density of states with LLR |
| 07 | `07_su2_llr` | SU(2) gauge density of states with LLR + replica exchange |
| 08 | `08_volume_scaling` | Kernel throughput vs volume for scalar and gauge actions |
| 09 | `09_u1_llr_smoothed` | Smoothed LLR vs vanilla LLR at the 4D U(1) bulk transition |
| 10 | `10_smoothed_synthetic_benchmark` | Python-only synthetic benchmark of smoothed-LLR convergence (no binary) |
