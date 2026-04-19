# reticolo

Basic toolkit for generic Lattice Field Theory calculations

## Build

The project uses CMake presets for the common developer flows.

Basic workflow:

```sh
cmake --preset default
cmake --build --preset default
ctest --preset default
```

Useful CMake options:

- `RETICOLO_ENABLE_OPENMP=ON|OFF`
- `RETICOLO_BUILD_TESTS=ON|OFF`
- `RETICOLO_FETCH_DEPS=ON|OFF`
- `RETICOLO_WARNINGS_AS_ERRORS=ON|OFF`
- `RETICOLO_COMPILER_PROFILE=AUTO|APPLE_CLANG|LLVM_CLANG|GCC|MSVC`
- `RETICOLO_HDF5_NATIVE_COMPLEX=ON|OFF`

Platform notes:

- macOS: the default Xcode Apple Clang toolchain does not provide OpenMP in the way this project expects. Use `macos-appleclang` for a no-OpenMP build, or `macos-llvm` if you want the Homebrew LLVM + libomp toolchain.
  - the `macos-llvm` preset uses `cmake/toolchains/macos-homebrew-llvm.cmake` to set the compiler, Homebrew prefixes, and active macOS SDK path before configuration
- Linux: use the `linux-gcc` or `linux-clang` preset as a starting point.
- Windows: the project no longer blocks MSVC at configure time; use the `windows-msvc` preset from a developer shell with dependencies available.

Compiler policy:

- `APPLE_CLANG`: portable macOS build, no assumption that OpenMP is available
- `LLVM_CLANG`: Clang/LLVM toolchain, OpenMP allowed when found
- `GCC`: GCC toolchain, OpenMP allowed when found
- `MSVC`: MSVC toolchain, warning flags and compiler policy adjusted accordingly

HDF5 policy:

- default builds keep the legacy compound representation for complex-valued data, even under HDF5 2.x
- `RETICOLO_HDF5_NATIVE_COMPLEX=ON` enables HDF5 2.x native complex datatypes
- native complex mode is opt-in because it changes the on-disk schema and is not backward-compatible with HDF5 1.x readers

Recommended presets:

```sh
cmake --preset macos-appleclang
cmake --preset macos-llvm
cmake --preset linux-gcc
cmake --preset linux-clang
```

# Modules

## Lattices
- [] local field
- [] gauge field

## Actions
- [] virtual action_base
- [] Relativistic Bose gas (\lambda \phi^4)
- [] Weak field GR

## Algorithms 
- Markov chain Monte Carlo
- Hamiltonian Monte Carlo

- Linear Logarithmic Relaxation (LLR)
- Complex Linear Logarithmic Relaxation (Comp-LLR)
  - TO-DO: write overall LLR parameters as attributes in the controller hdf5 file
  - TO-DO: write specific LLR parameters as attributes in the worker hdf5 file
  - TO-DO: implement the replica exchange
