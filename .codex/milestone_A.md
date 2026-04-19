# Milestone A Progress Log

Date started: 2026-03-13

## Goal

Milestone A targets a portable build foundation for macOS, Linux, and Windows without changing simulation behavior.

Definition of done:

- project configures on mainstream toolchains without hard-coded local paths
- MSVC is no longer blocked at CMake level
- OpenMP is optional rather than mandatory
- dependency handling is more reproducible
- there are documented presets for common platforms
- obvious source-level portability blockers are removed

## Baseline Findings

Initial state before edits:

- root CMake required OpenMP unconditionally
- root CMake explicitly rejected MSVC
- dependency fetching used floating references (`HEAD`, `master`)
- `build.sh` was tied to a specific Homebrew/LLVM/HDF5 setup
- `src/reticolo/lattice/indexing.hpp` included `<machine/limits.h>`
- `tests/hdf5_expandable_lattice.cpp` included `<unistd.h>` and used `getpid()`

Local baseline checks:

- `cmake -S . -B /tmp/reticolo-milestone-a-baseline -DBUILD_TESTING=ON`
  - failed at configure time because `OpenMP_CXX` was not found with the default Apple toolchain
- `ctest --test-dir build --output-on-failure`
  - existing test binaries in `build/` failed at runtime because they were linked against an old Homebrew HDF5 install path (`libhdf5.310.dylib`)

Implication:

- the old build setup is fragile both at configure time and at runtime

## Work In Progress

Planned Milestone A edit set:

1. modernize root CMake options and dependency discovery
2. make OpenMP optional and provide a no-OpenMP compatibility layer
3. remove obvious non-portable headers and platform-specific temp-file logic
4. add `CMakePresets.json`
5. update README and simplify `build.sh`
6. reconfigure and run tests in a fresh build tree

## Edit Log

### 2026-03-13

Started Milestone A implementation.

Observed existing user changes in:

- `CMakeLists.txt`
- `build.sh`
- `tests/CMakeLists.txt`
- HDF5-related files

Approach:

- preserve those changes where compatible
- build on top of them rather than reverting

Implemented:

- added `.codex/milestone_A.md` as the running project log
- added portable CMake options:
  - `RETICOLO_BUILD_TESTS`
  - `RETICOLO_ENABLE_OPENMP`
  - `RETICOLO_FETCH_DEPS`
  - `RETICOLO_WARNINGS_AS_ERRORS`
- removed the explicit MSVC hard-stop from the root CMake file
- replaced floating `FetchContent` references with pinned versions for `yaml-cpp` and `cxxopts`
- added package-first dependency lookup with `FetchContent` fallback
- made OpenMP optional at configure time instead of required
- added a portable OpenMP compatibility shim in `src/reticolo/core/tools/omp_compat.hpp`
- updated the interface target to define `RETICOLO_USE_OPENMP` only when OpenMP is actually found
- added `reticolo::reticolo` alias target
- replaced the Homebrew-specific `build.sh` contents with `cmake --preset default`
- added `CMakePresets.json`
- updated `README.md` with preset-based build instructions
- removed `<machine/limits.h>` from the indexing layer
- replaced `tests/hdf5_expandable_lattice.cpp` temp file naming so it no longer depends on `unistd.h`
- registered more tests with CTest:
  - `lattice`
  - `lattice_uptr`
  - `field_rw`
  - `hdf5_basic`
  - `hdf5_expandable_lattice`

Pending verification:

- clean configure/build with the default local toolchain
- identify whether `std::format` remains a blocker on default Apple Clang / libc++
- verify runtime linking against the currently installed HDF5

Follow-up adjustments after first verification attempt:

- changed the `default` preset to use `build/default` so it no longer collides with the pre-existing `build/` cache
- added an offline-friendly fallback that reuses already-populated `build/_deps/*-src` trees when they exist locally

Verification results:

- `cmake --preset default`
  - passed
- `cmake --build --preset default`
  - passed
- `cmake -S . -B /tmp/reticolo-milestone-a-check -DRETICOLO_ENABLE_OPENMP=OFF -DRETICOLO_BUILD_TESTS=ON -DBUILD_TESTING=ON -DRETICOLO_FETCH_DEPS=OFF`
  - passed
- `cmake --build /tmp/reticolo-milestone-a-check`
  - passed

Initial CTest result:

- 4/5 tests passed
- `hdf5_expandable_lattice` failed because the test reused one file for:
  - appendable dataset operations after `initFile()`
  - `saveLattice()`, which still uses exclusive-create semantics

Resolution:

- adjusted the test to use a separate temporary file for the lattice round-trip path
- this keeps library behavior unchanged while making the test valid

Additional test hardening:

- `field_rw` also reused a fixed `./out.hdf5` path and failed on reruns for the same reason
- updated it to use a temporary file and clean it up after execution

Final verification:

- `ctest --preset default`
  - passed
  - result: 5/5 tests passed
- `ctest --test-dir /tmp/reticolo-milestone-a-check --output-on-failure`
  - passed
  - result: 5/5 tests passed

## Milestone A Status

Current status: substantially completed

Completed outcomes:

- CMake no longer hard-stops on MSVC
- OpenMP is optional and the project configures without it
- default development flow is preset-based instead of Homebrew-path-based
- dependency fetches are pinned instead of floating on `HEAD` / `master`
- package-first dependency discovery is in place
- local cached dependency source trees can be reused in offline environments
- obvious source portability blockers were removed
- CTest covers five executable tests and passes in both checked build trees

Residual issues worth addressing later:

- the codebase still emits a non-trivial number of compiler warnings
- `saveLattice()` still uses exclusive-create semantics
  - this is a behavior choice, not changed in Milestone A
  - if desired, it should be discussed explicitly because it affects file overwrite behavior
- dependency handling is improved, but a cleaner vendor/package strategy would still help long-term reproducibility
- no CI has been added yet, so cross-platform support is improved structurally but not yet continuously enforced

Compiler policy follow-up:

- added explicit compiler profiles:
  - `AUTO`
  - `APPLE_CLANG`
  - `LLVM_CLANG`
  - `GCC`
  - `MSVC`
- CMake now distinguishes Apple Clang from upstream LLVM Clang and prints profile-specific guidance
- Apple Clang now gets explicit messaging that OpenMP should not be expected from the default Xcode toolchain
- added dedicated presets for:
  - `macos-appleclang`
  - `macos-llvm`
  - `linux-gcc`
  - `linux-clang`
  - `windows-msvc`
- the `macos-llvm` preset targets the Homebrew LLVM + libomp toolchain directly, which matches the local workflow used on this machine
- added `cmake/toolchains/macos-homebrew-llvm.cmake`
  - sets the Homebrew LLVM compiler pair
  - adds the Homebrew LLVM/libomp prefixes
  - queries `xcrun --show-sdk-path`
  - exports `SDKROOT`
  - sets `CMAKE_OSX_SYSROOT` before dependency configuration

Compiler-profile verification:

- `cmake --preset macos-appleclang`
  - passed
- `cmake --build build/macos-appleclang --target hdf5_basic`
  - passed
- `ctest --test-dir build/macos-appleclang --output-on-failure -R hdf5_basic`
  - passed
- `cmake -S . -B /tmp/reticolo-macos-llvm-toolchain -G Ninja -DCMAKE_TOOLCHAIN_FILE=/Users/olmo/Desktop/dev/reticolo/cmake/toolchains/macos-homebrew-llvm.cmake -DRETICOLO_COMPILER_PROFILE=LLVM_CLANG -DRETICOLO_ENABLE_OPENMP=ON -DRETICOLO_BUILD_TESTS=ON -DBUILD_TESTING=ON -DRETICOLO_FETCH_DEPS=OFF`
  - passed
- `cmake --build /tmp/reticolo-macos-llvm-toolchain --target hdf5_basic`
  - passed
- `ctest --test-dir /tmp/reticolo-macos-llvm-toolchain --output-on-failure -R hdf5_basic`
  - passed

Interpretation:

- Apple Clang profile works as the safe no-OpenMP macOS baseline
- Homebrew LLVM + libomp works when configured through the dedicated toolchain file
- this matches the intended local workflow for OpenMP-enabled macOS builds
