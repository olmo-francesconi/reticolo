# Milestone C Progress Log

Date started: 2026-03-13

## Goal

Milestone C prepares and executes the HDF5 2.x migration in the storage adapter layer without changing default simulation behavior or default file compatibility.

Constraints:

- no simulation semantics changes
- no default output schema changes
- HDF5 2.x support should be added in a way that still works with current HDF5 1.14 builds
- any use of HDF5 2.x-only file features should stay opt-in

## Initial Migration Notes

Official HDF Group references checked before implementation:

- HDF5 2.0 release information
- HDF5 2.0 migration notes
- HDF5 complex datatype reference

Main findings from upstream docs:

- HDF5 2.0.0 was released on 2025-11-10
- the file format has been updated to 4.0
- a new `H5T_COMPLEX` datatype class was added in 2.0
- complex datatypes created with `H5Tcomplex_create()` are not readable by previous HDF5 versions
- creating complex datatypes also requires compatibility with `H5F_LIBVER_V200`

Migration implication:

- reticolo should become HDF5 2.x-aware now
- reticolo should not switch its default complex on-disk representation to native HDF5 complex yet
- default behavior should continue using the existing compound-layout convention for backward compatibility

## Progress

### 2026-03-13

Implemented:

- added `RETICOLO_HDF5_NATIVE_COMPLEX` as an explicit CMake option
- added HDF5 version reporting and version normalization in `CMakeLists.txt`
- export HDF5 version and native-complex policy to the code through target compile definitions in `src/CMakeLists.txt`
- added `src/reticolo/core/storage/Hdf5Compat.hpp`
- updated `src/reticolo/core/storage/Hdf5PrimitiveTypeMappings.hpp` so:
  - default mode keeps the legacy compound complex schema
  - HDF5 2.x native complex uses `H5Tcomplex_create(...)` only when explicitly enabled
- updated `src/reticolo/core/storage/Hdf5StorageBackend.hpp` so file creation uses a version-aware file access property list
  - when native complex is enabled, file creation sets `H5F_LIBVER_V200` compatibility bounds as required by HDF5 2.x complex support
- updated `README.md` with the new HDF5 build option and compatibility policy

What changed structurally:

- the storage adapter now has an explicit HDF5 compatibility layer instead of assuming one API generation
- HDF5 2.x support is real and validated, but default output compatibility is preserved
- HDF5 2.x-only schema behavior is isolated behind one build option rather than leaking into the rest of the codebase

Verification:

Default compatibility mode:

- `cmake --build --preset default`
  - passed
- `ctest --preset default`
  - passed
  - result: 5/5 tests passed

HDF5 2.x native-complex mode:

- `cmake -S . -B /tmp/reticolo-hdf5-native-complex -DRETICOLO_HDF5_NATIVE_COMPLEX=ON -DRETICOLO_BUILD_TESTS=ON -DBUILD_TESTING=ON -DRETICOLO_ENABLE_OPENMP=OFF -DRETICOLO_FETCH_DEPS=OFF`
  - passed
- `cmake --build /tmp/reticolo-hdf5-native-complex`
  - passed
- `ctest --test-dir /tmp/reticolo-hdf5-native-complex --output-on-failure`
  - passed
  - result: 5/5 tests passed

Current assessment:

- reticolo is now prepared for HDF5 2.x at the adapter level
- the default build remains backward-compatible in its complex datatype schema
- the remaining work after Milestone C is cleanup:
  - remove or simplify legacy HDF5 helper paths that are now superseded by the compatibility layer
  - decide later whether to replace the aggregate mapping include with a slimmer registry mechanism

### 2026-03-13: Storage-Boundary Cleanup

Implemented:

- added `src/reticolo/core/storage/Hdf5TypeRegistry.hpp` as the storage-owned declaration point for `make_H5_Type<>`
- updated storage mapping headers to include the storage-owned type registry instead of the legacy `core/tools/hdf5_helpers.hpp`
- reduced `src/reticolo/core/tools/hdf5_helpers.hpp` to a compatibility forwarding include
- removed legacy HDF5 handler/helper includes from:
  - `src/reticolo/core.hpp`
  - `src/reticolo/modules/montecarlo/MonteCarloHandler.hpp`
  - `tests/field_rw.cpp`
  - `tests/hdf5_basic.cpp`
  - `tests/hdf5_expandable_lattice.cpp`
- updated backend error messages to refer to `Hdf5StorageBackend` rather than the old `HDF5Handler` name

What changed structurally:

- application/runtime persistence now flows through the storage facade/backend without depending on the old HDF5 helper header path
- the only remaining `core/tools/Hdf5Handler.hpp` file is a compatibility shim rather than an active storage implementation dependency
- the HDF5 type registry now belongs to the storage layer instead of the generic tools layer

Verification:

- `cmake --build --preset default`
  - passed
- `ctest --preset default`
  - passed
  - result: 5/5 tests passed

Cleanup assessment:

- application-side data storage is now effectively handled by the storage backend/facade boundary
- remaining raw HDF5 usage is intentional and limited to:
  - the backend implementation
  - storage-owned HDF5 mapping/compatibility headers
  - low-level verification in tests

### 2026-03-13: Legacy HDF5 Compatibility Removal

Implemented:

- deleted `src/reticolo/core/tools/Hdf5Handler.hpp`
- deleted `src/reticolo/core/tools/hdf5_helpers.hpp`

Result:

- the old `HDF5Handler` / `GlobalHdf5Handler` compatibility names are gone from the codebase
- the old generic-tools HDF5 helper path is gone from the codebase
- storage now relies only on the storage-owned facade/backend/type-registry headers

Verification:

- `rg -n "Hdf5Handler.hpp|hdf5_helpers.hpp|GlobalHdf5Handler|HDF5Handler" src tests apps`
  - no matches in source/test/app code
- `cmake --build --preset default`
  - passed
- `ctest --preset default`
  - passed
  - result: 5/5 tests passed

Final Milestone C storage-boundary status:

- all application-level persistence now flows through the storage facade/backend path
- the remaining HDF5-specific code lives only where it should:
  - storage backend implementation
  - storage-owned HDF5 mapping and compatibility headers
  - test-side low-level file inspection
