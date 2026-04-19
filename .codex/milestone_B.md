# Milestone B Progress Log

Date started: 2026-03-13

## Goal

Milestone B isolates the storage layer behind a cleaner internal boundary so the rest of the code stops depending directly on the HDF5 backend name and layout.

Constraints:

- no simulation behavior changes
- no output schema changes
- no user-visible workflow changes unless explicitly discussed

## Baseline Findings

Current persistence entry points are concentrated in `src/reticolo/core/tools/Hdf5Handler.hpp`, but the rest of the code still depends directly on:

- `HDF5Handler`
- `GlobalHdf5Handler`
- `make_H5_Type<>`
- raw HDF5 includes in several domain/type headers

Primary call sites currently using the storage backend directly:

- `src/reticolo/modules/montecarlo/MonteCarloHandler.hpp`
- `src/reticolo/modules/llr/LLRController.hpp`
- `tests/field_rw.cpp`
- `tests/hdf5_basic.cpp`
- `tests/hdf5_expandable_lattice.cpp`

Storage operations currently exposed:

- initialize/check file
- create group
- write fixed dataset
- setup appendable dataset
- append to dataset
- save lattice
- read lattice

## Planned Refactor

1. introduce a storage facade with backend composition
2. move the current HDF5 implementation behind that facade
3. refactor runtime/module/test call sites to use the storage facade
4. keep compatibility aliases where useful during the transition
5. run storage tests and existing regression coverage

## Progress

### 2026-03-13

Implemented:

- added `src/reticolo/core/storage/StorageFacade.hpp`
- introduced `reticolo::storage::StorageManager<Backend>`
- introduced `reticolo::storage::GlobalStorage`
- renamed the concrete backend class in `Hdf5Handler.hpp` to `Hdf5StorageBackend`
- kept compatibility via:
  - `using HDF5Handler = Hdf5StorageBackend`
  - `inline HDF5Handler GlobalHdf5Handler`

Refactored main call sites to use the storage facade:

- `src/reticolo/modules/montecarlo/MonteCarloHandler.hpp`
- `src/reticolo/modules/llr/LLRController.hpp`
- `tests/field_rw.cpp`
- `tests/hdf5_basic.cpp`
- `tests/hdf5_expandable_lattice.cpp`

Interface shape after refactor:

- semantic facade methods:
  - `initialize_file`
  - `file_exists`
  - `ensure_group`
  - `write_dataset`
  - `setup_appendable_dataset`
  - `append_dataset`
  - `save_lattice`
  - `load_lattice`
- compatibility wrappers remain available on the facade and backend

Verification:

- `cmake --build --preset default`
  - passed
- `ctest --preset default`
  - passed
  - result: 5/5 tests passed

Current Milestone B status:

- the runtime and tests are now decoupled from the HDF5 backend name
- the concrete backend remains HDF5-backed
- datatype registration is still spread across domain/type headers
- raw HDF5 headers are still included in domain/type headers

That means Milestone B is started and the backend boundary exists, but the type-mapping layer is not yet fully separated from the physics/domain code.

### 2026-03-13: Type Mapping Separation

Implemented:

- added storage-owned mapping headers:
  - `src/reticolo/core/storage/Hdf5PrimitiveTypeMappings.hpp`
  - `src/reticolo/core/storage/Hdf5MonteCarloTypeMappings.hpp`
  - `src/reticolo/action/storage/RelativisticBoseGasHdf5.hpp`
  - `src/reticolo/action/storage/WeakFieldEuclideanGRHdf5.hpp`

Moved HDF5 schema/type code out of:

- `src/reticolo/core/types/real.hpp`
- `src/reticolo/core/types/complex.hpp`
- `src/reticolo/core/types/hfield.hpp`
- `src/reticolo/modules/montecarlo/MonteCarloData.hpp`
- `src/reticolo/action/RelativisticBoseGas.hpp`
- `src/reticolo/action/WeakFieldEuclideanGR.hpp`

What changed structurally:

- core/domain types keep their data layout and math behavior
- storage-specific `make_H5_Type<>` specializations now live in storage-owned headers
- `Hdf5Handler.hpp` depends on primitive/core storage mappings
- `MonteCarloHandler.hpp` pulls in storage mappings for Monte Carlo data and action observables

Verification:

- `cmake --build --preset default`
  - passed
- `ctest --preset default`
  - passed
  - result: 5/5 tests passed

Updated assessment:

- backend boundary now exists
- runtime code now talks to a storage facade
- HDF5 datatype/schema code is no longer owned by the core physics/type headers
- remaining HDF5 coupling is now mostly concentrated in:
  - storage backend implementation
  - storage mapping headers

This is the first point where the storage layer is structurally distinct enough to support a future second backend or a cleaner HDF5 2.x migration path.

### 2026-03-13: Backend Formalization

Implemented:

- added `src/reticolo/core/storage/StorageBackendConcept.hpp`
- added `src/reticolo/core/storage/Hdf5StorageBackend.hpp`
- constrained `StorageManager` with the explicit `StorageBackend` concept
- moved the concrete HDF5 backend implementation out of `core/tools` and into `core/storage`
- reduced `src/reticolo/core/tools/Hdf5Handler.hpp` to a compatibility wrapper

What this means structurally:

- storage now has an explicit backend contract
- HDF5 is now a backend plugged into the storage layer rather than the storage layer itself
- legacy includes can still include `Hdf5Handler.hpp` without breaking

Expected benefit:

- a second backend can now be introduced into `core/storage` without threading HDF5 naming through the rest of the codebase

Verification:

- `cmake --build --preset default`
  - passed
- `ctest --preset default`
  - passed
  - result: 5/5 tests passed

Follow-up adjustment during verification:

- fixed the `StorageBackend` concept to use the required `backend.template setupExpandableDataset<double>(...)` form for a dependent template expression

Updated assessment:

- the storage layer now has a concrete backend contract rather than an implicit HDF5-shaped API
- the default backend is still HDF5, but it now sits behind an explicit `StorageManager<Backend>` boundary
- the old `core/tools/Hdf5Handler.hpp` include path remains as a compatibility shim, so the architectural move did not require a broad include migration

### 2026-03-13: Backend-Neutral Schema Surface

Implemented:

- added `src/reticolo/core/storage/StorageSchema.hpp`
- introduced storage-owned schema descriptors:
  - `schema::ObjectPath`
  - `schema::AppendableDatasetSpec`
- introduced schema helpers for current persisted objects:
  - `schema::lattice::field(...)`
  - `schema::montecarlo::run_group(...)`
  - `schema::montecarlo::observables_dataset(...)`
  - `schema::montecarlo::monte_carlo_dataset(...)`
  - `schema::montecarlo::observables_stream(...)`
  - `schema::montecarlo::monte_carlo_stream(...)`
  - `schema::llr::run_group(...)`
  - `schema::llr::ak_history_dataset(...)`

Facade changes:

- added `StorageManager` overloads that accept schema descriptors for:
  - groups
  - fixed datasets
  - appendable datasets
  - lattice save/load

Refactored call sites:

- `src/reticolo/modules/montecarlo/MonteCarloHandler.hpp`
- `src/reticolo/modules/llr/LLRController.hpp`
- `tests/field_rw.cpp`
- `tests/hdf5_basic.cpp`
- `tests/hdf5_expandable_lattice.cpp`

What changed structurally:

- runtime code no longer needs to hand-assemble backend object paths for the main Monte Carlo and LLR persistence flows
- the current file layout is now defined in a storage-owned schema header rather than being scattered through module logic
- the facade API can still accept raw strings, so this is a non-breaking transition step

Verification:

- `cmake --build --preset default`
  - passed
- `ctest --preset default`
  - passed
  - result: 5/5 tests passed

Updated assessment:

- backend contract and backend-neutral schema are now both explicit
- the remaining HDF5-specific coupling is mostly down to:
  - HDF5 datatype mapping headers
  - HDF5 attribute/dataset implementation details inside `Hdf5StorageBackend`
- this is a better base for the HDF5 2.x migration because the runtime now expresses storage intent separately from HDF5 path construction

### 2026-03-13: HDF5 Registry And Metadata Consolidation

Implemented:

- added `src/reticolo/core/storage/Hdf5TypeMappings.hpp` as the storage-owned aggregate include for all currently supported HDF5 type mappings
- updated `src/reticolo/core/storage/Hdf5StorageBackend.hpp` to depend on that aggregate registry instead of only primitive mappings
- removed direct HDF5 mapping includes from `src/reticolo/modules/montecarlo/MonteCarloHandler.hpp`
- moved lattice metadata names into the storage schema layer:
  - `schema::lattice::field_sizes_attribute`
  - `schema::lattice::rng_state_attribute`
- updated `Hdf5StorageBackend` to use those schema constants for attribute read/write

What changed structurally:

- runtime/module code no longer needs to know which HDF5 mapping headers must be visible for persistence to compile
- the HDF5 backend is now self-contained with respect to:
  - supported type registration visibility
  - lattice metadata attribute naming policy
- the remaining HDF5-specific behavior is concentrated in one place:
  - `src/reticolo/core/storage/Hdf5StorageBackend.hpp`
  - the storage-owned HDF5 mapping headers it aggregates

Verification:

- `cmake --build --preset default`
  - passed
- `ctest --preset default`
  - passed
  - result: 5/5 tests passed

Updated assessment:

- Milestone B is now close to the point where HDF5 2.x work can proceed against a clearly bounded adapter layer
- remaining pre-migration work is mostly cleanup rather than architecture:
  - decide whether to keep the aggregate mapping include model or replace it later with a slimmer registration mechanism
  - define the actual HDF5 2.x compatibility strategy inside the backend implementation layer
