# Workstream 04: Storage Semantics

Priority: medium
Status: in progress

## Goal

Make storage creation and overwrite behavior consistent and obvious.

## Main Target

- `src/reticolo/core/storage/Hdf5StorageBackend.hpp`

## Current Mismatch

- `initialize_file()` truncates existing files
- `saveLattice()` uses exclusive-create semantics

## Desired End State

- storage API names match behavior
- overwrite/fail-if-exists semantics are intentional and documented
- tests cover the chosen behavior

## Progress

### 2026-03-13

Chosen policy:

- `initialize_file()` remains whole-file truncate/create
- `save_lattice()` now means:
  - create the file if missing
  - open the file if it already exists
  - ensure parent groups for the lattice object path
  - overwrite the target lattice dataset if it already exists

Implemented:

- updated `src/reticolo/core/storage/Hdf5StorageBackend.hpp`
  - added parent-group creation for object paths
  - changed `saveLattice()` from exclusive-create semantics to create-or-open plus dataset overwrite
- updated `tests/hdf5_expandable_lattice.cpp`
  - lattice round-trip now reuses the same file as appendable datasets
  - `save_lattice()` is called twice on the same target to verify overwrite behavior

Verification:

- `ctest --test-dir build/default -R hdf5_expandable_lattice --output-on-failure`
  - passed
- `ctest --preset default --output-on-failure`
  - passed
  - result: `7/7` tests passed

Current result:

- storage semantics are now consistent with the rest of the facade
- `save_lattice()` is composable inside existing HDF5 files instead of being a special fail-if-exists path
