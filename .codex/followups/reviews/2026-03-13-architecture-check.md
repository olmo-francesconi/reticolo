# Architecture Check

Date: 2026-03-13

This review compares the implemented overhaul against the plan and milestone logs in `.codex/`.

## Validation Snapshot

Verified locally on 2026-03-13:

- `cmake --preset default`
- `cmake --build --preset default`
- `ctest --preset default --output-on-failure`

Result:

- configure passed
- build passed
- tests passed: `5/5`

## Main Findings

### 1. Registry/Metadata Architecture Has No Dedicated Tests

Severity: high

What is true now:

- the new storage tests are present
- the old lattice tests are present
- the new registry/manifest/runtime-metadata path is not covered by dedicated tests

Main affected areas:

- `src/reticolo/modules/factory/ModuleFactory.hpp`
- `src/reticolo/modules/factory/MCAlgorithmFactory.hpp`
- `src/reticolo/runtime/BuiltinMetadata.hpp`
- `src/reticolo/action/registration/ActionCatalog.hpp`

Consequence:

- the most important architectural refactor can regress silently
- the milestone claims around validation and metadata discovery are not enforced

Recommended action:

- add focused tests for:
  - valid `(module, action)` creation
  - invalid module/action diagnostics
  - metadata query results
  - algorithm availability by action family
  - invalid algorithm diagnostics

### 2. Runtime Global State Is Still Fragile

Severity: medium

Main affected area:

- `src/reticolo/runtime/ReticoloCore.hpp`

What is true now:

- singleton state still lives in a header
- static storage is still defined there
- config loading still terminates the process directly on failure

Consequence:

- weak library boundary
- harder unit testing
- awkward embedding/reuse outside the current executable path

Recommended action:

- move to `inline static` or a `.cpp` definition boundary
- prefer explicit runtime context over process-global singleton state
- let the app own process termination

### 3. Runtime Error Handling Still Exits Deep In The Stack

Severity: medium

Main affected areas:

- `src/reticolo/reticolo.hpp`
- `src/reticolo/runtime/ReticoloCore.hpp`
- parts of module setup code

Consequence:

- weak composability
- difficult failure-path tests
- architectural cleanup is only partial because the process model still leaks everywhere

Recommended action:

- convert runtime/config failures to exceptions
- terminate only in `apps/reticolo_run.cpp`

### 4. Manifest-Driven Registration Still Has An Implicit Double-Precision Bias

Severity: medium

Main affected area:

- `src/reticolo/action/registration/ActionModuleRegistrationSupport.hpp`

What is true now:

- aliases register to the double-precision module type
- the default action name also currently maps to the double-precision module type

Consequence:

- this works for the current two families
- it is not a robust general model for future action families

Recommended action:

- encode canonical precision ownership in the manifest
- bind aliases to the canonical target explicitly

### 5. Storage File-Creation Semantics Are Still Mixed

Severity: low

Main affected area:

- `src/reticolo/core/storage/Hdf5StorageBackend.hpp`

What is true now:

- `initialize_file()` truncates
- `saveLattice()` still fails if the file already exists

Consequence:

- repeated runs and mixed workflows are still easy to misuse

Recommended action:

- choose one explicit policy and document it
- ideally express that policy in storage-level API names
