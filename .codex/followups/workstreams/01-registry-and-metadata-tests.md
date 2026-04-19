# Workstream 01: Registry And Metadata Tests

Priority: highest
Status: in progress

## Goal

Add regression coverage for the architecture introduced in Milestones D and E.

## Scope

- runtime metadata queries
- module/action validation
- valid module construction
- algorithm validation and creation
- error messages for unsupported combinations

## Success Criteria

- a dedicated test target exists for the built-in registry/metadata path
- failures in descriptor metadata, registration bootstrap, or validation logic are caught by CTest

## First Slice

Add one focused executable test that checks:

- available modules contain `MonteCarlo`
- `WeakFieldEuclideanGR` exposes `Metropolis` only
- `RelativisticBoseGas` exposes `Metropolis`, `HMC`, and `LLRMetropolis`
- `ModuleFactory::MakeModule()` succeeds for built-in action families
- invalid module/action and invalid updater names raise informative errors

## Progress

### 2026-03-13

Implemented:

- added `tests/registry_metadata.cpp`
- added the `registry_metadata` CTest target

Current coverage in that test:

- `runtime::metadata::available_modules()`
- `runtime::metadata::actions_for_module(...)`
- `runtime::metadata::describe_action(...)`
- `ModuleFactory::MakeModule(...)`
- `ModuleFactory::ValidateModuleAction(...)`
- `AlgorithmFactory::MakeUpdater(...)`
- `AlgorithmFactory::ValidateUpdaterName(...)`

Verification:

- `cmake --build --preset default`
  - passed
- `ctest --preset default --output-on-failure`
  - passed
  - result: `6/6` tests passed

Follow-up:

- add alias-focused assertions if/when built-in aliases are introduced
- add a direct test for descriptor/catalog consistency if the manifest model grows further
