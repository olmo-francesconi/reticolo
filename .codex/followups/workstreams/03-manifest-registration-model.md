# Workstream 03: Manifest Registration Model

Priority: medium
Status: in progress

## Goal

Make manifest-driven registration fully explicit about canonical precision and alias binding.

## Main Target

- `src/reticolo/action/registration/ActionModuleRegistrationSupport.hpp`

## Problem

The current helper assumes aliases should bind to the double-precision module type.

## Desired End State

- canonical action target is expressed in metadata
- aliases bind to that canonical target explicitly
- adding float-only or unusual action families does not require helper surgery

## Progress

### 2026-03-13

Implemented:

- extended `src/reticolo/action/registration/ActionDescriptor.hpp`
  - added `ActionPrecisionBinding`
  - added explicit manifest fields for:
    - `canonical_precision`
    - `alias_precision`
- updated built-in action descriptors so canonical and alias routing are explicit in metadata
- updated `src/reticolo/action/registration/ActionModuleRegistrationSupport.hpp`
  - canonical action-name registration now routes through manifest metadata
  - alias registration now routes through manifest metadata
  - canonical registration no longer depends on double precision existing
- updated `src/reticolo/runtime/BuiltinMetadata.hpp`
  - runtime metadata now exposes canonical and alias precision binding
- updated `tests/registry_metadata.cpp`
  - asserts canonical precision metadata
  - asserts canonical action names construct the declared handler type

Verification:

- `ctest --test-dir build/default -R registry_metadata --output-on-failure`
  - passed
- `ctest --preset default --output-on-failure`
  - passed
  - result: `7/7` tests passed

Current result:

- canonical/default registration is no longer an implicit double-precision convention
- alias routing is no longer hard-coded in the helper
- the model is now structurally ready for future float-canonical or float-only families
