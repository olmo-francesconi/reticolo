# Workstream 02: Runtime Context And Errors

Priority: high
Status: in progress

## Goal

Replace the remaining process-global runtime/error handling with a cleaner executable-owned boundary.

## Main Targets

- `src/reticolo/runtime/ReticoloCore.hpp`
- `src/reticolo/reticolo.hpp`
- `apps/reticolo_run.cpp`

## Desired End State

- no header-defined mutable singleton storage
- runtime setup failures throw structured exceptions
- only the app entrypoint decides whether to print and exit

## Risks

- touches the public execution path
- should be done without changing user-visible YAML behavior

## Progress

### 2026-03-13

Implemented:

- updated `src/reticolo/runtime/ReticoloCore.hpp`
  - replaced header-defined mutable static definitions with `inline static` members
  - changed configuration-load failure from `exit()` to `std::runtime_error`
- updated `src/reticolo/runtime/runtime.hpp`
  - added `RuntimeExit` for non-error process exits such as help output
  - changed `reticolo_init()` to throw instead of terminating
  - changed `reticolo_run()` to log and rethrow instead of terminating
- updated `apps/reticolo_run.cpp`
  - app entrypoint now owns exit-code handling and stderr/stdout emission
- added `tests/runtime_api.cpp`
  - verifies missing config file throws
  - verifies `reticolo_init()` without `--config` throws `RuntimeExit`

Verification:

- `cmake --build --preset default`
  - passed
- `ctest --preset default --output-on-failure`
  - passed
  - result: `7/7` tests passed

Remaining work in this stream:

- the runtime still relies on global singleton config state even though the top-level error boundary is improved
- there is still no explicit runtime-context object; setup/config is still accessed through `ReticoloCore`

### 2026-03-13: Module Setup Error Propagation

Implemented:

- updated `src/reticolo/modules/montecarlo/MonteCarloHandler.hpp`
  - workspace initialization now throws contextual `std::runtime_error`
  - logger initialization now throws contextual `std::runtime_error`
- updated `src/reticolo/modules/llr/LLRController.hpp`
  - workspace initialization now throws contextual `std::runtime_error`
  - logger initialization now throws contextual `std::runtime_error`

Verification:

- `rg -n "exit\\(EXIT_FAILURE\\)|exit\\(EXIT_SUCCESS\\)" src/reticolo/modules/montecarlo src/reticolo/modules/llr src/reticolo/runtime apps`
  - no matches
- `cmake --build --preset default`
  - passed
- `ctest --preset default --output-on-failure`
  - passed
  - result: `7/7` tests passed
