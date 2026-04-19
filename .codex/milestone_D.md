# Milestone D Progress Log

Date started: 2026-03-13

## Goal

Milestone D replaces the current hard-coded module/action/algorithm factory switchboards with explicit registration-based registries.

Constraints:

- no YAML interface changes
- no simulation behavior changes
- no action/module availability changes
- built-in registrations should remain explicit and easy to audit

## Baseline Findings

Current hard-coded dispatch points:

- `src/reticolo/modules/factory/ModuleFactory.hpp`
- `src/reticolo/modules/factory/MCAlgorithmFactory.hpp`

Current issues:

- every new action requires edits in central factory code
- precision variants are manually enumerated in switches/maps
- runtime looks generic, but extensibility is still switchboard-driven
- Monte Carlo algorithm selection is encoded as `if constexpr` plus hard-coded string checks

## Refactor Target

Replace the hard-coded dispatch with:

- a module registry keyed by `(module_name, action_name)`
- an algorithm registry keyed by algorithm name and specialized by action type
- a bootstrap layer that registers the current built-in combinations once

This preserves the current user-facing YAML while making future additions local and explicit.

## Progress

### 2026-03-13

Implemented:

- added `src/reticolo/modules/factory/ModuleRegistry.hpp`
- added `src/reticolo/modules/factory/MCAlgorithmRegistry.hpp`
- replaced the old enum/switch implementation in `src/reticolo/modules/factory/ModuleFactory.hpp`
- replaced the old hard-coded `if constexpr` dispatch in `src/reticolo/modules/factory/MCAlgorithmFactory.hpp`

Current built-in registration model:

- `ModuleFactory` bootstraps the current built-in `(module, action)` combinations once into `ModuleRegistry`
- `MCAlgorithmFactory` bootstraps the current built-in algorithm names once per action type into `MCAlgorithmRegistry<Action, TGen>`

Built-in registrations preserved:

- module `MonteCarlo`
  - action `RelativisticBoseGas`
  - action `RelativisticBoseGas_F`
  - action `RelativisticBoseGas_D`
  - action `WeakFieldEuclideanGR`
  - action `WeakFieldEuclideanGR_F`
  - action `WeakFieldEuclideanGR_D`
- algorithms per action capability
  - `Metropolis`
  - `HMC`
  - `LLRMetropolis` (still mapped to the current `HMC<Action>` implementation, preserving existing behavior)

What changed structurally:

- factory dispatch is now registration-based rather than enum/switch-based
- adding a new built-in combination now means registering a creator instead of editing nested switches
- the YAML interface remains unchanged

Verification:

- `cmake --build --preset default`
  - passed
- `ctest --preset default`
  - passed
  - result: 5/5 tests passed

### 2026-03-13: Unified Descriptor Metadata Surface

Implemented:

- added `src/reticolo/action/registration/ActionCatalog.hpp`
- introduced `ActionFamilyInfo` as a unified metadata record containing:
  - module name
  - default/float/double action names
  - aliases
  - exposed algorithms
  - capability flags
- added catalog helpers:
  - `builtin_action_catalog()`
  - `available_modules()`
  - `actions_for_module(...)`
  - `find_action_family(...)`

Current usage:

- `ModuleFactory` validation now uses the unified catalog to resolve valid actions and identify known action families
- the descriptor/capability layer remains the source for algorithm exposure

What changed structurally:

- there is now a single metadata surface for built-in module/action/algorithm discovery
- factory validation no longer needs to infer everything from registry state alone
- this catalog is intentionally lightweight so it can be reused later during the planned CLI/runtime rewrite

Verification:

- `cmake --build --preset default`
  - passed
- `ctest --preset default`
  - passed
  - result: 5/5 tests passed

Updated assessment:

- Milestone D now has:
  - registration-based factories
  - decentralized built-in registration
  - action descriptors for naming and capabilities
  - a unified metadata catalog for discovery and validation
- this is likely enough modularity infrastructure to stop before the later CLI/runtime rewrite, unless you want one more pass to remove duplicated capability flags from the action classes themselves

Assessment:

- the project now has a real registry-based factory foundation
- current registration is still bootstrapped centrally, but the switchboard architecture is gone
- this is a strong base for later decentralizing registration if desired

### 2026-03-13: Registration Decentralization

Implemented:

- added action-owned module registration headers:
  - `src/reticolo/action/registration/RelativisticBoseGasModuleRegistration.hpp`
  - `src/reticolo/action/registration/WeakFieldEuclideanGRModuleRegistration.hpp`
- added `src/reticolo/modules/factory/BuiltinModuleRegistration.hpp` as a thin bootstrap aggregator
- moved built-in module registration logic out of `ModuleFactory.hpp`
- added algorithm-owned registration helpers in:
  - `src/reticolo/modules/montecarlo/algorithms/Metropolis.hpp`
  - `src/reticolo/modules/montecarlo/algorithms/HMC.hpp`
- added `src/reticolo/modules/factory/BuiltinAlgorithmRegistration.hpp` as a thin bootstrap aggregator
- moved built-in algorithm registration logic out of `MCAlgorithmFactory.hpp`

Additional cleanup:

- fixed the HDF5 version compile-definition export in `src/CMakeLists.txt` to use the normalized `RETICOLO_HDF5_VERSION_*` variables

What changed structurally:

- central factory headers now depend on registration bootstrap headers rather than owning the concrete built-in registrations
- built-in module registration now lives next to action/module composition concerns
- built-in algorithm registration now lives next to algorithm definitions
- future built-in additions no longer require editing the main factory headers

Verification:

- `cmake --build --preset default`
  - passed
- `ctest --preset default`
  - passed
  - result: 5/5 tests passed

Updated assessment:

- the registry-based architecture is now meaningfully decentralized
- there are still thin built-in bootstrap aggregators, but the factories themselves are no longer the place where built-in combinations are encoded

### 2026-03-13: Action Descriptor Layer

Implemented:

- added `src/reticolo/action/registration/ActionDescriptor.hpp`
- introduced action-family descriptors for:
  - `RelativisticBoseGas`
  - `WeakFieldEuclideanGR`
- introduced `register_monte_carlo_action_family<Descriptor, ActionTemplate>()`

Descriptor responsibilities:

- canonical/default action name
- float-precision action name
- double-precision action name
- additional aliases array

Refactored registrations:

- `src/reticolo/action/registration/RelativisticBoseGasModuleRegistration.hpp`
- `src/reticolo/action/registration/WeakFieldEuclideanGRModuleRegistration.hpp`

What changed structurally:

- action naming metadata is now defined once per action family instead of duplicated through manual registration calls
- module registration for templated action families is now generated from descriptor metadata rather than handwritten per precision variant
- this is the first real step toward “one place to describe an action family”

Verification:

- `cmake --build --preset default`
  - passed
- `ctest --preset default`
  - passed
  - result: 5/5 tests passed

Updated assessment:

- adding a new action family now has a clearer pattern:
  - define the action
  - define its descriptor metadata
  - add a thin registration entrypoint
- the next refinement would be extending descriptors beyond names into capabilities/metadata that can also drive algorithm registration and runtime reporting

### 2026-03-13: Capability Metadata In Action Descriptors

Implemented:

- extended `src/reticolo/action/registration/ActionDescriptor.hpp` so action-family descriptors now define:
  - naming metadata
  - capability flags:
    - `supports_metropolis`
    - `supports_hmc`
    - `supports_llr`
  - algorithm exposure list via `algorithms`
- added descriptor-to-concrete-action mapping through `ActionDescriptorFor<Action>` / `action_descriptor_t<Action>`
- added `available_algorithms<Action>()` as an initial descriptor-driven introspection helper
- updated `src/reticolo/modules/factory/BuiltinAlgorithmRegistration.hpp` so algorithm bootstrap now derives from action descriptor capabilities
- split module-registration support into `src/reticolo/action/registration/ActionModuleRegistrationSupport.hpp` to avoid include cycles between descriptors and algorithm factories

What changed structurally:

- action descriptors are now the source of truth for both:
  - how an action family is named
  - which algorithm families it exposes
- algorithm bootstrap no longer has to carry its own separate knowledge of action capabilities
- the descriptor layer is now positioned to support later runtime introspection or config validation

Verification:

- `cmake --build --preset default`
  - passed
- `ctest --preset default`
  - passed
  - result: 5/5 tests passed

Updated assessment:

- Milestone D now has a meaningful action descriptor layer rather than just a naming shim
- the next step, if desired, would be surfacing descriptor metadata into runtime validation/help text or using it to reduce duplicated capability flags inside the action classes themselves

### 2026-03-13: Descriptor As Sole Capability Source

Implemented:

- removed the old per-action capability flags from:
  - `src/reticolo/action/RelativisticBoseGas.hpp`
  - `src/reticolo/action/WeakFieldEuclideanGR.hpp`
- updated the legacy LLR worker concepts to read LLR capability from action descriptors instead of action-class booleans:
  - `src/reticolo/modules/llr/LLRHMCWorker.hpp`
  - `src/reticolo/modules/llr/LLRMetWorker.hpp`
  - `src/reticolo/modules/llr/LLRHMCMetWorker.hpp`
- kept descriptor capability checks in the registry/bootstrap layer, not in the concrete algorithm headers:
  - `src/reticolo/modules/factory/BuiltinAlgorithmRegistration.hpp` now decides which algorithms to expose from descriptor metadata
  - `src/reticolo/modules/montecarlo/algorithms/Metropolis.hpp` now only provides unconditional registration of the `Metropolis` implementation
  - `src/reticolo/modules/montecarlo/algorithms/HMC.hpp` now only provides unconditional registration helpers for `HMC` and `LLRMetropolis`

What changed structurally:

- action descriptors are now the only source of capability truth in the current codebase
- action classes no longer carry duplicated `IsMetropolisCapable` / `IsHmcCapable` / `IsLLRCapable` metadata
- the concrete algorithm headers no longer need direct knowledge of descriptor types, which avoids brittle include/namespace interactions in older action specialization paths
- capability decisions now happen at the correct architectural level:
  - descriptors define capabilities
  - bootstrap/validation reads capabilities
  - algorithm implementations only provide implementations

Verification:

- `cmake --build --preset default`
  - passed
- `ctest --preset default`
  - passed
  - result: 5/5 tests passed

Updated assessment:

- Milestone D now has a clean single-source capability model
- this is a good stopping point before the later CLI/runtime rewrite, because the remaining runtime path can now query descriptors rather than action internals

### 2026-03-13: Descriptor-Driven Validation

Implemented:

- added module-registry introspection in `src/reticolo/modules/factory/ModuleRegistry.hpp`
  - `actions_for_module(...)`
- added module-factory validation helpers in `src/reticolo/modules/factory/ModuleFactory.hpp`
  - `EnsureBuiltinsRegistered()`
  - `AvailableActions(...)`
  - `ValidateModuleAction(...)`
- added algorithm-name introspection helper in `src/reticolo/action/registration/ActionDescriptor.hpp`
  - `available_algorithm_names<Action>()`
- added algorithm validation in `src/reticolo/modules/factory/MCAlgorithmFactory.hpp`
  - `ValidateUpdaterName<Action, TGen>(...)`
- updated `src/reticolo/runtime/runtime.hpp` to validate module/action combinations before module creation

What changed behavior-wise:

- invalid module/action combinations now fail with a more specific message listing valid actions for that module
- invalid Monte Carlo algorithm names now fail with a descriptor-driven message listing valid algorithms for the selected action family
- the successful path is unchanged

Scope note:

- this is still intentionally a narrow validation improvement inside the existing runtime
- it is not a broader CLI/runtime redesign, which remains deferred to the planned later rewrite

Verification:

- `cmake --build --preset default`
  - passed
- `ctest --preset default`
  - passed
  - result: 5/5 tests passed
