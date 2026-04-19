# Milestone E Progress Log

Date started: 2026-03-13

## Goal

Milestone E turns the current descriptor layer into a fuller built-in manifest and metadata system, then uses that system to clean up the remaining integration coupling before the later CLI/runtime rewrite.

This milestone sits after the registry/storage/HDF5 work and before the planned runtime rewrite.

## Why This Milestone Exists

Milestone D solved the main switchboard problem:

- factories are registry-based
- action families have descriptors
- validation already uses descriptor/catalog data

What remains is a softer but still important modularity gap:

- descriptors are not yet the full operational manifest for an action family
- some action headers still mix physics code with integration glue
- built-in registration still depends on thin central bootstrap lists
- there is not yet a clean metadata API that a future CLI/runtime rewrite can consume directly

Milestone E addresses those gaps without changing simulation behavior or starting the runtime rewrite early.

## Constraints

- no YAML interface changes
- no simulation semantics changes
- no output/storage behavior changes unless separately approved
- do not begin the full CLI/runtime rewrite in this milestone
- prefer additive metadata surfaces first, then cleanup/refactoring around them

## Definition Of Done

Milestone E is complete enough when:

- descriptors/manifests are the authoritative source for built-in action metadata
- metadata can answer, through code, at least:
  - available modules
  - actions for a module
  - algorithms for an action family
  - precision variants and aliases
  - declared capabilities
- registration/bootstrap logic reads from that metadata rather than duplicating it
- remaining mixed integration glue is reduced or clearly separated from action core code
- legacy paths that do not fit the descriptor model are either normalized or explicitly isolated

## Proposed Work Breakdown

### Phase E1: Descriptor Manifest Expansion

Objectives:

- extend the current action descriptor concept into a fuller manifest

Targets:

- `src/reticolo/action/registration/ActionDescriptor.hpp`
- `src/reticolo/action/registration/ActionCatalog.hpp`

Expected work:

- formalize one manifest record per action family
- represent:
  - module exposure
  - action family names
  - aliases
  - precision support
  - algorithm exposure
  - capability flags
  - optional storage/schema identifiers where useful
- reduce duplicated metadata fields spread across helpers/catalog records

Success signal:

- built-in action metadata can be described from one authoritative family-level definition

### Phase E2: Metadata Query Surface

Objectives:

- provide a clean introspection API for the current built-in system

Targets:

- `src/reticolo/action/registration/ActionCatalog.hpp`
- `src/reticolo/modules/factory/ModuleFactory.hpp`
- `src/reticolo/modules/factory/MCAlgorithmFactory.hpp`
- possibly a new metadata-oriented header under `src/reticolo/runtime/` or `src/reticolo/action/registration/`

Expected work:

- expose query helpers for:
  - modules
  - actions by module
  - algorithms by action family
  - canonical name vs aliases
  - precision availability
- make runtime validation and future help-generation consume this API
- keep the API stable enough for the later CLI/runtime rewrite

Success signal:

- the future CLI can ask the codebase what exists instead of hard-coding that information again

### Phase E3: Integration Boundary Cleanup

Objectives:

- reduce mixed concerns inside action headers and related glue code

Targets:

- action headers under `src/reticolo/action/`
- registration headers under `src/reticolo/action/registration/`
- Monte Carlo specializations currently embedded in action headers

Expected work:

- separate, where practical:
  - action core physics code
  - registration/manifest code
  - action-specific Monte Carlo adaptation code
  - storage mapping declarations
- avoid broad behavioral rewrites; prefer boundary cleanup and file layout cleanup

Success signal:

- adding or modifying an action family touches fewer unrelated concerns

### Phase E4: Legacy LLR Normalization Or Isolation

Objectives:

- stop older LLR structure from weakening the new modularity model

Targets:

- `src/reticolo/modules/llr/`

Expected work:

- identify which LLR paths are part of the current supported runtime
- normalize active paths toward descriptor/registry conventions
- isolate or mark legacy-only code paths that should not shape new abstractions

Success signal:

- the descriptor model remains authoritative even in the presence of older LLR code

### Phase E5: Registration Bootstrap Simplification

Objectives:

- shrink the remaining central built-in bootstrap surface as far as practical without introducing hidden magic

Targets:

- `src/reticolo/modules/factory/BuiltinModuleRegistration.hpp`
- `src/reticolo/modules/factory/BuiltinAlgorithmRegistration.hpp`
- action-family registration entrypoints

Expected work:

- reduce central enumeration where metadata/manifest-driven expansion is sufficient
- keep built-in registrations explicit and auditable
- avoid auto-registration patterns that are fragile across translation units

Success signal:

- new built-in action families require only local declaration plus one clear registration hook, not edits across multiple central lists

## Proposed Implementation Order

1. expand descriptors into fuller manifests
2. expose a stable metadata query API
3. redirect more validation/bootstrap logic to the metadata layer
4. split mixed integration glue out of action headers where the payoff is clear
5. normalize or isolate the remaining legacy LLR paths
6. simplify the remaining built-in bootstrap layer

## Risks And Tradeoffs

- the main risk is overreaching into the future runtime rewrite too early
- another risk is introducing excessive compile-time abstraction without reducing practical coupling
- self-registration via static initialization is intentionally not the default direction here because it is harder to reason about in a template-heavy mostly-header-only codebase

Preferred bias:

- explicit metadata
- explicit bootstrap
- clean query APIs
- minimal behavioral churn

## Files Likely To Matter Most

- `src/reticolo/action/registration/ActionDescriptor.hpp`
- `src/reticolo/action/registration/ActionCatalog.hpp`
- `src/reticolo/action/registration/ActionModuleRegistrationSupport.hpp`
- `src/reticolo/action/registration/RelativisticBoseGasModuleRegistration.hpp`
- `src/reticolo/action/registration/WeakFieldEuclideanGRModuleRegistration.hpp`
- `src/reticolo/modules/factory/BuiltinModuleRegistration.hpp`
- `src/reticolo/modules/factory/BuiltinAlgorithmRegistration.hpp`
- `src/reticolo/modules/factory/ModuleFactory.hpp`
- `src/reticolo/modules/factory/MCAlgorithmFactory.hpp`
- `src/reticolo/modules/montecarlo/algorithms/Metropolis.hpp`
- `src/reticolo/modules/montecarlo/algorithms/HMC.hpp`
- `src/reticolo/modules/llr/`
- selected action headers under `src/reticolo/action/`

## Current Status

## Progress

### 2026-03-13: Phase E1 Manifest Expansion

Implemented:

- expanded `src/reticolo/action/registration/ActionDescriptor.hpp` from a loose descriptor-only header into a manifest-bearing metadata layer
- added `ActionManifestView`, which now carries:
  - module name
  - canonical/default action name
  - float and double precision names
  - aliases
  - exposed algorithms
  - precision availability flags
  - capability flags
  - storage schema identifier
- added:
  - `descriptor_manifest<Descriptor>()`
  - `action_manifest<Action>()`
- updated action-family descriptors so they now declare:
  - `module_name`
  - `has_float_precision`
  - `has_double_precision`
  - `storage_schema`
  - existing capability and algorithm metadata

Catalog/query cleanup:

- refactored `src/reticolo/action/registration/ActionCatalog.hpp` to use manifest views directly rather than rebuilding a parallel `ActionFamilyInfo` structure
- `builtin_action_catalog()` is now a stable manifest catalog over built-in action families
- kept and updated metadata queries:
  - `available_modules()`
  - `actions_for_module(...)`
  - `find_action_family(module_name, action_name)`
- added:
  - `find_action_family(action_name)`
  - `algorithms_for_action(action_name)`

Registration cleanup:

- updated `src/reticolo/action/registration/ActionModuleRegistrationSupport.hpp` so module registration now reads module/name/precision metadata from the manifest instead of directly from scattered descriptor fields

What changed structurally:

- descriptor metadata and catalog metadata are no longer separate parallel descriptions of the same built-in action families
- there is now one manifest-level representation that can support:
  - registration bootstrap
  - validation
  - future CLI/runtime discovery
- action-family metadata is richer without changing runtime behavior

Verification:

- `cmake --build --preset default`
  - passed
- `ctest --preset default`
  - passed
  - result: 5/5 tests passed

Updated assessment:

- Phase E1 is complete enough to move on
- the next best step is Phase E2: route more runtime-facing discovery and help/introspection queries through the manifest catalog instead of ad hoc factory helpers

### 2026-03-13: Phase E2 Metadata Query Surface

Implemented:

- added `src/reticolo/runtime/BuiltinMetadata.hpp` as a runtime-facing metadata API over the built-in manifest catalog

The new API currently exposes:

- `available_modules()`
- `actions_for_module(...)`
- `describe_action(...)`
- `canonical_action_name(...)`
- `algorithms_for_action(...)`

Metadata shape:

- `ActionInfo` contains:
  - module name
  - canonical action name
  - optional float and double precision names
  - aliases
  - algorithms
  - capability flags
  - storage schema id

Factory/runtime integration:

- updated `src/reticolo/modules/factory/ModuleFactory.hpp` so action discovery and module/action validation now read through the runtime metadata API
- updated `src/reticolo/modules/factory/MCAlgorithmFactory.hpp` so algorithm validation now gets available algorithms through the runtime metadata API instead of reaching into descriptor helpers directly

What changed structurally:

- built-in metadata now has a runtime-facing surface that can be consumed without coupling callers to registration internals
- the future CLI/runtime rewrite now has a stable place to ask:
  - which modules exist
  - which actions belong to a module
  - which algorithms belong to an action family
  - which precision variants and aliases are available
- factory validation is now one layer closer to the eventual desired architecture:
  - manifests define metadata
  - runtime metadata exposes it
  - factories consume it

Verification:

- `cmake --build --preset default`
  - passed
- `ctest --preset default`
  - passed
  - result: 5/5 tests passed

Updated assessment:

- Phase E2 is complete enough
- the next best step is Phase E3: reduce mixed integration glue in action headers, especially action-specific Monte Carlo adaptation code that still lives inside large action headers

### 2026-03-13: Phase E3 Integration Boundary Cleanup, Slice 1

Implemented:

- extracted the `WeakFieldEuclideanGR`-specific Metropolis specialization out of the action header into:
  - `src/reticolo/action/adapters/WeakFieldEuclideanGRMonteCarlo.hpp`
- removed the embedded `MMonteCarlo::Metropolis<WeakFieldEuclideanGR<...>>::updateField(...)` specialization block from:
  - `src/reticolo/action/WeakFieldEuclideanGR.hpp`
- wired the built-in registration path to include the adapter header through:
  - `src/reticolo/action/registration/WeakFieldEuclideanGRModuleRegistration.hpp`

What changed structurally:

- the `WeakFieldEuclideanGR` physics/action header no longer directly owns Monte Carlo integration glue for its custom Metropolis specialization
- action-specific Monte Carlo adaptation now has a dedicated adapter location under `src/reticolo/action/adapters/`
- the specialization remains available through the built-in registration/include path, so behavior is unchanged

Why this slice matters:

- this is the clearest current example of physics code and runtime integration code being mixed together in one large header
- moving it out establishes a repeatable pattern for future cleanup:
  - action core in `src/reticolo/action/`
  - integration adapters in `src/reticolo/action/adapters/`
  - registration in `src/reticolo/action/registration/`

Verification:

- `cmake --build --preset default`
  - passed
- `ctest --preset default`
  - passed
  - result: 5/5 tests passed

Updated assessment:

- Phase E3 is underway and the highest-value current mixed-concern case has been separated
- the next E3 decision is whether to continue extracting smaller integration-adjacent pieces now, or move to the later LLR normalization/isolation work in Phase E4

### 2026-03-13: Phase E3 Integration Boundary Cleanup, Slice 2

Implemented:

- added `src/reticolo/action/adapters/BuiltinMonteCarloAdapters.hpp` as the built-in Monte Carlo adapter aggregation point
- moved the `WeakFieldEuclideanGR` Monte Carlo adapter include to the algorithm-registration side by updating:
  - `src/reticolo/modules/factory/BuiltinAlgorithmRegistration.hpp`
- removed the Monte Carlo adapter include from:
  - `src/reticolo/action/registration/WeakFieldEuclideanGRModuleRegistration.hpp`

What changed structurally:

- module registration no longer needs to know about action-specific Monte Carlo adapter specializations
- built-in Monte Carlo adapters now live behind the algorithm bootstrap layer, which is the correct boundary for action-specific algorithm integration
- the separation of concerns is now cleaner:
  - action registration registers action/module families
  - algorithm registration brings in built-in algorithm and adapter wiring
  - action adapters remain isolated under `src/reticolo/action/adapters/`

Verification:

- `cmake --build --preset default`
  - passed
- `ctest --preset default`
  - passed
  - result: 5/5 tests passed

Updated assessment:

- Phase E3 now has a cleaner adapter placement model, not just an extracted file
- the next meaningful architectural choice is whether to continue with smaller E3 cleanup or shift to Phase E4 and formally normalize/isolate the older LLR path

### 2026-03-13: Phase E4 Legacy LLR Isolation

Implemented:

- added `src/reticolo/modules/llr/LegacyLLRSupport.hpp` as the single compatibility boundary between the legacy LLR subsystem and the modern descriptor-driven architecture

That support header now centralizes:

- the descriptor-based `llr_capable_action` concept used by the old LLR worker headers
- the explicit status flag:
  - `is_registry_integrated = false`

Updated legacy LLR headers:

- `src/reticolo/modules/llr/LLRHMCWorker.hpp`
- `src/reticolo/modules/llr/LLRMetWorker.hpp`
- `src/reticolo/modules/llr/LLRHMCMetWorker.hpp`
- `src/reticolo/modules/llr/LLRController.hpp`

What changed structurally:

- the old LLR subsystem no longer reaches directly into descriptor internals from multiple places
- its coupling to the modern action-capability model is isolated in one support header
- the code now makes the architectural status explicit:
  - current built-in runtime/registry flow is the modern path
  - the old LLR subsystem is a legacy path, not yet integrated into that flow

Why this matters:

- it prevents the legacy LLR subsystem from continuing to shape new abstractions implicitly
- it creates one place to normalize or replace later when LLR is eventually brought into the modern registry/runtime model
- it reduces the chance that future descriptor or metadata changes will require scattered edits across old LLR headers

Verification:

- `cmake --build --preset default`
  - passed
- `ctest --preset default`
  - passed
  - result: 5/5 tests passed

Updated assessment:

- Phase E4 is complete enough for the current milestone scope
- the legacy LLR code is now clearly isolated rather than half-participating in the modern modularity system
- the remaining Milestone E work is mostly Phase E5 polish around registration bootstrap simplification

### 2026-03-13: Phase E5 Bootstrap Simplification

Implemented:

- introduced a single shared built-in family list in:
  - `src/reticolo/action/registration/BuiltinActionFamilies.def`

That list now drives both:

- built-in manifest catalog construction in:
  - `src/reticolo/action/registration/ActionCatalog.hpp`
- built-in module bootstrap in:
  - `src/reticolo/modules/factory/BuiltinModuleRegistration.hpp`

What changed structurally:

- the built-in action family list is no longer duplicated across:
  - metadata/catalog assembly
  - module bootstrap registration
- built-in registration remains explicit and auditable
- no static-initialization magic was introduced
- the final shape is intentionally simple:
  - one explicit built-in family list
  - different consumers project that list into their own needs

Why this is the right stopping point:

- it removes the remaining obvious central duplication without over-engineering the bootstrap layer
- it keeps the built-in system easy to review and easy to extend
- it gives the later runtime/CLI rewrite a stable metadata/bootstrap foundation

Verification:

- `cmake --build --preset default`
  - passed
- `ctest --preset default`
  - passed
  - result: 5/5 tests passed

Milestone E assessment:

- Phase E1: complete enough
- Phase E2: complete enough
- Phase E3: complete enough for the current scope
- Phase E4: complete enough
- Phase E5: complete enough

Overall result:

- Milestone E is complete enough to close
- the codebase now has:
  - manifest-driven built-in metadata
  - a runtime-facing metadata API
  - cleaner separation between action core, adapters, registration, and algorithm wiring
  - explicit isolation of the legacy LLR subsystem
  - a simpler and less duplicated built-in bootstrap surface
