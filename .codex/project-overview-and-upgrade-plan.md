# Reticolo Project Overview And Upgrade Plan

Date: 2026-03-13

## Scope

This document captures:

- the current structure and architecture of the repository
- the main technical strengths and weaknesses
- the concrete risks around HDF5 and portability
- a detailed plan to upgrade the codebase without changing simulation behavior unless explicitly approved

The current mandate is architectural and build-system oriented. Any change that alters simulation semantics, input meaning, output schema, numerical behavior, or runtime defaults should be treated as a functional change and discussed first.

## Repository Overview

Top-level layout:

- `src/`: header-only library code
- `apps/`: command-line entrypoint (`reticolo_run`)
- `tests/`: small executable-based tests
- `scripts/input/`: YAML examples
- `scripts/analysis/`: Python analysis helpers
- `build/`: existing local build tree

Primary build system:

- CMake at the root
- `FetchContent` used for `yaml-cpp` and `cxxopts`
- HDF5 located via `find_package(HDF5 REQUIRED C)`
- OpenMP required unconditionally

Current project character:

- header-only library style
- template-heavy action and algorithm composition
- YAML-driven runtime selection
- direct HDF5 C API usage
- no packaging/install story beyond basic CMake skeletons

## Execution Model

The executable path is straightforward:

1. `apps/reticolo_run.cpp` calls `reticolo_init()`, `reticolo_run()`, `reticolo_end()`.
2. `reticolo_init()` parses `--config`, loads YAML through the `ReticoloCore` singleton, and initializes logging.
3. `reticolo_run()` iterates over `setup["workflows"]`.
4. For each workflow, `ModuleFactory::MakeModule()` creates a module based on module name and action name.
5. The module is configured once and then executed once per `runs[]` entry.

This is a reasonable user-facing model for "minimum amount of work for the end user": configuration-driven workflows with modules and actions selected from YAML. The main limitation is that the implementation is less modular internally than the interface suggests.

## Current Architecture

### 1. Public API surface

`src/reticolo/reticolo.hpp` is the umbrella header. It pulls in:

- core types/tools
- lattice/indexing
- concrete actions
- runtime routines

This gives users one include, but it also means the public API is tightly coupled to all currently implemented actions and their dependencies.

### 2. Runtime/config layer

`ReticoloCore` is a singleton storing a global `YAML::Node`.

Observations:

- static data members are defined in the header, which is fragile in a multi-TU header-only design
- error handling exits the process rather than propagating structured failures
- `std::filesystem::canonical()` is used during config loading, which fails if the path is not resolvable exactly as expected

### 3. Lattice/indexing layer

The lattice is implemented as `class Lattice : public std::vector<TField>`, with indexing data shared through `std::shared_ptr<Indexing>`.

Strengths:

- simple storage model
- neighbor accessors are cheap after index precomputation
- suitable for templated field/action code

Weaknesses:

- inheritance from `std::vector` is not ideal as a long-term library boundary
- indexing uses a platform-specific include: `#include <machine/limits.h>`
- there is a bug-risk around `Indexing::MaxSize` naming because the small-lattice branch defines `max_size` instead

### 4. Action layer

Actions implement a common `ActionBase` interface and expose:

- setup
- lattice synchronization
- global/local action evaluation
- HMC force evaluation
- optional LLR-related hooks
- observables

Implemented actions:

- `RelativisticBoseGas`
- `WeakFieldEuclideanGR`

This is the strongest architectural part of the codebase. The action abstraction is close to what is needed for a modular simulation framework.

### 5. Module and algorithm factories

The factories currently hard-code the matrix of supported modules, actions, and algorithms:

- `ModuleFactory.hpp`
- `MCAlgorithmFactory.hpp`

Implications:

- every new action requires factory edits
- every action precision variant is manually enumerated
- module/action registration is compile-time hard-coded, not extensible
- the runtime looks generic, but the internals are still a switchboard

This is the main architectural gap relative to the stated goal of minimizing end-user and developer work when adding new actions.

### 6. Monte Carlo module

`MonteCarloHandler<Action>` combines:

- workspace/log setup
- action allocation
- lattice allocation
- algorithm setup
- thermalization
- measurement buffering
- HDF5 output

This works, but the class mixes too many responsibilities:

- simulation orchestration
- filesystem policy
- logging
- persistence
- run-level state management

That makes it harder to test and harder to evolve output or scheduling behavior independently.

## HDF5 Status

The codebase uses the HDF5 C API directly in many places, especially:

- `src/reticolo/core/tools/Hdf5Handler.hpp`
- `src/reticolo/core/tools/hdf5_helpers.hpp`
- action-specific `make_H5_Type<>` specializations
- Monte Carlo data HDF5 compound type specializations
- tests that validate basic write/append/read behavior

### Positive state

- HDF5 access is already centralized more than expected
- `HDF5Handler` uses RAII wrappers for most raw HDF5 handles
- there are tests for idempotent create/overwrite and lattice round-trip
- compound datatype generation is explicit and understandable

### Current technical issues

1. The implementation is tightly coupled to the HDF5 1.x C API surface.

2. HDF5 support is spread across domain types and actions via `make_H5_Type<>` specializations, which makes storage concerns bleed into physics/model code.

3. `saveLattice()` still uses `H5F_ACC_EXCL`, unlike the more recent idempotent behavior elsewhere.

4. Compression assumes `deflate` availability without capability checks.

5. The code currently links `hdf5::hdf5` from CMake, which is not the most portable target spelling across installations and package managers.

6. There is no explicit compatibility layer to isolate an HDF5 API migration.

### HDF5 2.0 implications

Without browsing external docs here, the safe conclusion is:

- an HDF5 2.0 migration should not be attempted as a global search-and-replace
- the code first needs an internal storage abstraction layer
- all direct API contact points should be narrowed to one adapter module
- type mapping, dataset creation policy, attributes, and appendable datasets should be expressed in project-level concepts rather than raw HDF5 calls

That adapter-first approach reduces migration risk and allows dual-support for current HDF5 and future HDF5 2.0 during the transition.

## Build And Portability Status

### Current build assumptions

The current root `CMakeLists.txt` encodes several portability problems:

- MSVC is explicitly rejected
- Intel is explicitly rejected
- OpenMP is required, not optional
- release flags are injected globally and are compiler-specific
- dependencies are fetched from GitHub at configure time
- there is no Windows-specific handling for HDF5/OpenMP/runtime paths

### Concrete portability blockers found in source

1. `src/reticolo/lattice/indexing.hpp` includes `<machine/limits.h>`, which is not portable and will break on Linux/Windows toolchains.

2. `tests/hdf5_expandable_lattice.cpp` includes `<unistd.h>` for `getpid()`, which is POSIX-specific and not Windows-safe.

3. The code uses `std::format` heavily. This is a known portability risk, especially for Apple Clang/libc++ combinations and some Windows toolchains depending on version.

4. `build.sh` is Homebrew/Apple-Clang specific:
   - hard-coded `/opt/homebrew`
   - hard-coded LLVM paths
   - hard-coded HDF5 root
   - hard-coded libomp assumptions

5. The project is effectively tested only in one local macOS-style environment:
   - current `build/CMakeCache.txt` shows Apple/Homebrew paths
   - HDF5 currently resolves to version `1.14.6`

6. `FetchContent` with live Git references (`HEAD`, `master`) makes builds non-reproducible and problematic in restricted or offline environments.

### Less obvious portability issues

- Header-only design pushes all compile burden to consumers and magnifies compiler/library incompatibilities.
- Global compile flags use `CMAKE_CXX_FLAGS_*`, which is weaker than target-based compile options.
- Unconditional OpenMP makes Windows and Apple Clang adoption harder than necessary.
- Filesystem canonicalization can fail in edge cases where weakly canonical resolution would succeed.

## Testing Status

The project has lightweight executable-based tests for:

- lattice behavior
- lattice ownership variants
- field read/write
- HDF5 basics
- expandable HDF5 datasets

This is useful, but current gaps remain:

- no CTest coverage for every built test executable
- no matrix testing across compilers/platforms
- no schema-level validation for HDF5 outputs
- no regression tests for YAML parsing and workflow dispatch
- no separation between unit tests and integration tests

## Architectural Assessment Against Project Goals

Goal: ease of use and modularity for many actions with minimal end-user work.

Current strengths:

- YAML-driven workflows are the right direction
- actions follow a common conceptual interface
- module/action/algorithm concepts are already present
- output is already organized around reproducible artifacts

Current mismatches:

- adding an action still requires editing central factories
- simulation orchestration, persistence, and workspace policy are coupled
- storage details leak into domain classes
- build system is not portable enough for broad adoption
- current dependency discovery is too environment-specific

Conclusion:

The project has a strong conceptual shape, but it is still in an "early framework" stage rather than a fully modular simulation platform. The next phase should focus on making the internal architecture match the external promise.

## Recommended Upgrade Strategy

Do not start by changing physics code or algorithm behavior.

The highest-value order is:

1. stabilize build and portability foundations
2. isolate persistence/HDF5 behind a clean adapter
3. refactor registration/factory mechanisms for real modularity
4. strengthen tests and CI
5. only then expand supported actions/modules aggressively

## Detailed Plan

### Phase 0: Baseline And Safety Rails

Objectives:

- preserve current behavior
- make future refactors measurable

Tasks:

- create a baseline document for current supported workflows, actions, algorithms, and outputs
- freeze dependency versions in CMake instead of using `HEAD` and `master`
- add a developer option to disable network-based dependency fetching
- ensure every existing test is registered with CTest
- add one smoke test that configures and runs `reticolo_run` with a tiny YAML input

Deliverable:

- reproducible baseline build with a documented support matrix

### Phase 1: Cross-Platform Build Modernization

Objectives:

- build on macOS, Linux, and Windows
- remove environment-specific assumptions

Tasks:

- replace global compiler flag manipulation with target-based compile options
- stop rejecting MSVC outright
- make OpenMP optional:
  - enabled when found
  - disabled cleanly otherwise
- replace `FetchContent` live branch references with pinned versions
- add package-first dependency discovery with fallback fetch:
  - HDF5
  - yaml-cpp
  - cxxopts
- introduce standard project options:
  - `RETICOLO_BUILD_TESTS`
  - `RETICOLO_ENABLE_OPENMP`
  - `RETICOLO_FETCH_DEPS`
  - `RETICOLO_WARNINGS_AS_ERRORS`
- replace or demote `build.sh` with documented CMake presets
- add `CMakePresets.json` for:
  - macOS-clang
  - linux-gcc
  - linux-clang
  - windows-msvc

Code changes expected:

- root `CMakeLists.txt`
- `src/CMakeLists.txt`
- `apps/CMakeLists.txt`
- `tests/CMakeLists.txt`
- new `CMakePresets.json`
- README build instructions rewrite

Risk level:

- low functional risk
- high integration value

### Phase 2: Portability Cleanup In Source

Objectives:

- remove obvious platform-specific code paths

Tasks:

- replace `<machine/limits.h>` with standard headers only
- replace `unistd.h`/`getpid()` test-only logic with portable temp file naming
- audit filesystem calls and use safer path resolution patterns
- evaluate `std::format` support:
  - if target compilers support it reliably, keep it
  - otherwise introduce `fmt` or a small formatting wrapper

Note:

Replacing `std::format` is not intended to change functionality. It is a portability hardening step.

### Phase 3: HDF5 Adapter Layer

Objectives:

- isolate all HDF5-specific behavior behind internal abstractions
- prepare for HDF5 2.0 without large-scale breakage

Tasks:

- define a storage backend interface for:
  - file lifecycle
  - group creation
  - fixed dataset write
  - appendable dataset setup
  - append operation
  - attribute read/write
- split `HDF5Handler.hpp` into:
  - public storage facade
  - HDF5 backend implementation
  - datatype mapping utilities
- move datatype registration out of physics/action headers where possible
- define explicit schema objects for:
  - observables
  - Monte Carlo metadata
  - saved lattice fields
  - RNG state
- add tests that validate schema stability for current files

Deliverable:

- one narrow HDF5 backend boundary that can be ported to HDF5 2.0

### Phase 4: HDF5 2.0 Migration

Objectives:

- adopt the HDF5 2.0 framework through the new backend boundary

Tasks:

- map current HDF5 backend responsibilities to the HDF5 2.0 API/framework equivalents
- implement a compatibility backend or transitional adapter
- verify:
  - dataset creation
  - appendable datasets
  - compound types
  - string attributes
  - round-trip lattice persistence
- keep old and new backends selectable during transition if needed

Important constraint:

If HDF5 2.0 requires output schema or behavior changes visible to users, that must be discussed first.

### Phase 5: Real Modularity For Actions And Algorithms

Objectives:

- reduce developer work for adding new actions
- make the runtime architecture match the user-facing promise

Tasks:

- replace switch-based factories with registration-based factories
- separate action registration from module registration
- make precision variants systematic rather than hand-coded per action
- define clear action capability traits:
  - metropolis-capable
  - hmc-capable
  - llr-capable
- remove storage-specific template specializations from action headers when possible

Possible shape:

- `ActionRegistry`
- `ModuleRegistry`
- `AlgorithmRegistry`
- action descriptors that bundle:
  - user-facing name
  - precision support
  - capabilities
  - constructors

This phase should not change simulation semantics. It should reduce internal wiring cost.

### Phase 6: Descriptor-Driven Metadata And Integration Cleanup

Objectives:

- make descriptors the single operational manifest for built-in action families
- reduce header coupling between physics code, module glue, and runtime integration
- prepare the future CLI/runtime rewrite to consume metadata instead of hard-coded logic

Tasks:

- expand action descriptors into fuller manifests that can describe:
  - supported modules
  - exposed algorithms
  - precision variants
  - aliases
  - capability flags
  - storage/schema identity where needed
- introduce a stable metadata query surface for built-in capabilities:
  - available modules
  - actions per module
  - algorithms per action family
  - precision support
- move remaining integration glue out of action headers where practical:
  - action-specific Monte Carlo specializations
  - registration/bootstrap glue
  - descriptor metadata
  - storage mapping declarations
- reduce or isolate legacy code paths that still rely on older naming/layout conventions, especially around LLR
- make validation and later help/introspection generation descriptor-driven rather than registry-structure-driven

Deliverable:

- a descriptor/manifest layer that is rich enough to drive future CLI/runtime discovery and validation without reintroducing hard-coded switchboards

Important constraint:

- this phase should not attempt the full CLI/runtime rewrite
- it should prepare for that rewrite by making metadata authoritative and queryable

### Phase 7: Monte Carlo Runtime Decomposition

Objectives:

- make the core Monte Carlo module easier to test and extend

Tasks:

- split `MonteCarloHandler` into smaller components:
  - run configuration parser
  - workspace/output manager
  - measurement sink
  - simulation loop executor
  - checkpoint/configuration saver
- make run state explicit rather than spread across mutable member fields
- isolate measurement buffering from file I/O

Benefits:

- cleaner tests
- easier checkpointing/resume support later
- easier future multi-backend persistence

### Phase 8: CI And Developer Experience

Objectives:

- keep the codebase working across supported environments

Tasks:

- add CI for:
  - macOS latest
  - Ubuntu latest
  - Windows latest
- test matrix across:
  - Clang
  - GCC
  - MSVC
- add formatting and warnings checks
- document supported compilers and dependency installation paths
- provide example configure/build/test commands using presets

## Proposed Execution Order For Real Work

The practical order I would follow in this repo is:

1. modernize CMake and dependency handling
2. remove portable-header and temp-file issues
3. standardize tests and add CI
4. introduce the storage facade around current HDF5 code
5. migrate the HDF5 backend toward the HDF5 2.0 framework
6. refactor registry/factory architecture
7. enrich descriptors into manifests and clean integration boundaries
8. decompose Monte Carlo orchestration

## Suggested Near-Term Milestones

### Milestone A

Portable build foundation.

Definition of done:

- project configures on macOS, Linux, Windows
- MSVC is no longer blocked
- OpenMP is optional
- dependencies are pinned and reproducible

### Milestone B

Storage abstraction and test hardening.

Definition of done:

- all HDF5 access goes through one backend
- current HDF5 tests pass through the abstraction
- output schema is documented

### Milestone C

HDF5 2.0 backend adoption.

Definition of done:

- equivalent current persistence functionality on the new backend
- schema compatibility validated or migration path documented

### Milestone D

Registry-based modularity.

Definition of done:

- adding a new action no longer requires editing a central switch statement
- action capabilities are declarative

### Milestone E

Descriptor-driven manifests and metadata cleanup.

Definition of done:

- descriptors are the authoritative source for built-in action/module/algorithm metadata
- a stable metadata query API exists for runtime discovery and future CLI generation
- remaining mixed integration glue is reduced or clearly separated from action core code
- legacy paths that conflict with the descriptor model are either normalized or explicitly isolated

## Immediate Recommendations

If the next implementation step starts now, the best first work item is:

- build-system modernization and portability cleanup

Reason:

- it is high leverage
- it does not require changing simulation behavior
- it reduces friction for every subsequent HDF5 and architectural refactor

## Files Most Relevant For The Next Refactor

- `CMakeLists.txt`
- `src/CMakeLists.txt`
- `tests/CMakeLists.txt`
- `build.sh`
- `src/reticolo/core/tools/Hdf5Handler.hpp`
- `src/reticolo/core/tools/hdf5_helpers.hpp`
- `src/reticolo/modules/factory/ModuleFactory.hpp`
- `src/reticolo/modules/factory/MCAlgorithmFactory.hpp`
- `src/reticolo/modules/montecarlo/MonteCarloHandler.hpp`
- `src/reticolo/lattice/indexing.hpp`
- `tests/hdf5_basic.cpp`
- `tests/hdf5_expandable_lattice.cpp`
