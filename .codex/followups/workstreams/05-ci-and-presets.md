# Workstream 05: CI And Presets

Priority: medium
Status: in progress

## Goal

Turn the new portability/preset work into continuously checked behavior.

## Suggested First Matrix

- macOS Apple Clang preset
- Linux GCC preset

## Desired End State

- the supported preset story is verified by automation
- portability claims stop relying on one local machine

## Progress

### 2026-03-13

Implemented:

- added GitHub Actions workflow:
  - `.github/workflows/ci.yml`
- CI matrix currently covers:
  - `linux-gcc` on `ubuntu-latest`
  - `macos-appleclang` on `macos-latest`
- added missing preset parity in `CMakePresets.json`:
  - build presets for `linux-gcc` and `linux-clang`
  - test presets for `linux-gcc` and `linux-clang`

Current CI policy:

- install system HDF5 and Ninja from the runner package manager
- configure via the same CMake presets documented for local development
- build via preset
- test via preset

Why this scope:

- it verifies the two most credible baseline support paths first
- it avoids over-claiming Windows or Homebrew-LLVM OpenMP support before dedicated runner setup is in place
