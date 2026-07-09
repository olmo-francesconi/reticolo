---
name: check
description: >
  Run reticolo's local CI gates for C++ changes — clang-format (pinned 20.1.7),
  clang-tidy (pinned 18, WarningsAsErrors), and build + ctest — via
  tools/check.sh. Use before committing or pushing any C++ change, or whenever
  asked to check format / lint / tidy / build for this repo. Prevents the
  tool-version mismatches that make ad-hoc clang-format / clang-tidy runs
  disagree with CI.
---

# reticolo — format + clang-tidy + build gate

Use `tools/check.sh` for all format / tidy / build checks. It pins the EXACT
tool versions CI uses. Do NOT run `clang-format` / `clang-tidy` straight from
PATH: homebrew's defaults (clang-format 22, clang-tidy 22) disagree with CI —
format reflows differently, and clang-tidy 22 fires whole-tree noise the project
deliberately disables AND cannot parse a newer-clang compile DB. That mismatch is
exactly the trap this skill exists to avoid.

## Commands
- `tools/check.sh`               — all gates: format check → tidy → build + ctest
- `tools/check.sh --fix`         — auto-apply clang-format + clang-tidy fixes
- `tools/check.sh format [--fix]`
- `tools/check.sh tidy   [--fix]`
- `tools/check.sh build`         — build + ctest only
- `PRESET=macos-llvm tools/check.sh build` — choose the build/test preset

Run it (at minimum `format` + `tidy`) before every commit/push of C++ changes.
Green here ⇒ green CI; if it isn't green locally, don't push.

## One-time requirements (script auto-detects, errors clearly if missing)
- **clang-format 20.1.7** — CI pin. On this machine: `~/.local/bin/clang-format`.
- **clang-tidy 18** — `brew install llvm@18` (macOS) / `apt install clang-tidy-18`
  (Linux). The script builds a dedicated **clang-18** compile DB in `build/tidy18`
  so clang-tidy 18 parses without frontend errors. Tidy scope matches CI exactly:
  `src/lint/amalgamation.cpp` (umbrella → all headers) + `src/io/writer.cpp`.

## Naming warnings — DO NOT mass-rename
`clang-tidy --fix` for `readability-identifier-naming` **breaks the build**: a
member `foo_` and its accessor `foo()` both become `foo` (duplicate), and `L → l`
collides with parameters. The `.clang-tidy` already encodes the real convention
via `IgnoredRegexp` (trailing-`_` internal helpers, physics names with capitals
like `L`/`dE`/`sinP`, lowercase type aliases `cplx`/`real_scalar`, std-style RAII
guards `team_scope`/`slab_scope`). For a NEW naming warning: widen the relevant
`IgnoredRegexp` or add a targeted `NOLINT` — never a blanket rename. Full context
in the `project_clang_tidy_gate` memory.
