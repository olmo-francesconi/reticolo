---
name: check
description: >
  Run reticolo's local CI gates for C++ changes — clang-format (pinned 22.1.5),
  clang-tidy (pinned 22, WarningsAsErrors), and build + ctest — via
  tools/check.sh. Use before committing or pushing any C++ change, or whenever
  asked to check format / lint / tidy / build for this repo. Prevents the
  tool-version and compile-DB mismatches that make ad-hoc clang-format /
  clang-tidy runs disagree with CI.
---

# reticolo — format + clang-tidy + build gate

Use `tools/check.sh` for all format / tidy / build checks. It pins the EXACT
tool versions CI uses AND builds the matching compile DB. Do NOT run
`clang-format` / `clang-tidy` straight from PATH: a different clang-format major
reflows differently, and clang-tidy 22 must parse a compile DB built by a
clang-22 — pointed at the appleclang or `-march=native` build DB it fails with
frontend errors, not real findings. That mismatch is exactly the trap this skill
exists to avoid.

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
- **clang-format 22.1.5** — CI pin (the PyPI build; homebrew's 22.1.8 emits
  identical formatting, but pin the pip one so macOS and Linux match).
  Install: `pip install --user clang-format==22.1.5` → `~/.local/bin/clang-format`.
- **clang-tidy 22** — `brew install llvm` (macOS, → `/opt/homebrew/opt/llvm`) /
  `apt install clang-tidy-22` (Linux; on CI via apt.llvm.org `llvm.sh 22`). The
  script builds a dedicated **clang-22** compile DB in `build/tidy22` so
  clang-tidy 22 parses without frontend errors. Tidy scope matches CI exactly:
  `src/lint/amalgamation.cpp` (umbrella → all headers) + `src/io/writer.cpp`.

## Naming warnings — DO NOT mass-rename
`clang-tidy --fix` for `readability-identifier-naming` **breaks the build**: a
member `foo_` and its accessor `foo()` both become `foo` (duplicate), and `L → l`
collides with parameters. The `.clang-tidy` already encodes the real convention
via `IgnoredRegexp` (trailing-`_` internal helpers, physics names with capitals
like `L`/`dE`/`sinP`, lowercase type aliases `cplx`/`real_scalar`, std-style RAII
guards `team_scope`/`slab_scope`). For a NEW naming warning: widen the relevant
`IgnoredRegexp` or add a targeted `NOLINT` — never a blanket rename. Full context
in the `project_clang_tidy_gate` memory, including the fix/disable split behind
the 22 bump (unchecked-container-access, pragma-once, math-parens,
concise-preprocessor, integer-sign-comparison are disabled by design).
