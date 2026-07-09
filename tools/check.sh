#!/usr/bin/env bash
# Local mirror of CI's gates — format + clang-tidy + build + test — using the
# EXACT pinned tool versions CI uses, so a green run here means a green CI.
#
#   clang-format  20.1.7   (CI pins this; a different major reflows differently)
#   clang-tidy    18       (CI enforces WarningsAsErrors with this version; a
#                           newer clang-tidy fires whole-tree noise the project
#                           disables, and can't parse a newer-clang compile DB)
#
# Usage:
#   tools/check.sh                 # check all gates (format, tidy, build+test)
#   tools/check.sh --fix           # auto-apply clang-format + clang-tidy fixes
#   tools/check.sh format [--fix]  # just format
#   tools/check.sh tidy   [--fix]  # just clang-tidy (the two src/ TUs CI tidies)
#   tools/check.sh build           # just build + ctest
#
# Env: PRESET (build/test preset, default macos-appleclang)
#      CLANG_FORMAT / CLANG_TIDY  (override the auto-detected binaries)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PRESET="${PRESET:-macos-appleclang}"
TIDY_DB="build/tidy18"

say()  { printf '\033[1m== %s\033[0m\n' "$*"; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# --- tool resolution ---------------------------------------------------------
resolve_clang_format() {
    local c
    for c in "${CLANG_FORMAT:-}" "$HOME/.local/bin/clang-format" clang-format-20 clang-format; do
        [ -n "$c" ] && command -v "$c" >/dev/null 2>&1 || continue
        case "$("$c" --version 2>/dev/null)" in *"version 20."*) echo "$c"; return;; esac
    done
    die "clang-format 20.x not found (CI pins 20.1.7). Set CLANG_FORMAT=/path."
}
resolve_clang_tidy() {
    local c
    for c in "${CLANG_TIDY:-}" /opt/homebrew/opt/llvm@18/bin/clang-tidy clang-tidy-18 clang-tidy; do
        [ -n "$c" ] && command -v "$c" >/dev/null 2>&1 || continue
        case "$("$c" --version 2>/dev/null)" in *"version 18."*) echo "$c"; return;; esac
    done
    die "clang-tidy 18 not found. macOS: 'brew install llvm@18'. Linux: 'apt install clang-tidy-18'."
}

# --- format ------------------------------------------------------------------
do_format() {
    local cf; cf="$(resolve_clang_format)"
    local files; mapfile -t files < <(git ls-files '*.hpp' '*.cpp' '*.cuh' '*.cu')
    if [ "${1:-}" = fix ]; then
        say "clang-format --i ($(basename "$cf"), ${#files[@]} files)"
        printf '%s\0' "${files[@]}" | xargs -0 "$cf" -i
        echo "formatted."
    else
        say "clang-format --dry-run -Werror (${#files[@]} files)"
        local bad=0
        for f in "${files[@]}"; do
            "$cf" --dry-run -Werror "$f" >/dev/null 2>&1 || { echo "  needs formatting: $f"; bad=1; }
        done
        [ "$bad" = 0 ] && echo "format clean" || die "format issues (run: tools/check.sh format --fix)"
    fi
}

# --- clang-tidy --------------------------------------------------------------
# CI tidies src/**/*.cpp = the umbrella amalgamation (covers every header via
# HeaderFilterRegex) + writer.cpp. We build a dedicated clang-18 compile DB so
# clang-tidy-18 parses it without frontend errors (a newer-clang DB would fail).
ensure_tidy_db() {
    local ct="$1" cxx
    cxx="$(dirname "$ct")/clang++"
    [ -x "$cxx" ] || cxx="$(command -v clang++-18 || true)"
    [ -n "$cxx" ] && [ -x "$(command -v "$cxx" || echo "$cxx")" ] || die "matching clang++ 18 not found next to clang-tidy"
    if [ ! -f "$TIDY_DB/compile_commands.json" ]; then
        say "configuring clang-18 tidy DB ($TIDY_DB)"
        cmake -S . -B "$TIDY_DB" -G Ninja \
            -DCMAKE_CXX_COMPILER="$cxx" -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DRETICOLO_ENABLE_OPENMP=OFF \
            -DRETICOLO_BUILD_APPS=OFF -DRETICOLO_BUILD_TESTS=OFF -DRETICOLO_BUILD_EXAMPLES=OFF \
            -DRETICOLO_TUNE_NATIVE=OFF -DRETICOLO_WARNINGS_AS_ERRORS=OFF >/dev/null
    fi
    # Only <sleef.h> is needed for the umbrella to parse; the full sleef lib is
    # irrelevant to tidy and some scalar variants fail to build on clang-18/arm,
    # so don't let that abort the run.
    cmake --build "$TIDY_DB" --target sleef >/dev/null 2>&1 || true
    [ -f "$TIDY_DB"/_deps/sleef-build/include/sleef.h ] || die "sleef.h not generated in $TIDY_DB"
}
do_tidy() {
    local ct; ct="$(resolve_clang_tidy)"
    ensure_tidy_db "$ct"
    local tus=(src/lint/amalgamation.cpp src/io/writer.cpp)
    if [ "${1:-}" = fix ]; then
        say "clang-tidy --fix ($(basename "$ct"))"
        "$ct" -p "$TIDY_DB" --fix "${tus[@]}" || true
        echo "applied fixes — re-run 'tools/check.sh tidy' to confirm clean, then format --fix"
    else
        say "clang-tidy ($(basename "$ct"), WarningsAsErrors) — amalgamation + writer.cpp"
        "$ct" -p "$TIDY_DB" "${tus[@]}" && echo "tidy clean" || die "clang-tidy findings"
    fi
}

# --- build + test ------------------------------------------------------------
do_build() {
    say "cmake --build --preset $PRESET"
    cmake --build --preset "$PRESET"
    say "ctest --preset $PRESET"
    ctest --preset "$PRESET" --output-on-failure
}

# --- dispatch ----------------------------------------------------------------
FIX=""; STAGE="all"
for a in "$@"; do
    case "$a" in
        --fix) FIX=fix;;
        format|tidy|build) STAGE="$a";;
        *) die "unknown arg: $a";;
    esac
done
case "$STAGE" in
    format) do_format "$FIX";;
    tidy)   do_tidy "$FIX";;
    build)  do_build;;
    all)    do_format "$FIX"; do_tidy "$FIX"; do_build
            printf '\033[32m== all gates passed\033[0m\n';;
esac
