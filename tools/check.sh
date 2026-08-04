#!/usr/bin/env bash
# Local mirror of CI's gates — format + clang-tidy + build + test — using the
# EXACT pinned tool versions CI uses, so a green run here means a green CI.
#
#   clang-format  22.1.5   (CI pins the pip build; a different major reflows differently)
#   clang-tidy    22       (CI enforces WarningsAsErrors with this version; the
#                           enabled-check set is version-sensitive, and clang-tidy
#                           must parse a compile DB built by the SAME clang major)
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
shopt -s globstar nullglob  # src/**/*.cpp must recurse (matches CI's tidy glob)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PRESET="${PRESET:-macos-appleclang}"
TIDY_DB="build/tidy22"

say()  { printf '\033[1m== %s\033[0m\n' "$*"; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# --- tool resolution ---------------------------------------------------------
resolve_clang_format() {
    local c
    for c in "${CLANG_FORMAT:-}" "$HOME/.local/bin/clang-format" /opt/homebrew/opt/llvm/bin/clang-format clang-format-22 clang-format; do
        [ -n "$c" ] && command -v "$c" >/dev/null 2>&1 || continue
        case "$("$c" --version 2>/dev/null)" in *"version 22."*) echo "$c"; return;; esac
    done
    die "clang-format 22.x not found (CI pins 22.1.5). Install: 'pip install --user clang-format==22.1.5'. Set CLANG_FORMAT=/path."
}
resolve_clang_tidy() {
    local c
    for c in "${CLANG_TIDY:-}" /opt/homebrew/opt/llvm/bin/clang-tidy clang-tidy-22 clang-tidy; do
        [ -n "$c" ] && command -v "$c" >/dev/null 2>&1 || continue
        case "$("$c" --version 2>/dev/null)" in *"version 22."*) echo "$c"; return;; esac
    done
    die "clang-tidy 22 not found. macOS: 'brew install llvm'. Linux: 'apt install clang-tidy-22'."
}

# --- format ------------------------------------------------------------------
do_format() {
    local cf; cf="$(resolve_clang_format)"
    # --cached AND --others: a brand-new file is untracked until its first
    # commit, so scanning only tracked files lets it pass the gate, land, and
    # fail CI afterwards. --exclude-standard keeps .gitignore'd trees (build/,
    # and every fetched dependency under it) out.
    local files; mapfile -t files < <(
        git ls-files --cached --others --exclude-standard '*.hpp' '*.cpp' '*.cuh' '*.cu')
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
# HeaderFilterRegex) + writer.cpp. We build a dedicated clang-22 compile DB so
# clang-tidy-22 parses it without frontend errors (a mismatched-clang DB would fail).
ensure_tidy_db() {
    local ct="$1" cxx
    cxx="$(dirname "$ct")/clang++"
    [ -x "$cxx" ] || cxx="$(command -v clang++-22 || true)"
    [ -n "$cxx" ] && [ -x "$(command -v "$cxx" || echo "$cxx")" ] || die "matching clang++ 22 not found next to clang-tidy"
    if [ ! -f "$TIDY_DB/compile_commands.json" ]; then
        say "configuring clang-22 tidy DB ($TIDY_DB)"
        cmake -S . -B "$TIDY_DB" -G Ninja \
            -DCMAKE_CXX_COMPILER="$cxx" -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DRETICOLO_ENABLE_OPENMP=OFF \
            -DRETICOLO_BUILD_APPS=OFF -DRETICOLO_BUILD_TESTS=OFF -DRETICOLO_BUILD_EXAMPLES=OFF \
            -DRETICOLO_TUNE_NATIVE=OFF -DRETICOLO_WARNINGS_AS_ERRORS=OFF >/dev/null
    fi
    # Only <sleef.h> is needed for the umbrella to parse; the full sleef lib is
    # irrelevant to tidy and some scalar variants fail to build on arm, so don't
    # let that abort the run.
    cmake --build "$TIDY_DB" --target sleef >/dev/null 2>&1 || true
    [ -f "$TIDY_DB"/_deps/sleef-build/include/sleef.h ] || die "sleef.h not generated in $TIDY_DB"
}
do_tidy() {
    local ct; ct="$(resolve_clang_tidy)"
    ensure_tidy_db "$ct"
    # Match CI exactly: every production TU under src/ (amalgamation.cpp gives
    # full public-header coverage; writer.cpp/reader.cpp/cli are the impl TUs).
    # Globbing here — not a fixed two-file list — is what stops a src/ TU (e.g.
    # reader.cpp) from being tidied by CI but skipped locally.
    #
    # SCOPE, deliberately: the LIBRARY is fully tidied — every header via the
    # amalgamation TU, every impl TU via this glob. CONSUMER code is not:
    # apps/, tests/ and examples/ are excluded uniformly (the tidy DB is even
    # configured with BUILD_{APPS,TESTS,EXAMPLES}=OFF, so their TUs are absent
    # from the compile database). That is one boundary applied to all consumers,
    # not a per-directory exception. Consumers still get the full warning set
    # via reticolo_configure_warnings + -Werror at build time.
    local tus=(src/**/*.cpp)
    if [ "${1:-}" = fix ]; then
        say "clang-tidy --fix ($(basename "$ct"))"
        "$ct" -p "$TIDY_DB" --fix "${tus[@]}" || true
        echo "applied fixes — re-run 'tools/check.sh tidy' to confirm clean, then format --fix"
    else
        say "clang-tidy ($(basename "$ct"), WarningsAsErrors) — src/**/*.cpp"
        "$ct" -p "$TIDY_DB" "${tus[@]}" && echo "tidy clean" || die "clang-tidy findings"
    fi
}

# --- build + test ------------------------------------------------------------
# --- test ---------------------------------------------------------------------
# The suite is partitioned by CTest label (see tests/CMakeLists.txt — every test
# carries exactly one). Running each category separately costs nothing extra and
# buys a per-area breakdown: which part of the tree broke is visible without
# reading the failure log, and a category that silently registers zero tests
# shows up as `skipped` instead of vanishing into a green total.
CATEGORIES=(core rng physics io app cuda)

# Elapsed seconds with a fractional part, portable across macOS/Linux bash.
now_s() { python3 -c 'import time; print(f"{time.time():.3f}")'; }

do_test() {
    say "ctest --preset $PRESET — by category"

    local -a fail_names=()
    local total=0 passed=0 failed=0 skipped=0
    local t_start; t_start="$(now_s)"

    local cat
    for cat in "${CATEGORIES[@]}"; do
        local n
        n="$(ctest --preset "$PRESET" -L "^${cat}\$" -N 2>/dev/null | awk '/Total Tests:/{print $NF}')"
        [ -n "$n" ] || n=0

        if [ "$n" -eq 0 ]; then
            printf '  \033[2m%-9s ▸ skipped — no tests registered for this preset\033[0m\n' "$cat"
            skipped=$((skipped + 1))
            continue
        fi

        local out rc t0 t1 dt
        t0="$(now_s)"
        set +e
        out="$(ctest --preset "$PRESET" -L "^${cat}\$" --output-on-failure 2>&1)"
        rc=$?
        set -e
        t1="$(now_s)"
        dt="$(python3 -c "print(f'{$t1-$t0:.2f}')")"

        local nfail
        nfail="$(printf '%s' "$out" | awk '/tests failed out of/{print $4}')"
        [ -n "$nfail" ] || nfail=0
        local npass=$((n - nfail))

        total=$((total + n)); passed=$((passed + npass)); failed=$((failed + nfail))

        if [ "$rc" -eq 0 ]; then
            printf '  \033[32m%-9s\033[0m ▸ %4d tests   %4d passed   %4d failed   %6ss\n' \
                   "$cat" "$n" "$npass" "$nfail" "$dt"
        else
            # Full --output-on-failure diff first (that is what you actually
            # debug from), then the row — so the table at the end stays a
            # scannable overview rather than a replacement for the log.
            printf '\n\033[31m--- %s failures ------------------------------------------\033[0m\n' "$cat"
            # Drop the per-test "Passed" progress chatter; keep the failing
            # tests' captured output and the summary.
            printf '%s\n\n' "$out" | grep -vE '^[[:space:]]*Start[[:space:]]+[0-9]+:|\.\.\.\.*   Passed '
            printf '  \033[31m%-9s\033[0m ▸ %4d tests   %4d passed   \033[31m%4d FAILED\033[0m   %6ss\n' \
                   "$cat" "$n" "$npass" "$nfail" "$dt"
            # ctest lists these as "  <N> - <name> (Failed) <label>": take the
            # text after " - ", then strip the trailing "(Status) label".
            printf '%s\n' "$out" | sed -n '/The following tests FAILED/,$p' | \
                sed -n 's/^[[:space:]]*[0-9][0-9]* - //p' | \
                sed 's/ ([A-Za-z][^)]*).*$//' | while read -r name; do
                    printf '              \033[31m✖\033[0m %s\n' "$name"
                    done
            fail_names+=("$cat")
        fi
    done

    local t_end dt_all; t_end="$(now_s)"
    dt_all="$(python3 -c "print(f'{$t_end-$t_start:.2f}')")"
    printf '  %s\n' "──────────────────────────────────────────────────────────────"
    if [ "$failed" -eq 0 ]; then
        printf '  \033[1m%-9s\033[0m ▸ %4d tests   \033[32m%4d passed\033[0m   %4d failed   %6ss\n' \
               TOTAL "$total" "$passed" "$failed" "$dt_all"
    else
        printf '  \033[1m%-9s\033[0m ▸ %4d tests   %4d passed   \033[31m%4d FAILED\033[0m   %6ss\n' \
               TOTAL "$total" "$passed" "$failed" "$dt_all"
    fi
    [ "$skipped" -gt 0 ] && printf '  \033[2m%d categor%s skipped\033[0m\n' \
        "$skipped" "$([ "$skipped" -eq 1 ] && echo y || echo ies)"

    # Thread-invariance cases are only meaningful with OpenMP; without it
    # exec::traverse_threads() is hardcoded to 1 and they compare serial against
    # serial. Say so rather than let them read as real coverage.
    if ! grep -q '^RETICOLO_ENABLE_OPENMP:BOOL=ON' "build/$PRESET/CMakeCache.txt" 2>/dev/null; then
        printf '  \033[2mnote: OpenMP off on this preset — thread-invariance cases in `core`\n'
        printf '        ran degenerate (team size forced to 1).\033[0m\n'
    fi

    if [ "$failed" -gt 0 ]; then
        printf '\n  rerun: \033[1mctest --preset %s -L %s --output-on-failure\033[0m\n' \
               "$PRESET" "${fail_names[0]}"
        return 1
    fi
    return 0
}

do_build() {
    say "cmake --build --preset $PRESET"
    cmake --build --preset "$PRESET"
    do_test
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
