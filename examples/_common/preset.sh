# Shared prelude for example sweeps. Source me as:
#     source "$(dirname "${BASH_SOURCE[0]}")/../_common/preset.sh" "$@"
# Sets: $here, $root, $preset, $build_dir. Consumes --preset / --preset=X / -h.
# After sourcing, $@ holds the remaining (non-preset) args; callers can `shift`
# past them or just use the parsed $preset directly.
#
# Examples are standalone consumers of reticolo: each owns a CMakeLists.txt that
# links reticolo::reticolo via find-or-fetch. Call `build_example` to configure
# + build the current example in place; it sets $example_bin to the directory
# holding the freshly built binaries. Override the compiler with CC/CXX (e.g.
# the macos-llvm toolchain for OpenMP).

here=$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)
root=$(cd "$here/../.." && pwd)
preset=${RETICOLO_PRESET:-macos-appleclang}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)   preset="$2";      shift 2 ;;
        --preset=*) preset="${1#*=}"; shift   ;;
        -h|--help)
            echo "Usage: $(basename "$0") [--preset <name>]"
            echo "  --preset <name>   label for the example build dir, and the"
            echo "                    reticolo build it nests (default:"
            echo "                    \$RETICOLO_PRESET or macos-appleclang)."
            echo "                    Set CC/CXX for a specific toolchain."
            exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done
build_dir="$root/build/$preset"

# Configure + build the current example ($here) as its own CMake project and
# leave the binaries in $example_bin. Demonstrates downstream consumption: the
# example's find-or-fetch resolves reticolo from this very checkout (sibling
# path) when run in-tree, or fetches it from git when copied out.
build_example() {
    example_bin="$here/build/$preset"
    cmake -S "$here" -B "$example_bin" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$example_bin"
}
