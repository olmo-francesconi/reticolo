# Shared preset prelude for example sweeps. Source me as:
#     source "$(dirname "${BASH_SOURCE[0]}")/../_common/preset.sh" "$@"
# Sets: $here, $root, $preset, $build_dir. Consumes --preset / --preset=X / -h.
# After sourcing, $@ holds the remaining (non-preset) args; callers can `shift`
# past them or just use the parsed $preset directly.

here=$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)
root=$(cd "$here/../.." && pwd)
preset=${RETICOLO_PRESET:-macos-appleclang}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)   preset="$2";      shift 2 ;;
        --preset=*) preset="${1#*=}"; shift   ;;
        -h|--help)
            echo "Usage: $(basename "$0") [--preset <name>]"
            echo "  --preset <name>   CMake build preset whose apps/ dir holds the"
            echo "                    binaries (default: \$RETICOLO_PRESET or"
            echo "                    macos-appleclang). Use macos-llvm for OpenMP."
            exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done
build_dir="$root/build/$preset"
