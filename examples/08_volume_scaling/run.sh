#!/usr/bin/env bash
# Volume-scaling sweep of `s_full` and `compute_force` for the four
# representative actions: phi4, compact_u1, wilson_su2, wilson_su3.
#
# Single invocation of the bench binary walks the full (ndim, L) grid.
# Each cell runs each kernel under a twin budget (max dof updates OR max
# seconds, whichever hits first), so noise is comparable across volumes
# and the SU(3)-4D-L=16 tail can't run away. Output is a CSV consumed by
# analyze.py.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../_common/preset.sh" "$@"

build_example
bin="$example_bin/bench_volume_scaling"

results="$here/results"
mkdir -p "$results"
csv="$results/scaling.csv"
rm -f "$csv"

ndims=${NDIMS:-2,3,4}
sizes=${SIZES:-4,6,8,12,16}
actions=${ACTIONS:-phi4,compact_u1,wilson_su2,wilson_su3}
budget_dofs=${BUDGET_DOFS:-2e9}
budget_seconds=${BUDGET_SECONDS:-2.0}
seed=${SEED:-42}

printf '[%s] sweeping ndims=%s sizes=%s actions=%s\n' \
    "$(date +%H:%M:%S)" "$ndims" "$sizes" "$actions"

"$bin" \
    --ndims="$ndims" \
    --sizes="$sizes" \
    --actions="$actions" \
    --budget_dofs="$budget_dofs" \
    --budget_seconds="$budget_seconds" \
    --seed="$seed" \
    --out="$csv"

printf '[%s] wrote %s\n' "$(date +%H:%M:%S)" "$csv"

if command -v python3 >/dev/null 2>&1; then
    python3 "$here/analyze.py" "$csv" "$results/scaling.pdf"
    printf '[%s] wrote %s\n' "$(date +%H:%M:%S)" "$results/scaling.pdf"
else
    echo "python3 not found — skipping plot" >&2
fi
