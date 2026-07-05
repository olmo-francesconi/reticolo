# 07 — volume scaling

Walks the kernel cost of `s_full` and `compute_force` across `(ndim, L)`
for four representative actions:

| action       | field type                            |
|--------------|---------------------------------------|
| `phi4`       | scalar `Lattice<double>`              |
| `compact_u1` | `MatrixLinkLattice<U1>` (angle per link) |
| `wilson_su2` | `MatrixLinkLattice<SU2>`              |
| `wilson_su3` | `MatrixLinkLattice<SU3>`              |

`phi4` exercises the plain scalar kernels; `compact_u1` exercises the
hand-tuned U(1) link kernels; the two Wilson rows exercise the full
matrix-link gauge path including SU(2)/SU(3) staple sums.

## What `dofs` counts

`dofs` is the number of **real numbers stored in the field**, not
sites or links. This puts the four action types onto a fair common
axis:

| action       | reals stored                                |
|--------------|---------------------------------------------|
| `phi4`       | `nsites`  (1 double per site)               |
| `compact_u1` | `nlinks = ndim·nsites` (1 angle per link)   |
| `wilson_su2` | `8·ndim·nsites` (SU(2) link = 8 reals)      |
| `wilson_su3` | `18·ndim·nsites` (SU(3) link = 18 reals)    |

The throughput column `dof_per_s` is then "real numbers swept per
second" — directly comparable across actions.

## Timing model

Each kernel runs under a **twin budget**: it stops at either
`BUDGET_DOFS` cumulative dof updates (`n_calls × dofs_per_call`) or
`BUDGET_SECONDS` wall time, whichever hits first. Small lattices loop
many times to reach the dof budget; large lattices may run only once or
twice before the time cap fires. Reporting both `n_calls` and `wall_s`
lets the analyser see the sample count behind each timing point.

This replaces the older adaptive-batch best-of-N timer (still used by
`bench_actions`): noise is now comparable across volumes, and the
SU(3)-4D-large-L tail can't run away.

## Building & running

Standalone build:

```sh
cd examples/07_volume_scaling
cmake -S . -B build && cmake --build build
```

Or let `run.sh` handle it automatically:

```sh
examples/07_volume_scaling/run.sh         # or: --preset linux-gcc
```

## Run

Env knobs (all optional):
- `NDIMS=2,3,4`
- `SIZES=4,6,8,12,16`
- `ACTIONS=phi4,compact_u1,wilson_su2,wilson_su3`
- `BUDGET_DOFS=2e9`, `BUDGET_SECONDS=2.0`, `SEED=42`

Output:
- `results/scaling.csv` — one row per `(ndim, L, action, kernel)`
- `results/scaling.pdf` — log-log wall-time vs. dofs, rows = `ndim`,
  cols = `kernel`, one line per action, fitted slope in the legend

## Reading the plots

A perfectly-local kernel touches O(dofs) memory and does O(dofs) flops,
so log-log slope **α ≈ 1**. Slopes above 1 indicate super-linear
behaviour (cache cliffs, allocator warmup, NUMA effects on Linux). The
absolute height ratios between actions are the per-dof cost
difference — Wilson rows sit well above `phi4` because each link
operation hits several SU(N) matmuls.
