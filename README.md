# reticolo

Lightweight C++23 library for **serial** Monte Carlo simulations of
scalar quantum field theories on a lattice.

The design priorities, in order, are:

1. **Small public surface.** A user writes a hand-rolled `main()` that
   wires a `Lattice`, an action struct, an updater, and an HDF5 writer.
   ~30–60 lines, no framework.
2. **Performance at the core.** No virtual dispatch in the hot loop;
   action methods inline into the templated updater. Indexing tables
   are shared across lattices of the same shape.
3. **One way to do each thing.** One writer, one CLI helper, one
   logger. No central registries.

Scope (deliberate): serial only, scalar fields only. No OpenMP, no MPI,
no gauge theories. The architecture leaves room for any of these later
without renegotiating the public surface — see
[`proposals/rewrite_plan_v3.md`](proposals/rewrite_plan_v3.md).

## Status

Work-in-progress rewrite on `rewrite/v3`. Milestone M0 — scaffold,
tooling, CI — is in flight. No usable library code yet.

## Build

```sh
cmake --preset macos-appleclang
cmake --build --preset macos-appleclang
ctest --preset macos-appleclang        # once tests land at M1
```

Available presets: `macos-appleclang`, `macos-llvm`, `linux-gcc`,
`linux-clang`, `debug` (asan + ubsan).

## Layout

```
include/reticolo/      header-only core (lattice, action concepts, algorithms)
src/io/                single TU that owns libhdf5 (lands at M5)
apps/                  hand-written reference simulations (M7+)
tests/                 physics + unit tests (M1+)
proposals/             design docs
```

## License

TBD.
