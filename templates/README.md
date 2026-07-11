# templates/

Copy-and-fill scaffolds for the three main extension points of the library. Each
file is a **skeleton**: all the structural boilerplate (includes, class shape,
the members the relevant concept requires, logging, the RNG-ownership invariant)
is already in place, and the parts that are *yours to write* are marked with
`// FILL IN ⓝ` banners.

These files are **not built** — they live outside every CMake target and each
carries an `#error` guard, so they can't compile until you copy one out and fill
it in. They are reference copies to start from, nothing more. When a template
drifts from the real files it mirrors, treat the real file as truth.

## How to use one

1. **Copy** the file(s) to the destination in the table below, renaming
   `template` → `yourname` (and the `My*` identifiers inside).
2. **Fill in** every `// FILL IN ⓝ` section — search the file for `FILL IN`.
3. **Delete** the `#error` line near the top.
4. **Register** it as noted in the file header and the table.

## What's here

| Template | Copy to | Register in |
| --- | --- | --- |
| `action/template_formula.hpp` | `include/reticolo/action/nn/formula/<name>_formula.hpp` | (included by the leaf) |
| `action/template_action.hpp` | `include/reticolo/action/nn/<name>.hpp` | add `#include` to `action/nn.hpp` |
| `updater/template_updater.hpp` | `include/reticolo/updater/<name>/<name>.hpp` | add `#include` to `reticolo.hpp`; apps instantiate it directly |
| `orchestrator/template_worker.hpp` | `include/reticolo/orch/<name>/worker.hpp` | (aggregated by `orch/<name>.hpp`) |
| `orchestrator/template_driver.hpp` | `include/reticolo/orch/<name>/driver.hpp` | new `orch/<name>.hpp` → `#include` in `reticolo.hpp` |

## Which one do I want?

- **Action** — a new physics action (φ⁴-style nearest-neighbour scalar). The
  common case. Two files: the shared per-site `formula/` (host+device) and the
  `NNAction` leaf that binds it. Complex (sign-problem) and gauge actions follow
  the same 3-step shape via `ComplexAction`+`ImagPart` / `Wilson<G>` — start from
  a real `action/complex/*` / `action/gauge/*` leaf instead.
- **Updater** — a new update algorithm *other than HMC* (Langevin, heat-bath,
  overrelaxation, …). One header modelling the `updater::Updater` concept; the
  apps and both orchestrators pick it up unchanged because they only depend on
  that concept.
- **Orchestrator** — a new *concurrent-simulation workflow* (a new way to drive
  an ensemble of workers, like `orch::span` or `orch::llr`). A `Worker` type +
  a two-phase `setup()/run()` orchestrator object over `parallel_workers`.

See `docs/writing_an_app.md` and `docs/architecture.md` for the full picture,
and the real files each template mirrors (named in every header).
