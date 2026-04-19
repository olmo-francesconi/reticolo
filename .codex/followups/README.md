# Follow-Up Work

Date: 2026-03-13

This folder tracks post-overhaul follow-up work that came out of the architecture review.

## Structure

- `reviews/`
  - point-in-time review notes and findings
- `workstreams/`
  - one file per concrete improvement area

## Current Priorities

1. Add regression coverage for the new registry/manifest/runtime-metadata layers.
2. Clean up runtime ownership and error propagation around `ReticoloCore` and `reticolo_init()/reticolo_run()`.
3. Tighten manifest-driven registration so precision and alias behavior are explicit.
4. Make storage file-creation semantics consistent and documented.
5. Add CI coverage for the supported preset matrix.
