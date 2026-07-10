# 05 — Relativistic Bose gas density of states with complex LLR

End-to-end demo of the complex-action stack (`action/complex/` → `BoseGas`) on
the 4D relativistic Bose gas at finite chemical potential μ — the sign-problem
workhorse. The phase-quenched HMC samples `exp(−S_R)`; LLR reconstructs `ρ(S_I)`
of the imaginary action, from which the phase factor `⟨e^{iS_I}⟩` and the
overlap free energy follow.

## What this example does

Two-stage pipeline per μ:

1. **Stage 1 — phase-quenched HMC** (`bose_gas_hmc`). Samples `exp(−S_R(μ))`,
   recording the `S_I` time series per trajectory; its `(lo, hi)` percentiles
   bracket the LLR window range.
2. **Stage 2 — complex LLR** (`bose_gas_llr`) over `[E_min, E_max]` with auto-δ
   for ~30 replicas. Reconstructs `ln ρ(S_I)` in the phase-quenched ensemble.
3. **Diagnostics** (`analyze.py`). Per-μ LLR-vs-HMC overlays of `ln ρ` / `ρ` /
   the converged slopes `a_n`, plus a second figure with `⟨e^{iS_I}⟩` and the
   overlap free-energy density `−ln⟨e^{iS_I}⟩/V` vs μ (LLR vs HMC). LLR errors
   are the spread over independent-seed replica runs. `paper_data_raw/` holds
   reference values.

## Run

```sh
./run.sh          # build (find-or-fetch reticolo), sweep μ, then analyze
```

Like every `examples/` entry this is a standalone consumer project: its
`CMakeLists.txt` links `reticolo::reticolo` via the inline find-or-fetch block.
