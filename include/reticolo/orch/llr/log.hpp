#pragma once

#include <reticolo/core/log/log.hpp>

#include <cstddef>
#include <string_view>

// LLR-specific logging helpers. Sits next to <reticolo/core/log/log_helpers.hpp>
// so the core logger stays domain-agnostic — anything that knows about
// Robbins-Monro / Newton-Raphson sweeps lives here.

namespace reticolo::orch::llr {

// Per-iteration row for an LLR phase (NR warm-up or RM sweep). Identical schema
// across every LLR app — phase is the only non-numeric column.
//
// `acc` is the sampler's acceptance over THIS iteration's trajectories. Under
// HMC it is the knob n_md is tuned against: the windowed action carries a stiff
// (Q−E_n)²/2δ² term on top of the base action, so a trajectory length that is
// fine for the plain action can be far past the integrator's stability limit
// here. Without it in the row there is no way to size n_md from a run's own logs.
// NOLINTNEXTLINE(readability-identifier-naming, bugprone-easily-swappable-parameters)
inline void iter(std::string_view phase,
                 std::size_t k,
                 std::size_t n_iters,
                 double a,
                 double dE,
                 double delta,
                 double acc = -1.0) {
    double const ratio = (delta != 0.0) ? (dE / delta) : 0.0;
    if (acc < 0.0) {
        log::info("repl",
                  "{}  {:>3}/{}  a={:+.3f}  ⟨dE⟩={:+.3e}  ⟨dE⟩/δ={:+.3f}",
                  phase,
                  k,
                  n_iters,
                  a,
                  dE,
                  ratio);
        return;
    }
    log::info("repl",
              "{}  {:>3}/{}  a={:+.3f}  ⟨dE⟩={:+.3e}  ⟨dE⟩/δ={:+.3f}  acc={:.3f}",
              phase,
              k,
              n_iters,
              a,
              dE,
              ratio,
              acc);
}

}  // namespace reticolo::orch::llr
