#pragma once

#include <reticolo/core/log.hpp>

#include <cstddef>
#include <string_view>

// LLR-specific logging helpers. Sits next to <reticolo/core/log_helpers.hpp>
// so the core logger stays domain-agnostic — anything that knows about
// Robbins-Monro / Newton-Raphson sweeps lives here.

namespace reticolo::llr {

// Per-iteration row for an LLR phase (NR warm-up or RM sweep). Identical
// 6-field schema across every LLR app — phase is the only non-numeric column.
// NOLINTNEXTLINE(readability-identifier-naming) physics convention: dE = <E - E_n>
inline void iter(
    std::string_view phase, std::size_t k, std::size_t n_iters, double a, double dE, double delta) {
    double const ratio = (delta != 0.0) ? (dE / delta) : 0.0;
    log::info("repl",
              "{}  {:>3}/{}  a={:+.3f}  ⟨dE⟩={:+.3e}  ⟨dE⟩/δ={:+.3f}",
              phase,
              k,
              n_iters,
              a,
              dE,
              ratio);
}

}  // namespace reticolo::llr
