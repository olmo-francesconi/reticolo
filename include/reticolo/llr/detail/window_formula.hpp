#pragma once

#include <reticolo/core/hd.hpp>

// Mode-A LLR window formula — the single source of truth shared by the CPU
// `llr::WindowedAction` and the CUDA device kernels (cuda/llr/windowed_action.cuh).
// Annotated RETICOLO_HD so the SAME expression compiles for both backends; the
// CPU/device windowed math cannot silently diverge (same discipline as the
// per-site action formulas). Real-valued only (mode A); the complex mode B path
// stays inlined in the CPU WindowedAction until it is ported.

namespace reticolo::llr::detail {

// S_LLR = (1 + a) * S + (S - e_n)^2 / (2 * delta^2)
template <class R>
[[nodiscard]] RETICOLO_HD R windowed_value(R s, R a, R e_n, R delta) noexcept {
    R const d = s - e_n;
    return ((R{1} + a) * s) + ((d * d) / (R{2} * delta * delta));
}

// d S_LLR / dS = (1 + a) + (S - e_n) / delta^2 — the multiplier on F_base.
template <class R>
[[nodiscard]] RETICOLO_HD R force_scale(R s, R a, R e_n, R delta) noexcept {
    return (R{1} + a) + ((s - e_n) / (delta * delta));
}

}  // namespace reticolo::llr::detail
