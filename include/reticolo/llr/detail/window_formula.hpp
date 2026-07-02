#pragma once

#include <reticolo/core/hd.hpp>

// LLR window formula — the single source of truth shared by the CPU
// `llr::WindowedAction` and the CUDA device kernels (cuda/llr/windowed_action.cuh).
// Annotated RETICOLO_HD so the SAME expression compiles for both backends; the
// CPU/device windowed math cannot silently diverge (same discipline as the
// per-site action formulas). Mode A constrains the base action S directly; mode B
// (complex actions, e.g. BoseGas) samples the real part S_R and constrains the
// imaginary observable q = S_I.

namespace reticolo::llr::detail {

// --- mode A (real LLR): the window constrains the base action S ---------------

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

// --- mode B (complex LLR): sample S_R, constrain q = S_I ----------------------

// S_LLR = S_R + a * q + (q - e_n)^2 / (2 * delta^2)
template <class R>
[[nodiscard]] RETICOLO_HD R windowed_value_complex(R s_r, R q, R a, R e_n, R delta) noexcept {
    R const d = q - e_n;
    return s_r + (a * q) + ((d * d) / (R{2} * delta * delta));
}

// The multiplier on F_I in F_LLR = F_R + scale * F_I: scale = a + (q - e_n)/delta^2.
template <class R>
[[nodiscard]] RETICOLO_HD R force_scale_imag(R q, R a, R e_n, R delta) noexcept {
    return a + ((q - e_n) / (delta * delta));
}

}  // namespace reticolo::llr::detail
