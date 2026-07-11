#pragma once

#include <reticolo/core/hd.hpp>

// LLR window formula — the single source of truth shared by the CPU
// `action::WindowedAction` and the CUDA device kernels (cuda/llr/windowed_action.cuh).
// Annotated RETICOLO_HD so the SAME expression compiles for both backends; the
// CPU/device windowed math cannot silently diverge (same discipline as the
// per-site action formulas). Mode A constrains the base action S directly; mode B
// (complex actions, e.g. BoseGas) samples the real part S_R and constrains the
// imaginary observable q = S_I.

namespace reticolo::action::formula {

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

// --- replica exchange ---------------------------------------------------------

// Acceptance exponent for swapping two Gaussian-window replicas i, j with tilts
// a_*, window centres e_n_*, widths delta_*, and constraint values q_* — the
// swap accepts with min(1, exp(log_p)). Linear-tilt term + the window quadratic
// cross-term; the latter is exact for the Gaussian window and vanishes only when
// both replicas share E_n and delta. Shared by the CPU config-swap
// (orch::llr::try_exchange) and the CUDA param-swap driver so the two backends
// cannot drift on this subtle acceptance.
template <class R>
[[nodiscard]] RETICOLO_HD R
exchange_log_p(R a_i, R a_j, R e_n_i, R e_n_j, R q_i, R q_j, R delta_i, R delta_j) noexcept {
    R const dq   = q_i - q_j;
    R const qsum = q_i + q_j;
    R const lin  = (a_i - a_j) * dq;
    R const win  = (dq / R{2}) * (((qsum - (R{2} * e_n_i)) / (delta_i * delta_i)) -
                                  ((qsum - (R{2} * e_n_j)) / (delta_j * delta_j)));
    return lin + win;
}

}  // namespace reticolo::action::formula
