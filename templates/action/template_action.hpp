#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// TEMPLATE — a nearest-neighbour scalar action leaf.  COPY, don't edit here.
//
//   1. Copy to  include/reticolo/action/nn/<name>.hpp   (rename MyAction → …).
//   2. Copy templates/action/template_formula.hpp alongside and #include it.
//      (CPU-ONLY, no GPU port planned? Skip the formula file and write the math
//       directly in the force_kernel / action_kernel lambdas below — the
//       separate formula only buys bit-identical CUDA sharing. See that file.)
//   3. Fill every `// FILL IN` section.
//   4. Delete the `#error` line below.
//   5. Register:  add `#include <reticolo/action/nn/<name>.hpp>` to
//      include/reticolo/action/nn.hpp — then it's usable as `act::MyAction`.
//   6. Smoke it with a force-vs-finite-difference test (see
//      tests/physics/test_phi4_force_consistency.cpp) and, if useful, an app
//      (copy apps/phi4_hmc.cpp).
//
// A leaf is a designated-init aggregate: just the couplings + describe() + two
// kernels that bind the shared formula. `NNAction` supplies s_full /
// compute_force / caching / the LLR fast path through the parallel sweep engine.
//
// A "kernel" here is a member returning a LAMBDA that captures your couplings BY
// VALUE — the sweep engine calls that lambda once per site, and the by-value
// capture lets it inline the couplings into the vectorised hot loop exactly as a
// hand-written action would. You write the tiny lambda body; it just forwards to
// your formula function.
//
// For a COMPLEX (sign-problem) or GAUGE action, start from a real leaf in
// action/complex/*.hpp or action/gauge/*.hpp instead — same 3-step shape.
//
// Mirrors include/reticolo/action/nn/phi4.hpp — open it next to this file; the
// worked examples below ARE that file's contents.
// ═══════════════════════════════════════════════════════════════════════════

#error "template: fill in the FILL IN sections, then delete this #error line."

#include <reticolo/action/nn/formula/myaction_formula.hpp>  // FILL IN ⓪: your formula
#include <reticolo/action/nn/nn_action.hpp>
#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/log.hpp>

#include <cstddef>

namespace reticolo::action {

// One-line description of the action + its S formula here.
template <class T = double>
struct MyAction : NNAction<MyAction<T>, T> {
    using value_type = T;

    // ── FILL IN ① — your couplings (plain members; the leaf is a designated- ──
    //   init aggregate, so callers still write `MyAction{.kappa=…, .lambda=…}`).
    //   Worked example (Phi4 has two):   T kappa = T{0};   T lambda = T{0};
    T coupling = T{0};  // ← replace with your coupling(s)
    // ──────────────────────────────────────────────────────────────────────────

    void describe(log::Entry& e) const {
        e.line("MyAction<{}>", scalar_name<T>());
        // ── FILL IN ② — one e.param() per coupling (shown in the run banner) ──
        //   Worked example (Phi4):
        //       e.param("κ={:.3f}", kappa);
        //       e.param("λ={:.3f}", lambda);
        e.param("c={:.3f}", coupling);
        // ────────────────────────────────────────────────────────────────────────
    }

    // force_kernel(): return a lambda the sweep calls at EVERY site for −dS/dφ.
    // Capture your couplings by value (`[k = kappa, …]`). The lambda's arguments
    // are fixed by the engine:
    //   std::size_t i  — flat site index (usually unused; keep it /*i*/).
    //   T           phi  — φ at this site.
    //   T           nbrs — Σ φ over ALL 2·d neighbours (see the formula header).
    // The body just forwards to your formula's force function.
    [[nodiscard]] auto force_kernel() const noexcept {
        // ── FILL IN ③ — bind your formula's force ────────────────────────────
        // Worked example (Phi4, capturing κ and λ):
        //     return [k = kappa, lam = lambda](std::size_t /*i*/, T phi, T nbrs) {
        //         return formula::phi4_force_site<T>(phi, nbrs, k, lam);
        //     };
        return [c = coupling](std::size_t /*i*/, T phi, T nbrs) {
            return formula::myaction_force_site<T>(phi, nbrs /*, c*/);
        };
        // ──────────────────────────────────────────────────────────────────────
    }

    // action_kernel(): return a lambda the sweep calls at every site for S_site.
    // Capture couplings by value, same as force_kernel. Arguments (fixed):
    //   T phi — φ at this site.
    //   T fwd — Σ φ over the d FORWARD (positive-μ) neighbours ONLY.
    [[nodiscard]] auto action_kernel() const noexcept {
        // ── FILL IN ④ — bind your formula's action ───────────────────────────
        // Worked example (Phi4):
        //     return [k = kappa, lam = lambda](T phi, T fwd) {
        //         return formula::phi4_action_site<T>(phi, fwd, k, lam);
        //     };
        return [c = coupling](T phi, T fwd) {
            return formula::myaction_action_site<T>(phi, fwd /*, c*/);
        };
        // ──────────────────────────────────────────────────────────────────────
    }

    // OPTIONAL, delete if not needed. Defaults (in NNAction) cover the common
    // "self + neighbour-sum" case (Phi4/Phi6/SineGordon). Override ONLY for:
    //   auto action_combine() const;  // per-bond COMBINE, e.g. XY's cos(self−nbr)
    //   auto force_combine()  const;  // per-bond COMBINE for the force
    //   T    action_scale()   const;  // total-S prefactor applied post-reduce
    //   void prep(Lattice<T> const&) const;  // fill scratch() before a force pass
    //   double s_full_and_force(Lattice<T> const&, Lattice<T>&) const;  // LLR fused
    //          fast path — see phi4.hpp's staged_force_energy use.
};

}  // namespace reticolo::action
