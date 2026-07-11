#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// TEMPLATE — action per-site formula (shared CPU + CUDA).  COPY, don't edit here.
//
//   1. Copy to  include/reticolo/action/nn/formula/<name>_formula.hpp
//      renaming  myaction_* → <name>_* .
//   2. Fill both `// FILL IN` bodies with your per-site physics.
//   3. Delete the `#error` line below.
//   4. It gets included by your leaf (templates/action/template_action.hpp).
//
// This file is the SINGLE SOURCE OF TRUTH for the per-site math — the CPU action
// (the NNAction leaf) and any CUDA device functor both call into these, so host
// and device run bit-identical arithmetic. Keep it RETICOLO_HD and free of any
// loop / lattice / field types: just scalar `T` in, scalar `T` out.
//
// WHY A SEPARATE FILE? Only so a CUDA device functor can #include this math ALONE
// (RETICOLO_HD, no host/lattice deps) and stay bit-identical to the CPU. If you
// are CPU-ONLY and never porting to GPU, you don't need this file — write the
// math directly in the leaf's force_kernel / action_kernel lambdas
// (templates/action/template_action.hpp) and drop the `formula::` forwarding.
//
// Mirrors include/reticolo/action/nn/formula/phi4_formula.hpp — open it next to
// this file; the worked examples below ARE that file's contents.
// ═══════════════════════════════════════════════════════════════════════════

#error "template: fill in the FILL IN sections, then delete this #error line."

#include <reticolo/core/hd.hpp>

namespace reticolo::action::formula {

// ─────────────────────────────────────────────────────────────────────────────
// THE TWO NEIGHBOUR SUMS — the one non-obvious part; get it right and the rest
// is just algebra. The sweep engine hands each of your two functions a DIFFERENT
// sum over the nearest neighbours:
//
//   • ACTION counts each bond ONCE  → its kernel gets  `fwd`  = Σ φ(x+μ) over the
//     d FORWARD (positive-μ) neighbours only.
//   • FORCE is a derivative, so ∂/∂φ(x) touches every bond from BOTH ends → its
//     kernel gets `nbrs` = Σ φ(x±μ) over ALL 2·d neighbours (+μ and −μ).
//
// Example: a symmetric hopping term  −κ·φ(x)·Σ_{μ>0} φ(x+μ)  contributes
//     to S:      −κ·φ·fwd            (each bond once)
//     to −dS/dφ: +2κ·nbrs           (∂/∂φ hits the bond from both directions)
//
// So force_site MUST equal −∂/∂φ of action_site (with the both-direction sum).
// tests/physics/test_phi4_force_consistency.cpp verifies this by finite
// differences — copy it for your action and it flags a wrong sign/factor at once.
// ─────────────────────────────────────────────────────────────────────────────

// −dS/dφ(x), the force at one site.  Parameters:
//   phi   — φ(x), this site's field value.
//   nbrs  — Σ φ(y) over ALL 2·d nearest neighbours (both +μ and −μ).
//   …     — add your couplings as trailing `T` params; the leaf passes them in.
template <class T>
[[nodiscard]] RETICOLO_HD inline T myaction_force_site(T phi, T nbrs
                                                       /* FILL IN: , T kappa, T lambda, … */) {
    // ── FILL IN ① — return −dS/dφ(x) ─────────────────────────────────────────
    // Worked example — the real Phi4 force, −dS/dφ = 2κ·nbrs − 2φ − 4λ·φ·(φ²−1):
    //     return (T{2} * kappa * nbrs) - (T{2} * phi)
    //          - (T{4} * lambda * phi * ((phi * phi) - T{1}));
    (void)phi;
    (void)nbrs;
    return T{0};  // ← replace with your −dS/dφ(x)
    // ─────────────────────────────────────────────────────────────────────────
}

// S_site(x): this site's contribution to the total action S = Σ_x S_site(x).
//   phi   — φ(x).
//   fwd   — Σ φ(x+μ) over the d FORWARD (positive-μ) neighbours ONLY (bond once).
//   …     — the SAME couplings as the force.
template <class T>
[[nodiscard]] RETICOLO_HD inline T myaction_action_site(T phi, T fwd
                                                        /* FILL IN: , T kappa, T lambda, … */) {
    // ── FILL IN ② — return S_site(x) ─────────────────────────────────────────
    // Worked example — the real Phi4 action, S_site = −2κ·φ·fwd + φ² + λ·(φ²−1)²:
    //     T const phi2 = phi * phi;
    //     T const dev  = phi2 - T{1};
    //     return (T{-2} * kappa * phi * fwd) + phi2 + (lambda * dev * dev);
    (void)phi;
    (void)fwd;
    return T{0};  // ← replace with your S_site(x)
    // ─────────────────────────────────────────────────────────────────────────
}

}  // namespace reticolo::action::formula
