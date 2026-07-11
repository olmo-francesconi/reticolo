#pragma once

#include <concepts>

namespace reticolo::alg {

// The contract every update algorithm satisfies — the surface that apps and the
// orchestration layer (orch::llr::Replica, orch::span::Chain) depend on. It is
// the updater-level analogue of action::HmcAction: duck-typed, no base class,
// checked at the use site. `alg::Hmc` is the sole implementation today; a new
// updater is a class modelling this concept (reusing the same integrator
// type-parameters), and the orchestrators pick it up unchanged.
//
// The three members are exactly what a driver needs from a sampler:
//   - step()          run one trajectory, returning {dH, accepted}
//   - last_s_full()   S of the current config after the most recent trajectory
//                     (so a driver reads the action without a fresh O(V) sweep)
//   - rng()           the owned RNG state, for checkpoint / resume
template <class U>
concept Updater = requires(U& u, U const& cu) {
    { u.step().accepted } -> std::convertible_to<bool>;
    { u.step().dH } -> std::convertible_to<double>;
    { cu.last_s_full() } -> std::convertible_to<double>;
    u.rng();
};

}  // namespace reticolo::alg
