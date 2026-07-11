#pragma once

#include <cmath>
#include <utility>

namespace reticolo::orch::llr {

// Replica-exchange Metropolis acceptance for the Gaussian-window LLR action
//     S_n(q) = a_n·q + (q − E_n)² / (2·δ_n²)     (q = constraint observable)
// Swapping the two configs (constraint values q_i, q_j) accepts with
//     P_accept = min{ 1, exp(log_p) },
//     log_p = (a_i − a_j)(q_i − q_j)                              [linear tilt]
//           + (q_i − q_j)/2 · [ (q_i+q_j−2E_i)/δ_i²
//                             − (q_i+q_j−2E_j)/δ_j² ]            [window term]
// The window term is exact for the Gaussian window and vanishes only when the
// two replicas share E_n and δ (the degenerate "hard-window" case where the
// linear form alone is exact). Dropping it biases the swap when adjacent
// windows differ by O(δ), which they do by construction.
//
// q is the LLR constraint observable via `Replica::energy()`: S_base in real
// mode A, S_I in complex mode B — the same observable `a` couples to.
//
// On accept, the two Lattice objects are swapped in place. Each replica's
// HMC holds a `Lattice<F>&` bound to its own Lattice member's address;
// std::swap on the Lattice values moves the underlying `std::vector` and
// `std::shared_ptr<Indexing>` (both O(1)) without changing those addresses,
// so the HMC references stay valid.
//
// The app owns the sweep loop (even/odd alternation, parity choice, etc.).
template <class Replica, class Rng>
[[nodiscard]] bool try_exchange(Replica& ri, Replica& rj, Rng& rng) {
    auto const q_i    = ri.energy();
    auto const q_j    = rj.energy();
    auto const dq     = q_i - q_j;
    auto const q_sum  = q_i + q_j;
    auto const d_i    = ri.delta();
    auto const d_j    = rj.delta();
    auto const linear = (ri.a() - rj.a()) * dq;
    auto const window = (dq / 2) * (((q_sum - (2 * ri.E_n())) / (d_i * d_i)) -
                                    ((q_sum - (2 * rj.E_n())) / (d_j * d_j)));
    auto const log_p  = linear + window;
    bool const accept = (log_p >= 0.0) || (rng.uniform() < std::exp(log_p));
    if (accept) {
        using std::swap;
        swap(ri.field(), rj.field());
        ri.swap_energy_cache(rj);
    }
    return accept;
}

}  // namespace reticolo::orch::llr
