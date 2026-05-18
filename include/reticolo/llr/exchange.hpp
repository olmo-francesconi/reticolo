#pragma once

#include <cmath>
#include <utility>

namespace reticolo::llr {

// =============================================================================
//  Replica-exchange Metropolis acceptance for LLR with E = S_base:
//      P_accept = min{ 1, exp( (a_i - a_j) * (E_i - E_j) ) }
//
//  On accept, the two Lattice objects are swapped in place. Each replica's
//  HMC holds a `Lattice<F>&` bound to its own Lattice member's address;
//  std::swap on the Lattice values moves the underlying `std::vector` and
//  `std::shared_ptr<Indexing>` (both O(1)) without changing those addresses,
//  so the HMC references stay valid.
//
//  The app owns the sweep loop (even/odd alternation, parity choice, etc.).
// =============================================================================
template <class Replica, class Rng>
[[nodiscard]] bool try_exchange(Replica& ri, Replica& rj, Rng& rng) {
    auto const e_i    = ri.energy();
    auto const e_j    = rj.energy();
    auto const log_p  = (ri.a() - rj.a()) * (e_i - e_j);
    bool const accept = (log_p >= 0.0) || (rng.uniform() < std::exp(log_p));
    if (accept) {
        using std::swap;
        swap(ri.phi(), rj.phi());
    }
    return accept;
}

}  // namespace reticolo::llr
