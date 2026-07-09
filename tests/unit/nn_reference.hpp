#pragma once

// Bit-exact nearest-neighbour reference for the site-stencil unit tests. Since
// the Indexing neighbour tables are gone, the reference gathers via the COMPUTED
// Indexing::next/prev — same values, same fold order as the vectorised nests
// (per axis mu ascending: +mu then -mu), so the map comparison stays bit-exact.

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/site.hpp>

#include <cstddef>

namespace reticolo::test {

// body(i, phi, nbrs): nbrs = Σ_mu (data[next(i,mu)] + data[prev(i,mu)]).
template <class T, class Body>
inline void visit_nn_ref(Lattice<T> const& l, Body const& body) {
    auto const& idx = l.indexing_ref();
    T const* const d = l.data();
    std::size_t const nd = l.ndims();
    for (std::size_t i = 0; i < l.nsites(); ++i) {
        T nb{0};
        for (std::size_t mu = 0; mu < nd; ++mu) {
            nb += d[idx.next(Site{i}, mu).value()];
            nb += d[idx.prev(Site{i}, mu).value()];
        }
        body(i, d[i], nb);
    }
}

// body(phi, fwd) -> Acc, summed over sites. fwd = Σ_mu data[next(i,mu)].
template <class T, class Acc = T, class Body>
[[nodiscard]] inline Acc reduce_fwd_ref(Lattice<T> const& l, Body const& body) {
    auto const& idx = l.indexing_ref();
    T const* const d = l.data();
    std::size_t const nd = l.ndims();
    Acc total{};
    for (std::size_t i = 0; i < l.nsites(); ++i) {
        T fwd{0};
        for (std::size_t mu = 0; mu < nd; ++mu) {
            fwd += d[idx.next(Site{i}, mu).value()];
        }
        total += body(d[i], fwd);
    }
    return total;
}

}  // namespace reticolo::test
