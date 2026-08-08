#pragma once

#include <reticolo/core/exec/parallel.hpp>
#include <reticolo/core/field/lattice.hpp>

#include <algorithm>
#include <array>
#include <cstddef>

// Checkerboard (red/black) traversal — the LOCAL-update member of the same
// traversal family as nn_stencil's `nn_visit` / `nn_reduce`, and used the same
// way: an ACTION FAMILY drives it to evaluate its kernels over the lattice.
// Nothing above the family layer sees it.
//
// It is a separate nest rather than a Policy on nn_stencil because three things
// differ at the loop level, each fatal to reuse:
//
//   * the inner dim-0 loop runs at STRIDE 2, phased per row by the parity of the
//     row's higher coordinates — nn_stencil's peeled literal-bound x-sweep exists
//     precisely to keep that loop unit-stride and branch-free, and is not worth
//     perturbing for a loop that cannot vectorise anyway;
//   * the body WRITES the element it reads, which nn_visit's write-disjoint
//     contract forbids;
//   * the body needs the neighbour fold DEFERRED, not a pre-summed aggregate. A
//     local move changes `self`, and a bond combine reads `self` — so the two
//     candidate values genuinely fold to different sums.
//
// Race-freedom is the bipartite argument: every neighbour of a site is of the
// opposite colour, so one colour's sweep reads only frozen data. It needs every
// extent EVEN — an odd extent wraps a colour onto itself under periodic BCs.
// `checkerboard_ok` is the check; this nest assumes it.
//
// Work items are the field's CANONICAL partition, the same items every other
// per-site pass uses, folded in canonical item order — so the result is
// deterministic for a fixed (team, slabs) config, exactly like `nn_reduce`.

namespace reticolo::exec {

// Reduce `body(i, self, gather) -> Acc` over the sites of one colour, where
//
//     gather(fn)   calls   fn(mu, nbr_value)   for each of the 2·ndims neighbours
//
// The body owns the write to the lattice (that is the point — this is the local
// update) and returns its per-site contribution to the fold.
//
// The neighbours are delivered one at a time WITH their direction rather than
// pre-summed, for two reasons. A bond combine reads the candidate value, so the
// two candidates of a local move genuinely fold to different sums; and an
// anisotropic action (complex/, whose last direction carries a different weight)
// needs to separate the last direction from the rest. It is also the same shape
// as the device functor protocol's `accumulate(mu, nbr)`, so the CPU and GPU
// local-update paths consume neighbours identically.
template <class Acc, class T, class Body>
[[nodiscard]] inline Acc nn_color_reduce(Lattice<T>& l, int color, Body const& body) noexcept {
    T* const data        = l.data();
    auto const& sh       = l.shape();
    std::size_t const d  = l.ndims();
    std::size_t const l0 = sh[0];

    std::array<std::size_t, 4> stride{};
    stride[0] = 1;
    for (std::size_t mu = 1; mu < d; ++mu) {
        stride[mu] = stride[mu - 1] * sh[mu - 1];
    }

    Partition const p  = partition(l);
    int const nthreads = traverse_threads(p.n_sites, l.bytes_per_site());
    note_slicing(sh.data(), d, l.bytes_per_site(), p, nthreads);

    return parallel_reduce<Acc>(nthreads, p.n_items, [&](std::size_t it) {
        std::size_t const base = it * p.item_sites;
        std::size_t const cnt  = std::min(p.item_sites, p.n_sites - base);
        Acc acc{};

        if (d == 1) {
            // 1D: the partition hands out flat chunks, not whole rows, so parity
            // is the coordinate itself and the chunk carries its own wrap ends.
            std::size_t const x0 = base + ((base ^ static_cast<std::size_t>(color)) & 1U);
            for (std::size_t x = x0; x < base + cnt; x += 2) {
                std::size_t const xp = (x + 1 == l0) ? 0 : (x + 1);
                std::size_t const xm = (x == 0) ? (l0 - 1) : (x - 1);
                auto const gather    = [&](auto const& fn) {
                    fn(std::size_t{0}, data[xp]);
                    fn(std::size_t{0}, data[xm]);
                };
                acc += body(x, data[x], gather);
            }
            return acc;
        }

        // d ≥ 2: an item is a contiguous span of WHOLE rows — the partition splits
        // at split_dim ≥ 1 with every dim below it full, so item_sites is a
        // multiple of l0 and the row walk is exact.
        for (std::size_t row = base; row < base + cnt; row += l0) {
            std::size_t par = 0;
            std::array<std::size_t, 4> rp{};
            std::array<std::size_t, 4> rm{};
            for (std::size_t mu = 1; mu < d; ++mu) {
                std::size_t const c = (row / stride[mu]) % sh[mu];
                par += c;
                rp[mu] =
                    (c + 1 == sh[mu]) ? (row - ((sh[mu] - 1) * stride[mu])) : (row + stride[mu]);
                rm[mu] = (c == 0) ? (row + ((sh[mu] - 1) * stride[mu])) : (row - stride[mu]);
            }
            // site parity = (x + Σ_{mu≥1} c_mu) & 1 — solve for the first x of
            // this colour, then stride 2.
            std::size_t const x0 = (par + static_cast<std::size_t>(color)) & 1U;
            for (std::size_t x = x0; x < l0; x += 2) {
                std::size_t const i  = row + x;
                std::size_t const xp = (x + 1 == l0) ? row : (i + 1);
                std::size_t const xm = (x == 0) ? (row + l0 - 1) : (i - 1);
                auto const gather    = [&](auto const& fn) {
                    fn(std::size_t{0}, data[xp]);
                    fn(std::size_t{0}, data[xm]);
                    for (std::size_t mu = 1; mu < d; ++mu) {
                        fn(mu, data[rp[mu] + x]);
                        fn(mu, data[rm[mu] + x]);
                    }
                };
                acc += body(i, data[i], gather);
            }
        }
        return acc;
    });
}

// Reduce `body(i) -> Acc` over the SITE INDICES of one colour, with no neighbour
// gather. The link families use this: a gauge staple reaches sites two hops away
// (x+μ̂−ν̂) and in other directions, so the nearest-neighbour gather above is the
// wrong shape and the caller walks its own geometry through `next`/`prev`. Same
// partition, same canonical fold order, same determinism.
//
// `Field` is any field with the site geometry (a Lattice or a MatrixLinkLattice);
// the colour is over SITES, so a link field's own colouring folds the direction
// in at the caller (see action::formula::wilson_metropolis_stencil).
template <class Acc, class Field, class Body>
[[nodiscard]] inline Acc color_index_reduce(Field const& f, int color, Body const& body) noexcept {
    auto const& sh       = f.shape();
    std::size_t const d  = f.ndims();
    std::size_t const l0 = sh[0];

    std::array<std::size_t, 4> stride{};
    stride[0] = 1;
    for (std::size_t mu = 1; mu < d; ++mu) {
        stride[mu] = stride[mu - 1] * sh[mu - 1];
    }

    Partition const p  = partition(f);
    int const nthreads = traverse_threads(p.n_sites, f.bytes_per_site());

    return parallel_reduce<Acc>(nthreads, p.n_items, [&](std::size_t it) {
        std::size_t const base = it * p.item_sites;
        std::size_t const cnt  = std::min(p.item_sites, p.n_sites - base);
        Acc acc{};
        if (d == 1) {
            std::size_t const x0 = base + ((base ^ static_cast<std::size_t>(color)) & 1U);
            for (std::size_t x = x0; x < base + cnt; x += 2) {
                acc += body(x);
            }
            return acc;
        }
        for (std::size_t row = base; row < base + cnt; row += l0) {
            std::size_t par = 0;
            for (std::size_t mu = 1; mu < d; ++mu) {
                par += (row / stride[mu]) % sh[mu];
            }
            std::size_t const x0 = (par + static_cast<std::size_t>(color)) & 1U;
            for (std::size_t x = x0; x < l0; x += 2) {
                acc += body(row + x);
            }
        }
        return acc;
    });
}

// Every extent even — the precondition for the bipartite argument above.
template <class Field>
[[nodiscard]] inline bool checkerboard_ok(Field const& l) noexcept {
    for (std::size_t mu = 0; mu < l.ndims(); ++mu) {
        if (l.shape()[mu] % 2 != 0) {
            return false;
        }
    }
    return true;
}

}  // namespace reticolo::exec
