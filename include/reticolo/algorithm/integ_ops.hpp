#pragma once

#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/parallel.hpp>

#include <cstddef>

namespace reticolo::alg::integ {

// Field-generic integrator atoms — the elementwise drift `field += c·mom`
// and the additive kick `mom += k·force`. Factored out of the integrator
// bodies so they can be overloaded on the field type without touching
// Leapfrog/Omelyan2/Omelyan4 themselves.
//
// Defaults below cover any field that exposes `data()` + a `value_type` +
// a free `flat_size()` overload — i.e. `Lattice<T>` and `LinkLattice<T>`.
// Matrix-group link fields (`MatrixLinkLattice<G, T>`) override
// `drift_field` with the group-exponential update `U ← exp(i·dt·P)·U`,
// while `kick_add` stays generic because the algebra is a vector space and
// the additive update is correct for the momentum buffer regardless of
// whether the *field* lives on a group.

// The MD time step is always a real scalar even when the field is complex —
// using `real_scalar_t<F>` for the coefficient turns `c * p[i]` into a
// real×complex (2 mults, no inf/NaN-recovery branch) instead of a full
// complex×complex, which also unblocks vectorisation. `__restrict` on the
// output tells the vectoriser the store buffer can't alias the input (field,
// mom, and force are always distinct sibling lattices).
template <class Field, class Mom>
inline void drift_field(Field& field, Mom const& mom, double cdt) noexcept {
    using F                  = typename Field::value_type;
    std::size_t const n      = flat_size(field);
    real_scalar_t<F> const c = static_cast<real_scalar_t<F>>(cdt);
    F* const fd              = field.data();
    auto const* const pd     = mom.data();
    // Elementwise axpy: threshold/chunk from the element size; each element is
    // independent → bit-identical for any partition. __restrict is re-applied
    // inside the worker so the vectoriser keeps the no-alias guarantee.
    reticolo::exec::parallel_map_ranges(
        n, sizeof(F), 1, [fd, pd, c](std::size_t base, std::size_t cnt) {
            F* __restrict const f          = fd;
            auto const* __restrict const p = pd;
            std::size_t const end          = base + cnt;
            for (std::size_t i = base; i < end; ++i) {
                f[i] += c * p[i];
            }
        });
}

template <class Mom, class Force>
inline void kick_add(Mom& mom, Force const& force, double kdt) noexcept {
    using F                  = typename Mom::value_type;
    std::size_t const n      = flat_size(mom);
    real_scalar_t<F> const c = static_cast<real_scalar_t<F>>(kdt);
    F* const md              = mom.data();
    auto const* const fd     = force.data();
    reticolo::exec::parallel_map_ranges(
        n, sizeof(F), 1, [md, fd, c](std::size_t base, std::size_t cnt) {
            F* __restrict const m           = md;
            auto const* __restrict const fp = fd;
            std::size_t const end           = base + cnt;
            for (std::size_t i = base; i < end; ++i) {
                m[i] += c * fp[i];
            }
        });
}

// Matrix-link drift overload: U ← exp(dt·P)·U per direction, dispatched
// through the group model's `expi_lmul_slab` so SU(2)/SU(3)/U(1) all reuse
// the same per-direction loop.
template <class G, class T>
inline void drift_field(MatrixLinkLattice<G, T>& field,
                        MatrixLinkLattice<G, T> const& mom,
                        double cdt) noexcept {
    std::size_t const d  = field.ndims();
    std::size_t const ns = field.nsites();
    // Compute-bound matrix-exponential drift → worksplit like the staple force,
    // when the group exposes a pure per-range worker (SU(3)). Each thread applies
    // it to an 8-aligned site chunk (matching the group's pass-5 batch), so the
    // result is bit-identical to the serial slab for any thread count. Own
    // fork/join, per-op model. Groups without a range worker stay serial.
    if constexpr (requires(T* uu, T const* pp) {
                      G::expi_lmul_range(
                          uu, pp, double{}, std::size_t{}, std::size_t{}, std::size_t{});
                  }) {
        // gran = pass-5 k_b=8 batch so chunks stay batch-aligned; threshold/chunk
        // from the (large) gauge footprint, bit-identical to the serial slab per
        // chunk.
        reticolo::exec::field_visit(field, 8, [&](std::size_t base, std::size_t cnt) {
            for (std::size_t mu = 0; mu < d; ++mu) {
                G::expi_lmul_range(
                    field.mu_block_data(mu), mom.mu_block_data(mu), cdt, ns, base, cnt);
            }
        });
    } else {
        for (std::size_t mu = 0; mu < d; ++mu) {
            G::expi_lmul_slab(field.mu_block_data(mu), mom.mu_block_data(mu), cdt, ns);
        }
    }
}

}  // namespace reticolo::alg::integ
