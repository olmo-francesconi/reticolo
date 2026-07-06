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
    using F                        = typename Field::value_type;
    F* __restrict const f          = field.data();
    auto const* __restrict const p = mom.data();
    std::size_t const n            = flat_size(field);
    real_scalar_t<F> const c       = static_cast<real_scalar_t<F>>(cdt);
    // Worksplit across the enclosing per-trajectory region (Model B); serial when
    // called outside one. Kept lambda-free so __restrict survives for the vectoriser.
    if (reticolo::detail::g_in_traverse_region) {
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            f[i] += c * p[i];
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            f[i] += c * p[i];
        }
    }
}

template <class Mom, class Force>
inline void kick_add(Mom& mom, Force const& force, double kdt) noexcept {
    using F                         = typename Mom::value_type;
    F* __restrict const m           = mom.data();
    auto const* __restrict const fp = force.data();
    std::size_t const n             = flat_size(mom);
    real_scalar_t<F> const c        = static_cast<real_scalar_t<F>>(kdt);
    if (reticolo::detail::g_in_traverse_region) {
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < n; ++i) {
            m[i] += c * fp[i];
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            m[i] += c * fp[i];
        }
    }
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
    for (std::size_t mu = 0; mu < d; ++mu) {
        G::expi_lmul_slab(field.mu_block_data(mu), mom.mu_block_data(mu), cdt, ns);
    }
}

}  // namespace reticolo::alg::integ
