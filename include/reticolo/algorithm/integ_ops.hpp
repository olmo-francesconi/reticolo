#pragma once

#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>

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

template <class Field, class Mom>
inline void drift_field(Field& field, Mom const& mom, double cdt) noexcept {
    using F             = typename Field::value_type;
    F* const f          = field.data();
    auto const* const p = mom.data();
    std::size_t const n = flat_size(field);
    F const c           = static_cast<F>(cdt);
    for (std::size_t i = 0; i < n; ++i) {
        f[i] += c * p[i];
    }
}

template <class Mom, class Force>
inline void kick_add(Mom& mom, Force const& force, double kdt) noexcept {
    using F              = typename Mom::value_type;
    F* const m           = mom.data();
    auto const* const fp = force.data();
    std::size_t const n  = flat_size(mom);
    F const c            = static_cast<F>(kdt);
    for (std::size_t i = 0; i < n; ++i) {
        m[i] += c * fp[i];
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
