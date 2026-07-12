#pragma once

#include <reticolo/core/field/field_traits.hpp>
#include <reticolo/core/field/matrix_link_lattice.hpp>
#include <reticolo/core/exec/parallel.hpp>

#include <cstddef>

namespace reticolo::updater::integ {

// Field-generic integrator atoms — the elementwise drift `field += c·mom`
// and the additive kick `mom += k·force`. Factored out of the integrator
// bodies so they can be overloaded on the field type without touching
// Leapfrog/Omelyan2/Omelyan4 themselves.
//
// The generic templates cover the scalar/complex site field `Lattice<T>`,
// which is always site-major (`flat_size == nsites`) and so rides the
// canonical field partition directly. Matrix-group link fields
// (`MatrixLinkLattice<G, T>`) take the overloads below: `drift_field` does the
// group-exponential update `U ← exp(i·dt·P)·U`, and `kick_add` walks the
// per-direction component slabs on the SAME site partition so the momentum
// buffer keeps the thread↔slab ownership every other pass uses.

// The MD time step is always a real scalar even when the field is complex —
// using `real_scalar_t<F>` for the coefficient turns `c * p[i]` into a
// real×complex (2 mults, no inf/NaN-recovery branch) instead of a full
// complex×complex, which also unblocks vectorisation. `__restrict` on the
// output tells the vectoriser the store buffer can't alias the input (field,
// mom, and force are always distinct sibling lattices).
template <class Field, class Mom>
inline void drift_field(Field& field, Mom const& mom, double cdt) noexcept {
    using F              = Field::value_type;
    auto const c         = static_cast<real_scalar_t<F>>(cdt);
    F* const fd          = field.data();
    auto const* const pd = mom.data();
    // Elementwise axpy on the canonical field partition — each element is
    // independent → bit-identical for any partition / thread count. __restrict
    // is re-applied inside the worker so the vectoriser keeps the no-alias
    // guarantee (field, mom are distinct sibling lattices).
    reticolo::exec::field_visit(field, 1, [fd, pd, c](std::size_t base, std::size_t cnt) {
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
    using F              = Mom::value_type;
    auto const c         = static_cast<real_scalar_t<F>>(kdt);
    F* const md          = mom.data();
    auto const* const fd = force.data();
    reticolo::exec::field_visit(mom, 1, [md, fd, c](std::size_t base, std::size_t cnt) {
        F* __restrict const m           = md;
        auto const* __restrict const fp = fd;
        std::size_t const end           = base + cnt;
        for (std::size_t i = base; i < end; ++i) {
            m[i] += c * fp[i];
        }
    });
}

// Matrix-link drift overload: U ← exp(dt·P)·U per direction, dispatched through
// the group model's `expi_lmul_range` so SU(2)/SU(3)/U(1) all reuse the same
// per-direction worker. Compute-bound matrix-exponential drift → worksplit through
// the shared field partition (contiguous row-aligned items); each item runs
// expi_lmul_range over its own site range and handles the CH batch/scalar tail
// itself, so the result is bit-identical to a serial sweep for any thread count.
// Own fork/join, per-op model.
template <class G, class T>
inline void drift_field(MatrixLinkLattice<G, T>& field,
                        MatrixLinkLattice<G, T> const& mom,
                        double cdt) noexcept {
    std::size_t const d    = field.ndims();
    std::size_t const span = field.link_span();  // padded component stride
    // gran = pass-5 k_b=8 batch, kept for the kernel's tail handling; the field
    // partition itself is row-aligned (whole inner-dim products), not gran-aligned.
    reticolo::exec::field_visit(field, 8, [&](std::size_t base, std::size_t cnt) {
        for (std::size_t mu = 0; mu < d; ++mu) {
            G::expi_lmul_range(
                field.mu_block_data(mu), mom.mu_block_data(mu), cdt, span, base, cnt);
        }
    });
}

// Matrix-link kick overload: the algebra is a vector space, so the momentum
// update is a plain additive axpy over the real component storage. It walks the
// SAME canonical site partition as the drift (each of the d·nc component slabs
// is a stride-1 nsites array, so the site range [base, cnt) maps to a contiguous
// span in every slab), keeping the momentum buffer's thread↔slab ownership
// aligned with every other pass. Elementwise-independent → bit-identical for any
// thread count.
template <class G, class T>
inline void
kick_add(MatrixLinkLattice<G, T>& mom, MatrixLinkLattice<G, T> const& force, double kdt) noexcept {
    constexpr std::size_t nc = MatrixLinkLattice<G, T>::n_real_components;
    std::size_t const nblk   = mom.ndims() * nc;
    std::size_t const span   = mom.link_span();  // padded component stride
    T const c                = static_cast<T>(kdt);
    T* const md              = mom.data();
    T const* const fd        = force.data();
    reticolo::exec::field_visit(mom, 1, [md, fd, c, nblk, span](std::size_t base, std::size_t cnt) {
        for (std::size_t blk = 0; blk < nblk; ++blk) {
            T* __restrict const m        = md + (blk * span) + base;
            T const* __restrict const fp = fd + (blk * span) + base;
            for (std::size_t i = 0; i < cnt; ++i) {
                m[i] += c * fp[i];
            }
        }
    });
}

}  // namespace reticolo::updater::integ
