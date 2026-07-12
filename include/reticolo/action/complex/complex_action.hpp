#pragma once

#include <reticolo/action/cache.hpp>
#include <reticolo/action/sweep/complex.hpp>
#include <reticolo/core/field/lattice.hpp>

#include <complex>
#include <cstddef>

// ComplexAction<Derived, T> — the REAL (phase-quenched) part of a complex scalar
// action with a sign problem, i.e. the S_R / F_R that HMC actually samples of a
// total S = S_R + i·S_I. It is an anisotropic nearest-neighbour action: the last
// direction ("time") may carry a different weight (e.g. cosh(mu)), so it runs on
// the split-last drivers that hand the leaf the full neighbour sum AND the
// last-direction sum separately. All physics is in the leaf's kernels, which
// receive the lattice (to read geometry-dependent prefactors like 2d + m²):
//
//   auto action_kernel(l) const;  // (phi, fwd_total, fwd_last) -> T       (S_R site)
//   auto force_kernel(l)  const;  // (i, phi, nbrs_total, nbrs_last) -> complex (F_R)
//
// The IMAGINARY part (S_I, F_I — the LLR constraint observable) is an orthogonal
// extension: the leaf derives from `ImagPart` (action/complex/imag_part.hpp)
// ALONGSIDE this base, mirroring how WindowedAction decorates a base action. So a
// complex leaf is `struct BoseGas : ComplexAction<…>, ImagPart<…>`.

namespace reticolo::action {

template <class Derived, class T>
struct ComplexAction : SFullCache {
    using complex_t = std::complex<T>;

    [[nodiscard]] double s_full(Lattice<complex_t> const& l) const noexcept {
        auto kern             = derived_().action_kernel(l);
        complex_t const total = sweep::reduce_fwd_split_last<complex_t>(
            l, [&kern](complex_t phi, complex_t fwd_total, complex_t fwd_last) {
                return complex_t{kern(phi, fwd_total, fwd_last), T{0}};
            });
        auto const s = static_cast<double>(std::real(total));
        last_s_full_ = s;
        return s;
    }

    void compute_force(Lattice<complex_t> const& l, Lattice<complex_t>& force) const noexcept {
        auto kern            = derived_().force_kernel(l);
        complex_t* const out = force.data();
        sweep::visit_nn_split_last<complex_t>(
            l, [&kern, out](std::size_t i, complex_t phi, complex_t nt, complex_t nl) {
                out[i] = kern(i, phi, nt, nl);
            });
    }

    void compute_force_and_kick(Lattice<complex_t> const& l,
                                Lattice<complex_t>& mom,
                                T k_dt) const noexcept {
        auto kern          = derived_().force_kernel(l);
        complex_t* const m = mom.data();
        sweep::visit_nn_split_last<complex_t>(
            l, [&kern, m, k_dt](std::size_t i, complex_t phi, complex_t nt, complex_t nl) {
                m[i] += k_dt * kern(i, phi, nt, nl);
            });
    }

private:
    [[nodiscard]] Derived const& derived_() const noexcept {
        return static_cast<Derived const&>(*this);
    }
};

}  // namespace reticolo::action
