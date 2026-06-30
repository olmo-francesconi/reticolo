#pragma once

#include <reticolo/action/detail/concepts.hpp>
#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/link_lattice.hpp>

#include <complex>
#include <cstddef>
#include <optional>

namespace reticolo::llr {

// LLR window parameters (a, E_n, delta) stay real-valued even when the field
// is complex — `scalar_of_t` is the canonical `reticolo::real_scalar_t`.
template <class T>
using scalar_of_t = real_scalar_t<T>;

// LLR-tilted action with Gaussian-penalty window. One template for both
// scalar (`Field = Lattice<T>`) and gauge (`Field = LinkLattice<T>`) base
// actions. Two modes picked at compile time from the base action's
// interface (only the scalar-field BoseGas hits mode B today):
//
// Mode A — real LLR, `Base` does NOT satisfy `HasImagPart`. The window
// constrains the base action itself:
//
//     S_LLR    = (1 + a) * S_base + (S_base - E_n)^2 / (2 * delta^2)
//     F_LLR(x) = (1 + a + (S_base - E_n) / delta^2) * F_base(x)
//
// Mode B — complex LLR, `Base` satisfies `HasImagPart`. HMC samples on the
// real (phase-quenched) part S_R and the window constrains the imaginary
// observable S_I = `base.s_imag(field)`:
//
//     S_LLR    = S_R + a * S_I + (S_I - E_n)^2 / (2 * delta^2)
//     F_LLR(x) = F_R(x) + (a + (S_I - E_n) / delta^2) * F_I(x)
//
// where F_R = base.compute_force, F_I = base.compute_force_imag. The NR/RM
// loop watches <S_I - E_n>; reconstructing ln rho(S_I) recovers the DoS of
// the imaginary part in the phase-quenched ensemble.

template <class Base, class T = typename Base::value_type, class Field = Lattice<T>>
struct WindowedAction {
    using value_type = T;
    using field_type = Field;
    using scalar_t   = scalar_of_t<T>;

    // Owned by value (not reference): each Replica carries its own base so any
    // mutable scratch on the action stays per-replica. Required for OpenMP
    // parallelism over replicas.
    Base base;
    scalar_t a = scalar_t{0};
    // NOLINTNEXTLINE(readability-identifier-naming) physics convention E_n = window centre
    scalar_t E_n   = scalar_t{0};
    scalar_t delta = scalar_t{1};

    static constexpr bool k_complex = action::HasImagPart<Base, Field>;

    [[nodiscard]] scalar_t constraint_value(Field const& l) const noexcept {
        if constexpr (k_complex) {
            return base.s_imag(l);
        } else {
            return base.s_full(l);
        }
    }

    [[nodiscard]] scalar_t s_full(Field const& l) const noexcept {
        scalar_t const q   = constraint_value(l);
        scalar_t const dq  = q - E_n;
        scalar_t const inv = scalar_t{1} / (scalar_t{2} * delta * delta);
        if constexpr (k_complex) {
            return base.s_full(l) + (a * q) + (dq * dq * inv);
        } else {
            return ((scalar_t{1} + a) * q) + (dq * dq * inv);
        }
    }

    // Raw-action cache pass-through to the base. The LLR Replica reads these
    // after a trajectory to skip the post-trajectory constraint sweep; HMC
    // snapshots and restores them across reject rollbacks. See
    // `HasCacheRollback` in `algorithm/hmc.hpp`.
    [[nodiscard]] scalar_t last_s_full() const noexcept { return base.last_s_full(); }
    void restore_last_s_full(scalar_t v) const noexcept { base.restore_last_s_full(v); }

    [[nodiscard]] scalar_t last_s_imag() const noexcept
        requires action::HasImagPart<Base, Field>
    {
        return base.last_s_imag();
    }
    void restore_last_s_imag(scalar_t v) const noexcept
        requires action::HasImagPart<Base, Field>
    {
        base.restore_last_s_imag(v);
    }

    void compute_force(Field const& l, Field& force) const noexcept {
        if constexpr (k_complex) {
            // Combined: F_R + (a + (S_I - E_n)/delta^2) * F_I.
            base.compute_force(l, force);
            scalar_t const s     = base.s_imag(l);
            scalar_t const scale = a + ((s - E_n) / (delta * delta));
            Field& imag_force    = scratch_(force.indexing());
            base.compute_force_imag(l, imag_force);
            T* const fp         = force.data();
            T const* const ip   = imag_force.data();
            std::size_t const n = flat_size(force);
            for (std::size_t i = 0; i < n; ++i) {
                fp[i] += scale * ip[i];
            }
        } else {
            // Fused base kernel when available: one neighbour pass yields
            // both S_base and the force, dropping the separate full-lattice
            // `s_full` sweep per force call.
            scalar_t s{};
            if constexpr (requires { base.s_full_and_force(l, force); }) {
                s = static_cast<scalar_t>(base.s_full_and_force(l, force));
            } else {
                base.compute_force(l, force);
                s = static_cast<scalar_t>(base.s_full(l));
            }
            scalar_t const scale = scalar_t{1} + a + ((s - E_n) / (delta * delta));
            T* const fp          = force.data();
            std::size_t const n  = flat_size(force);
            for (std::size_t i = 0; i < n; ++i) {
                fp[i] *= scale;
            }
        }
    }

    void compute_force_and_kick(Field const& l, Field& mom, scalar_t k_dt) const noexcept
        requires action::HasFusedKick<Base, Field>
    {
        if constexpr (k_complex) {
            scalar_t const s     = base.s_imag(l);
            scalar_t const scale = a + ((s - E_n) / (delta * delta));
            // If the base action exposes a fused combined-force kernel
            // (F_R + scale·F_I in one pass directly into mom), use it —
            // skips the imag_force scratch buffer and the merge pass.
            // Otherwise fall back to the two-pass form.
            if constexpr (requires {
                              base.compute_force_combined_and_kick(
                                  l, mom, scalar_t{1}, scale, k_dt);
                          }) {
                base.compute_force_combined_and_kick(l, mom, scalar_t{1}, scale, k_dt);
            } else {
                base.compute_force_and_kick(l, mom, k_dt);
                Field& imag_force = scratch_(mom.indexing());
                base.compute_force_imag(l, imag_force);
                T* const mp         = mom.data();
                T const* const ip   = imag_force.data();
                std::size_t const n = flat_size(mom);
                scalar_t const k    = k_dt * scale;
                for (std::size_t i = 0; i < n; ++i) {
                    mp[i] += k * ip[i];
                }
            }
        } else {
            if constexpr (requires(Field& f) { base.s_full_and_force(l, f); }) {
                // Fused: one neighbour pass yields S_base and the force into
                // the scratch field; a linear merge applies the scaled kick.
                // Replaces the separate full-lattice `s_full` sweep that the
                // fallback pays on every MD step.
                Field& f             = scratch_(mom.indexing());
                auto const s         = static_cast<scalar_t>(base.s_full_and_force(l, f));
                scalar_t const scale = scalar_t{1} + a + ((s - E_n) / (delta * delta));
                scalar_t const c     = k_dt * scale;
                T* const mp          = mom.data();
                T const* const fp    = f.data();
                std::size_t const n  = flat_size(mom);
                for (std::size_t i = 0; i < n; ++i) {
                    mp[i] += c * fp[i];
                }
            } else {
                scalar_t const s     = base.s_full(l);
                scalar_t const scale = scalar_t{1} + a + ((s - E_n) / (delta * delta));
                base.compute_force_and_kick(l, mom, k_dt * scale);
            }
        }
    }

    // Lazy scratch field: the complex-LLR imag-force pass (mode B) and the
    // mode-A fused force+merge path stage their force here. Allocated on
    // first force call and reused; avoids the per-MD-step malloc/free that
    // used to dominate the bose_gas_llr force path. `mutable` because the
    // force methods are const.
    mutable std::optional<Field> scratch_storage{};

private:
    [[nodiscard]] Field& scratch_(std::shared_ptr<Indexing const> idx) const noexcept {
        if (!scratch_storage) {
            scratch_storage.emplace(std::move(idx));
        }
        return *scratch_storage;
    }
};

}  // namespace reticolo::llr
