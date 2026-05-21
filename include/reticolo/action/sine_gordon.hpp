#pragma once

#include <reticolo/action/detail/helpers.hpp>
#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/math/vec_libm.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

namespace reticolo::action {

// Sine-Gordon scalar action: same NN hopping + on-site mass term as Phi4,
// plus a periodic potential -alpha * cos(phi):
//
//   S = sum_x [ -2 kappa phi(x) sum_{mu>0} phi(x+mu)
//               + phi(x)^2
//               - alpha * cos(phi(x)) ]

template <class T = double>
struct SineGordon {
    using value_type = T;

    T kappa = T{0};
    T alpha = T{0};

    void describe(log::Entry& e) const {
        e.line("SineGordon<{}>", scalar_name<T>());
        e.param("κ={:.3f}", kappa);
        e.param("α={:.3f}", alpha);
    }

    [[nodiscard]] T s_local(Lattice<T> const& l, Site x) const noexcept {
        T const phi  = l[x];
        T const nbrs = nn_neighbour_sum(l, x);
        return (T{-2} * kappa * phi * nbrs) + (phi * phi) - (alpha * std::cos(phi));
    }

    [[nodiscard]] T ds_local(Lattice<T> const& l, Site x, T new_v) const noexcept {
        return ds_local_from_nbrs(l[x], new_v, nn_neighbour_sum(l, x));
    }

    [[nodiscard]] T ds_local_from_nbrs(T phi, T new_v, T nbrs) const noexcept {
        T const hop  = T{-2} * kappa * (new_v - phi) * nbrs;
        T const mass = (new_v * new_v) - (phi * phi);
        T const pot  = -alpha * (std::cos(new_v) - std::cos(phi));
        return hop + mass + pot;
    }

    [[nodiscard]] T s_full(Lattice<T> const& l) const noexcept {
        T s{};
        if constexpr (std::is_same_v<T, double>) {
            std::size_t const n = l.nsites();
            ensure_scratch_(n);
            math::cos_batch(scratch.data(), l.data(), n);
            T const k         = kappa;
            T const alp       = alpha;
            T const* const cs = scratch.data();
            // Walk sites in the same row-major order as reduce_fwd; we
            // accumulate -alpha*cos contributions via the scratch index
            // since reduce_fwd's body has no i.
            T cos_sum = T{0};
            for (std::size_t i = 0; i < n; ++i) {
                cos_sum += cs[i];
            }
            T const hopping = detail::reduce_fwd<T>(
                l, [k](T phi, T fwd_sum) { return (T{-2} * k * phi * fwd_sum) + (phi * phi); });
            s = hopping - (alp * cos_sum);
        } else {
            T const k   = kappa;
            T const alp = alpha;
            s           = detail::reduce_fwd<T>(l, [k, alp](T phi, T fwd_sum) {
                return (T{-2} * k * phi * fwd_sum) + (phi * phi) - (alp * std::cos(phi));
            });
        }
        last_s_full_ = s;
        return s;
    }

    [[nodiscard]] T last_s_full() const noexcept { return last_s_full_; }
    void restore_last_s_full(T v) const noexcept { last_s_full_ = v; }

    // force(x) = -dS/dphi(x)
    //         = 2 kappa sum_{mu, +-} phi(x+mu) - 2 phi(x) - alpha sin(phi(x))
    void compute_force(Lattice<T> const& l, Lattice<T>& force) const noexcept {
        T const k    = kappa;
        T const alp  = alpha;
        T* const out = force.data();
        if constexpr (std::is_same_v<T, double>) {
            std::size_t const n = l.nsites();
            ensure_scratch_(n);
            math::sin_batch(scratch.data(), l.data(), n);
            T const* const sp = scratch.data();
            detail::visit_nn<T>(l, [k, alp, out, sp](std::size_t i, T phi, T nbrs) {
                out[i] = (T{2} * k * nbrs) - (T{2} * phi) - (alp * sp[i]);
            });
        } else {
            detail::visit_nn<T>(l, [k, alp, out](std::size_t i, T phi, T nbrs) {
                out[i] = (T{2} * k * nbrs) - (T{2} * phi) - (alp * std::sin(phi));
            });
        }
    }

    // Fused force + leapfrog kick.
    void compute_force_and_kick(Lattice<T> const& l, Lattice<T>& mom, T k_dt) const noexcept {
        T const k   = kappa;
        T const alp = alpha;
        T* const m  = mom.data();
        if constexpr (std::is_same_v<T, double>) {
            std::size_t const n = l.nsites();
            ensure_scratch_(n);
            math::sin_batch(scratch.data(), l.data(), n);
            T const* const sp = scratch.data();
            detail::visit_nn<T>(l, [k, alp, k_dt, m, sp](std::size_t i, T phi, T nbrs) {
                T const F = (T{2} * k * nbrs) - (T{2} * phi) - (alp * sp[i]);
                m[i] += k_dt * F;
            });
        } else {
            detail::visit_nn<T>(l, [k, alp, k_dt, m](std::size_t i, T phi, T nbrs) {
                T const F = (T{2} * k * nbrs) - (T{2} * phi) - (alp * std::sin(phi));
                m[i] += k_dt * F;
            });
        }
    }

    // Lazy per-site sin/cos scratch — populated by vector libm at the start
    // of each force / s_full call and reused across MD steps. `mutable` so
    // the public force/s_full methods can stay const.
    mutable std::vector<T> scratch{};

    mutable T last_s_full_ = std::numeric_limits<T>::quiet_NaN();

private:
    void ensure_scratch_(std::size_t n) const noexcept {
        if (scratch.size() < n) {
            scratch.resize(n);
        }
    }
};

}  // namespace reticolo::action
